"""4Fe-4S 72q GPU recovery on ROQUO (GB200) from a saved pool. RHF by default.

Runs the SBD diagonalizer on GPU (diag-gpu / diag-gpu_uhf, built with nvc++ -cuda) inside a GB200
Slurm allocation. Supports single-node and multi-node (up to 64+ GPUs) via MPI grid.

Single-node: HPC_TARGET="slurm" (default), each solve is a separate sbatch job.
Multi-node: HPC_TARGET="local", runs in-allocation to avoid occ handoff across nodes.

    FE4S4_POOL=<npz> python run_fe4s4_gpu_roquo.py

Env (set by run_fe4s4_gpu_roquo.sh or run_fe4s4_64gpu_d5e9.sh):
    FE4S4_METHOD         : rhf (default) | uhf  -> selects diag-gpu vs diag-gpu_uhf
    FE4S4_POOL           : persisted raw_samples npz path(s)
    FE4S4_SQD_DIM        : subspace dim (default 3e8)
    FE4S4_RECSTEPS/FE4S4_NBATCH : recovery passes / K-batches (default 5 / 5)
    ROQUO_OMPTHREADS     : CPU threads for the host side (default 140)
    FE4S4_ADET_COMM_SIZE : MPI grid a-dimension (default 1; for 64-GPU: 8)
    FE4S4_BDET_COMM_SIZE : MPI grid b-dimension (default 1; for 64-GPU: 8)
    FE4S4_HPC_TARGET     : "slurm" (single-node) or "local" (multi-node in-allocation)
    FE4S4_NCCL_PRECMD    : NCCL env exports for multi-node (H2+H3 fixes)
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import fe2s2_common as C

METHOD = os.environ.get("FE4S4_METHOD", "rhf")
SQD_DIM = int(float(os.environ.get("FE4S4_SQD_DIM", "300000000")))
RECSTEPS = int(os.environ.get("FE4S4_RECSTEPS", "5"))
# n_batches=1: the ConcurrentTaskRunner launches all batches at once, and N concurrent diag-gpu
# procs on 1 GPU multiply GPU memory -> OOM (job 2305). One batch per recovery step fits and still
# gives the full recovery-effect trajectory. Bump only with more GPUs / a memory-serialized runner.
NBATCH = int(os.environ.get("FE4S4_NBATCH", "1"))
OMP = int(os.environ.get("ROQUO_OMPTHREADS", "140"))
# Multi-node MPI grid (a x b = total ranks). Default a=1, b=1 (single GPU).
# For multi-node: a>=2 avoids the empty-range OMP wedge. E.g. 64-GPU: a=8, b=8.
ADET_COMM_SIZE = int(os.environ.get("FE4S4_ADET_COMM_SIZE", "1"))
BDET_COMM_SIZE = int(os.environ.get("FE4S4_BDET_COMM_SIZE", "1"))
TASK_COMM_SIZE = int(os.environ.get("FE4S4_TASK_COMM_SIZE", "1"))
_TOTAL_RANKS = ADET_COMM_SIZE * BDET_COMM_SIZE * TASK_COMM_SIZE
# Davidson ITERATION CAP (--iteration). Was hardcoded to 5, which is why steps ended on the cap
# rather than on --tolerance: at d=1e11 the 5th iteration still gained 4.5 mHa with tol=0.043
# against a 0.01 target. Default kept at 5 so existing launchers are unchanged.
DAVIDSON_ITERATION = os.environ.get("FE4S4_DAVIDSON_ITERATION", "5")

# Extra flags appended verbatim to the solver command line, comma-separated. Chiefly for
# warm-starting Davidson from the previous step's CI vector:
#   FE4S4_SOLVER_USER_ARGS="--savename,wf_warm.bin,--loadname,wf_warm.bin"
SOLVER_USER_ARGS = os.environ.get("FE4S4_SOLVER_USER_ARGS", "").strip()

# Davidson subspace size (--block, "max_nb" in davidson_thrust.h). Real GPU memory driver:
# per-rank Davidson working set = (2*block+1) * W_size * 8 bytes, W_size = (dets_per_spin/a)
# * (dets_per_spin/b) (confirmed by reading sbd_mpi/include/sbd/chemistry/tpb/sbdiag.h +
# davidson_thrust.h directly, not inferred). Default 20 preserves the proven d3e10 behavior
# exactly; d4e10 at the same grid has 1.33x d3e10's per-rank W (200,000^2 vs 173,205^2 dets),
# so block=20 there needs ~91GB/rank in Davidson vectors alone (SLURM sacct confirmed ~115GB/
# rank average on the crashed job) vs d3e10's ~68GB/rank -- block=15 brings d4e10 back down to
# d3e10's own proven footprint without touching the grid/node count at all.
DAVIDSON_BLOCK = int(os.environ.get("FE4S4_DAVIDSON_BLOCK", "20"))
# HPC target: "slurm" (per-solve sbatch) or "local" (in-allocation, for multi-node)
HPC_TARGET = os.environ.get("FE4S4_HPC_TARGET", "slurm")
# NCCL pre-commands (H2+H3 multi-node fixes from qcsc-multinode-fix)
NCCL_PRECMD = os.environ.get("FE4S4_NCCL_PRECMD", "")
# Classically seed the SQD subspace with HF excitations (QSCI+SD). Generic knob, any molecule:
#   0 = off, 1 = singles only, 2 = doubles only, 3 = singles + doubles.
# The seed is capped to fit the requested sqd_dim, so it runs at 9M etc. (see sqd.py _merge_with_seed).
SEED_CISD = int(os.environ.get("SEED_CISD", "0"))
# Fraction of the subspace budget the CISD seed may take (<1.0 reserves room for the sample's
# higher-excitation dets on top of S+D -- "partial-CISD + heavy mixing", the lever that can beat
# the pure-CISD S+D energy ceiling).
SEED_CISD_FRAC = float(os.environ.get("SEED_CISD_FRAC", "1.0"))
# Orbital optimization: DO_RDM != 0 makes the solver write per-block RDMs and enables OO between DE
# trials. OO only acts across DE iterations, so ITERATIONS must be >= 2 (and NUM_WALKERS >= 4 for
# the DE mutation, per differential_evolution_trial). Defaults keep the prior single-pass behavior.
DO_RDM = int(os.environ.get("DO_RDM", "0"))
ITERATIONS = int(os.environ.get("ITERATIONS", "1"))
NUM_WALKERS = int(os.environ.get("NUM_WALKERS", "1"))
# OO stopping controls: tight trust radius + few L-BFGS iters/trial => small orbital steps so the
# gradient can decrease gradually toward oo_grad_tol without over-rotating on the fixed RDMs.
OO_TRUST_RADIUS = float(os.environ.get("OO_TRUST_RADIUS", "0.5"))
OO_MAXITER = int(os.environ.get("OO_MAXITER", "300"))

# HCI (heat-bath-CI-flavored) importance booster. OPT-IN, default off (matches every other
# opt-in booster in sqd.py: FlowParameters default hci_boost=0).
HCI_BOOST = int(os.environ.get("HCI_BOOST", "0"))
HCI_BOOST_MAX = int(os.environ.get("HCI_BOOST_MAX", "20000"))
HCI_BOOST_NCORE = int(os.environ.get("HCI_BOOST_NCORE", "200"))

# Carryover type, hardcoded to 1 in every driver invocation until now. Made env-overridable to
# test types 2/3 (SinglesExtendHalfdets). See sbdiag.h:695-738.
CARRYOVER_TYPE = os.environ.get("CARRYOVER_TYPE", "1")
# OO_RESOLVE_RDMS=1: rigorous self-consistent path -- re-diagonalize the fixed CI subspace in the
# rotated basis (in-process solve_fermion) each orbital step so the gradient is the true MCSCF
# gradient and the energy stays variational (>= FCI). Fast (in-process); the correct convergent mode.
OO_RESOLVE_RDMS = int(os.environ.get("OO_RESOLVE_RDMS", "0"))
# Truncated subspace size for the in-process OO re-solve. Must be << full sqd_dim to be fast: the
# orbital rotation is set by the dominant low-excitation dets, so ~sqrt(this) dets/spin suffices.
# Default 250k (~500 dets/spin -> ms-scale re-solve). Keep small for speed / 200q scaling.
OO_RESOLVE_MAXDIM = int(float(os.environ.get("OO_RESOLVE_MAXDIM", "250000")))

# Node-local Prefect DB (Lustre home is slow/locks); keep the persisted storage on Lustre.
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("PREFECT_TELEMETRY_ENABLED", "false")
os.environ.setdefault("SBD_TASK_RUNNER", "concurrent")
os.environ.setdefault("PREFECT_SERVER_DATABASE_TIMEOUT", "120")
os.environ.setdefault("PREFECT_SERVER_DATABASE_CONNECTION_TIMEOUT", "60")
os.environ.setdefault("PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS", "120")
# The client's HTTP request timeout defaults to 60s. During a long child-solve poll the single
# ephemeral server can take >60s to answer an API call (e.g. block load) -> httpx.ReadTimeout that
# kills the flow on step 1 of a multi-step recovery. Bump it way up.
os.environ.setdefault("PREFECT_API_REQUEST_TIMEOUT", "600")
os.environ.setdefault("OMP_NUM_THREADS", str(OMP))
# launcher + mpi options via ENV (comma-split by create_blocks' env_csv). argparse --mpi-options
# rejects '-'-prefixed tokens like --gpu-bind=closest, so pass them here instead. srun = PMIx.
# The nvhpc-openmpi (ompi4) build only supports mpirun -- srun --mpi=pmix aborts before
# MPI_Init on it (ROQUO MPI manual sec 4). The standalone hpcx/2.50 (ompi5) build is the
# opposite: srun --mpi=pmix is its native launch path (project memory qcsc-multinode-fix:
# "srun --mpi=pmix --gpu-bind=closest (PMIx; mpirun nested fails)"). Mixing these up is
# exactly the bug found 2026-07-31 (srun rejecting --map-by on the ompi4 build) -- so the
# launcher AND its options must be selected together per stack, not just the launcher name.
if os.environ.get("FE4S4_TALKATIVE", "0") == "1" or os.environ.get("FE4S4_CONTAINER_TALKATIVE", "0") == "1" or os.environ.get("FE4S4_CONTAINER_TALKATIVE_DEBUGTIMING", "0") == "1":
    os.environ["SBD_LAUNCHER"] = "srun"
    # ROQUO support's rank-to-node mapping suggestion: our app uses a 2D adet x bdet grid
    # while their nccl-tests reproducer uses a flat layout; --distribution=block:block forces
    # a different rank<->node assignment than srun's default (cyclic across the allocation)
    # to test whether ring formation order (which depends on rank ordering) matters.
    os.environ["SBD_MPI_OPTIONS"] = "--mpi=pmix,--gpu-bind=closest,--distribution=block:block"
elif os.environ.get("FE4S4_HPCX250", "0") == "1":
    os.environ["SBD_LAUNCHER"] = "srun"
    # NOTE: do NOT add --propagate=NOFILE here -- tested 2026-08-01 and it broke IB RDMA
    # memory registration ("ibv_reg_mr ... Bad address", ucp_mm.c failed to register host
    # buffer on mlx5_0) on a 16-node/64-rank run within 8 minutes. --propagate=<LIMIT>
    # replaces srun's default resource-limit propagation wholesale, apparently dragging in a
    # restrictive RLIMIT_MEMLOCK from the submitting shell that compute nodes don't otherwise
    # see. The plain `ulimit -n 65536` in the launch script (no --propagate) is harmless and
    # sufficient to test the FD-exhaustion hypothesis without this side effect.
    # DO NOT add --cpu-bind / --cpus-per-task here. Tried 2026-08-29 and it BREAKS the run:
    #   NCCL error invalid usage at framework/nccl_utility.h:75 (ncclCommInitRank), 38s in.
    # Reason: main.cc picks its GPU with  myDevice = mpi_rank % numDevices  (line ~40), relying on
    # every rank seeing ALL 4 GPUs. Measured CUDA_VISIBLE_DEVICES per rank:
    #   --gpu-bind=closest alone            -> 0,1,2,3 for every rank -> rank%4 = 0,1,2,3  (correct)
    #   + --cpu-bind=cores --cpus-per-task=36 -> 0,1 for every rank   -> rank%2 = 0,1,0,1  (2 ranks
    #                                             per GPU -> NCCL rejects the duplicate device)
    # Pinning CPUs narrows what SLURM considers the closest GPUs. Any CPU-binding change here
    # would need main.cc to derive its device from SLURM_LOCALID instead of mpi_rank%numDevices.
    os.environ["SBD_MPI_OPTIONS"] = "--mpi=pmix,--gpu-bind=closest"
elif os.environ.get("FE4S4_NVHPCMPI", "0") == "1" or os.environ.get("FE4S4_CC100", "0") == "1":
    os.environ["SBD_LAUNCHER"] = "mpirun"
    # --map-by ppr:4:node --bind-to none is the ROQUO MPI manual's own tested mpirun recipe
    # (sec 7.1) for whole-node (gpu:4) CUDA-aware MPI.
    os.environ["SBD_MPI_OPTIONS"] = "--map-by,ppr:4:node,--bind-to,none"
else:
    os.environ["SBD_LAUNCHER"] = "srun"
    # BUG FIX (2026-09-02): plain `srun --gpu-bind=closest` with no explicit rank count inherits
    # the OUTER allocation's full task count. For a single-rank (a=1 x b=1) job this is a real
    # correctness bug, not just a performance one -- verified directly on ROQUO:
    #   outer --ntasks=1 -> inner srun fails outright: "No step_layout given for pending step"
    #     (the parent Python subprocess then hangs forever waiting on a step that never starts --
    #     0% GPU util, ~100% CPU, indefinitely; cost 6 wasted probe jobs before being traced).
    #   outer --ntasks=4, inner srun with NO -n -> FOUR duplicate diag-gpu_uhf processes launch
    #     against the same a=1xb=1 input, all pinned to the same GPU (measured: 45 GB combined
    #     memory vs one rank's correct ~11 GB) -- silently wrong, not even a crash.
    #   outer --ntasks=4, inner `srun --gpu-bind=closest -n1 --overlap` -> exactly one correct
    #     process, verified end to end (real solve, correct timing, correct output).
    # _TOTAL_RANKS is computed at module top (ADET_COMM_SIZE*BDET_COMM_SIZE*TASK_COMM_SIZE, all
    # defined before this point) specifically so it is genuinely in scope here. Only the
    # total_ranks==1 case is touched; the multi-node path (total_ranks>1, e.g. the working
    # 144-rank d3e10 production runs) stays byte-identical, per the standing "DO NOT add
    # --cpu-bind/--cpus-per-task" warning on the FE4S4_HPCX250 branch above: per-rank resource
    # narrowing on the multi-node path has previously broken NCCL's ncclCommInitRank by changing
    # which GPUs each rank can see, and -n/--overlap on a >1-rank srun risks the same class of harm.
    if _TOTAL_RANKS == 1:
        os.environ["SBD_MPI_OPTIONS"] = "--gpu-bind=closest,-n1,--overlap"
    else:
        os.environ["SBD_MPI_OPTIONS"] = "--gpu-bind=closest"


def _pool_paths() -> list[str]:
    raw = os.environ.get("FE4S4_POOL", "").strip()
    if not raw:
        raise SystemExit("Set FE4S4_POOL to the persisted raw_samples npz path(s).")
    return [p if p.startswith("file://") else f"file://{p}"
            for p in raw.replace(",", " ").split()]


def main() -> None:
    if C.MOLECULE not in C.MOLECULES:
        raise SystemExit(f"unknown FE_MOL={C.MOLECULE}")
    pools = _pool_paths()
    p = C.sbd_paths()
    # Single-GPU (no MPI) binaries only work at total_ranks==1. Multi-node needs the -mpi variant;
    # default to the safe-allreduce build (fixes the Mpi2dSlide::Sync rendezvous hang) whenever
    # ADET/BDET comm sizes actually span more than one rank. FE4S4_SAFE_ALLREDUCE=0 falls back to
    # the plain -mpi binary for A/B comparison.
    total_ranks_for_binary = ADET_COMM_SIZE * BDET_COMM_SIZE
    if total_ranks_for_binary > 1:
        if os.environ.get("FE4S4_TALKATIVE", "0") == "1":
            # DIAGNOSTIC ONLY: identical build to -mpi-hpcx250, but wrapped so each rank
            # writes its own full, untruncated, real-time-flushed log file directly to disk
            # (runs/nccl_capture_d1e10/rank_<N>.log), bypassing Prefect's in-memory pipe
            # capture + MAX_LOG_SIZE=10_000-char-truncated logger.error() in
            # packages/qcsc-prefect-executor/.../local/run.py -- which silently discards
            # NCCL_DEBUG output on success (truncation) and entirely on a killed/timed-out
            # run (buffered pipe data never surfacing). See docs/2026-08-02 root-cause note.
            mpi_suffix = "-mpi-hpcx250-talkative"
        elif os.environ.get("FE4S4_PRISTINE_TALKATIVE", "0") == "1":
            # Decisive isolation test: helper.h/nccl_utility.h reverted to PRISTINE (no cr/an
            # hoist, no extra diagnostics) via git stash, main.cc restored (build script gate
            # requires the CUDA-before-MPI_Init fix present), built TODAY. If this ALSO fails
            # with the same rank=0/1 symptom, it is conclusively an environment/toolchain drift
            # issue since July 31, unrelated to ANY of our source changes.
            mpi_suffix = "-mpi-hpcx250-pristine-talkative"
        elif os.environ.get("FE4S4_FRESHTEST_TALKATIVE", "0") == "1":
            # Sanity check: a FRESH rebuild of the EXACT same working recipe (build_sbd_mpi_uhf_
            # hpcx250.sh, unmodified), rebuilt today rather than July 31, to test whether ANY
            # fresh rebuild fails the same way genexcfix did (environment/toolchain drift) even
            # with zero source patches applied. Talkative-wrapped for full per-rank log capture.
            mpi_suffix = "-mpi-hpcx250-freshtest-talkative"
        elif os.environ.get("FE4S4_OLDMODULE_TALKATIVE", "0") == "1":
            # Last cheap isolation test before the container-build path: binary built with
            # module use --prepend pointing at the pre-2026-08-02 hpcx/2.50.lua backup (i.e.
            # WITHOUT the admin-added UCX_MAX_RNDV_RAILS auto-tune block), everything else
            # identical to -freshtest (fresh rebuild of the unmodified working recipe). If
            # this STILL hits rank=0/1 on MPI_Init_thread, the module change is conclusively
            # ruled out as the cause -- confirms the bug is unrelated to that one admin diff.
            mpi_suffix = "-mpi-oldmodule-talkative"
        elif os.environ.get("FE4S4_CONTAINER_TALKATIVE_DEBUGTIMING", "0") == "1":
            # One-off profiling build: identical to FE4S4_CONTAINER_TALKATIVE but points at a
            # binary compiled with -DSBD_DEBUG_MULT -DSBD_DEBUG_DAVIDSON (separate binary name,
            # never overwrites the production diag-gpu_uhf-mpi-container). Added 2026-09-03 to
            # measure real time_mult/time_comm/time_slid split per mult.run() call, to check
            # whether the un-batched per-pair InnerProduct Allreduces in davidson_thrust.h are a
            # real bottleneck at 144+ rank scale. Not used by any production launcher.
            mpi_suffix = "-mpi-container-talkative-debugtiming"
        elif os.environ.get("FE4S4_CONTAINER_TALKATIVE", "0") == "1":
            # Multi-node test of the container-isolated build (nvhpc-26.5-devel.sif, own
            # bundled hpcx-2.50, ZERO ROQUO shared-module involvement during compilation).
            # Single-rank sanity already passed (real MPI_Init_thread, real GPU visible via
            # --nv, reached real FCIDUMP-loading application logic) -- this is the decisive
            # multi-rank test: if THIS also hits rank=0/1 across ranks, the bug is proven to
            # be unrelated to ROQUO's shared module filesystem entirely (something in the
            # SLURM/PMIx/UCX fabric bootstrap itself, or genuinely compile-date-dependent
            # in a way neither environment change explains).
            mpi_suffix = "-mpi-container-talkative"
        elif os.environ.get("FE4S4_GENEXCFIX_TALKATIVE", "0") == "1":
            # Same as FE4S4_GENEXCFIX but wrapped for full per-rank stdout/stderr capture to
            # disk (runs/nccl_capture_genexcfix_test/rank_<N>.log) -- two prior live
            # multi-node genexcfix attempts failed without a captured root cause (Prefect's
            # log pipe silently truncates/discards native stderr on a killed/failed run).
            mpi_suffix = "-mpi-hpcx250-genexcfix-talkative"
        elif os.environ.get("FE4S4_GENEXCFIX", "0") == "1":
            # Validation build for the GenerateExcitation cr/an-heap-allocation fix (helper.h,
            # 2026-08-04): cr/an were reconstructed on the heap every outer-loop iteration,
            # which under -gpu=mem:unified livelocks NVHPC's managed-memory pool allocator
            # lock under concurrent OMP threads (ROQUO Issue #73 root cause). Fix hoists them
            # outside the parallelized loop (one alloc/thread, reused via .clear()). This
            # build lets a real recovery step run at full OMP_NUM_THREADS (no workaround)
            # to confirm the fix, not just the OMP_NUM_THREADS=1 mitigation.
            mpi_suffix = "-mpi-hpcx250-genexcfix"
        elif os.environ.get("FE4S4_CONTROL_UNPATCHED", "0") == "1":
            # Control for the GenerateExcitation fix isolation test: fresh rebuild of the
            # CURRENT checkout's UNPATCHED helper.h (everything else identical to
            # -genexcfix's build, including the pre-existing uncommitted main.cc/
            # nccl_utility.h changes). Used to prove the "MPI Size of twister" crash seen
            # on -genexcfix is a property of any fresh rebuild, not the helper.h patch.
            mpi_suffix = "-mpi-hpcx250-control-unpatched"
        elif os.environ.get("FE4S4_HPCX250", "0") == "1":
            # Consistent build+run against the standalone hpcx/2.50 module (not nvhpc's
            # bundled hpcx, not nvhpc-openmpi/26.5) -- testing whether this stack's NCCL
            # bootstrap avoids the ring-allgather deadlock hit at 64-100 ranks on nvhpc-openmpi.
            mpi_suffix = "-mpi-hpcx250"
        elif os.environ.get("FE4S4_CC100", "0") == "1":
            # A/B test: same build as -mpi-nvhpcmpi but SBD_GPU_ARCH=cc100 (Blackwell) instead
            # of the preset's default cc90 (Hopper) -- GB200 reports compute_cap=10.0 via
            # nvidia-smi, so cc90-targeted device code is architecture-mismatched on ROQUO.
            mpi_suffix = "-mpi-nvhpcmpi-cc100"
        elif os.environ.get("FE4S4_NVHPCMPI", "0") == "1":
            mpi_suffix = "-mpi-nvhpcmpi"
        elif os.environ.get("FE4S4_PR79", "0") == "1":
            mpi_suffix = "-mpi-pr79"
        else:
            mpi_suffix = "-mpi-safeallreduce" if os.environ.get("FE4S4_SAFE_ALLREDUCE", "1") == "1" else "-mpi"
    else:
        mpi_suffix = ""

    # Explicit escape hatch, applied AFTER the chain above so it composes with any selection.
    # Set FE4S4_BINARY_SUFFIX to override the binary suffix outright, e.g.
    #     FE4S4_BINARY_SUFFIX=-mpi-upstream-main-talkative
    # which selects the wrapper that writes each rank's stdout/stderr to
    # runs/solver_logs/rank_<N>.log (FE4S4_TALKATIVE_LOGDIR to relocate).
    #
    # This exists because the Prefect local runtime launches the solver with stdout=PIPE +
    # `await proc.communicate()` (adapters/.../fugaku/runtime.py:39-42), which reads until EOF --
    # so the solver's per-iteration "Davidson iteration <it>.<ib> <E...>" lines
    # (davidson_thrust.h:579) are invisible for the entire run. On the 9-hour d=1e11 step this
    # left no way to tell which Davidson iteration was in flight or what the energy was; gdb
    # could not recover them either, since -Kfast optimizes `it` and `E` out of the DWARF.
    _suffix_override = os.environ.get("FE4S4_BINARY_SUFFIX", "")
    if _suffix_override:
        print(f"[launcher] FE4S4_BINARY_SUFFIX overrides {mpi_suffix!r} -> {_suffix_override!r}")
        mpi_suffix = _suffix_override

    diag_gpu = os.path.join(p["diag"], f"diag-gpu{mpi_suffix}")
    diag_gpu_uhf = os.path.join(p["diag"], f"diag-gpu_uhf{mpi_suffix}")

    base = C.run_dir(METHOD)          # runs/fe4s4_<method>/ (Lustre)
    (base / "recover").mkdir(parents=True, exist_ok=True)
    # Per-run isolation: PREFECT_HOME/work_dir MUST be unique per orchestrator, otherwise two
    # concurrent runs share one ephemeral SQLite DB and corrupt it ("sqlite3.DatabaseError: file is
    # not a database" -> create_blocks fails). Tag by SLURM job id (falls back to seed+pid) so
    # runs launched in parallel (e.g. seed levels 1/2/3 + an XL run) never collide. Keep everything
    # on Lustre (base): the diag-gpu solve FAILS when work_dir is on $SLURM_SCRATCH (local NVMe).
    run_tag = os.environ.get("SLURM_JOB_ID") or f"s{SEED_CISD}_{os.getpid()}"
    prefect_home = base / f"prefect_home_gpu_{run_tag}"
    work_dir = base / f"work_gpu_recover_{run_tag}"
    storage_dir = prefect_home / "storage"
    prefect_home.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    storage_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PREFECT_HOME"] = str(prefect_home)
    os.environ["PREFECT_LOCAL_STORAGE_PATH"] = str(storage_dir)

    # Solver block: SLURM target + GPU mode (the manual-blessed, stable Prefect pattern, like
    # Fugaku/Miyabi). Each recovery-step solve becomes its own sbatch GPU job the orchestrator
    # submits + polls (sacct) -- the orchestrator never runs diag-gpu in-process, so the ephemeral
    # Prefect server no longer stalls (which killed the `local` target under multi-step load).
    # The generated .slurm carries: --gres=gpu:1, --account, module load cuda/13.2 hpcx/2.50, and
    # launches via `srun --gpu-bind=closest diag-gpu` (PMIx, per ROQUO manual sec 4.5). 1 GPU/solve
    # (multi-rank OOMs); --ompthreads 36 = the qtr-plan cores.
    # For multi-node: HPC_TARGET="local" runs in-allocation (all steps share one work dir, no handoff).
    total_ranks = ADET_COMM_SIZE * BDET_COMM_SIZE * TASK_COMM_SIZE
    create_blocks_cmd = [
        p["python"], os.path.join(p["sbd"], "create_blocks.py"),
        "--hpc-target", HPC_TARGET, "--method", METHOD, "--solver-mode", "gpu",
        "--slurm-account", os.environ.get("ROQUO_ACCOUNT", "q0000219"),
        "--slurm-partition", os.environ.get("ROQUO_PARTITION", "roquo"),
        "--slurm-gres", f"gpu:{min(4, total_ranks)}",  # max 4 GPUs per node
        # launcher=srun + mpi-options come from SBD_LAUNCHER/SBD_MPI_OPTIONS env (set above).
        "--num-nodes", str(max(1, (total_ranks + 3) // 4)),  # ceil(ranks / 4)
        "--mpiprocs", str(total_ranks),
        "--ompthreads", "36",
        "--walltime", "12:00:00",
        "--carryover-ratio", "0.5", "--carryover-type", CARRYOVER_TYPE,
        "--do-rdm", str(DO_RDM),
        # Raised from 43200 (12h): both d3e10 and d4e10 trajectories have hit real per-step
        # times exceeding 12h (d4e10 step 3 was killed mid-solve by this exact timeout,
        # confirmed via the "Local command timed out after 43200.0 seconds: srun" error) --
        # 20h gives real headroom above the observed ceiling while still catching a genuinely
        # hung solve eventually, rather than disabling the timeout outright.
        "--solver-timeout-seconds", "72000",
        "--work-dir", str(work_dir),
        "--saved-samples", ",".join(pools),
        "--iteration", DAVIDSON_ITERATION, "--block", str(DAVIDSON_BLOCK),
        "--sbd-executable", diag_gpu, "--sbd-executable-uhf", diag_gpu_uhf,
    ]
    # Pass MPI grid comm sizes (multi-node only; single-GPU defaults to a=1,b=1)
    if SOLVER_USER_ARGS:
        # MUST use --flag=value, not ["--flag", value]: the value itself starts with \"--\"
        # (e.g. \"--savename,wf_warm.bin\") and argparse then treats it as another option, failing
        # with \"expected one argument\" (exit 2). Measured: job 122132 died in 3 s this way.
        create_blocks_cmd.append(f"--solver-user-args={SOLVER_USER_ARGS}")
        print(f"[launcher] solver user args: {SOLVER_USER_ARGS}")
    if ADET_COMM_SIZE > 1 or BDET_COMM_SIZE > 1:
        create_blocks_cmd.extend(["--adet-comm-size", str(ADET_COMM_SIZE)])
        create_blocks_cmd.extend(["--bdet-comm-size", str(BDET_COMM_SIZE)])
    if TASK_COMM_SIZE > 1:
        create_blocks_cmd.extend(["--task-comm-size", str(TASK_COMM_SIZE)])
    # Inject NCCL pre-commands for multi-node (H2+H3 fixes)
    # LocalRuntime rejects modules/pre_commands outright ("Local execution does not
    # support modules or pre_commands") -- for HPC_TARGET=local the orchestrator process
    # already inherits the NCCL env vars from this SBATCH script's own shell, so skip it.
    if HPC_TARGET != "local":
        create_blocks_cmd.extend(["--modules", "cuda/13.2", "hpcx/2.50"])
    if NCCL_PRECMD and HPC_TARGET != "local":
        create_blocks_cmd.extend(["--pre-commands", NCCL_PRECMD])

    # create_blocks.py shells out to a bare `prefect` CLI (to set a Variable) -- put the
    # interpreter's own bin/ dir on PATH so that resolves to the same venv's `prefect`.
    cb_env = {**os.environ, "PATH": f"{os.path.dirname(p['python'])}{os.pathsep}{os.environ['PATH']}"}
    subprocess.run(create_blocks_cmd, cwd=p["sbd"], check=True, env=cb_env)

    from sbd.flow_params import CircuitParameters, DEParameters, FlowParameters
    from sbd.main import riken_sqd_de

    t0 = time.perf_counter()
    params = FlowParameters(
        fcidump=C.FCIDUMP,
        sqd_dim=SQD_DIM,
        n_recovery_steps=RECSTEPS,
        n_batches=NBATCH,
        seed_cisd=SEED_CISD,
        seed_budget_frac=SEED_CISD_FRAC,
        oo_trust_radius=OO_TRUST_RADIUS,
        oo_maxiter=OO_MAXITER,
        oo_resolve_rdms=bool(OO_RESOLVE_RDMS),
        oo_resolve_maxdim=OO_RESOLVE_MAXDIM,
        hci_boost=HCI_BOOST,
        hci_boost_max=HCI_BOOST_MAX,
        hci_boost_ncore=HCI_BOOST_NCORE,
        quantum_source="saved",
        solver_block_ref="sbd_solver_job/davidson-solver-gpu",
        circ_params=CircuitParameters(n_lucj_layers=1),
        de_params=DEParameters(num_walkers=NUM_WALKERS, iterations=ITERATIONS,
                               randomization_factor=0.2, fxc=0.5),
    )
    best = riken_sqd_de(params)
    dt = time.perf_counter() - t0

    trace = _read_recovery_trace()
    out = {
        "method": METHOD, "molecule": "fe4s4", "cluster": "roquo-gpu",
        "sqd_dim": SQD_DIM, "n_batches": NBATCH, "max_recovery": RECSTEPS,
        "seed_cisd": SEED_CISD, "seed_budget_frac": SEED_CISD_FRAC,
        "do_rdm": DO_RDM, "iterations": ITERATIONS, "num_walkers": NUM_WALKERS,
        "best_energy": best, "elapsed_s": dt, "pools": pools, "recovery_trace": trace,
    }
    seed_tag = (f"_seed{SEED_CISD}" + (f"f{SEED_CISD_FRAC:g}" if SEED_CISD_FRAC < 1.0 else "")) if SEED_CISD else ""
    oo_tag = f"_oo_it{ITERATIONS}w{NUM_WALKERS}" if DO_RDM else ""
    fn = f"recover_{METHOD}_roquogpu_rec{RECSTEPS}_k{NBATCH}_dim{SQD_DIM}{seed_tag}{oo_tag}.json"
    (base / "recover" / fn).write_text(json.dumps(out, indent=2))
    print(
        f"CAPTURE_FE4S4_GPU {METHOD}: best_E={best:.6f} steps={len(trace)} dim={SQD_DIM} "
        f"({dt:.0f}s) -> {base / 'recover' / fn}",
        flush=True,
    )


def _read_recovery_trace() -> list[dict]:
    try:
        from prefect.client.orchestration import get_client
        from prefect.client.schemas.filters import ArtifactFilter, ArtifactFilterKey

        with get_client(sync_client=True) as client:
            arts = client.read_artifacts(
                artifact_filter=ArtifactFilter(key=ArtifactFilterKey(any_=["sqd-telemetry"])),
            )
            # There can be several sqd-telemetry artifacts (some empty); scan ALL of them and
            # return the longest recovery_trace found. (Earlier bug: only arts[0] was read, which
            # was often the empty one -> empty trace even though another artifact had it.)
            best: list[dict] = []
            for art in arts:
                try:
                    data = json.loads(art.data)
                except Exception:
                    continue
                for rec in data if isinstance(data, list) else []:
                    tr = rec.get("recovery_trace") if isinstance(rec, dict) else None
                    if tr and len(tr) > len(best):
                        best = tr
            return best
    except Exception as exc:
        print(f"[warn] could not read recovery_trace ({exc}).")
    return []


if __name__ == "__main__":
    main()
