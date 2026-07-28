# Generic RHF/UHF/BS-UHF SQD Recovery

`run_recover.py` is a molecule-agnostic template for the "sample once, then re-diagonalize
offline for many recovery steps" pattern used throughout this repo's Fe2S2/Fe4S4 work — with no
Fe-specific defaults. Point it at any FCIDUMP (and, for broken-symmetry UHF, your own AF-groups
JSON) and a persisted sample pool, and it runs a checkpointed, multi-step SQD recovery on Fugaku,
Slurm (e.g. ROQUO), or locally.

See [`docs/tutorials/run_uhf_bsuhf_any_molecule.md`](../../docs/tutorials/run_uhf_bsuhf_any_molecule.md)
for the full walkthrough (getting an FCIDUMP, building the AF-groups spec, building the solver
binaries) and [`docs/reference/hpc_resource_sizing.md`](../../docs/reference/hpc_resource_sizing.md)
for how to choose `--nodes/--ranks-per-node/--omp-threads/--adet/--bdet`.

[`walkthrough.ipynb`](walkthrough.ipynb) demonstrates the same pipeline end-to-end on a small
molecule with no HPC required, before you scale up with this script.

## Prerequisites

```bash
cd /path/to/qcsc-prefect
uv sync
cd algorithms/sbd && uv sync          # installs the `sbd` package into the active venv
```

Then build the native solver for your platform/method — see
[`algorithms/sbd/native/README.md`](../../algorithms/sbd/native/README.md). You need the `_uhf`
binary variant for both plain UHF and BS-UHF FCIDUMPs.

You also need a persisted sample pool (a `raw_samples_*.npz`) from a prior `quantum_source=
"real-device"` (or `"random"`, for a dry run) sampling run — this script only re-diagonalizes,
it does not sample.

## Quick correctness check (no HPC, no real pool)

```bash
python run_recover.py \
  --fcidump /path/to/your.fcidump --method uhf \
  --pool /path/to/any_pool.npz --quantum-source random \
  --sqd-dim 2000 --recovery-steps 1 --n-batches 1 \
  --run-dir /tmp/sbd_recover_test \
  --hpc-target local --solver-mode cpu \
  --nodes 1 --ranks-per-node 1 --omp-threads 1 --adet 1 --bdet 1 \
  --sbd-executable /path/to/native/diag --sbd-executable-uhf /path/to/native/diag_uhf
```

`--quantum-source random` ignores `--pool`'s contents (deterministic random bitstrings instead) —
useful to confirm the FCIDUMP, AF-groups spec, and block/flow plumbing all work before you touch a
real pool or a multi-node allocation. `--pool` is still required by the CLI even though its
contents aren't read in this mode.

## Fugaku example (deep recovery, checkpointed)

```bash
python run_recover.py \
  --fcidump /path/to/your.fcidump --method uhf \
  --af-groups '{"metal_a":[2,3,4,5,6],"metal_b":[7,8,9,10,11],"up":["metal_a"],"down":["metal_b"]}' \
  --pool file:///path/to/raw_samples.npz \
  --sqd-dim 1000000000 --recovery-steps 50 --n-batches 1 \
  --ckpt-dir /path/to/ckpt --run-dir /path/to/run \
  --hpc-target fugaku --group ra000000 --queue large \
  --nodes 2304 --ranks-per-node 1 --omp-threads 48 --adet 48 --bdet 48 \
  --sbd-executable /path/to/native/diag --sbd-executable-uhf /path/to/native/diag_uhf
```

Re-running the exact same command (same `--ckpt-dir`) after a wall-time kill resumes automatically
from the last completed step — look for `[ckpt] RESUME from ... completed N step(s)` in the log.

## ROQUO example (in-allocation GPU, single node / 4 GPUs)

```bash
python run_recover.py \
  --fcidump /path/to/your.fcidump --method uhf --af-groups @af_groups.json \
  --pool file:///path/to/raw_samples.npz \
  --sqd-dim 300000000 --recovery-steps 10 --n-batches 4 \
  --ckpt-dir /path/to/ckpt --run-dir /path/to/run \
  --hpc-target local --solver-mode gpu \
  --nodes 1 --ranks-per-node 4 --omp-threads 1 --adet 1 --bdet 4 \
  --launcher srun --mpi-options --gpu-bind=closest \
  --sbd-executable /path/to/native/diag-gpu --sbd-executable-uhf /path/to/native/diag-gpu_uhf
```

Wrap this in an `sbatch --gres=gpu:4 --ntasks-per-node=4 ...` submission — `--hpc-target local`
runs every Davidson solve in-allocation (no nested `sbatch`), which is the fast path on ROQUO.

## Miyabi-G example (JCAHPC, GH200/H100, 1 GPU/node — genuine multi-node)

```bash
python run_recover.py \
  --fcidump /path/to/your.fcidump --method uhf --af-groups @af_groups.json \
  --pool /path/to/raw_samples.npz \
  --sqd-dim 8000000000 --recovery-steps 60 --n-batches 1 \
  --ckpt-dir /path/to/ckpt --run-dir /path/to/run \
  --hpc-target miyabi --group <your_group> --queue regular-g --solver-mode gpu \
  --solver-block-ref sbd_solver_job/davidson-solver-gpu \
  --launcher mpirun --modules nvidia/25.9 --walltime 48:00:00 \
  --nodes 64 --ranks-per-node 1 --omp-threads 1 --adet 8 --bdet 8 \
  --sbd-executable /path/to/native/diag-gpu_uhf-mpi-miyabi \
  --sbd-executable-uhf /path/to/native/diag-gpu_uhf-mpi-miyabi
```

Miyabi-G is 1 H100/node (not 4 like ROQUO's GB200), so **any** multi-rank run is genuinely
cross-node — there is no single-node multi-GPU case to fall back to. Build the multi-node binary
with [`algorithms/sbd/native/build_sbd_mpi_uhf_miyabi.sh`](../../algorithms/sbd/native/build_sbd_mpi_uhf_miyabi.sh)
(it auto-clones the fork and applies both required source patches — see that script's header).

Real, hard-won quirks found by actually running this on Miyabi-G — every one of these produced a
silent or confusing failure before being fixed:

- **`--solver-block-ref sbd_solver_job/davidson-solver-gpu`, not the `davidson-solver` default.**
  `solver_mode=gpu` on `hpc_target=miyabi` gets its own block name (`-gpu` suffix); omitting this
  flag fails with `Unable to find block document named davidson-solver`.
- **`--queue` is not actually honored for GPU jobs.** `create_blocks.py` hardcodes the GPU queue
  to `regular-g` regardless of what you pass — pass it anyway for clarity, but don't expect
  `debug-g` to work for a quick GPU smoke test today.
- **`--launcher mpirun`, not the default `mpiexec.hydra`.** The generic Miyabi template defaults
  to Intel MPI's launcher (Miyabi-C convention); this NVHPC/HPC-X-built binary needs `mpirun`, or
  the job fails immediately with `mpiexec.hydra: command not found`.
- **`--modules nvidia/25.9` is required, not optional.** PBS batch jobs on Miyabi-G do **not**
  inherit the submitting shell's loaded modules — without this, the generated script has no
  `module load` line at all and the solver binary fails with a missing-shared-library error on
  the very first rank.
- **Even with `--modules`, cross-node library loading can still fail on remote ranks.** Confirmed
  on real hardware: `module load`'s `LD_LIBRARY_PATH` reaches rank 0 (launched directly by the
  shell `mpirun` runs in) but not rank 1+ on other nodes — a 2-node test failed with
  `libnccl.so.2: cannot open shared object file` specifically on the remote rank. The build script
  now embeds the real runtime library directories as RPATH at link time, so every rank is
  self-sufficient regardless of environment forwarding. If you rebuild by hand, make sure your
  linker flags include `-Wl,-rpath,<nccl>:<cublas>:<mpi>:<cuda>:<nvhpc-compilers>/lib`.
- **PBS resource requests have no `ngpus=` clause.** `#PBS -l select=N:mpiprocs=1` alone gets the
  GPU — Miyabi-G's 1-GPU-per-node topology means the node allocation implies the GPU. Don't add
  an `ngpus=` request; it isn't part of this queue's resource model.
- **The `sbd` package needs an extra install step beyond `algorithms/sbd`'s own `uv sync`.** The
  4 `qcsc-prefect-*` packages (`core`/`adapters`/`blocks`/`executor`) are not resolved by
  `algorithms/sbd`'s own dependency tree on a fresh clone — run (from `algorithms/sbd/`):
  ```bash
  uv pip install --python .venv/bin/python --no-deps \
    -e ../../packages/qcsc-prefect-core -e ../../packages/qcsc-prefect-adapters \
    -e ../../packages/qcsc-prefect-blocks -e ../../packages/qcsc-prefect-executor
  ```
  Without this, `create_blocks.py` fails with `ModuleNotFoundError: No module named
  'qcsc_prefect_blocks'`.
- **Pin `qiskit-ibm-runtime==0.47.0` if a fresh `uv sync` picks up `0.48.0`.** `uv.lock` is
  gitignored in this repo, so a brand-new clone re-resolves dependencies from scratch; at the time
  this was tested, `qiskit-ibm-runtime==0.48.0` broke `prefect_qiskit`'s own import chain
  (`ModuleNotFoundError: No module named 'qiskit_ibm_runtime.utils.result_decoder'`).
- **`/home` has a tiny quota (~50GB) on Miyabi; clone and build under `/work/<group>/<user>/`
  instead.** `MIYABI_REPO` is a required env var for the multi-node build script for exactly this
  reason — there is no safe universal default to guess.
- **2FA on every fresh SSH connection.** Not something to fix in code — use SSH `ControlMaster`/
  `ControlPersist` so one manually-authenticated connection covers everything else for hours.

## Flag reference

| Group | Flag | Notes |
| --- | --- | --- |
| molecule | `--fcidump` | required |
| | `--method {rhf,uhf}` | default `uhf` |
| | `--af-groups` | raw JSON, `@path/to/file.json`, or the `fe4s4` convenience keyword. Omit for plain UHF/RHF |
| | `--af-pol`, `--af-free-s` | fractional-polarization / free-bridging-fragment knobs, see the tutorial |
| SQD | `--pool` | one or more sample-pool paths/URIs, required |
| | `--sqd-dim`, `--recovery-steps`, `--n-batches` | subspace size, recovery passes, K-batches |
| | `--quantum-source {saved,random}` | `saved` is the production path |
| checkpoint | `--ckpt-dir` | omit to disable checkpoint/resume |
| HPC | `--hpc-target {fugaku,miyabi,slurm,local}` | `slurm`/`local` cover ROQUO (`local` = in-allocation, fast) |
| | `--nodes`, `--ranks-per-node`, `--omp-threads`, `--cores-per-node` | validated against each other before anything is submitted |
| | `--adet`, `--bdet`, `--task-comm-size` | must multiply to `nodes × ranks-per-node` |
| | `--launcher`, `--mpi-options` | e.g. `mpiexec` (Fugaku), `srun`/`mpirun` (ROQUO), `mpirun` (Miyabi-G) |
| | `--modules` | modules to `module load` inside the generated batch script — required on PBS targets (Fugaku, Miyabi) whose jobs don't inherit the submitting shell's environment |
| | `--walltime` | per-solver-job wall-clock limit (default `02:00:00` — too short once you scale up) |
| | `--queue`, `--group`, `--fugaku-gfscache`, `--fugaku-mpi-options-for-pjm` | Fugaku/Miyabi (PBS); `--group` doubles as PBS `group_list` on Miyabi |
| | `--slurm-account`, `--slurm-partition`, `--slurm-gres` | Slurm-only (ROQUO) |
| solver | `--sbd-executable`, `--sbd-executable-uhf` | paths to the built binaries |
| | `--block`, `--iteration`, `--carryover-ratio`, `--carryover-type`, `--solver-timeout-seconds` | Davidson solver knobs, defaults match the production Fe4S4 launchers |
| misc | `--run-dir` | required; holds `work/`, `prefect_home/`, `result.json` |

Run `python run_recover.py --help` for the exact, always-current flag list.
