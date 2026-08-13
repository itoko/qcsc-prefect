"""4Fe-4S 72q sampling on ROQUO (IBM kobe from a compute node) -> save pool. RHF by default.

ROQUO compute nodes reach IBM Quantum, so we can sample here without touching Fugaku's queue.
The sampling pass persists the merged 5M-shot pool before any diagonalization; the tiny post-sample
solve runs on 1 GPU (diag-gpu). Reuse the saved pool later for GPU recovery at large sqd_dim.

    METHOD=rhf python run_fe4s4_sample_roquo.py           # real device (needs .env.local creds)
    FE2S2_QSRC=random METHOD=rhf python run_fe4s4_sample_roquo.py   # dry-run, no IBM
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import fe2s2_common as C

METHOD = os.environ.get("FE4S4_METHOD", "rhf")
OMP = int(os.environ.get("ROQUO_OMPTHREADS", "140"))

os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("PREFECT_TELEMETRY_ENABLED", "false")
os.environ.setdefault("SBD_TASK_RUNNER", "concurrent")
os.environ.setdefault("PREFECT_SERVER_DATABASE_TIMEOUT", "60")
os.environ.setdefault("PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS", "120")
os.environ.setdefault("OMP_NUM_THREADS", str(OMP))
# diag-gpu is MPI: run the small post-sample solve on 1 GPU (see GPU runner notes).
os.environ["SBD_LAUNCHER"] = "mpirun"
os.environ["SBD_MPI_OPTIONS"] = "-n,1"


# Total shots and how many device submissions to split them into (generic, any molecule).
SHOTS = int(float(os.environ.get("SHOTS", "5000000")))
NSHOT_BATCHES = int(os.environ.get("NSHOT_BATCHES", "5"))


def main() -> None:
    if C.MOLECULE not in C.MOLECULES:
        raise SystemExit(f"unknown FE_MOL={C.MOLECULE}")
    qsrc = os.environ.get("FE2S2_QSRC", "real-device")
    p = C.sbd_paths()
    diag_gpu = os.path.join(p["diag"], "diag-gpu")
    diag_gpu_uhf = os.path.join(p["diag"], "diag-gpu_uhf")

    base = C.run_dir(METHOD)
    (base / "samples").mkdir(parents=True, exist_ok=True)
    # PREFECT_HOME MUST be on Lustre (not node-local $SLURM_SCRATCH): save_ndarray persists the pool
    # to settings.home/storage/sqd_data (= PREFECT_HOME/storage), so a node-local PREFECT_HOME gets
    # WIPED at job end and the 5M-shot pool is lost (exactly what happened to the first RHF sample).
    prefect_home = base / "prefect_home"
    prefect_home.mkdir(parents=True, exist_ok=True)
    os.environ["PREFECT_HOME"] = str(prefect_home)
    os.environ["PREFECT_LOCAL_STORAGE_PATH"] = str(prefect_home / "storage")
    (prefect_home / "storage").mkdir(parents=True, exist_ok=True)

    # Idempotency: reuse an existing persisted pool.
    existing = C.find_saved_pools(METHOD)
    if existing and os.environ.get("FE4S4_FORCE") != "1":
        C.samples_manifest_path(METHOD).write_text(json.dumps(existing, indent=2))
        print(f"[skip] {METHOD}: {len(existing)} pool(s) present; reuse (FE4S4_FORCE=1 to redo).")
        for x in existing:
            print(f"  {x}")
        return

    subprocess.run(
        [
            p["python"], os.path.join(p["sbd"], "create_blocks.py"),
            "--hpc-target", "local", "--method", METHOD, "--solver-mode", "gpu",
            "--num-nodes", "1", "--mpiprocs", "1", "--ompthreads", str(OMP),
            "--walltime", "06:00:00",
            "--carryover-ratio", "0.5", "--carryover-type", "1",
            "--solver-timeout-seconds", "43200",
            "--work-dir", str(base / "work_sample"),
            "--shots", str(SHOTS), "--n-shot-batches", str(NSHOT_BATCHES),
            "--iteration", "5", "--block", "20",
            "--dynamical-decoupling", "--dd-sequence", "XY4", "--measure-twirling",
            "--sbd-executable", diag_gpu, "--sbd-executable-uhf", diag_gpu_uhf,
        ],
        cwd=p["sbd"], check=True,
    )

    if qsrc == "real-device":
        C.save_ibm_runner_block()

    from sbd.flow_params import CircuitParameters, DEParameters, FlowParameters
    from sbd.main import riken_sqd_de

    t0 = time.perf_counter()
    params = FlowParameters(
        fcidump=C.FCIDUMP,
        sqd_dim=3_000_000,          # sampling pass; the raw 5M-shot pool is what we keep
        n_recovery_steps=1,
        n_batches=1,
        quantum_source=qsrc,
        solver_block_ref="sbd_solver_job/davidson-solver-gpu",
        circ_params=CircuitParameters(n_lucj_layers=1),
        de_params=DEParameters(num_walkers=1, iterations=1, randomization_factor=0.2, fxc=0.5),
    )
    e = riken_sqd_de(params)
    dt = time.perf_counter() - t0

    pools = C.find_saved_pools(METHOD)
    C.samples_manifest_path(METHOD).write_text(json.dumps(pools, indent=2))
    print(
        f"CAPTURE_FE4S4_SAMPLE_ROQUO {METHOD}: E={e:.6f} qsrc={qsrc} "
        f"pools={len(pools)} ({dt:.0f}s)",
        flush=True,
    )
    for x in pools:
        print(f"  {x}")


if __name__ == "__main__":
    main()
