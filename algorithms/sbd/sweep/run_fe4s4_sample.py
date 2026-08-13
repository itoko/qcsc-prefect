"""4Fe-4S 72-qubit sampling on Fugaku: 5M shots, save the merged pool (no heavy diagonalization).

Model of the proven Fe2S2 FE_sample run, scaled to 72q (NORB=36, NELEC=54, MS2=0). The sampling
pass only needs to persist the raw bitstring pool BEFORE any diagonalization, so a modest node
count suffices here; the heavy sqd_dim=3e8 diagonalization is the separate recovery job.

    FE2S2_QSRC=random python run_fe4s4_sample.py   # cheap dry-run, no IBM call
    python run_fe4s4_sample.py                      # real device (ibm_kobe), 5M shots

Idempotent: if a persisted pool already exists under runs/fe4s4_rhf/samples/, sampling is skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import fe2s2_common as C  # FE_MOL=fe4s4 selects the 4Fe-4S FCIDUMP + runs/fe4s4_* layout

TAG = "fe4s4_sample"
METHOD = os.environ.get("FE4S4_METHOD", "uhf")  # UHF (open-shell) case
# Large-node solver profile: square comm grid (adet x bdet x task). 100 = 10x10x1. Override with
# FE4S4_NODES + FE4S4_ADET/FE4S4_BDET (must satisfy adet*bdet*task == nodes, a perfect square grid).
NODES = int(os.environ.get("FE4S4_NODES", "3600"))
ADET = int(os.environ.get("FE4S4_ADET", "60"))
BDET = int(os.environ.get("FE4S4_BDET", "60"))
QUEUE = os.environ.get("FE4S4_QUEUE", "large")

# Prefect's own usage telemetry contends on the ephemeral SQLite DB ("database is locked" on the
# TELEMETRY_SESSION insert). Disable it; it is unrelated to our workflow data.
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("PREFECT_TELEMETRY_ENABLED", "false")
os.environ.setdefault("SBD_TASK_RUNNER", "concurrent")
os.environ.setdefault("PREFECT_SERVER_DATABASE_TIMEOUT", "60")
os.environ.setdefault("PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS", "120")


def main() -> None:
    if C.MOLECULE != "fe4s4":
        raise SystemExit(f"set FE_MOL=fe4s4 (got {C.MOLECULE})")
    C.run_subdirs(METHOD)  # ensure samples/recover/post exist
    qsrc = os.environ.get("FE2S2_QSRC", "real-device")
    p = C.sbd_paths()

    # Idempotency: reuse an already-persisted pool (never re-charge the device).
    existing = C.find_saved_pools(METHOD)
    if existing and os.environ.get("FE4S4_FORCE") != "1":
        C.samples_manifest_path(METHOD).write_text(json.dumps(existing, indent=2))
        print(f"[skip] {METHOD}: {len(existing)} pool(s) present; reuse (FE4S4_FORCE=1 to redo).")
        for x in existing:
            print(f"  {x}")
        return

    os.environ["PREFECT_HOME"] = str(C.prefect_home(METHOD))
    os.environ["PREFECT_LOCAL_STORAGE_PATH"] = str(C.prefect_home(METHOD) / "storage")
    C.prefect_home(METHOD).mkdir(parents=True, exist_ok=True)

    # 1. Solver block on a large-node Fugaku grid (UHF). The sampling pass still runs the solver
    #    once, so give it a real comm grid (adet x bdet x task, square, product == NODES) matching
    #    the rec10 1600-node pattern. DD(XY4)+measure-twirling, carryover on, 5M shots / 5 batches.
    subprocess.run(
        [
            p["python"], os.path.join(p["sbd"], "create_blocks.py"),
            "--hpc-target", "fugaku",
            "--method", METHOD,
            "--solver-mode", "fugaku",
            "--project", "ra010014", "--group", "ra010014",
            "--queue", QUEUE,
            "--fugaku-gfscache", "/vol0004:/vol0002",
            "--num-nodes", str(NODES), "--mpiprocs", str(NODES), "--ompthreads", "48",
            "--launcher", "mpiexec", "--mpi-options", f"-n {NODES}",
            "--fugaku-mpi-options-for-pjm", "max-proc-per-node=1",
            "--adet-comm-size", str(ADET), "--bdet-comm-size", str(BDET), "--task-comm-size", "1",
            "--carryover-ratio", "0.5", "--carryover-type", "1",
            "--solver-timeout-seconds", "43200",
            "--work-dir", str(C.run_dir(METHOD) / "work_sample"),
            "--shots", "5000000", "--n-shot-batches", "5",
            "--iteration", "5", "--block", "20",
            "--dynamical-decoupling", "--dd-sequence", "XY4", "--measure-twirling",
            "--sbd-executable", p["diag_rhf"],
            "--sbd-executable-uhf", p["diag_uhf"],
        ],
        cwd=p["sbd"], check=True,
    )

    # 2. IBM runner (real device only).
    if qsrc == "real-device":
        C.save_ibm_runner_block()

    # 3. One flow pass at a small sqd_dim just to sample + persist the 5M-shot pool.
    from sbd.flow_params import CircuitParameters, DEParameters, FlowParameters
    from sbd.main import riken_sqd_de

    t0 = time.perf_counter()
    params = FlowParameters(
        fcidump=C.FCIDUMP,
        sqd_dim=3_000_000,       # sampling pass; the raw pool is what we keep
        n_recovery_steps=1,
        n_batches=1,
        quantum_source=qsrc,
        solver_block_ref="sbd_solver_job/davidson-solver",
        circ_params=CircuitParameters(n_lucj_layers=1),
        de_params=DEParameters(num_walkers=1, iterations=1, randomization_factor=0.2, fxc=0.5),
    )
    e = riken_sqd_de(params)
    dt = time.perf_counter() - t0

    pools = C.find_saved_pools(METHOD)
    C.samples_manifest_path(METHOD).write_text(json.dumps(pools, indent=2))
    print(
        f"CAPTURE_FE4S4_SAMPLE {METHOD}: E_sampling={e:.6f} qsrc={qsrc} "
        f"pools={len(pools)} ({dt:.0f}s)",
        flush=True,
    )
    for x in pools:
        print(f"  {x}")
    if not pools:
        print("[warn] no pool persisted -- check quantum_source and the [persist] log lines.")


if __name__ == "__main__":
    main()
