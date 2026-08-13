"""4Fe-4S 72q UHF multi-iteration recovery from the SAVED kobe pool (no hardware).

Reuses the 5M-shot pool persisted by the sampling run (quantum_source="saved") and runs a deep
recovery sweep at a large subspace. NO IBM call. The heavy diagonalization runs on a large-node
Fugaku PJM grid, submitted by the orchestrator (which itself runs on mem2/x86_64).

    python run_fe4s4_recover.py    # (invoked by run_fe4s4_recover.sh on mem2)

Env knobs (set by the .sh):
    FE4S4_POOL     : path(s) to the persisted raw_samples npz (comma-separated per walker)
    FE4S4_SQD_DIM  : subspace dim (default 3e8)
    FE4S4_RECSTEPS : recovery passes (default 5)
    FE4S4_NBATCH   : K-batch count (default 5)
    FE4S4_NODES/ADET/BDET/QUEUE : solver comm grid (default 3600 = 60x60, large)

Data-area rule: PREFECT_HOME + solver work_dir MUST be under a Fugaku data area (/vol0002, i.e.
$MY_SPACE), else pjsub refuses ("current directory is not a data area").
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import fe2s2_common as C

METHOD = os.environ.get("FE4S4_METHOD", "uhf")
SQD_DIM = int(float(os.environ.get("FE4S4_SQD_DIM", "300000000")))   # 3e8
RECSTEPS = int(os.environ.get("FE4S4_RECSTEPS", "5"))
NBATCH = int(os.environ.get("FE4S4_NBATCH", "5"))
NODES = int(os.environ.get("FE4S4_NODES", "3600"))
ADET = int(os.environ.get("FE4S4_ADET", "60"))
BDET = int(os.environ.get("FE4S4_BDET", "60"))
QUEUE = os.environ.get("FE4S4_QUEUE", "large")

# Data-area base for PREFECT_HOME + work_dir (avoid /vol0006 home; use $MY_SPACE on /vol0002).
SPACE = os.environ.get("MY_SPACE", str(C.sweep_root()))
BASE = Path(SPACE) / "sweep" / "runs" / f"fe4s4_{METHOD}"

os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("PREFECT_TELEMETRY_ENABLED", "false")
os.environ.setdefault("SBD_TASK_RUNNER", "concurrent")
os.environ.setdefault("PREFECT_SERVER_DATABASE_TIMEOUT", "60")
os.environ.setdefault("PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS", "120")


def _pool_paths() -> list[str]:
    raw = os.environ.get("FE4S4_POOL", "").strip()
    if not raw:
        raise SystemExit("Set FE4S4_POOL to the persisted raw_samples npz path(s).")
    out = []
    for p in raw.replace(",", " ").split():
        # walker_sqd loads via load_ndarray, which expects a file:// URI for local files.
        out.append(p if p.startswith("file://") else f"file://{p}")
    return out


def main() -> None:
    if C.MOLECULE != "fe4s4":
        raise SystemExit(f"set FE_MOL=fe4s4 (got {C.MOLECULE})")
    pools = _pool_paths()
    p = C.sbd_paths()
    work_dir = BASE / "work_recover"
    prefect_home = BASE / "prefect_home_recover"
    (BASE / "recover").mkdir(parents=True, exist_ok=True)
    prefect_home.mkdir(parents=True, exist_ok=True)
    os.environ["PREFECT_HOME"] = str(prefect_home)
    os.environ["PREFECT_LOCAL_STORAGE_PATH"] = str(prefect_home / "storage")

    # Solver block: saved-samples wired in, quantum_source="saved" downstream. Large-node grid.
    subprocess.run(
        [
            p["python"], os.path.join(p["sbd"], "create_blocks.py"),
            "--hpc-target", "fugaku", "--method", METHOD, "--solver-mode", "fugaku",
            "--project", "ra010014", "--group", "ra010014",
            "--queue", QUEUE, "--fugaku-gfscache", "/vol0004:/vol0002",
            "--num-nodes", str(NODES), "--mpiprocs", str(NODES), "--ompthreads", "48",
            "--launcher", "mpiexec", "--mpi-options", f"-n {NODES}",
            "--fugaku-mpi-options-for-pjm", "max-proc-per-node=1",
            "--adet-comm-size", str(ADET), "--bdet-comm-size", str(BDET), "--task-comm-size", "1",
            "--carryover-ratio", "0.5", "--carryover-type", "1",
            "--solver-timeout-seconds", "43200",
            "--work-dir", str(work_dir),
            "--saved-samples", ",".join(pools),
            "--iteration", "5", "--block", "20",
            "--sbd-executable", p["diag_rhf"], "--sbd-executable-uhf", p["diag_uhf"],
        ],
        cwd=p["sbd"], check=True,
    )

    from sbd.flow_params import CircuitParameters, DEParameters, FlowParameters
    from sbd.main import riken_sqd_de

    t0 = time.perf_counter()
    params = FlowParameters(
        fcidump=C.FCIDUMP,
        sqd_dim=SQD_DIM,
        n_recovery_steps=RECSTEPS,
        n_batches=NBATCH,
        quantum_source="saved",
        solver_block_ref="sbd_solver_job/davidson-solver",
        circ_params=CircuitParameters(n_lucj_layers=1),
        de_params=DEParameters(num_walkers=1, iterations=1, randomization_factor=0.2, fxc=0.5),
    )
    best = riken_sqd_de(params)
    dt = time.perf_counter() - t0

    trace = _read_recovery_trace()
    out = {
        "method": METHOD, "molecule": "fe4s4", "cluster": "fugaku",
        "sqd_dim": SQD_DIM, "n_batches": NBATCH, "max_recovery": RECSTEPS,
        "nodes": NODES, "best_energy": best, "elapsed_s": dt, "pools": pools,
        "recovery_trace": trace,
    }
    fn = f"recover_{METHOD}_fugaku_rec{RECSTEPS}_k{NBATCH}_dim{SQD_DIM}.json"
    out_path = BASE / "recover" / fn
    out_path.write_text(json.dumps(out, indent=2))
    print(
        f"CAPTURE_FE4S4_RECOVER {METHOD}: best_E={best:.6f} steps={len(trace)} "
        f"dim={SQD_DIM} ({dt:.0f}s) -> {out_path}",
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
            if not arts:
                return []
            data = json.loads(arts[0].data)
            for rec in reversed(data):
                if isinstance(rec, dict) and rec.get("recovery_trace"):
                    return rec["recovery_trace"]
    except Exception as exc:
        print(f"[warn] could not read recovery_trace ({exc}).")
    return []


if __name__ == "__main__":
    main()
