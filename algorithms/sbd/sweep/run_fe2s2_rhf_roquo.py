"""Fe2S2 RHF SQD run on ROQUO CPU (aarch64 `roquo` partition, Slurm), large-scale.

Same FCIDUMP and same conditions as the Fugaku UHF `FE_sample` run, but --method rhf and using as
many CPU cores as the ROQUO node allows. The orchestrator (this Python flow) and the SBD solver both
run inside ONE roquo Slurm allocation: the solver uses launcher="single" (runs the aarch64 `diag`
binary directly in the allocation, no nested sbatch), so the create_blocks HPC target is "slurm"
purely to render a valid profile -- there is no queue-within-queue.

    FE2S2_QSRC=random python run_fe2s2_rhf_roquo.py     # cheap dry-run, no IBM call
    python run_fe2s2_rhf_roquo.py                        # real device (needs IBM_* env)

Idempotent on the sample pool: if runs/fe2s2_rhf/samples/ already holds a persisted pool it is
reused (quantum_source stays real-device only to sample once; recovery reuses the pool in-run).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import fe2s2_common as C

# ROQUO specifics -------------------------------------------------------------------------------
METHOD = "rhf"
ACCOUNT = os.environ.get("ROQUO_ACCOUNT", "q0000219")
PARTITION = os.environ.get("ROQUO_PARTITION", "roquo")
# "as many CPUs as allowed": a full roquo node is ~144 cores; leave a few for the orchestrator/OS.
OMP_THREADS = int(os.environ.get("ROQUO_OMPTHREADS", "140"))


def _repo_sbd() -> Path:
    """The sbd package dir on ROQUO (~/qcsc-prefect/algorithms/sbd), via MY_PROJECT or this file."""
    proj = os.environ.get("MY_PROJECT")
    if proj:
        return Path(proj) / "algorithms/sbd"
    return C.sweep_root().parent


def main() -> None:
    dirs = C.run_subdirs(METHOD)
    qsrc = os.environ.get("FE2S2_QSRC", "real-device")
    sbd = _repo_sbd()
    diag = sbd / "native" / "diag"
    diag_uhf = sbd / "native" / "diag_uhf"

    # Prefect's ephemeral API + SQLite DB must live on FAST node-local storage: on Lustre home the
    # ephemeral server times out ("Timed out while attempting to connect to ephemeral Prefect API").
    # Put PREFECT_HOME on /tmp; keep the persisted sample pool (LOCAL_STORAGE_PATH) under the run
    # dir on Lustre so pools survive after the allocation. Bump the ephemeral startup timeout too.
    node_tmp = Path(os.environ.get("TMPDIR", f"/tmp/prefect_{os.environ.get('USER', 'u')}"))
    prefect_home = node_tmp / f"ph_{C.MOLECULE}_{METHOD}"
    prefect_home.mkdir(parents=True, exist_ok=True)
    os.environ["PREFECT_HOME"] = str(prefect_home)
    storage = C.prefect_home(METHOD) / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    os.environ["PREFECT_LOCAL_STORAGE_PATH"] = str(storage)
    os.environ.setdefault("PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS", "120")
    # The Ray task runner + a single SQLite DB deadlocks ("database is locked") when the recovery/
    # batch tasks write concurrently. This run is single-walker, so parallelism buys nothing: use
    # the lightweight ConcurrentTaskRunner and give SQLite a long busy-timeout so any contended
    # write waits instead of erroring.
    os.environ.setdefault("SBD_TASK_RUNNER", "concurrent")
    os.environ.setdefault("PREFECT_SERVER_DATABASE_TIMEOUT", "60")
    os.environ.setdefault("PREFECT_SERVER_DATABASE_CONNECTION_TIMEOUT", "60")
    os.environ.setdefault("OMP_NUM_THREADS", str(OMP_THREADS))

    # 1. Solver block on the Slurm target. --method rhf selects the RHF `diag` binary. Conditions
    #    (shots, error mitigation, carryover) match the Fugaku FE_sample run exactly.
    import subprocess

    # hpc-target "local": run the diag binary directly as a subprocess INSIDE this Slurm allocation
    # (no per-solve sbatch round-trip). The queued "slurm" target works too but adds ~2 min queue
    # latency per solve -- fatal for a 5x5 recovery x batch loop. The allocation itself is obtained
    # by run_fe2s2_rhf_roquo.sh (sbatch/srun); create_blocks just needs the executable + threads.
    cmd = [
        str(sbd / ".venv/bin/python"), str(sbd / "create_blocks.py"),
        "--hpc-target", "local",
        "--method", METHOD,
        "--solver-mode", "cpu",
        "--num-nodes", "1",
        "--mpiprocs", "1",              # single diag process...
        "--ompthreads", str(OMP_THREADS),  # ...with OMP_THREADS threads
        "--launcher", "single",
        "--walltime", "02:00:00",
        "--carryover-ratio", "0.5",
        "--carryover-type", "1",
        "--solver-timeout-seconds", "43200",
        "--work-dir", str(C.run_dir(METHOD) / "work_recover"),
        "--shots", "5000000",
        "--n-shot-batches", "5",
        "--iteration", "5",
        "--block", "20",
        "--dynamical-decoupling", "--dd-sequence", "XY4", "--measure-twirling",
        "--sbd-executable", str(diag),
        "--sbd-executable-uhf", str(diag_uhf),
    ]
    subprocess.run(cmd, cwd=str(sbd), check=True)

    # 2. IBM runner block (only when hitting the real device).
    if qsrc == "real-device":
        C.save_ibm_runner_block()

    # 3. Run the flow: match FE_sample scale (sqd_dim=3e6, rec5, k5), single walker/pass, --method
    #    rhf comes from the solver block above.
    from sbd.flow_params import CircuitParameters, DEParameters, FlowParameters
    from sbd.main import riken_sqd_de

    t0 = time.perf_counter()
    params = FlowParameters(
        fcidump=C.FCIDUMP,
        sqd_dim=3_000_000,
        n_recovery_steps=5,
        n_batches=5,
        quantum_source=qsrc,
        solver_block_ref="sbd_solver_job/davidson-solver",
        circ_params=CircuitParameters(n_lucj_layers=1),
        de_params=DEParameters(num_walkers=1, iterations=1, randomization_factor=0.2, fxc=0.5),
    )
    best_energy = riken_sqd_de(params)
    dt = time.perf_counter() - t0

    # 4. Persist recovery trajectory JSON for the RHF-vs-UHF plot.
    trace = _read_recovery_trace()
    out = {
        "method": METHOD, "molecule": C.MOLECULE, "cluster": "roquo",
        "partition": PARTITION, "omp_threads": OMP_THREADS,
        "sqd_dim": 3_000_000, "n_batches": 5, "max_recovery": 5,
        "best_energy": best_energy, "elapsed_s": dt, "qsrc": qsrc,
        "recovery_trace": trace,
    }
    out_path = dirs["recover"] / f"recover_{METHOD}_roquo_rec5_k5.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(
        f"CAPTURE_ROQUO_RHF: best_E={best_energy:.6f}  steps={len(trace)}  qsrc={qsrc}  "
        f"threads={OMP_THREADS}  ({dt:.0f}s) -> {out_path}",
        flush=True,
    )


def _read_recovery_trace() -> list[dict]:
    """Pull recovery_trace out of the most recent sqd-telemetry artifact (self-contained JSON)."""
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
        print(f"[warn] could not read recovery_trace ({exc}); JSON will omit it.")
    return []


if __name__ == "__main__":
    main()
