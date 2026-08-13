"""Offline multi-iteration configuration-recovery sweep from a persisted pool.

    python fe2s2_recover.py --method uhf --max-recovery 10

Reuses the sample pool persisted by fe2s2_sample.py (quantum_source="saved") -- NO device call,
no credentials. Runs one flow whose walker performs --max-recovery self-consistent recovery
passes, then writes the full telemetry (including the per-step recovery_trace added to
sbd/sqd.py) as JSON into runs/<mol>_<method>/recover/.

Because the recovery loop already keeps the best pass AND now records every pass in
recovery_trace, a single run with n_recovery_steps=max gives the whole per-iteration curve -- we
do not need to launch one job per depth.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import fe2s2_common as C


def _load_pool_manifest(method: str) -> list[str]:
    mani = C.samples_manifest_path(method)
    if mani.is_file():
        pools = json.loads(mani.read_text())
        if pools:
            return pools
    # Fall back to scanning the storage dir directly.
    pools = C.find_saved_pools(method)
    if not pools:
        raise SystemExit(
            f"No persisted pool for {method}. Run fe2s2_sample.py --method {method} first."
        )
    return pools


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(C.METHODS))
    ap.add_argument("--max-recovery", type=int, default=10)
    ap.add_argument("--n-batches", type=int, default=5, help="K-batch count per recovery pass.")
    ap.add_argument("--sqd-dim", type=int, default=3_000_000)
    ap.add_argument("--n-lucj-layers", type=int, default=1)
    args = ap.parse_args()

    method = args.method
    dirs = C.run_subdirs(method)
    pools = _load_pool_manifest(method)

    os.environ["PREFECT_HOME"] = str(C.prefect_home(method))
    os.environ["PREFECT_LOCAL_STORAGE_PATH"] = str(C.prefect_home(method) / "storage")

    # Solver block with the saved-samples list wired in and quantum_source="saved" downstream.
    C.run_create_blocks(
        method,
        work_dir=C.run_dir(method) / "work_recover",
        saved_samples=pools,
        iteration=1,
        block=20,
    )

    from sbd.flow_params import CircuitParameters, DEParameters, FlowParameters
    from sbd.main import riken_sqd_de

    t0 = time.perf_counter()
    params = FlowParameters(
        fcidump=C.FCIDUMP,
        sqd_dim=args.sqd_dim,
        n_recovery_steps=args.max_recovery,
        n_batches=args.n_batches,
        quantum_source="saved",
        solver_block_ref="sbd_solver_job/davidson-solver",
        circ_params=CircuitParameters(n_lucj_layers=args.n_lucj_layers),
        de_params=DEParameters(num_walkers=1, iterations=1, randomization_factor=0.2, fxc=0.5),
    )
    best_energy = riken_sqd_de(params)
    dt = time.perf_counter() - t0

    # The per-step trajectory is on the sqd-telemetry artifact. We re-read it from the Prefect DB
    # so the recovery JSON is self-contained even if the artifact is later pruned.
    trace = _read_recovery_trace()
    out = {
        "method": method,
        "molecule": C.MOLECULE,
        "sqd_dim": args.sqd_dim,
        "n_batches": args.n_batches,
        "max_recovery": args.max_recovery,
        "best_energy": best_energy,
        "elapsed_s": dt,
        "pools": pools,
        "recovery_trace": trace,
    }
    out_path = dirs["recover"] / f"recover_{method}_rec{args.max_recovery}_k{args.n_batches}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(
        f"CAPTURE_RECOVER {method}: best_E={best_energy:.6f}  steps={len(trace)}  "
        f"({dt:.0f}s) -> {out_path}",
        flush=True,
    )


def _read_recovery_trace() -> list[dict]:
    """Pull the recovery_trace out of the most recent sqd-telemetry artifact of this flow run."""
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
            # data is a list of per-walker telemetry dicts; take the last walker's trace.
            for rec in reversed(data):
                if isinstance(rec, dict) and rec.get("recovery_trace"):
                    return rec["recovery_trace"]
    except Exception as exc:  # never let telemetry read-back abort the run
        print(f"[warn] could not read recovery_trace artifact ({exc}); JSON will omit it.")
    return []


if __name__ == "__main__":
    main()
