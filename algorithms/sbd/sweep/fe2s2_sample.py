"""Sample the Fe2S2 LUCJ circuit ONCE for a given method and persist the merged pool.

    python fe2s2_sample.py --method uhf            # real device
    FE2S2_QSRC=random python fe2s2_sample.py --method uhf   # cheap local dry-run, no IBM call

Idempotent: if a persisted pool already exists for this method (see fe2s2_common.find_saved_pools),
sampling is SKIPPED and the existing pool manifest is printed. This is the "sample once, reuse"
guarantee -- the expensive device shots are never re-taken.

After a successful device run the persisted pool URIs are written to
runs/<mol>_<method>/samples/pool_manifest.json, which fe2s2_recover.py feeds straight back in as
--saved-samples for the offline recovery sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import fe2s2_common as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(C.METHODS))
    ap.add_argument(
        "--shots", type=int, default=5_000_000,
        help="Total device shots (split into --n-shot-batches batches).",
    )
    ap.add_argument("--n-shot-batches", type=int, default=5)
    ap.add_argument(
        "--sqd-dim", type=int, default=3_000_000,
        help="Subspace dim for the sampling pass; only the raw pool is reused downstream.",
    )
    ap.add_argument("--n-lucj-layers", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="Re-sample even if a pool exists.")
    # ── Error-aware LUCJ layout + alpha-beta coupling density ──────────────────────────────────
    ap.add_argument(
        "--error-aware-layout", action="store_true",
        help="Use ffsim.generate_lucj_pass_manager (LUCJ-aware, error-aware) instead of Sabre.",
    )
    ap.add_argument(
        "--ab-stride", type=int, default=4,
        help="Alpha-beta coupling stride: 4=stock heavy-hex, 2=~2x denser, 1=full per-orbital.",
    )
    ap.add_argument("--twoq-error-threshold", type=float, default=1.0)
    ap.add_argument("--readout-error-threshold", type=float, default=0.1)
    ap.add_argument("--layout-connectivity", default="heavy-hex", choices=("heavy-hex", "square"))
    args = ap.parse_args()

    method = args.method
    C.run_subdirs(method)  # ensure samples/recover/post exist
    qsrc = os.environ.get("FE2S2_QSRC", "real-device")

    # --- Idempotency guard: do not re-sample if a pool is already persisted -------------------
    existing = C.find_saved_pools(method)
    if existing and not args.force:
        C.samples_manifest_path(method).write_text(json.dumps(existing, indent=2))
        print(
            f"[skip] {method}: {len(existing)} persisted pool(s) already present; "
            f"reusing (pass --force to re-sample).",
            flush=True,
        )
        for p in existing:
            print(f"  {p}")
        return

    # Isolated Prefect home so the persisted pool lands under runs/<mol>_<method>/prefect_home/.
    os.environ["PREFECT_HOME"] = str(C.prefect_home(method))
    os.environ["PREFECT_LOCAL_STORAGE_PATH"] = str(C.prefect_home(method) / "storage")
    C.prefect_home(method).mkdir(parents=True, exist_ok=True)

    # 1. Solver block for this method (RHF vs UHF lives here, not in FlowParameters).
    C.run_create_blocks(
        method,
        work_dir=C.run_dir(method) / "work_sample",
        shots=args.shots,
        n_shot_batches=args.n_shot_batches,
        # A single evaluation pass is enough to sample+persist; recovery depth is swept offline.
        iteration=1,
        block=20,
    )

    # 2. IBM runner block (only when actually hitting the device).
    if qsrc == "real-device":
        C.save_ibm_runner_block()

    # 3. Run one flow pass; the persist step in walker_sqd saves the merged pool before diag.
    from sbd.flow_params import CircuitParameters, DEParameters, FlowParameters
    from sbd.main import riken_sqd_de

    t0 = time.perf_counter()
    params = FlowParameters(
        fcidump=C.FCIDUMP,
        sqd_dim=args.sqd_dim,
        n_recovery_steps=1,          # sampling pass only; the pool is what we keep
        n_batches=1,
        quantum_source=qsrc,
        solver_block_ref="sbd_solver_job/davidson-solver",
        circ_params=CircuitParameters(
            n_lucj_layers=args.n_lucj_layers,
            ab_stride=args.ab_stride,
            use_error_aware_layout=args.error_aware_layout,
            two_qubit_error_threshold=args.twoq_error_threshold,
            readout_error_threshold=args.readout_error_threshold,
            layout_connectivity=args.layout_connectivity,
        ),
        de_params=DEParameters(num_walkers=1, iterations=1, randomization_factor=0.2, fxc=0.5),
    )
    energy = riken_sqd_de(params)
    dt = time.perf_counter() - t0

    # 4. Record the persisted pool paths for the recovery sweep.
    pools = C.find_saved_pools(method)
    C.samples_manifest_path(method).write_text(json.dumps(pools, indent=2))
    print(
        f"CAPTURE_SAMPLE {method}: E_sampling={energy:.6f}  qsrc={qsrc}  "
        f"pools={len(pools)}  ({dt:.0f}s)",
        flush=True,
    )
    for p in pools:
        print(f"  {p}")
    if not pools:
        print(
            "[warn] no persisted pool found -- check that quantum_source was 'real-device' and "
            "the persist step ran (see [persist] log lines).",
            flush=True,
        )


if __name__ == "__main__":
    main()
