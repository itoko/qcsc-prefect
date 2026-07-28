#!/usr/bin/env python
"""Generic RHF/UHF/BS-UHF multi-step SQD recovery for any molecule, any HPC target.

Molecule-agnostic version of the pattern proven on Fe2S2/Fe4S4: sample once, persist the pool,
then re-diagonalize offline (``quantum_source="saved"``) for many recovery steps with
checkpoint/resume, on Fugaku, Slurm (e.g. ROQUO), or locally. See README.md for worked examples
and ``docs/tutorials/run_uhf_bsuhf_any_molecule.md`` for the full tutorial this script backs.

Prerequisites (same as any other qcsc-prefect flow):
    uv sync   # repo root
    cd algorithms/sbd && uv sync   # installs the `sbd` package + its deps into the active venv
    # then build native/diag (+ native/diag_uhf for UHF) per algorithms/sbd/native/README.md

Run with the same Python environment `algorithms/sbd` was installed into.

Example (Fugaku, deep UHF recovery from a saved pool):
    python run_recover.py \\
        --fcidump /path/to/your.fcidump --method uhf \\
        --af-groups '{"metal_a":[2,3,4,5,6],"metal_b":[7,8,9,10,11],"up":["metal_a"],"down":["metal_b"]}' \\
        --pool file:///path/to/raw_samples.npz \\
        --sqd-dim 1000000000 --recovery-steps 50 --n-batches 1 \\
        --ckpt-dir /path/to/ckpt --run-dir /path/to/run \\
        --hpc-target fugaku --group ra000000 --queue large \\
        --nodes 2304 --ranks-per-node 1 --omp-threads 48 --adet 48 --bdet 48

Example (ROQUO, in-allocation GPU, single node / 4 GPUs):
    python run_recover.py \\
        --fcidump /path/to/your.fcidump --method uhf --af-groups @af_groups.json \\
        --pool file:///path/to/raw_samples.npz \\
        --sqd-dim 300000000 --recovery-steps 10 --n-batches 4 \\
        --ckpt-dir /path/to/ckpt --run-dir /path/to/run \\
        --hpc-target local --solver-mode gpu \\
        --nodes 1 --ranks-per-node 4 --omp-threads 1 --adet 1 --bdet 4 \\
        --launcher srun --mpi-options --gpu-bind=closest \\
        --sbd-executable /path/to/diag-gpu --sbd-executable-uhf /path/to/diag-gpu_uhf
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_SBD_DIR = REPO_ROOT / "algorithms" / "sbd"


def _read_af_groups(raw: str | None) -> str | None:
    """Accept a raw JSON string, an ``@path/to/file.json`` reference, or a convenience keyword
    (e.g. "fe4s4") that ``_parse_af_groups()`` in ``chem.py`` expands internally. Returns the
    exact string to put in ``FE4S4_AF_GROUPS`` (validated as JSON only if it isn't a keyword)."""
    if not raw:
        return None
    if raw.startswith("@"):
        raw = Path(raw[1:]).read_text()
    if raw.strip().lower() not in {"fe4s4"}:
        try:
            json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--af-groups is neither a known keyword nor valid JSON: {exc}")
    return raw


def _pool_uris(pools: list[str]) -> list[str]:
    return [p if "://" in p else f"file://{p}" for p in pools]


def _check_sizing(args: argparse.Namespace) -> None:
    """Mirror the checks create_blocks.py itself performs, but fail fast with a pointer to the
    sizing reference doc before spending any time building blocks."""
    total_ranks = args.nodes * args.ranks_per_node
    if args.cores_per_node is not None:
        used = args.ranks_per_node * args.omp_threads
        if used > args.cores_per_node:
            raise SystemExit(
                f"--ranks-per-node({args.ranks_per_node}) x --omp-threads({args.omp_threads}) "
                f"= {used} > --cores-per-node({args.cores_per_node}). See "
                "docs/reference/hpc_resource_sizing.md."
            )
    grid = args.adet * args.bdet * args.task_comm_size
    if grid != total_ranks:
        raise SystemExit(
            f"--adet({args.adet}) x --bdet({args.bdet}) x --task-comm-size({args.task_comm_size}) "
            f"= {grid} != total ranks {total_ranks} (= --nodes x --ranks-per-node). See "
            "docs/reference/hpc_resource_sizing.md."
        )


def build_create_blocks_cmd(args: argparse.Namespace, *, python: str, sbd_dir: Path,
                             work_dir: Path, pools: list[str]) -> list[str]:
    cmd = [
        python, str(sbd_dir / "create_blocks.py"),
        "--hpc-target", args.hpc_target,
        "--method", args.method,
        "--solver-mode", args.solver_mode,
        "--work-dir", str(work_dir),
        "--num-nodes", str(args.nodes),
        "--mpiprocs", str(args.ranks_per_node),
        "--ompthreads", str(args.omp_threads),
        "--task-comm-size", str(args.task_comm_size),
        "--adet-comm-size", str(args.adet),
        "--bdet-comm-size", str(args.bdet),
        "--block", str(args.block),
        "--iteration", str(args.iteration),
        "--carryover-ratio", str(args.carryover_ratio),
        "--carryover-type", str(args.carryover_type),
        "--solver-timeout-seconds", str(args.solver_timeout_seconds),
        "--walltime", args.walltime,
        "--saved-samples", *pools,
        "--sbd-executable", args.sbd_executable,
    ]
    if args.sbd_executable_uhf:
        cmd += ["--sbd-executable-uhf", args.sbd_executable_uhf]
    if args.launcher:
        cmd += ["--launcher", args.launcher]
    if args.mpi_options:
        cmd += ["--mpi-options", *args.mpi_options]
    if args.modules:
        cmd += ["--modules", *args.modules]

    if args.hpc_target == "fugaku":
        if not args.group:
            raise SystemExit("--group is required for --hpc-target fugaku.")
        cmd += ["--group", args.group, "--queue", args.queue]
        if args.fugaku_gfscache:
            cmd += ["--fugaku-gfscache", args.fugaku_gfscache]
        if args.fugaku_mpi_options_for_pjm:
            cmd += ["--fugaku-mpi-options-for-pjm", *args.fugaku_mpi_options_for_pjm]
    elif args.hpc_target == "miyabi":
        if not args.group:
            raise SystemExit("--group is required for --hpc-target miyabi (PBS group_list).")
        if not args.queue:
            raise SystemExit(
                "--queue is required for --hpc-target miyabi (e.g. regular-g for Miyabi-G GPU "
                "nodes, regular-c for Miyabi-C CPU nodes)."
            )
        cmd += ["--group", args.group, "--queue", args.queue]
        # Miyabi-G is 1 GPU/node -- confirmed from a working reference PBS script (not in this
        # repo) that `#PBS -l select=N:mpiprocs=M` alone gets the GPU, with no separate `ngpus=`
        # resource request and `mpirun -n <total>` as the launcher (not mpiexec.hydra). This is
        # NOT yet independently verified against this repo's own Miyabi adapter template --
        # confirm the generated .pbs script's select= line looks right before trusting it blind.
    elif args.hpc_target == "slurm":
        if not args.slurm_account:
            raise SystemExit("--slurm-account is required for --hpc-target slurm.")
        if not args.slurm_partition:
            raise SystemExit("--slurm-partition is required for --hpc-target slurm.")
        cmd += ["--slurm-account", args.slurm_account, "--slurm-partition", args.slurm_partition]
        if args.slurm_gres:
            cmd += ["--slurm-gres", args.slurm_gres]
    # "local" needs none of the above: no scheduler, diag runs as a direct subprocess.
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mol = p.add_argument_group("molecule")
    mol.add_argument("--fcidump", required=True, help="Path to your molecule's FCIDUMP.")
    mol.add_argument("--method", choices=["rhf", "uhf"], default="uhf")
    mol.add_argument(
        "--af-groups", default=None,
        help='Broken-symmetry-UHF fragment spec: raw JSON, "@path/to/file.json", or the "fe4s4" '
             "convenience keyword. Omit for plain UHF/RHF (no localized guess).",
    )
    mol.add_argument("--af-pol", type=float, default=None, help="FE4S4_AF_POL override (0-1].")
    mol.add_argument("--af-free-s", action="store_true", help="Sets FE4S4_AF_FREE_S=1.")

    sqd = p.add_argument_group("SQD / recovery")
    sqd.add_argument("--pool", nargs="+", required=True,
                      help="Persisted sample-pool path(s) (one per walker). Bare paths are "
                           "converted to file:// URIs automatically.")
    sqd.add_argument("--sqd-dim", type=int, default=1_000_000)
    sqd.add_argument("--recovery-steps", type=int, default=5)
    sqd.add_argument("--n-batches", type=int, default=1)
    sqd.add_argument("--n-lucj-layers", type=int, default=1)
    sqd.add_argument("--quantum-source", choices=["saved", "random"], default="saved",
                      help="'saved' (the production path: re-diagonalize the --pool, no IBM "
                           "call) or 'random' (ignore --pool, deterministic random bitstrings — "
                           "for a dry run / plumbing check only).")
    sqd.add_argument("--random-seed", type=int, default=24)

    ckpt = p.add_argument_group("checkpoint")
    ckpt.add_argument("--ckpt-dir", default=None,
                       help="Enables checkpoint/resume (FE4S4_CKPT_DIR). Omit to disable.")

    hpc = p.add_argument_group("HPC target + sizing (see docs/reference/hpc_resource_sizing.md)")
    hpc.add_argument("--hpc-target", choices=["fugaku", "miyabi", "slurm", "local"], required=True)
    hpc.add_argument("--solver-mode", choices=["cpu", "gpu", "fugaku"], default=None,
                      help="Default: 'fugaku' for --hpc-target fugaku, else 'cpu'. Pass 'gpu' "
                           "for a GPU solver run (e.g. ROQUO).")
    hpc.add_argument("--nodes", type=int, default=1)
    hpc.add_argument("--ranks-per-node", type=int, default=1)
    hpc.add_argument("--omp-threads", type=int, default=1)
    hpc.add_argument("--cores-per-node", type=int, default=None,
                      help="If set, validated against ranks-per-node x omp-threads before "
                           "submitting anything (Fugaku A64FX: 48; ROQUO GB200: 144).")
    hpc.add_argument("--adet", type=int, default=1, help="adet_comm_size.")
    hpc.add_argument("--bdet", type=int, default=1, help="bdet_comm_size.")
    hpc.add_argument("--task-comm-size", type=int, default=1)
    hpc.add_argument("--launcher", default=None, help="e.g. mpiexec (Fugaku), srun/mpirun (ROQUO).")
    hpc.add_argument("--mpi-options", nargs="*", default=None)
    hpc.add_argument("--modules", nargs="*", default=None,
                      help="Modules to 'module load' inside the generated batch script -- PBS "
                           "targets (Fugaku, Miyabi) do not inherit the submitting shell's "
                           "loaded modules, so anything the solver binary needs on PATH/"
                           "LD_LIBRARY_PATH (e.g. nvidia/25.9 on Miyabi-G) must be listed here.")
    hpc.add_argument("--queue", default=None, help="Fugaku rscgrp.")
    hpc.add_argument("--group", default=None, help="Fugaku group.")
    hpc.add_argument("--fugaku-gfscache", default=None)
    hpc.add_argument("--fugaku-mpi-options-for-pjm", nargs="*", default=None)
    hpc.add_argument("--slurm-account", default=None)
    hpc.add_argument("--slurm-partition", default=None)
    hpc.add_argument("--slurm-gres", default=None, help="e.g. gpu:4 (ROQUO).")

    solver = p.add_argument_group("solver")
    solver.add_argument("--sbd-executable", default=str(DEFAULT_SBD_DIR / "native" / "diag"))
    solver.add_argument("--sbd-executable-uhf",
                         default=str(DEFAULT_SBD_DIR / "native" / "diag_uhf"))
    solver.add_argument("--block", type=int, default=20)
    solver.add_argument("--iteration", type=int, default=5)
    solver.add_argument("--carryover-ratio", type=float, default=0.5)
    solver.add_argument("--carryover-type", type=int, default=1)
    solver.add_argument("--solver-timeout-seconds", type=float, default=43_200)
    solver.add_argument("--walltime", default="02:00:00",
                         help="Per-solver-job wall-clock limit (PBS/Slurm HH:MM:SS), NOT the "
                              "total run time -- one solve per recovery step gets its own job "
                              "at this limit. create_blocks.py's own default (2h) is usually too "
                              "short for a large sqd_dim; size it to your expected per-step time.")
    solver.add_argument("--solver-block-ref", default="sbd_solver_job/davidson-solver")

    misc = p.add_argument_group("misc")
    misc.add_argument("--run-dir", required=True,
                       help="Where this run's work_dir/prefect_home/result.json live.")
    misc.add_argument("--sbd-dir", default=str(DEFAULT_SBD_DIR))
    misc.add_argument("--python", default=sys.executable,
                       help="Python to invoke create_blocks.py with (default: this interpreter).")

    args = p.parse_args()
    if args.solver_mode is None:
        args.solver_mode = "fugaku" if args.hpc_target == "fugaku" else "cpu"
    _check_sizing(args)

    sbd_dir = Path(args.sbd_dir)
    run_dir = Path(args.run_dir)
    work_dir = run_dir / "work"
    prefect_home = run_dir / "prefect_home"
    work_dir.mkdir(parents=True, exist_ok=True)
    prefect_home.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
    os.environ.setdefault("PREFECT_TELEMETRY_ENABLED", "false")
    os.environ.setdefault("SBD_TASK_RUNNER", "concurrent")
    os.environ["PREFECT_HOME"] = str(prefect_home)
    os.environ["PREFECT_LOCAL_STORAGE_PATH"] = str(prefect_home / "storage")

    af_groups = _read_af_groups(args.af_groups)
    if af_groups is not None:
        os.environ["FE4S4_AF_GROUPS"] = af_groups
        if args.af_pol is not None:
            os.environ["FE4S4_AF_POL"] = str(args.af_pol)
        if args.af_free_s:
            os.environ["FE4S4_AF_FREE_S"] = "1"
    if args.ckpt_dir:
        Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
        os.environ["FE4S4_CKPT_DIR"] = args.ckpt_dir

    pools = _pool_uris(args.pool)
    cb_cmd = build_create_blocks_cmd(args, python=args.python, sbd_dir=sbd_dir,
                                      work_dir=work_dir, pools=pools)
    print("Generating Prefect blocks:", " ".join(cb_cmd), flush=True)
    # create_blocks.py shells out to a bare `prefect` CLI (e.g. to set Variables) -- put the
    # interpreter's own bin/ dir on PATH so that resolves to the same venv's `prefect`.
    cb_env = {**os.environ, "PATH": f"{Path(args.python).parent}{os.pathsep}{os.environ['PATH']}"}
    subprocess.run(cb_cmd, cwd=sbd_dir, check=True, env=cb_env)

    from sbd.flow_params import CircuitParameters, DEParameters, FlowParameters
    from sbd.main import riken_sqd_de

    params = FlowParameters(
        fcidump=args.fcidump,
        sqd_dim=args.sqd_dim,
        n_recovery_steps=args.recovery_steps,
        n_batches=args.n_batches,
        quantum_source=args.quantum_source,
        random_seed=args.random_seed,
        solver_block_ref=args.solver_block_ref,
        circ_params=CircuitParameters(n_lucj_layers=args.n_lucj_layers),
        de_params=DEParameters(num_walkers=1, iterations=1, randomization_factor=0.2, fxc=0.5),
    )
    t0 = time.perf_counter()
    best = riken_sqd_de(params)
    dt = time.perf_counter() - t0

    trace = _read_recovery_trace()
    out = {
        "fcidump": args.fcidump, "method": args.method, "hpc_target": args.hpc_target,
        "sqd_dim": args.sqd_dim, "n_batches": args.n_batches,
        "recovery_steps": args.recovery_steps, "nodes": args.nodes,
        "ranks_per_node": args.ranks_per_node, "omp_threads": args.omp_threads,
        "best_energy": best, "elapsed_s": dt, "pools": pools,
        "recovery_trace": trace,
    }
    out_path = run_dir / "result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"CAPTURE_RECOVER {args.method}: best_E={best:.6f} steps={len(trace)} "
          f"dim={args.sqd_dim} ({dt:.0f}s) -> {out_path}", flush=True)


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
