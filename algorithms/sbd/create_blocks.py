from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import tomllib


def _import_sbd_solver_block():
    # Supports direct execution:
    # python algorithms/sbd/create_blocks.py ...
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from sbd.solver_job import SBDSolverJob

    return SBDSolverJob


def _register_block_types(*custom_block_classes) -> None:
    from qcsc_prefect_blocks.common.blocks import (
        CommandBlock,
        ExecutionProfileBlock,
        HPCProfileBlock,
    )

    block_types = [
        CommandBlock,
        ExecutionProfileBlock,
        HPCProfileBlock,
        *custom_block_classes,
    ]
    for block_cls in block_types:
        register = getattr(block_cls, "register_type_and_schema", None)
        if callable(register):
            register()


def _set_variable(variable_name: str, value: Any) -> None:
    subprocess.run(
        ["prefect", "variable", "set", variable_name, json.dumps(value), "--overwrite"],
        check=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Prefect blocks for SBD workflow (Miyabi or Fugaku)."
    )
    parser.add_argument("--config", type=Path, help="Path to TOML/JSON config file.")

    parser.add_argument("--hpc-target", choices=["miyabi", "fugaku", "slurm", "local"])
    parser.add_argument("--project")
    parser.add_argument("--group")
    parser.add_argument("--queue")
    parser.add_argument("--work-dir")
    parser.add_argument("--sbd-executable")
    parser.add_argument(
        "--sbd-executable-uhf",
        help="Path to the UHF (open-shell) diag_uhf binary. Required when --method uhf.",
    )

    parser.add_argument("--launcher")
    parser.add_argument("--walltime")
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--mpiprocs", type=int)
    parser.add_argument("--ompthreads", type=int)
    parser.add_argument(
        "--mem",
        help=(
            "Memory per node for the PBS select chunk, e.g. '32gb' (Miyabi only). "
            "Unset means the scheduler's own server-side default is used, which on Miyabi-G "
            "has been observed to reserve ~100GB/node regardless of actual usage -- set this "
            "explicitly to avoid large multi-node jobs stalling on cluster-wide memory "
            "availability rather than actual solver memory needs."
        ),
    )
    parser.add_argument("--modules", nargs="+")
    parser.add_argument("--mpi-options", nargs="*")
    parser.add_argument("--pre-commands", nargs="*")

    parser.add_argument("--fugaku-gfscache")
    parser.add_argument("--fugaku-spack-modules", nargs="+")
    parser.add_argument("--fugaku-mpi-options-for-pjm", nargs="*")

    # Slurm (e.g. ROQUO) knobs. --queue carries the partition and --project the account by default;
    # these override when set. slurm_qpu is emitted as #SBATCH --qpu (a ROQUO extension).
    parser.add_argument("--slurm-account")
    parser.add_argument("--slurm-partition")
    parser.add_argument("--slurm-qpu")
    parser.add_argument("--slurm-memory")
    parser.add_argument("--slurm-ntasks", type=int)
    parser.add_argument("--slurm-gres", help="Slurm --gres, e.g. gpu:1 (ROQUO GPU jobs).")

    parser.add_argument("--script-filename")
    parser.add_argument("--metrics-artifact-key")

    parser.add_argument("--command-block-name")
    parser.add_argument("--execution-profile-block-name")
    parser.add_argument("--hpc-profile-block-name")
    parser.add_argument("--solver-block-name")
    parser.add_argument("--options-variable-name")

    parser.add_argument("--task-comm-size", type=int)
    parser.add_argument("--adet-comm-size", type=int)
    parser.add_argument("--bdet-comm-size", type=int)
    parser.add_argument("--block", type=int)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--carryover-ratio", type=float)
    parser.add_argument("--carryover-type", type=int)
    parser.add_argument(
        "--do-rdm", type=int,
        help="Solver --rdm value. 0 (default) = occupancy only. Nonzero makes the solver compute "
             "and write the full per-spin-block 1- and 2-RDMs (rdm1_a/b.txt, rdm2_aa/ab/bb.txt) "
             "for orbital optimization. Requires a solver binary built with the RDM writers.",
    )
    parser.add_argument("--solver-timeout-seconds", type=float)
    parser.add_argument("--solver-mode", choices=["cpu", "gpu", "fugaku"])
    parser.add_argument("--method", choices=["rhf", "uhf"])
    parser.add_argument("--shots", type=int)
    parser.add_argument(
        "--n-shot-batches",
        dest="n_shot_batches",
        type=int,
        help=(
            "Split the quantum sampling into this many sampler jobs of shots//N each and mix them "
            "(BitArray.concatenate_shots), to reach a large effective shot count without one huge "
            "IBM job. 1 = single submission (default)."
        ),
    )
    parser.add_argument(
        "--saved-samples",
        dest="saved_samples",
        nargs="+",
        help=(
            "Diagonalize from persisted sample pool(s) instead of sampling. One saved-artifact path "
            "per walker (the file://... or S3 key printed by a prior real-device run's [persist] "
            "line). Requires quantum_source='saved'. No IBM call is made."
        ),
    )
    parser.add_argument(
        "--sqd-options-json",
        help='Raw JSON for Prefect variable value (e.g. \'{"params": {"shots": 50000}}\').',
    )
    parser.add_argument(
        "--dynamical-decoupling",
        dest="dynamical_decoupling",
        action="store_true",
        default=None,
        help="Enable dynamical decoupling on idle qubits during sampling (IBM SamplerV2).",
    )
    parser.add_argument(
        "--dd-sequence",
        dest="dd_sequence",
        choices=["XX", "XpXm", "XY4"],
        help="Dynamical-decoupling pulse sequence (default XY4 when DD is enabled).",
    )
    parser.add_argument(
        "--gate-twirling",
        dest="gate_twirling",
        action="store_true",
        default=None,
        help="Enable Pauli twirling of 2-qubit gates (IBM SamplerV2).",
    )
    parser.add_argument(
        "--measure-twirling",
        dest="measure_twirling",
        action="store_true",
        default=None,
        help="Enable measurement twirling (IBM SamplerV2).",
    )
    return parser.parse_args()


def _load_config_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file was not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("rb") as f:
        return tomllib.load(f)


def _pick_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"Expected list[str] or CSV string, got: {type(value)}")


def _normalize_str_dict(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        normalized = {
            str(key).strip(): str(val).strip() for key, val in value.items() if str(key).strip()
        }
        return normalized or None
    raise ValueError(f"Expected dict[str, str], got: {type(value)}")


def _default_block_names(*, hpc_target: str, solver_mode: str) -> dict[str, str]:
    is_gpu = solver_mode == "gpu"
    if hpc_target == "miyabi":
        return {
            "profile_name": "sbd-gpu" if is_gpu else "sbd-mpi",
            "execution_profile_block_name": "exec-sbd-gpu" if is_gpu else "exec-sbd-mpi",
            "hpc_profile_block_name": "hpc-miyabi-sbd-gpu" if is_gpu else "hpc-miyabi-sbd",
            "solver_block_name": "davidson-solver-gpu" if is_gpu else "davidson-solver",
        }
    if hpc_target == "slurm":
        return {
            "profile_name": "sbd-gpu" if is_gpu else "sbd-mpi",
            "execution_profile_block_name": "exec-sbd-slurm-gpu" if is_gpu else "exec-sbd-slurm",
            "hpc_profile_block_name": "hpc-slurm-sbd-gpu" if is_gpu else "hpc-slurm-sbd",
            "solver_block_name": "davidson-solver-gpu" if is_gpu else "davidson-solver",
        }
    if hpc_target == "local":
        return {
            "profile_name": "sbd-gpu" if is_gpu else "sbd-mpi",
            "execution_profile_block_name": "exec-sbd-local-gpu" if is_gpu else "exec-sbd-local",
            "hpc_profile_block_name": "hpc-local-sbd-gpu" if is_gpu else "hpc-local-sbd",
            "solver_block_name": "davidson-solver-gpu" if is_gpu else "davidson-solver",
        }
    return {
        "profile_name": "sbd-gpu" if is_gpu else "sbd-mpi",
        "execution_profile_block_name": "exec-sbd-fugaku-gpu" if is_gpu else "exec-sbd-fugaku",
        "hpc_profile_block_name": "hpc-fugaku-sbd-gpu" if is_gpu else "hpc-fugaku-sbd",
        "solver_block_name": "davidson-solver-gpu" if is_gpu else "davidson-solver",
    }


def _normalize_modules_for_target(
    *, is_miyabi: bool, solver_mode: str, modules: list[str] | None
) -> list[str]:
    if is_miyabi and solver_mode == "gpu":
        return []
    return list(modules or [])


def _env_values() -> dict[str, Any]:
    def env_int(name: str) -> int | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        return int(raw)

    def env_float(name: str) -> float | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        return float(raw)

    def env_csv(name: str) -> list[str] | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        return [item.strip() for item in raw.split(",") if item.strip()]

    def env_first_str(*names: str) -> str | None:
        for name in names:
            raw = os.getenv(name, "").strip()
            if raw:
                return raw
        return None

    def env_bool(name: str) -> bool | None:
        raw = os.getenv(name, "").strip().lower()
        if not raw:
            return None
        return raw in ("1", "true", "yes", "on")

    def env_first_int(*names: str) -> int | None:
        for name in names:
            value = env_int(name)
            if value is not None:
                return value
        return None

    def env_first_csv(*names: str) -> list[str] | None:
        for name in names:
            value = env_csv(name)
            if value:
                return value
        return None

    return {
        "hpc_target": env_first_str("SBD_HPC_TARGET"),
        "project": env_first_str("SBD_PROJECT", "MIYABI_PBS_PROJECT", "FUGAKU_PROJECT"),
        "group": env_first_str("SBD_GROUP", "FUGAKU_GROUP", "FUGAKU_PROJECT"),
        "queue": env_first_str("SBD_QUEUE", "MIYABI_PBS_QUEUE", "FUGAKU_RSCGRP"),
        "work_dir": env_first_str("SBD_WORK_DIR"),
        "sbd_executable": env_first_str("SBD_EXECUTABLE"),
        "sbd_executable_uhf": env_first_str("SBD_EXECUTABLE_UHF"),
        "launcher": env_first_str("SBD_LAUNCHER", "MIYABI_LAUNCHER", "FUGAKU_LAUNCHER"),
        "walltime": env_first_str("SBD_WALLTIME", "MIYABI_WALLTIME", "FUGAKU_WALLTIME"),
        "mem": env_first_str("SBD_MEM", "MIYABI_MEM", "FUGAKU_MEM"),
        "num_nodes": env_first_int("SBD_NUM_NODES", "MIYABI_NUM_NODES", "FUGAKU_NUM_NODES"),
        "mpiprocs": env_first_int("SBD_MPIPROCS", "MIYABI_MPIPROCS", "FUGAKU_MPIPROCS"),
        "ompthreads": env_first_int("SBD_OMPTHREADS", "MIYABI_OMPTHREADS", "FUGAKU_OMPTHREADS"),
        "modules": env_first_csv("SBD_MODULES", "MIYABI_MODULES"),
        "mpi_options": env_first_csv("SBD_MPI_OPTIONS", "MIYABI_MPI_OPTIONS", "FUGAKU_MPI_OPTIONS"),
        "pre_commands": env_first_csv("SBD_PRE_COMMANDS"),
        "fugaku_gfscache": env_first_str("FUGAKU_GFSCACHE"),
        "fugaku_spack_modules": env_first_csv("FUGAKU_SPACK_MODULES"),
        "fugaku_mpi_options_for_pjm": env_first_csv("FUGAKU_MPI_OPTIONS_FOR_PJM"),
        "slurm_account": env_first_str("SBD_SLURM_ACCOUNT"),
        "slurm_partition": env_first_str("SBD_SLURM_PARTITION"),
        "slurm_qpu": env_first_str("SBD_SLURM_QPU"),
        "slurm_memory": env_first_str("SBD_SLURM_MEMORY"),
        "slurm_ntasks": env_int("SBD_SLURM_NTASKS"),
        "slurm_gres": env_first_str("SBD_SLURM_GRES"),
        "script_filename": env_first_str("SBD_SCRIPT_FILENAME"),
        "metrics_artifact_key": env_first_str("SBD_METRICS_ARTIFACT_KEY"),
        "command_block_name": env_first_str("SBD_CMD_BLOCK_NAME"),
        "execution_profile_block_name": env_first_str("SBD_EXEC_PROFILE_BLOCK_NAME"),
        "hpc_profile_block_name": env_first_str("SBD_HPC_PROFILE_BLOCK_NAME"),
        "solver_block_name": env_first_str("SBD_SOLVER_BLOCK_NAME"),
        "options_variable_name": env_first_str("SBD_OPTIONS_VARIABLE"),
        "task_comm_size": env_int("SBD_TASK_COMM_SIZE"),
        "adet_comm_size": env_int("SBD_ADET_COMM_SIZE"),
        "bdet_comm_size": env_int("SBD_BDET_COMM_SIZE"),
        "block": env_int("SBD_BLOCK"),
        "iteration": env_int("SBD_ITERATION"),
        "tolerance": env_float("SBD_TOLERANCE"),
        "carryover_ratio": env_float("SBD_CARRYOVER_RATIO"),
        "carryover_type": env_int("SBD_CARRYOVER_TYPE"),
        "do_rdm": env_int("SBD_DO_RDM"),
        "solver_timeout_seconds": env_float("SBD_SOLVER_TIMEOUT_SECONDS"),
        "solver_mode": env_first_str("SBD_SOLVER_MODE"),
        "method": env_first_str("SBD_METHOD"),
        "shots": env_int("SBD_SHOTS"),
        "n_shot_batches": env_int("SBD_N_SHOT_BATCHES"),
        "saved_samples": env_first_str("SBD_SAVED_SAMPLES"),
        "sqd_options_json": env_first_str("SBD_OPTIONS_JSON"),
        "dynamical_decoupling": env_bool("SBD_DYNAMICAL_DECOUPLING"),
        "dd_sequence": env_first_str("SBD_DD_SEQUENCE"),
        "gate_twirling": env_bool("SBD_GATE_TWIRLING"),
        "measure_twirling": env_bool("SBD_MEASURE_TWIRLING"),
    }


def main() -> None:
    args = _parse_args()
    config = _load_config_file(args.config)
    env = _env_values()

    from qcsc_prefect_blocks.common.blocks import (
        CommandBlock,
        ExecutionProfileBlock,
        HPCProfileBlock,
    )

    sbd_solver_cls = _import_sbd_solver_block()
    _register_block_types(sbd_solver_cls)

    hpc_target = (
        str(_pick_value(args.hpc_target, config.get("hpc_target"), env.get("hpc_target"), "miyabi"))
        .strip()
        .lower()
    )
    if hpc_target not in {"miyabi", "fugaku", "slurm", "local"}:
        raise RuntimeError("'hpc_target' must be 'miyabi', 'fugaku', 'slurm', or 'local'.")
    is_miyabi = hpc_target == "miyabi"
    is_slurm = hpc_target == "slurm"
    is_local = hpc_target == "local"

    if is_local:
        # Local target runs the diag binary directly as a subprocess (no scheduler): no account,
        # partition, or queue needed. Used for in-allocation ROQUO runs and laptops.
        project = ""
    elif is_slurm:
        # For Slurm the "project" carries the Slurm --account (--slurm-account wins over --project).
        project = _pick_value(
            args.slurm_account,
            config.get("slurm_account"),
            env.get("slurm_account"),
            args.project,
            config.get("project"),
            env.get("project"),
            args.group,
            config.get("group"),
            env.get("group"),
        )
        if not project:
            raise RuntimeError(
                "Set the Slurm account via --slurm-account/SBD_SLURM_ACCOUNT (or --project)."
            )
    elif is_miyabi:
        project = _pick_value(
            args.project,
            config.get("project"),
            env.get("project"),
            args.group,
            config.get("group"),
            env.get("group"),
        )
    else:
        project = _pick_value(
            args.group,
            config.get("group"),
            env.get("group"),
            args.project,
            config.get("project"),
            env.get("project"),
        )
    if not project and not is_local:
        if is_miyabi:
            raise RuntimeError(
                "Set 'project' in --config/--project or SBD_PROJECT/MIYABI_PBS_PROJECT."
            )
        raise RuntimeError(
            "Set 'group' in --config/--group (preferred) or 'project' for compatibility, "
            "or use SBD_GROUP/FUGAKU_GROUP/FUGAKU_PROJECT."
        )

    # Local target needs no queue/partition.
    if is_local:
        queue = ""
    # For Slurm the "queue" carries the partition (--slurm-partition wins over --queue).
    elif is_slurm:
        queue = _pick_value(
            args.slurm_partition,
            config.get("slurm_partition"),
            env.get("slurm_partition"),
            args.queue,
            config.get("queue"),
            env.get("queue"),
        )
        if not queue:
            raise RuntimeError(
                "Set the Slurm partition via --slurm-partition/SBD_SLURM_PARTITION (or --queue)."
            )
    else:
        queue = _pick_value(args.queue, config.get("queue"), env.get("queue"))
        if not queue:
            raise RuntimeError(
                "Set 'queue' in --config/--queue or SBD_QUEUE/MIYABI_PBS_QUEUE/FUGAKU_RSCGRP."
            )

    work_dir = _pick_value(args.work_dir, config.get("work_dir"), env.get("work_dir"))
    if not work_dir:
        raise RuntimeError("Set 'work_dir' in --config, --work-dir, or SBD_WORK_DIR.")

    sbd_executable = _pick_value(
        args.sbd_executable, config.get("sbd_executable"), env.get("sbd_executable")
    )
    if not sbd_executable:
        raise RuntimeError("Set 'sbd_executable' in --config, --sbd-executable, or SBD_EXECUTABLE.")

    sbd_executable_uhf = _pick_value(
        args.sbd_executable_uhf,
        config.get("sbd_executable_uhf"),
        env.get("sbd_executable_uhf"),
    )

    if is_miyabi:
        launcher_default = "mpiexec.hydra"
    elif is_slurm or is_local:
        # Run the diag binary directly (no MPI launcher): in-allocation for slurm, subprocess
        # for local.
        launcher_default = "single"
    else:
        launcher_default = "mpiexec"
    launcher = str(
        _pick_value(args.launcher, config.get("launcher"), env.get("launcher"), launcher_default)
    ).strip()
    walltime = str(
        _pick_value(args.walltime, config.get("walltime"), env.get("walltime"), "02:00:00")
    ).strip()

    num_nodes = int(_pick_value(args.num_nodes, config.get("num_nodes"), env.get("num_nodes"), 1))
    mpiprocs = int(_pick_value(args.mpiprocs, config.get("mpiprocs"), env.get("mpiprocs"), 4))
    ompthreads_raw = _pick_value(args.ompthreads, config.get("ompthreads"), env.get("ompthreads"))
    ompthreads = int(ompthreads_raw) if ompthreads_raw is not None else None
    mem_raw = _pick_value(args.mem, config.get("mem"), env.get("mem"))
    mem = str(mem_raw).strip() if mem_raw is not None else None

    modules_default = ["intel/2023.2.0", "impi/2021.10.0"] if is_miyabi else []
    modules = _normalize_str_list(
        _pick_value(args.modules, config.get("modules"), env.get("modules"), modules_default)
    )
    mpi_options = _normalize_str_list(
        _pick_value(args.mpi_options, config.get("mpi_options"), env.get("mpi_options"), [])
    )
    pre_commands = (
        _normalize_str_list(
            _pick_value(args.pre_commands, config.get("pre_commands"), env.get("pre_commands"), [])
        )
        or []
    )
    environments = _normalize_str_dict(config.get("environments")) or {}

    fugaku_gfscache = str(
        _pick_value(
            args.fugaku_gfscache,
            config.get("fugaku_gfscache"),
            env.get("fugaku_gfscache"),
            "/vol0002",
        )
    ).strip()
    fugaku_spack_modules = _normalize_str_list(
        _pick_value(
            args.fugaku_spack_modules,
            config.get("fugaku_spack_modules"),
            env.get("fugaku_spack_modules"),
            [],
        )
    )
    fugaku_mpi_options_for_pjm = _normalize_str_list(
        _pick_value(
            args.fugaku_mpi_options_for_pjm,
            config.get("fugaku_mpi_options_for_pjm"),
            env.get("fugaku_mpi_options_for_pjm"),
            [],
        )
    )

    task_comm_size = int(
        _pick_value(args.task_comm_size, config.get("task_comm_size"), env.get("task_comm_size"), 1)
    )
    adet_comm_size = int(
        _pick_value(args.adet_comm_size, config.get("adet_comm_size"), env.get("adet_comm_size"), 1)
    )
    bdet_comm_size = int(
        _pick_value(args.bdet_comm_size, config.get("bdet_comm_size"), env.get("bdet_comm_size"), 1)
    )
    block = int(_pick_value(args.block, config.get("block"), env.get("block"), 4))
    iteration = int(_pick_value(args.iteration, config.get("iteration"), env.get("iteration"), 1))
    tolerance = float(
        _pick_value(args.tolerance, config.get("tolerance"), env.get("tolerance"), 1e-2)
    )
    carryover_ratio = float(
        _pick_value(
            args.carryover_ratio, config.get("carryover_ratio"), env.get("carryover_ratio"), 0.1
        )
    )
    carryover_type = int(
        _pick_value(
            args.carryover_type, config.get("carryover_type"), env.get("carryover_type"), 0
        )
    )
    do_rdm = int(
        _pick_value(args.do_rdm, config.get("do_rdm"), env.get("do_rdm"), 0)
    )
    # Total budget (queue wait + solve) the executor allows a single diag job. Raise this well
    # above the ~2h solve time when submitting to a congested queue (e.g. 2025-node large-queue
    # jobs) so the flow does not WaitTimeout while the job is still pending. Default 7200s.
    solver_timeout_seconds = float(
        _pick_value(
            args.solver_timeout_seconds,
            config.get("solver_timeout_seconds"),
            env.get("solver_timeout_seconds"),
            7200.0,
        )
    )
    solver_mode = str(
        _pick_value(args.solver_mode, config.get("solver_mode"), env.get("solver_mode"), "cpu")
    ).strip()
    method = str(
        _pick_value(args.method, config.get("method"), env.get("method"), "rhf")
    ).strip()
    resource_class = "gpu" if solver_mode == "gpu" else "cpu"
    user_args = _normalize_str_list(config.get("user_args")) or []
    if is_miyabi and solver_mode == "gpu":
        if "unset OMPI_MCA_mca_base_env_list" not in pre_commands:
            pre_commands.insert(0, "unset OMPI_MCA_mca_base_env_list")
        environments.setdefault("MIYABI", "G")
    modules = _normalize_modules_for_target(
        is_miyabi=is_miyabi, solver_mode=solver_mode, modules=modules
    )

    default_names = _default_block_names(hpc_target=hpc_target, solver_mode=solver_mode)

    command_block_name = str(
        _pick_value(
            args.command_block_name,
            config.get("command_block_name"),
            env.get("command_block_name"),
            "cmd-sbd-diag",
        )
    ).strip()

    execution_profile_block_name = str(
        _pick_value(
            args.execution_profile_block_name,
            config.get("execution_profile_block_name"),
            env.get("execution_profile_block_name"),
            default_names["execution_profile_block_name"],
        )
    ).strip()

    hpc_profile_block_name = str(
        _pick_value(
            args.hpc_profile_block_name,
            config.get("hpc_profile_block_name"),
            env.get("hpc_profile_block_name"),
            default_names["hpc_profile_block_name"],
        )
    ).strip()

    solver_block_name = str(
        _pick_value(
            args.solver_block_name,
            config.get("solver_block_name"),
            env.get("solver_block_name"),
            default_names["solver_block_name"],
        )
    ).strip()

    options_variable_name = str(
        _pick_value(
            args.options_variable_name,
            config.get("options_variable_name"),
            env.get("options_variable_name"),
            "sqd_options",
        )
    ).strip()

    if is_miyabi:
        script_default = "sbd_solver.pbs"
    elif is_slurm:
        script_default = "sbd_solver.slurm"
    else:
        script_default = "sbd_solver.pjm"
    script_filename = str(
        _pick_value(
            args.script_filename,
            config.get("script_filename"),
            env.get("script_filename"),
            script_default,
        )
    ).strip()

    metrics_artifact_key = str(
        _pick_value(
            args.metrics_artifact_key,
            config.get("metrics_artifact_key"),
            env.get("metrics_artifact_key"),
            "miyabi-sbd-metrics" if is_miyabi else "fugaku-sbd-metrics",
        )
    ).strip()

    shots_default = 500000 if is_miyabi else 50000
    shots = int(_pick_value(args.shots, config.get("shots"), env.get("shots"), shots_default))
    sqd_options_json = _pick_value(
        args.sqd_options_json, config.get("sqd_options_json"), env.get("sqd_options_json")
    )

    # Derive executable keys from backend x method, matching SBDSolverJob._executable_key:
    # sbd_diag (cpu/fugaku+rhf), sbd_diag_gpu, sbd_diag_uhf, sbd_diag_gpu_uhf.
    backend_suffix = "_gpu" if solver_mode == "gpu" else ""
    rhf_key = f"sbd_diag{backend_suffix}"
    uhf_key = f"sbd_diag{backend_suffix}_uhf"
    active_key = uhf_key if method == "uhf" else rhf_key

    if method == "uhf" and not sbd_executable_uhf:
        raise RuntimeError(
            "method='uhf' requires the UHF binary path: set 'sbd_executable_uhf' in --config, "
            "--sbd-executable-uhf, or SBD_EXECUTABLE_UHF (build it with `UHF=1 ./build_sbd_*.sh`)."
        )

    CommandBlock(
        command_name="sbd-diag",
        executable_key=active_key,
        description="SBD diagonalization executable",
        default_args=[],
    ).save(command_block_name, overwrite=True)

    ExecutionProfileBlock(
        profile_name=default_names["profile_name"],
        command_name="sbd-diag",
        resource_class=resource_class,
        num_nodes=num_nodes,
        mpiprocs=mpiprocs,
        ompthreads=ompthreads,
        walltime=walltime,
        mem=mem,
        launcher=launcher,
        mpi_options=mpi_options or [],
        modules=modules or [],
        pre_commands=pre_commands,
        environments=environments,
    ).save(execution_profile_block_name, overwrite=True)

    executable_path = str(Path(str(sbd_executable)).expanduser().resolve())

    # Register both the RHF and (when provided) the UHF binary so RHF and UHF solver blocks can
    # share one HPC profile. The active CommandBlock picks one via its executable_key.
    executable_map = {rhf_key: executable_path}
    if sbd_executable_uhf:
        executable_map[uhf_key] = str(Path(str(sbd_executable_uhf)).expanduser().resolve())

    if is_local:
        HPCProfileBlock(
            hpc_target="local",
            queue_cpu="",
            queue_gpu="",
            project_cpu="",
            project_gpu="",
            executable_map=executable_map,
        ).save(hpc_profile_block_name, overwrite=True)
    elif is_slurm:
        slurm_qpu = _pick_value(args.slurm_qpu, config.get("slurm_qpu"), env.get("slurm_qpu"))
        slurm_memory = _pick_value(
            args.slurm_memory, config.get("slurm_memory"), env.get("slurm_memory")
        )
        slurm_ntasks = _pick_value(
            args.slurm_ntasks, config.get("slurm_ntasks"), env.get("slurm_ntasks")
        )
        slurm_gres = _pick_value(args.slurm_gres, config.get("slurm_gres"), env.get("slurm_gres"))
        HPCProfileBlock(
            hpc_target="slurm",
            queue_cpu=str(queue),      # partition
            queue_gpu=str(queue),
            project_cpu=str(project),  # account
            project_gpu=str(project),
            executable_map=executable_map,
            slurm_qpu=str(slurm_qpu) if slurm_qpu else None,
            slurm_memory=str(slurm_memory) if slurm_memory else None,
            slurm_ntasks=int(slurm_ntasks) if slurm_ntasks else None,
            slurm_gres=str(slurm_gres) if slurm_gres else None,
        ).save(hpc_profile_block_name, overwrite=True)
    elif is_miyabi:
        HPCProfileBlock(
            hpc_target="miyabi",
            queue_cpu=str(queue),
            queue_gpu="regular-g",
            project_cpu=str(project),
            project_gpu=str(project),
            executable_map=executable_map,
        ).save(hpc_profile_block_name, overwrite=True)
    else:
        HPCProfileBlock(
            hpc_target="fugaku",
            queue_cpu=str(queue),
            queue_gpu=str(queue),
            project_cpu=str(project),
            project_gpu=str(project),
            executable_map=executable_map,
            gfscache=fugaku_gfscache or None,
            spack_modules=fugaku_spack_modules or [],
            mpi_options_for_pjm=fugaku_mpi_options_for_pjm or [],
        ).save(hpc_profile_block_name, overwrite=True)

    sbd_solver_cls(
        root_dir=str(Path(str(work_dir)).expanduser().resolve()),
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        script_filename=script_filename,
        metrics_artifact_key=metrics_artifact_key,
        task_comm_size=task_comm_size,
        adet_comm_size=adet_comm_size,
        bdet_comm_size=bdet_comm_size,
        block=block,
        iteration=iteration,
        tolerance=tolerance,
        carryover_ratio=carryover_ratio,
        carryover_type=carryover_type,
        do_rdm=do_rdm,
        timeout_seconds=solver_timeout_seconds,
        solver_mode=solver_mode,
        method=method,
        user_args=user_args,
    ).save(solver_block_name, overwrite=True)

    if sqd_options_json:
        options_value = json.loads(str(sqd_options_json))
    else:
        options_value = {"params": {"shots": shots}}

    # Error mitigation: inject dynamical decoupling and/or Pauli twirling into the SamplerV2
    # options. These pass through prefect_qiskit's REST gate (extra="ignore") to the IBM Runtime
    # server verbatim. Off by default so existing/legacy runs are byte-for-byte unchanged.
    dd_enabled = bool(_pick_value(args.dynamical_decoupling, config.get("dynamical_decoupling"),
                                  env.get("dynamical_decoupling"), False))
    gate_twirl = bool(_pick_value(args.gate_twirling, config.get("gate_twirling"),
                                  env.get("gate_twirling"), False))
    meas_twirl = bool(_pick_value(args.measure_twirling, config.get("measure_twirling"),
                                  env.get("measure_twirling"), False))
    dd_sequence = _pick_value(args.dd_sequence, config.get("dd_sequence"),
                              env.get("dd_sequence"), "XY4")
    if dd_enabled or gate_twirl or meas_twirl:
        params = options_value.setdefault("params", {})
        sampler_options = params.setdefault("options", {})
        if dd_enabled:
            sampler_options["dynamical_decoupling"] = {
                "enable": True,
                "sequence_type": str(dd_sequence),
            }
        if gate_twirl or meas_twirl:
            sampler_options["twirling"] = {
                "enable_gates": bool(gate_twirl),
                "enable_measure": bool(meas_twirl),
            }
        if gate_twirl:
            # IBM Runtime rejects gate twirling on fractional gates (error 1519). The LUCJ
            # ansatz transpiles to fractional rzz gates, and prefect_qiskit's get_target() does
            # not expose use_fractional_gates, so gate twirling cannot run through this stack.
            print(
                "  WARNING: gate twirling (enable_gates) is incompatible with the fractional "
                "rzz gates in the LUCJ circuit and will fail with IBM error 1519. Prefer "
                "--measure-twirling and --dynamical-decoupling instead."
            )

    # Shot batching: a harness-side loop count read by walker_sqd. It lives at the TOP LEVEL of the
    # options dict (NOT inside params, which is validated against the IBM REST schema).
    n_shot_batches = int(_pick_value(args.n_shot_batches, config.get("n_shot_batches"),
                                     env.get("n_shot_batches"), 1) or 1)
    if n_shot_batches > 1:
        options_value["n_shot_batches"] = n_shot_batches

    # Saved sample pool(s): a per-walker list of persisted-artifact paths, read by walker_sqd when
    # quantum_source='saved' to diagonalize offline without an IBM call. Top-level (harness-side),
    # NOT inside params. From CLI it is a list (nargs="+"); from env/config a whitespace/comma string.
    saved_samples = _pick_value(
        args.saved_samples, config.get("saved_samples"), env.get("saved_samples"), None
    )
    if saved_samples:
        if isinstance(saved_samples, str):
            saved_samples = [p for p in re.split(r"[,\s]+", saved_samples.strip()) if p]
        options_value["saved_samples"] = list(saved_samples)

    _set_variable(options_variable_name, options_value)

    print(f"Saved blocks and variable for SBD workflow (target={hpc_target})")
    print(f"  SBD solver block: {solver_block_name}")
    print(f"  Command block: {command_block_name}")
    print(f"  Execution profile block: {execution_profile_block_name}")
    print(f"  HPC profile block: {hpc_profile_block_name}")
    print(f"  Options variable: {options_variable_name}")
    print(f"  Work directory: {Path(str(work_dir)).expanduser().resolve()}")
    print(f"  Method: {method}")
    print(f"  SBD executable: {Path(str(sbd_executable)).expanduser().resolve()}")
    if sbd_executable_uhf:
        print(f"  SBD UHF executable: {Path(str(sbd_executable_uhf)).expanduser().resolve()}")
    print(f"  Script filename: {script_filename}")
    print(f"  Metrics artifact key: {metrics_artifact_key}")
    mitigation = options_value.get("params", {}).get("options", {})
    if mitigation:
        print(f"  Error mitigation: {json.dumps(mitigation)}")
    else:
        print("  Error mitigation: none")
    total_shots = options_value.get("params", {}).get("shots")
    if n_shot_batches > 1:
        print(f"  Shots: {total_shots} = {n_shot_batches} batches x "
              f"{int(total_shots) // n_shot_batches} (mixed via concatenate_shots)")
    else:
        print(f"  Shots: {total_shots} (single submission)")


if __name__ == "__main__":
    main()
