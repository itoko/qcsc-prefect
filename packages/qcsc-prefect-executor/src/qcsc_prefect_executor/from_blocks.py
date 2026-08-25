from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from qcsc_prefect_adapters.fugaku import builder as fugaku_builder
from qcsc_prefect_adapters.fugaku import runtime as fugaku_runtime
from qcsc_prefect_adapters.fugaku.builder import FugakuJobRequest
from qcsc_prefect_adapters.fugaku.runtime import FugakuPJMRuntime
from qcsc_prefect_adapters.local.runtime import LocalJobRequest
from qcsc_prefect_adapters.miyabi import builder as miyabi_builder
from qcsc_prefect_adapters.miyabi import runtime as miyabi_runtime
from qcsc_prefect_adapters.miyabi.builder import MiyabiJobRequest
from qcsc_prefect_adapters.miyabi.runtime import MiyabiPBSRuntime
from qcsc_prefect_adapters.slurm import builder as slurm_builder
from qcsc_prefect_adapters.slurm import runtime as slurm_runtime
from qcsc_prefect_adapters.slurm.builder import SlurmJobRequest
from qcsc_prefect_adapters.slurm.runtime import SlurmRuntime
from qcsc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock
from qcsc_prefect_core.models.execution_profile import ExecutionProfile
from qcsc_prefect_core.queue import QueueAwareSubmitGate, QueueProbe

from qcsc_prefect_executor.bulk.exceptions import (
    DuplicateJobKeyError,
    QueueFullError,
    SubmitError,
    TemporarySubmitError,
)
from qcsc_prefect_executor.bulk.models import (
    BulkJobRecord,
    BulkJobSpec,
    BulkJobStatus,
    BulkRunResult,
    SubmittedJob,
)
from qcsc_prefect_executor.bulk.native_manifest import create_native_bulk_group_manifests
from qcsc_prefect_executor.bulk.registry import BulkJobRegistry
from qcsc_prefect_executor.fugaku.run import run_fugaku_job
from qcsc_prefect_executor.local.run import run_local_job
from qcsc_prefect_executor.miyabi.run import run_miyabi_job
from qcsc_prefect_executor.slurm.run import run_slurm_job

_EXECUTION_PROFILE_OVERRIDE_KEYS = {
    "num_nodes",
    "mpiprocs",
    "ompthreads",
    "walltime",
    "mem",
    "launcher",
    "mpi_options",
    "modules",
    "pre_commands",
    "environments",
}
_SCRIPT_SUFFIX_BY_TARGET = {
    "miyabi": ".pbs",
    "fugaku": ".pjm",
    "slurm": ".slurm",
}
_KNOWN_SCRIPT_SUFFIXES = frozenset(_SCRIPT_SUFFIX_BY_TARGET.values())


@dataclass(frozen=True)
class SubmissionTarget:
    """Execution routing information resolved from Prefect blocks.

    Attributes:
        hpc_target: Runtime target name, such as ``"local"``, ``"miyabi"``,
            ``"fugaku"``, or ``"slurm"``.
        queue_name: Queue, partition, or resource-group name selected for the
            execution profile's resource class. Empty for local execution.
        project: Project, group, or account name selected for the resource
            class. Empty for local execution and scheduler targets that do not
            require an account.
    """

    hpc_target: str
    queue_name: str
    project: str


@dataclass(frozen=True)
class _PreparedBlockJob:
    submission_target: SubmissionTarget
    work_dir: Path
    script_filename: str | None
    exec_profile: ExecutionProfile
    req: Any


async def _resolve_loaded_block(value):
    if inspect.isawaitable(value):
        return await value
    return value


async def _load_block(block_cls, block_name: str):
    return await _resolve_loaded_block(block_cls.load(block_name))


def _resolve_submission_target_from_loaded_blocks(
    hpc_block: HPCProfileBlock, resource_class: str
) -> SubmissionTarget:
    if hpc_block.hpc_target == "local":
        return SubmissionTarget(hpc_target="local", queue_name="", project="")
    if resource_class == "gpu":
        return SubmissionTarget(
            hpc_target=hpc_block.hpc_target,
            queue_name=hpc_block.queue_gpu,
            project=hpc_block.project_gpu,
        )
    return SubmissionTarget(
        hpc_target=hpc_block.hpc_target,
        queue_name=hpc_block.queue_cpu,
        project=hpc_block.project_cpu,
    )


async def resolve_hpc_target(*, hpc_profile_block_name: str) -> str:
    """Load an ``HPCProfileBlock`` and return its execution target name.

    Args:
        hpc_profile_block_name: Prefect block document name for
            `qcsc_prefect_blocks.common.blocks.HPCProfileBlock`.

    Returns:
        The configured ``hpc_target`` value, for example ``"local"``,
        ``"miyabi"``, ``"fugaku"``, or ``"slurm"``.
    """

    hpc_block = await _load_block(HPCProfileBlock, hpc_profile_block_name)
    return str(hpc_block.hpc_target)


async def resolve_submission_target(
    *,
    hpc_profile_block_name: str,
    execution_profile_block_name: str,
) -> SubmissionTarget:
    """Resolve scheduler routing from block names without submitting a job.

    This helper is useful when a flow needs to inspect the target queue or
    project before it creates scheduler-specific filenames or logs. It loads
    the ``HPCProfileBlock`` and ``ExecutionProfileBlock`` and chooses CPU or
    GPU queue/project fields from the execution profile's ``resource_class``.

    Args:
        hpc_profile_block_name: Prefect block document name for target-specific
            scheduler settings.
        execution_profile_block_name: Prefect block document name for
            scheduler-independent execution settings.

    Returns:
        Resolved scheduler target, queue/partition/resource group, and
        project/account values.
    """

    hpc_block = await _load_block(HPCProfileBlock, hpc_profile_block_name)
    execution_profile_block = await _load_block(ExecutionProfileBlock, execution_profile_block_name)
    return _resolve_submission_target_from_loaded_blocks(
        hpc_block, execution_profile_block.resource_class
    )


def build_scheduler_script_filename(script_stem: str, hpc_target: str) -> str:
    """Build a scheduler-specific script filename from a logical stem.

    Existing scheduler suffixes are replaced, while names without a known
    scheduler suffix receive the target suffix appended. For example,
    ``"batch"`` becomes ``"batch.pbs"`` for Miyabi and ``"batch.slurm"`` for
    Slurm; ``"batch.pbs"`` becomes ``"batch.pjm"`` for Fugaku.

    Args:
        script_stem: Logical script name or existing scheduler script filename.
        hpc_target: Scheduler target name.

    Returns:
        Script filename with the suffix required by the scheduler target.

    Raises:
        NotImplementedError: If ``hpc_target`` is not supported.
    """

    suffix = _SCRIPT_SUFFIX_BY_TARGET.get(hpc_target)
    if suffix is None:
        raise NotImplementedError(f"Unsupported hpc_target for script naming: {hpc_target}")

    script_path = Path(script_stem)
    if script_path.suffix in _KNOWN_SCRIPT_SUFFIXES:
        script_path = script_path.with_suffix(suffix)
    else:
        script_path = script_path.with_name(script_path.name + suffix)
    return str(script_path)


async def resolve_scheduler_script_filename(
    *,
    script_stem: str,
    hpc_profile_block_name: str,
) -> str:
    """Resolve scheduler target from blocks and return a matching filename.

    Args:
        script_stem: Logical script name or existing scheduler script filename.
        hpc_profile_block_name: Prefect block document name used to determine
            the scheduler target.

    Returns:
        Scheduler-specific script filename.
    """

    hpc_target = await resolve_hpc_target(hpc_profile_block_name=hpc_profile_block_name)
    return build_scheduler_script_filename(script_stem, hpc_target)


def _build_execution_profile(
    *,
    command_block: CommandBlock,
    execution_profile_block: ExecutionProfileBlock,
    user_args: list[str] | None,
    execution_profile_overrides: dict[str, Any] | None,
) -> ExecutionProfile:
    arguments = list(command_block.default_args)
    if user_args:
        arguments.extend(user_args)

    profile_kwargs: dict[str, Any] = {
        "command_key": command_block.command_name,
        "num_nodes": execution_profile_block.num_nodes,
        "mpiprocs": execution_profile_block.mpiprocs,
        "ompthreads": execution_profile_block.ompthreads,
        "walltime": execution_profile_block.walltime,
        "mem": getattr(execution_profile_block, "mem", None),
        "launcher": execution_profile_block.launcher,
        "mpi_options": list(execution_profile_block.mpi_options),
        "modules": list(execution_profile_block.modules),
        "pre_commands": list(getattr(execution_profile_block, "pre_commands", [])),
        "environments": dict(execution_profile_block.environments),
        "arguments": arguments,
    }
    if execution_profile_overrides:
        invalid_keys = sorted(set(execution_profile_overrides) - _EXECUTION_PROFILE_OVERRIDE_KEYS)
        if invalid_keys:
            raise ValueError(
                "Unsupported execution_profile_overrides keys: " + ", ".join(invalid_keys)
            )
        for key, value in execution_profile_overrides.items():
            if key in {"mpi_options", "modules", "pre_commands"} and value is not None:
                profile_kwargs[key] = list(value)
            elif key == "environments" and value is not None:
                profile_kwargs[key] = dict(value)
            else:
                profile_kwargs[key] = value

    return ExecutionProfile(
        **profile_kwargs,
    )


def _default_fugaku_job_name(command_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", command_name).strip("-")
    if not normalized:
        return "prefect-job"
    return normalized[:63]


def _command_args_to_user_args(command_args: dict[str, Any] | None) -> list[str] | None:
    if not command_args:
        return None

    user_args: list[str] = []
    for key, value in command_args.items():
        option = str(key) if str(key).startswith("-") else "--" + str(key).replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            user_args.append(option)
        elif isinstance(value, (list, tuple)):
            for item in value:
                user_args.extend([option, str(item)])
        else:
            user_args.extend([option, str(value)])
    return user_args


def _resolve_named_argument(
    *,
    preferred: str | None,
    alias: str | None,
    label: str,
) -> str:
    value = preferred or alias
    if value is None:
        raise ValueError(f"{label} is required.")
    return value


def _ensure_registry_can_submit(*, registry: BulkJobRegistry, job_key: str) -> None:
    record = registry.get_job(job_key)
    if record is None:
        return
    if record.status.is_submit_candidate:
        return

    scheduler_part = (
        f", scheduler_job_id={record.scheduler_job_id}" if record.scheduler_job_id else ""
    )
    raise DuplicateJobKeyError(
        f"Bulk job key {job_key!r} already has status {record.status.value}"
        f"{scheduler_part}. Use a fresh job_key or registry for a new scheduler job."
    )


async def _resolve_default_bulk_queue_probe(
    *,
    hpc_profile_block: str,
    execution_profile_block: str,
    max_active_jobs: int,
    safety_margin: int,
    submit_mode: Literal["single", "native_bulk"] = "single",
) -> QueueProbe:
    submission_target = await resolve_submission_target(
        hpc_profile_block_name=hpc_profile_block,
        execution_profile_block_name=execution_profile_block,
    )
    if submission_target.hpc_target == "fugaku":
        from qcsc_prefect_adapters.fugaku.queue import FugakuQueueProbe

        return FugakuQueueProbe(
            max_active_jobs=max_active_jobs,
            safety_margin=safety_margin,
            project=submission_target.project,
            queue=submission_target.queue_name,
            capacity_mode="native_bulk" if submit_mode == "native_bulk" else "single",
        )

    raise ValueError(
        "queue_probe is required for bulk execution when hpc_target is "
        f"{submission_target.hpc_target!r}. Pass a scheduler-specific QueueProbe."
    )


def _build_bulk_run_result(
    *,
    registry: BulkJobRegistry,
    total_jobs: int,
) -> BulkRunResult:
    counts = registry.status_counts()
    failed_jobs = [
        record.job_key
        for record in registry.get_all_jobs()
        if record.status == BulkJobStatus.FAILED
    ]
    return BulkRunResult(
        total_jobs=total_jobs,
        status_counts=counts,
        succeeded=counts.get(BulkJobStatus.SUCCEEDED.value, 0),
        failed=counts.get(BulkJobStatus.FAILED.value, 0),
        cancelled=counts.get(BulkJobStatus.CANCELLED.value, 0),
        submit_deferred=counts.get(BulkJobStatus.SUBMIT_DEFERRED.value, 0),
        unknown=counts.get(BulkJobStatus.UNKNOWN.value, 0),
        registry_path=registry.path,
        failed_jobs=failed_jobs,
    )


def _has_failed_jobs(registry: BulkJobRegistry) -> bool:
    return registry.status_counts().get(BulkJobStatus.FAILED.value, 0) > 0


def _safe_bulk_group_key(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.=-]+", "-", value).strip("-")
    return safe or "bulk-group"


def _bulk_group_key_for_jobs(jobs: list[BulkJobRecord]) -> str:
    first = _safe_bulk_group_key(jobs[0].job_key)
    last = _safe_bulk_group_key(jobs[-1].job_key)
    stage = _safe_bulk_group_key(str(jobs[0].stage_id or "stage"))
    return f"native-bulk-{stage}-{first}-{last}-{len(jobs)}"[:180]


def _chunk_records(
    records: list[BulkJobRecord],
    *,
    chunk_size: int,
) -> list[list[BulkJobRecord]]:
    if chunk_size <= 0:
        raise ValueError("max_bulk_group_size must be positive.")
    return [records[index : index + chunk_size] for index in range(0, len(records), chunk_size)]


def _submit_limit_for_cycle(
    *,
    registry: BulkJobRegistry,
    initial_submit_count: int | None,
    max_submit_per_refill: int,
) -> int:
    if registry.bootstrap_done():
        return max(0, int(max_submit_per_refill))
    if initial_submit_count is None:
        return max(0, int(max_submit_per_refill))
    return max(0, int(initial_submit_count))


def _queue_available_slots(queue_probe: QueueProbe) -> int:
    try:
        capacity = queue_probe.get_capacity()
    except Exception:
        return 0
    return max(0, int(capacity.available_slots))


def _target_active_slots(
    *,
    registry: BulkJobRegistry,
    target_active_jobs: int | None,
) -> int | None:
    if target_active_jobs is None:
        return None
    return max(0, int(target_active_jobs) - registry.count_active_jobs())


def _native_bulk_submit_count(
    *,
    registry: BulkJobRegistry,
    queue_probe: QueueProbe,
    submit_limit: int,
    target_active_jobs: int | None,
) -> int:
    limits = [
        max(0, int(submit_limit)),
        _queue_available_slots(queue_probe),
        registry.count_submit_candidates(),
    ]
    target_slots = _target_active_slots(
        registry=registry,
        target_active_jobs=target_active_jobs,
    )
    if target_slots is not None:
        limits.append(target_slots)
    return max(0, min(limits))


def _validate_native_bulk_candidates(jobs: list[BulkJobRecord]) -> None:
    missing_stage = [
        job.job_key for job in jobs if not job.stage_id or not str(job.stage_id).strip()
    ]
    if missing_stage:
        raise ValueError(
            "submit_mode='native_bulk' requires stage_id for every selected job: "
            + ", ".join(missing_stage)
        )


def _mark_deferred_if_needed(
    *,
    registry: BulkJobRegistry,
    job_key: str,
    error: str | None,
) -> None:
    record = registry.get_job(job_key)
    if record is not None and record.status == BulkJobStatus.SUBMIT_DEFERRED:
        return
    registry.mark_submit_deferred(job_key, error=error)


def _mark_failed_if_needed(
    *,
    registry: BulkJobRegistry,
    job_key: str,
    error: str | None,
) -> None:
    record = registry.get_job(job_key)
    if record is not None and record.status in {
        BulkJobStatus.FAILED,
        BulkJobStatus.SUCCEEDED,
        BulkJobStatus.CANCELLED,
    }:
        return
    registry.mark_failed(job_key, error=error)


async def _prepare_job_from_blocks(
    *,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    work_dir: Path,
    script_filename: str | None,
    user_args: list[str] | None = None,
    fugaku_job_name: str | None = None,
    execution_profile_overrides: dict[str, Any] | None = None,
) -> _PreparedBlockJob:
    command_block = await _load_block(CommandBlock, command_block_name)
    execution_profile_block = await _load_block(ExecutionProfileBlock, execution_profile_block_name)
    hpc_block = await _load_block(HPCProfileBlock, hpc_profile_block_name)

    if execution_profile_block.command_name != command_block.command_name:
        raise ValueError(
            f"ExecutionProfileBlock '{execution_profile_block_name}' is for command "
            f"'{execution_profile_block.command_name}', but command block "
            f"'{command_block_name}' is '{command_block.command_name}'."
        )

    executable = hpc_block.executable_map.get(command_block.executable_key)
    if not executable:
        raise KeyError(
            f"Executable key '{command_block.executable_key}' was not found in "
            f"HPCProfileBlock '{hpc_profile_block_name}'."
        )

    submission_target = _resolve_submission_target_from_loaded_blocks(
        hpc_block, execution_profile_block.resource_class
    )
    if submission_target.hpc_target == "local":
        resolved_script_filename = None
    else:
        if not script_filename:
            raise ValueError("script_filename is required for scheduler execution targets.")
        resolved_script_filename = build_scheduler_script_filename(
            script_filename,
            submission_target.hpc_target,
        )
    if submission_target.hpc_target in {"miyabi", "fugaku"} and not submission_target.project:
        raise ValueError("Project/Group is empty. Update HPCProfileBlock project_cpu/project_gpu.")

    exec_profile = _build_execution_profile(
        command_block=command_block,
        execution_profile_block=execution_profile_block,
        user_args=user_args,
        execution_profile_overrides=execution_profile_overrides,
    )
    resolved_work_dir = Path(work_dir).expanduser().resolve()

    if submission_target.hpc_target == "local":
        req = LocalJobRequest(executable=executable)
    elif submission_target.hpc_target == "miyabi":
        req = MiyabiJobRequest(
            queue_name=submission_target.queue_name,
            project=submission_target.project,
            executable=executable,
        )
    elif submission_target.hpc_target == "fugaku":
        req = FugakuJobRequest(
            queue_name=submission_target.queue_name,
            project=submission_target.project,
            executable=executable,
            job_name=fugaku_job_name or _default_fugaku_job_name(command_block.command_name),
            gfscache=hpc_block.gfscache or "/vol0002",
            spack_modules=list(hpc_block.spack_modules) if hpc_block.spack_modules else [],
            mpi_options_for_pjm=list(hpc_block.mpi_options_for_pjm)
            if hpc_block.mpi_options_for_pjm
            else [],
            pjm_resources=list(hpc_block.pjm_resources) if hpc_block.pjm_resources else [],
        )
    elif submission_target.hpc_target == "slurm":
        req = SlurmJobRequest(
            partition=submission_target.queue_name,
            account=submission_target.project or None,
            executable=executable,
            qpu=hpc_block.slurm_qpu,
            memory=getattr(hpc_block, "slurm_memory", None),
            ntasks=getattr(hpc_block, "slurm_ntasks", None),
            gres=getattr(hpc_block, "slurm_gres", None),
        )
    else:
        raise NotImplementedError(
            f"hpc_target='{submission_target.hpc_target}' is not supported yet by "
            "run_job_from_blocks."
        )

    return _PreparedBlockJob(
        submission_target=submission_target,
        work_dir=resolved_work_dir,
        script_filename=resolved_script_filename,
        exec_profile=exec_profile,
        req=req,
    )


def _write_script_for_prepared_job(prepared: _PreparedBlockJob) -> Path:
    target = prepared.submission_target.hpc_target
    if prepared.script_filename is None:
        raise ValueError(f"hpc_target={target!r} does not use scheduler job scripts.")
    if target == "miyabi":
        script_text = miyabi_builder.render_script(
            work_dir=prepared.work_dir,
            exec_profile=prepared.exec_profile,
            req=prepared.req,
        )
        return miyabi_builder.write_script_file(
            work_dir=prepared.work_dir,
            filename=prepared.script_filename,
            text=script_text,
        )

    if target == "fugaku":
        script_basename = Path(prepared.script_filename).name
        script_text = fugaku_builder.render_script(
            work_dir=prepared.work_dir,
            exec_profile=prepared.exec_profile,
            req=prepared.req,
            script_basename=script_basename,
        )
        return fugaku_builder.write_script_file(
            work_dir=prepared.work_dir,
            filename=prepared.script_filename,
            text=script_text,
        )

    if target == "slurm":
        script_text = slurm_builder.render_script(
            work_dir=prepared.work_dir,
            exec_profile=prepared.exec_profile,
            req=prepared.req,
        )
        return slurm_builder.write_script_file(
            work_dir=prepared.work_dir,
            filename=prepared.script_filename,
            text=script_text,
        )

    raise NotImplementedError(f"Unsupported hpc_target for submit: {target}")


def _exception_text(exc: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        parts.append(str(current))
        current = current.__cause__
    return "\n".join(part for part in parts if part)


def _classify_submit_exception(exc: BaseException) -> SubmitError:
    if isinstance(exc, SubmitError):
        return exc

    message = _exception_text(exc).lower()
    queue_full_patterns = {
        "queue full",
        "job limit",
        "submit limit",
        "accept limit",
        "ru-accept",
        "too many jobs",
        "maximum number of jobs",
        "exceed",
        "exceeded",
    }
    temporary_patterns = {
        "temporar",
        "try again",
        "unavailable",
        "timeout",
        "timed out",
        "busy",
        "connection",
        "rate limit",
    }

    if any(pattern in message for pattern in queue_full_patterns):
        return QueueFullError(_exception_text(exc))
    if any(pattern in message for pattern in temporary_patterns):
        return TemporarySubmitError(_exception_text(exc))
    return SubmitError(_exception_text(exc))


async def _submit_prepared_job(
    prepared: _PreparedBlockJob,
    *,
    fugaku_no_check_directory: bool = False,
) -> str:
    target = prepared.submission_target.hpc_target
    if target == "local":
        raise ValueError(
            "Scheduler submit APIs do not support hpc_target='local'. "
            "Use run_job_from_blocks() for local execution."
        )
    script_path = _write_script_for_prepared_job(prepared)

    if target == "miyabi":
        submit = await MiyabiPBSRuntime().submit(script_path, cwd=prepared.work_dir)
    elif target == "fugaku":
        submit = await FugakuPJMRuntime(
            no_check_directory=fugaku_no_check_directory
        ).submit(script_path, cwd=prepared.work_dir)
    elif target == "slurm":
        submit = await SlurmRuntime().submit(script_path, cwd=prepared.work_dir)
    else:
        raise NotImplementedError(f"Unsupported hpc_target for submit: {target}")

    return submit.job_id


def _write_fugaku_native_bulk_script(
    *,
    prepared: _PreparedBlockJob,
    bulk_manifest_dir: Path,
) -> Path:
    if prepared.submission_target.hpc_target != "fugaku":
        raise ValueError("submit_mode='native_bulk' is only supported for Fugaku/PJM.")

    script_basename = Path(prepared.script_filename).name
    script_text = fugaku_builder.render_manifest_bulk_script(
        work_dir=prepared.work_dir,
        bulk_manifest_dir=bulk_manifest_dir,
        exec_profile=prepared.exec_profile,
        req=prepared.req,
        script_basename=script_basename,
    )
    return fugaku_builder.write_script_file(
        work_dir=prepared.work_dir,
        filename=prepared.script_filename,
        text=script_text,
    )


async def _submit_native_bulk_group_from_blocks(
    *,
    registry: BulkJobRegistry,
    jobs: list[BulkJobRecord],
    command_block: str,
    execution_profile_block: str,
    hpc_profile_block: str,
    fugaku_no_check_directory: bool = False,
) -> str:
    bulk_group_key = _bulk_group_key_for_jobs(jobs)
    bulk_group_dir = registry.path.parent / "native-bulk" / bulk_group_key
    manifest_group = create_native_bulk_group_manifests(
        bulk_group_dir=bulk_group_dir,
        jobs=jobs,
    )
    prepared = await _prepare_job_from_blocks(
        command_block_name=command_block,
        execution_profile_block_name=execution_profile_block,
        hpc_profile_block_name=hpc_profile_block,
        work_dir=manifest_group.bulk_group_dir,
        script_filename=bulk_group_key,
        user_args=None,
        fugaku_job_name=bulk_group_key[:63],
    )
    script_path = _write_fugaku_native_bulk_script(
        prepared=prepared,
        bulk_manifest_dir=manifest_group.manifest_dir,
    )
    parent_job_id = await FugakuPJMRuntime(
        no_check_directory=fugaku_no_check_directory
    ).submit_bulk(
        script_path,
        manifest_group.bulk_count,
        cwd=manifest_group.bulk_group_dir,
    )

    for bulk_index, job in enumerate(jobs):
        scheduler_subjob_id = f"{parent_job_id}[{bulk_index}]"
        registry.mark_submitted(
            job.job_key,
            scheduler_subjob_id,
            submit_mode="native_bulk",
            bulk_group_key=bulk_group_key,
            bulk_parent_job_id=parent_job_id,
            bulk_index=bulk_index,
            scheduler_subjob_id=scheduler_subjob_id,
        )

    return parent_job_id


async def _submit_native_bulk_cycle_from_blocks(
    *,
    registry: BulkJobRegistry,
    command_block: str,
    execution_profile_block: str,
    hpc_profile_block: str,
    queue_probe: QueueProbe,
    submit_limit: int,
    max_bulk_group_size: int,
    target_active_jobs: int | None,
    stop_on_first_failure: bool,
    fugaku_no_check_directory: bool = False,
) -> bool:
    submit_count = _native_bulk_submit_count(
        registry=registry,
        queue_probe=queue_probe,
        submit_limit=submit_limit,
        target_active_jobs=target_active_jobs,
    )
    if submit_count <= 0:
        return False

    selected_jobs = registry.get_submit_candidates_fifo(limit=submit_count)
    if not selected_jobs:
        return False

    _validate_native_bulk_candidates(selected_jobs)
    for chunk in _chunk_records(selected_jobs, chunk_size=max_bulk_group_size):
        try:
            await _submit_native_bulk_group_from_blocks(
                registry=registry,
                jobs=chunk,
                command_block=command_block,
                execution_profile_block=execution_profile_block,
                hpc_profile_block=hpc_profile_block,
                fugaku_no_check_directory=fugaku_no_check_directory,
            )
        except Exception as exc:
            classified = _classify_submit_exception(exc)
            if isinstance(classified, QueueFullError | TemporarySubmitError):
                for job in chunk:
                    _mark_deferred_if_needed(
                        registry=registry,
                        job_key=job.job_key,
                        error=str(classified),
                    )
                return True

            for job in chunk:
                _mark_failed_if_needed(
                    registry=registry,
                    job_key=job.job_key,
                    error=str(classified),
                )
            if stop_on_first_failure:
                return False

    return False


async def submit_job_from_blocks(
    *,
    work_dir: Path,
    job_key: str,
    command_block: str | None = None,
    execution_profile_block: str | None = None,
    hpc_profile_block: str | None = None,
    command_args: dict[str, Any] | None = None,
    registry: BulkJobRegistry | None = None,
    command_block_name: str | None = None,
    execution_profile_block_name: str | None = None,
    hpc_profile_block_name: str | None = None,
    fugaku_no_check_directory: bool = False,
) -> SubmittedJob:
    """Submit one block-defined HPC job without waiting for completion.

    Queue-full and retryable scheduler failures are recorded as
    ``SUBMIT_DEFERRED`` when a registry is provided, then raised so a future
    refill loop can stop submitting more jobs in the current cycle. Set
    ``fugaku_no_check_directory`` to opt into ``pjsub --no-check-directory`` for
    Fugaku submissions only.
    """

    resolved_command_block = _resolve_named_argument(
        preferred=command_block,
        alias=command_block_name,
        label="command_block",
    )
    resolved_execution_profile_block = _resolve_named_argument(
        preferred=execution_profile_block,
        alias=execution_profile_block_name,
        label="execution_profile_block",
    )
    resolved_hpc_profile_block = _resolve_named_argument(
        preferred=hpc_profile_block,
        alias=hpc_profile_block_name,
        label="hpc_profile_block",
    )

    if registry is not None:
        if registry.get_job(job_key) is None:
            registry.upsert_jobs(
                [
                    BulkJobSpec(
                        job_key=job_key,
                        work_dir=Path(work_dir),
                        command_args=dict(command_args or {}),
                    )
                ]
            )
        _ensure_registry_can_submit(registry=registry, job_key=job_key)

    prepared = await _prepare_job_from_blocks(
        command_block_name=resolved_command_block,
        execution_profile_block_name=resolved_execution_profile_block,
        hpc_profile_block_name=resolved_hpc_profile_block,
        work_dir=work_dir,
        script_filename=job_key,
        user_args=_command_args_to_user_args(command_args),
    )

    try:
        scheduler_job_id = await _submit_prepared_job(
            prepared,
            fugaku_no_check_directory=fugaku_no_check_directory,
        )
    except Exception as exc:
        classified = _classify_submit_exception(exc)
        if registry is not None:
            if isinstance(classified, QueueFullError | TemporarySubmitError):
                registry.mark_submit_deferred(job_key, error=str(classified))
            else:
                registry.mark_failed(job_key, error=str(classified))
        raise classified from exc

    if registry is not None:
        registry.mark_submitted(job_key, scheduler_job_id)

    return SubmittedJob(
        job_key=job_key,
        scheduler_job_id=scheduler_job_id,
        status=BulkJobStatus.SUBMITTED,
        work_dir=prepared.work_dir,
    )


def _parse_fugaku_pjstat_rows(stdout: str) -> dict[str, dict[str, Any]]:
    return fugaku_runtime.parse_pjstat_rows(stdout)


_FUGAKU_SUBJOB_ID_RE = re.compile(r"^(\d+)\[(\d+)\]$")
_FUGAKU_SUBJOB_RANGE_RE = re.compile(r"^(\d+)\[(\d+)-(\d+)\]$")
_PARENT_FALLBACK_JOB_ID_KEY = "_qcsc_parent_fallback_job_id"


def _parse_fugaku_subjob_id(job_id: str) -> tuple[str, int] | None:
    match = _FUGAKU_SUBJOB_ID_RE.match(str(job_id).strip())
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _parse_fugaku_subjob_range(job_id: str) -> tuple[str, int, int] | None:
    match = _FUGAKU_SUBJOB_RANGE_RE.match(str(job_id).strip())
    if match is None:
        return None
    start = int(match.group(2))
    end = int(match.group(3))
    if end < start:
        return None
    return match.group(1), start, end


def _format_fugaku_subjob_ranges(parent_job_id: str, indices: list[int]) -> list[str]:
    if not indices:
        return []

    formatted: list[str] = []
    sorted_indices = sorted(set(indices))
    range_start = sorted_indices[0]
    previous = range_start
    for index in sorted_indices[1:]:
        if index == previous + 1:
            previous = index
            continue

        formatted.append(
            f"{parent_job_id}[{range_start}]"
            if range_start == previous
            else f"{parent_job_id}[{range_start}-{previous}]"
        )
        range_start = previous = index

    formatted.append(
        f"{parent_job_id}[{range_start}]"
        if range_start == previous
        else f"{parent_job_id}[{range_start}-{previous}]"
    )
    return formatted


def _fugaku_pjstat_query_ids(scheduler_job_ids: list[str]) -> list[str]:
    parent_indices: dict[str, list[int]] = {}
    passthrough: list[str] = []

    for scheduler_job_id in dict.fromkeys(str(job_id).strip() for job_id in scheduler_job_ids):
        if not scheduler_job_id:
            continue
        parsed_subjob = _parse_fugaku_subjob_id(scheduler_job_id)
        if parsed_subjob is not None:
            parent_job_id, bulk_index = parsed_subjob
            parent_indices.setdefault(parent_job_id, []).append(bulk_index)
            continue

        parsed_range = _parse_fugaku_subjob_range(scheduler_job_id)
        if parsed_range is not None:
            parent_job_id, start, end = parsed_range
            passthrough.append(f"{parent_job_id}[{start}-{end}]")
            continue

        passthrough.append(scheduler_job_id)

    query_ids = list(passthrough)
    for parent_job_id, indices in parent_indices.items():
        query_ids.extend(_format_fugaku_subjob_ranges(parent_job_id, indices))
    return query_ids


def _select_fugaku_rows_for_requested_ids(
    *,
    requested_ids: list[str],
    rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for scheduler_job_id in requested_ids:
        if scheduler_job_id in rows:
            selected[scheduler_job_id] = rows[scheduler_job_id]
            continue

        parsed_subjob = _parse_fugaku_subjob_id(scheduler_job_id)
        if parsed_subjob is None:
            continue

        parent_job_id, _bulk_index = parsed_subjob
        parent_row = rows.get(parent_job_id)
        if parent_row is None:
            continue

        fallback_row = dict(parent_row)
        fallback_row["JOB_ID"] = scheduler_job_id
        fallback_row[_PARENT_FALLBACK_JOB_ID_KEY] = parent_job_id
        selected[scheduler_job_id] = fallback_row

    return selected


async def _query_fugaku_history_statuses(
    query_ids: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    suffix = tuple(query_ids or [])
    for args in [
        ("pjstat", "-v", "-H"),
        ("pjstat", "-H", "-v"),
        ("pjstat", "-H"),
    ]:
        try:
            stdout = await fugaku_runtime.run_command(*args, *suffix)
        except Exception:
            continue
        return _parse_fugaku_pjstat_rows(stdout)
    return {}


def _parse_slurm_sacct_rows(stdout: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 6:
            continue
        job_id = fields[0].strip()
        if not job_id or "." in job_id:
            continue
        rows[job_id] = {
            "JobID": job_id,
            "State": fields[1].strip(),
            "ExitCode": fields[2].strip(),
            "Elapsed": fields[3].strip(),
            "AllocCPUS": fields[4].strip(),
            "NodeList": fields[5].strip(),
        }
    return rows


def _parse_miyabi_qstat_rows(stdout: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    current_job_id: str | None = None
    current_key = ""
    current_row: dict[str, Any] = {}

    def save_current() -> None:
        if current_job_id:
            rows[current_job_id] = dict(current_row)

    for line in stdout.splitlines():
        if line.startswith("Job Id:"):
            save_current()
            current_job_id = line.split(":", 1)[1].strip()
            current_row = {"Job_Id": current_job_id}
            current_key = ""
            continue

        if current_job_id is None or not line.strip():
            continue

        if line.startswith("\t") and current_key:
            current_row[current_key] = str(current_row[current_key]) + line.strip()
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        current_key = key.strip()
        current_row[current_key] = value.strip()

    save_current()
    return rows


async def _query_scheduler_statuses(
    *,
    hpc_target: str,
    scheduler_job_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not scheduler_job_ids:
        return {}

    requested = set(scheduler_job_ids)
    if hpc_target == "fugaku":
        query_ids = _fugaku_pjstat_query_ids(scheduler_job_ids)
        active_stdout = await fugaku_runtime.run_command("pjstat", "-v", *query_ids)
        rows = _parse_fugaku_pjstat_rows(active_stdout)
        missing = sorted(requested - set(rows))
        if missing:
            rows.update(await _query_fugaku_history_statuses(_fugaku_pjstat_query_ids(missing)))
        return _select_fugaku_rows_for_requested_ids(
            requested_ids=scheduler_job_ids,
            rows=rows,
        )

    if hpc_target == "slurm":
        stdout = await slurm_runtime.run_command(
            "sacct",
            "-j",
            ",".join(scheduler_job_ids),
            "--format=JobID,State,ExitCode,Elapsed,AllocCPUS,NodeList",
            "--parsable2",
            "--noheader",
        )
        return {
            job_id: row
            for job_id, row in _parse_slurm_sacct_rows(stdout).items()
            if job_id in requested
        }

    if hpc_target == "miyabi":
        stdout = await miyabi_runtime.run_command("qstat", "-f", *scheduler_job_ids)
        rows = _parse_miyabi_qstat_rows(stdout)
        return {job_id: row for job_id, row in rows.items() if job_id in requested}

    raise NotImplementedError(f"Unsupported hpc_target for monitor: {hpc_target}")


def _bulk_status_from_scheduler_row(hpc_target: str, row: dict[str, Any]) -> BulkJobStatus:
    if hpc_target == "fugaku":
        state = str(row.get("ST", "")).strip().upper()
        if state == "ACC":
            return BulkJobStatus.SUBMITTED
        if state in {"QUE", "Q", "HLD"}:
            return BulkJobStatus.QUEUED
        if state in {"RUN", "R", "RNA", "RNE", "RNO", "RNP", "RSM", "SPD", "SPP"}:
            return BulkJobStatus.RUNNING
        if state == "EXT":
            return BulkJobStatus.UNKNOWN
        if state == "CCL":
            return BulkJobStatus.CANCELLED
        if state in {"ERR", "RJT"}:
            return BulkJobStatus.FAILED
        return BulkJobStatus.UNKNOWN

    if hpc_target == "slurm":
        state = str(row.get("State", "")).strip().upper().split()[0].rstrip("+")
        if state in {"PENDING", "CONFIGURING"}:
            return BulkJobStatus.QUEUED
        if state in {"RUNNING", "COMPLETING", "SUSPENDED"}:
            return BulkJobStatus.RUNNING
        if state == "COMPLETED":
            return BulkJobStatus.SUCCEEDED
        if state == "CANCELLED":
            return BulkJobStatus.CANCELLED
        if state in {
            "BOOT_FAIL",
            "DEADLINE",
            "FAILED",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "PREEMPTED",
            "TIMEOUT",
        }:
            return BulkJobStatus.FAILED
        return BulkJobStatus.UNKNOWN

    if hpc_target == "miyabi":
        state = str(row.get("job_state", row.get("state", ""))).strip().upper()
        exit_status = str(row.get("Exit_status", "")).strip()
        if state in {"Q", "H", "W", "T"}:
            return BulkJobStatus.QUEUED
        if state in {"R", "E"}:
            return BulkJobStatus.RUNNING
        if state in {"C", "F"} or exit_status:
            return BulkJobStatus.SUCCEEDED if exit_status in {"", "0"} else BulkJobStatus.FAILED
        return BulkJobStatus.UNKNOWN

    return BulkJobStatus.UNKNOWN


def _record_has_success_evidence(record) -> bool:
    if not record.expected_outputs:
        return False
    paths = [
        path if path.is_absolute() else record.work_dir / path for path in record.expected_outputs
    ]
    return all(path.exists() for path in paths)


def _monitor_status_from_scheduler_row(
    *,
    hpc_target: str,
    row: dict[str, Any],
    record: Any | None,
) -> tuple[BulkJobStatus, str | None]:
    status = _bulk_status_from_scheduler_row(hpc_target, row)
    if hpc_target != "fugaku":
        error = None if status != BulkJobStatus.UNKNOWN else "unknown scheduler state"
        return status, error

    state = str(row.get("ST", "")).strip().upper()
    parent_fallback_job_id = row.get(_PARENT_FALLBACK_JOB_ID_KEY)
    if parent_fallback_job_id is not None:
        if record is not None and _record_has_success_evidence(record):
            return BulkJobStatus.SUCCEEDED, None
        return (
            BulkJobStatus.UNKNOWN,
            f"subjob row was not found; parent job {parent_fallback_job_id} is weak evidence only",
        )

    if state == "EXT":
        if record is not None and record.expected_outputs:
            if _record_has_success_evidence(record):
                return BulkJobStatus.SUCCEEDED, None
            return (
                BulkJobStatus.FAILED,
                "PJM reported EXT but expected outputs are missing",
            )
        return (
            BulkJobStatus.UNKNOWN,
            "PJM reported EXT without expected_outputs evidence",
        )

    error = None if status != BulkJobStatus.UNKNOWN else "unknown scheduler state"
    return status, error


def _records_by_scheduler_id(
    registry: BulkJobRegistry | None,
) -> dict[str, Any]:
    if registry is None:
        return {}
    records_by_scheduler_id: dict[str, Any] = {}
    for record in registry.get_all_jobs():
        scheduler_id = record.effective_scheduler_job_id
        if scheduler_id:
            records_by_scheduler_id[str(scheduler_id)] = record
    return records_by_scheduler_id


def _update_registry_for_monitor_status(
    *,
    registry: BulkJobRegistry,
    job_key: str,
    status: BulkJobStatus,
    error: str | None = None,
) -> None:
    if status == BulkJobStatus.QUEUED:
        registry.record_monitor_attempt(job_key)
        registry.mark_queued(job_key)
    elif status == BulkJobStatus.RUNNING:
        registry.record_monitor_attempt(job_key)
        registry.mark_running(job_key)
    elif status == BulkJobStatus.SUCCEEDED:
        registry.record_monitor_attempt(job_key)
        registry.mark_succeeded(job_key)
    elif status == BulkJobStatus.FAILED:
        registry.record_monitor_attempt(job_key)
        registry.mark_failed(job_key, error=error)
    elif status == BulkJobStatus.CANCELLED:
        registry.record_monitor_attempt(job_key)
        registry.mark_cancelled(job_key, error=error)
    elif status == BulkJobStatus.UNKNOWN:
        registry.mark_unknown(job_key, error=error)
    else:
        registry.record_monitor_attempt(job_key)


async def monitor_jobs_many(
    *,
    scheduler_job_ids: list[str],
    hpc_profile_block: str | None = None,
    registry: BulkJobRegistry | None = None,
    hpc_profile_block_name: str | None = None,
) -> dict[str, BulkJobStatus]:
    """Monitor many scheduler jobs with one aggregated scheduler query per target."""

    resolved_hpc_profile_block = _resolve_named_argument(
        preferred=hpc_profile_block,
        alias=hpc_profile_block_name,
        label="hpc_profile_block",
    )
    hpc_target = await resolve_hpc_target(hpc_profile_block_name=resolved_hpc_profile_block)

    requested_ids = list(dict.fromkeys(scheduler_job_ids))
    if not requested_ids:
        return {}

    query_error: str | None = None
    try:
        scheduler_rows = await _query_scheduler_statuses(
            hpc_target=hpc_target,
            scheduler_job_ids=requested_ids,
        )
    except Exception as exc:
        scheduler_rows = {}
        query_error = _exception_text(exc)

    records_by_scheduler_id = _records_by_scheduler_id(registry)
    results: dict[str, BulkJobStatus] = {}

    for scheduler_job_id in requested_ids:
        row = scheduler_rows.get(scheduler_job_id)
        record = records_by_scheduler_id.get(scheduler_job_id)
        if record is not None and record.status == BulkJobStatus.SUCCEEDED:
            status = BulkJobStatus.SUCCEEDED
            error = None
        elif row is not None:
            status, error = _monitor_status_from_scheduler_row(
                hpc_target=hpc_target,
                row=row,
                record=record,
            )
        elif record is not None and _record_has_success_evidence(record):
            status = BulkJobStatus.SUCCEEDED
            error = None
        else:
            status = BulkJobStatus.UNKNOWN
            error = query_error or "job was not found in scheduler output"

        results[scheduler_job_id] = status
        if registry is not None and record is not None:
            _update_registry_for_monitor_status(
                registry=registry,
                job_key=record.job_key,
                status=status,
                error=error,
            )

    return results


async def run_jobs_from_blocks_bulk(
    *,
    jobs: list[BulkJobSpec],
    command_block: str,
    execution_profile_block: str,
    hpc_profile_block: str,
    registry_path: Path,
    queue_probe: QueueProbe | None = None,
    max_active_jobs: int = 1000,
    safety_margin: int = 20,
    max_submit_per_refill: int = 100,
    submit_mode: Literal["single", "native_bulk"] = "single",
    initial_submit_count: int | None = None,
    max_bulk_group_size: int = 100,
    target_active_jobs: int | None = None,
    poll_interval_seconds: int = 60,
    refill_interval_seconds: int = 60,
    stop_on_first_failure: bool = False,
    fugaku_no_check_directory: bool = False,
) -> BulkRunResult:
    """Run many block-defined HPC jobs through one queue-aware bulk loop.

    This API submits and monitors scheduler jobs from a shared pending pool. It
    does not create one Prefect task per scheduler job, and wave identifiers on
    ``BulkJobSpec`` remain registry metadata for downstream workflow readiness
    checks rather than submit units. The default ``submit_mode="single"`` keeps
    using one scheduler submit per logical job. Fugaku native bulk submission is
    an explicit opt-in path via ``submit_mode="native_bulk"``. Set
    ``fugaku_no_check_directory`` to opt into ``pjsub --no-check-directory`` for
    Fugaku submissions only.
    """

    registry = BulkJobRegistry(registry_path)
    registry.upsert_jobs(jobs)
    registry.refresh_completed_jobs_from_outputs()
    total_jobs = len({job.job_key for job in jobs})

    if submit_mode not in {"single", "native_bulk"}:
        raise ValueError("submit_mode must be 'single' or 'native_bulk'.")

    if registry.all_terminal() or (stop_on_first_failure and _has_failed_jobs(registry)):
        return _build_bulk_run_result(registry=registry, total_jobs=total_jobs)

    if submit_mode == "native_bulk":
        submission_target = await resolve_submission_target(
            hpc_profile_block_name=hpc_profile_block,
            execution_profile_block_name=execution_profile_block,
        )
        if submission_target.hpc_target != "fugaku":
            raise ValueError("submit_mode='native_bulk' is only supported for Fugaku/PJM.")
        if max_bulk_group_size <= 0:
            raise ValueError("max_bulk_group_size must be positive.")

    resolved_queue_probe = queue_probe or await _resolve_default_bulk_queue_probe(
        hpc_profile_block=hpc_profile_block,
        execution_profile_block=execution_profile_block,
        max_active_jobs=max_active_jobs,
        safety_margin=safety_margin,
        submit_mode=submit_mode,
    )
    submit_gate = QueueAwareSubmitGate(
        queue_probe=resolved_queue_probe,
        max_active_jobs=target_active_jobs if target_active_jobs is not None else max_active_jobs,
        safety_margin=safety_margin,
        max_submit_per_refill=max_submit_per_refill,
    )

    loop = asyncio.get_running_loop()
    next_refill_at = 0.0

    while not registry.all_terminal():
        monitorable_jobs = [
            job for job in registry.get_monitorable_jobs() if job.effective_scheduler_job_id
        ]
        if monitorable_jobs:
            await monitor_jobs_many(
                hpc_profile_block=hpc_profile_block,
                scheduler_job_ids=[str(job.effective_scheduler_job_id) for job in monitorable_jobs],
                registry=registry,
            )

        registry.refresh_completed_jobs_from_outputs()

        if stop_on_first_failure and _has_failed_jobs(registry):
            break

        now = loop.time()
        if now >= next_refill_at:
            submit_limit = _submit_limit_for_cycle(
                registry=registry,
                initial_submit_count=initial_submit_count,
                max_submit_per_refill=max_submit_per_refill,
            )
            stop_after_deferred_submit = False
            if submit_mode == "native_bulk":
                stop_after_deferred_submit = await _submit_native_bulk_cycle_from_blocks(
                    registry=registry,
                    command_block=command_block,
                    execution_profile_block=execution_profile_block,
                    hpc_profile_block=hpc_profile_block,
                    queue_probe=resolved_queue_probe,
                    submit_limit=submit_limit,
                    max_bulk_group_size=max_bulk_group_size,
                    target_active_jobs=target_active_jobs,
                    stop_on_first_failure=stop_on_first_failure,
                    fugaku_no_check_directory=fugaku_no_check_directory,
                )
            else:
                pre_candidates = registry.get_submit_candidates(limit=submit_limit)
                if pre_candidates:
                    submit_gate.max_submit_per_refill = submit_limit
                    submit_count = min(
                        submit_gate.allowed_submit_count(),
                        len(pre_candidates),
                    )
                    for job in pre_candidates[:submit_count]:
                        try:
                            await submit_job_from_blocks(
                                command_block=command_block,
                                execution_profile_block=execution_profile_block,
                                hpc_profile_block=hpc_profile_block,
                                work_dir=job.work_dir,
                                job_key=job.job_key,
                                command_args=job.command_args,
                                registry=registry,
                                fugaku_no_check_directory=fugaku_no_check_directory,
                            )
                        except QueueFullError as exc:
                            _mark_deferred_if_needed(
                                registry=registry,
                                job_key=job.job_key,
                                error=str(exc),
                            )
                            break
                        except TemporarySubmitError as exc:
                            _mark_deferred_if_needed(
                                registry=registry,
                                job_key=job.job_key,
                                error=str(exc),
                            )
                            break
                        except Exception as exc:
                            _mark_failed_if_needed(
                                registry=registry,
                                job_key=job.job_key,
                                error=_exception_text(exc),
                            )
                            if stop_on_first_failure:
                                break

            if stop_after_deferred_submit:
                break

            next_refill_at = now + max(0.0, float(refill_interval_seconds))

        if stop_on_first_failure and _has_failed_jobs(registry):
            break
        if registry.all_terminal():
            break

        if submit_mode == "native_bulk":
            active_jobs = registry.count_active_jobs()
            submit_candidates = registry.count_submit_candidates()
            if active_jobs == 0 and submit_candidates == 0:
                break

        sleep_seconds = max(0.0, float(poll_interval_seconds))
        if sleep_seconds == 0 and next_refill_at > loop.time():
            sleep_seconds = max(0.0, next_refill_at - loop.time())
        await asyncio.sleep(sleep_seconds)

    return _build_bulk_run_result(registry=registry, total_jobs=total_jobs)


async def run_job_from_blocks(
    *,
    command_block_name: str,
    execution_profile_block_name: str,
    hpc_profile_block_name: str,
    work_dir: Path,
    script_filename: str | None = None,
    user_args: list[str] | None = None,
    watch_poll_interval: float = 10.0,
    timeout_seconds: float | None = None,
    metrics_artifact_key: str = "hpc-job-metrics",
    fugaku_job_name: str | None = None,
    execution_profile_overrides: dict[str, Any] | None = None,
) -> Any:
    """Resolve Prefect blocks and execute a job on the configured target.

    This is the main block-driven entrypoint for workflow authors. It loads the
    command, execution profile, and HPC profile blocks; converts them into the
    internal runtime models; and dispatches to local execution or the Miyabi,
    Fugaku, or Slurm executor.

    Args:
        command_block_name: Prefect block document name for the command to run.
        execution_profile_block_name: Prefect block document name describing
            resources, launcher, environment, and default execution behavior.
        hpc_profile_block_name: Prefect block document name describing the
            execution target and executable mapping, plus scheduler routing
            fields when applicable.
        work_dir: Working directory for the process or scheduler job.
        script_filename: Logical or scheduler-specific script filename. The
            suffix is normalized for scheduler targets. It is ignored for local
            execution and may be omitted.
        user_args: Optional extra command-line arguments appended after the
            command block's default arguments.
        watch_poll_interval: Seconds to wait between scheduler status polls.
        timeout_seconds: Optional maximum wait time for terminal job status.
        metrics_artifact_key: Prefect artifact key used for job metrics.
        fugaku_job_name: Optional Fugaku PJM job name. When omitted, a safe name
            is derived from the command name.
        execution_profile_overrides: Optional runtime overrides for selected
            execution profile fields, such as ``num_nodes`` or ``walltime``.

    Returns:
        A target-specific result object: ``LocalRunResult``,
        ``MiyabiRunResult``, ``FugakuRunResult``, or ``SlurmRunResult``.

    Raises:
        ValueError: If the command and execution profile blocks refer to
            different command names, if a required project/group is missing,
            if local execution receives ``modules`` or ``pre_commands``, or if
            unsupported execution profile override keys are provided.
        KeyError: If the command's executable key is missing from the HPC
            profile's executable map.
        NotImplementedError: If the resolved ``hpc_target`` is unsupported.
    """
    prepared = await _prepare_job_from_blocks(
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        work_dir=work_dir,
        script_filename=script_filename,
        user_args=user_args,
        fugaku_job_name=fugaku_job_name,
        execution_profile_overrides=execution_profile_overrides,
    )

    if prepared.submission_target.hpc_target == "local":
        return await run_local_job(
            work_dir=prepared.work_dir,
            exec_profile=prepared.exec_profile,
            req=prepared.req,
            timeout_seconds=timeout_seconds,
            metrics_artifact_key=metrics_artifact_key,
        )

    if prepared.submission_target.hpc_target == "miyabi":
        return await run_miyabi_job(
            work_dir=prepared.work_dir,
            script_filename=prepared.script_filename,
            exec_profile=prepared.exec_profile,
            req=prepared.req,
            watch_poll_interval=watch_poll_interval,
            timeout_seconds=timeout_seconds,
            metrics_artifact_key=metrics_artifact_key,
        )

    if prepared.submission_target.hpc_target == "fugaku":
        return await run_fugaku_job(
            work_dir=prepared.work_dir,
            script_filename=prepared.script_filename,
            exec_profile=prepared.exec_profile,
            req=prepared.req,
            watch_poll_interval=watch_poll_interval,
            timeout_seconds=timeout_seconds,
            metrics_artifact_key=metrics_artifact_key,
        )

    if prepared.submission_target.hpc_target == "slurm":
        return await run_slurm_job(
            work_dir=prepared.work_dir,
            script_filename=prepared.script_filename,
            exec_profile=prepared.exec_profile,
            req=prepared.req,
            watch_poll_interval=watch_poll_interval,
            timeout_seconds=timeout_seconds,
            metrics_artifact_key=metrics_artifact_key,
        )

    raise NotImplementedError(
        f"hpc_target='{prepared.submission_target.hpc_target}' is not supported yet by "
        "run_job_from_blocks."
    )
