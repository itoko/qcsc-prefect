from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from qcsc_prefect_core.queue import QueueProbe

from qcsc_prefect_executor.bulk.exceptions import QueueFullError, TemporarySubmitError
from qcsc_prefect_executor.bulk.models import (
    BulkJobRecord,
    BulkJobSpec,
    BulkJobStatus,
    BulkTickResult,
    SubmittedJob,
    effective_execution_profile_block,
    effective_hpc_profile_block,
)
from qcsc_prefect_executor.bulk.registry import BulkJobRegistry


def _exception_text(exc: Exception) -> str:
    text = str(exc)
    return text or exc.__class__.__name__


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


async def monitor_jobs_many(*args, **kwargs):
    from qcsc_prefect_executor.from_blocks import monitor_jobs_many as _monitor_jobs_many

    return await _monitor_jobs_many(*args, **kwargs)


async def submit_job_from_blocks(*args, **kwargs):
    from qcsc_prefect_executor.from_blocks import submit_job_from_blocks as _submit_job_from_blocks

    return await _submit_job_from_blocks(*args, **kwargs)


async def _resolve_default_bulk_queue_probe(*args, **kwargs) -> QueueProbe:
    from qcsc_prefect_executor.from_blocks import (
        _resolve_default_bulk_queue_probe as _resolve_queue_probe,
    )

    return await _resolve_queue_probe(*args, **kwargs)


def _pending_fifo_candidates(
    registry: BulkJobRegistry,
    *,
    limit: int,
) -> list[BulkJobRecord]:
    if limit <= 0:
        return []

    records = registry.get_submit_candidates_fifo(limit=registry.count_submit_candidates())
    pending = [record for record in records if record.status == BulkJobStatus.PENDING]
    return pending[:limit]


@dataclass
class GlobalFugakuBulkRunner:
    """Non-blocking queue-aware runner for staged Fugaku workflows.

    The runner uses the existing single-submit path. Native PJM bulk submission
    remains an explicit experimental mode elsewhere and is not used here.
    Within one tick, selected pending jobs are submitted concurrently up to
    ``submit_workers`` without increasing the queue-aware batch size.
    """

    command_block: str
    execution_profile_block: str
    hpc_profile_block: str
    registry_path: Path
    queue_probe: QueueProbe | None = None
    max_active_jobs: int = 1000
    safety_margin: int = 20
    initial_submit_count: int | None = None
    max_submit_per_refill: int = 100
    target_active_jobs: int | None = None
    no_check_directory: bool = False
    submit_workers: int = 8

    def __post_init__(self) -> None:
        self.registry_path = Path(self.registry_path).expanduser()
        self.submit_workers = int(self.submit_workers)
        if self.submit_workers < 1:
            raise ValueError("submit_workers must be positive.")
        self.registry = BulkJobRegistry(self.registry_path)

    def register_jobs(self, jobs: list[BulkJobSpec]) -> None:
        """Register logical jobs idempotently and skip completed outputs."""

        self.registry.upsert_jobs(jobs)
        self.registry.refresh_completed_jobs_from_outputs()

    async def tick(self) -> BulkTickResult:
        """Run one monitor/refill cycle without waiting for terminal completion."""

        monitored = await self._monitor_once()
        self.registry.refresh_completed_jobs_from_outputs()
        submitted = await self._submit_once()
        return BulkTickResult(
            submitted=submitted,
            monitored=monitored,
            status_counts=self.status_counts(),
            registry_path=self.registry.path,
        )

    def all_submitted(self, stage_id: str) -> bool:
        """Return whether a stage has no pending or deferred logical jobs."""

        blocked_statuses = {
            BulkJobStatus.PENDING,
            BulkJobStatus.SUBMIT_DEFERRED,
        }
        return all(
            record.status not in blocked_statuses
            for record in self.registry.get_all_jobs()
            if record.stage_id == stage_id
        )

    def status_counts(self, stage_id: str | None = None) -> dict[str, int]:
        """Return status counts for all jobs or one stage."""

        counts: dict[str, int] = {}
        for record in self.registry.get_all_jobs():
            if stage_id is not None and record.stage_id != stage_id:
                continue
            counts[record.status.value] = counts.get(record.status.value, 0) + 1
        return counts

    async def _monitor_once(self) -> dict[str, BulkJobStatus]:
        monitorable_jobs = [
            job for job in self.registry.get_monitorable_jobs() if job.effective_scheduler_job_id
        ]
        if not monitorable_jobs:
            return {}

        grouped_scheduler_ids: dict[str, list[str]] = {}
        for job in monitorable_jobs:
            hpc_profile_block = effective_hpc_profile_block(job, self.hpc_profile_block)
            grouped_scheduler_ids.setdefault(hpc_profile_block, []).append(
                str(job.effective_scheduler_job_id)
            )

        results: dict[str, BulkJobStatus] = {}
        for hpc_profile_block, scheduler_job_ids in grouped_scheduler_ids.items():
            results.update(
                await monitor_jobs_many(
                    hpc_profile_block=hpc_profile_block,
                    scheduler_job_ids=scheduler_job_ids,
                    registry=self.registry,
                )
            )
        return results

    async def _submit_once(self) -> list[SubmittedJob]:
        submit_limit = _submit_limit_for_cycle(
            registry=self.registry,
            initial_submit_count=self.initial_submit_count,
            max_submit_per_refill=self.max_submit_per_refill,
        )
        submit_count = await self._submit_count(submit_limit=submit_limit)
        candidates = _pending_fifo_candidates(self.registry, limit=submit_count)
        if not candidates:
            return []

        semaphore = asyncio.Semaphore(self.submit_workers)

        async def _attempt(
            job: BulkJobRecord,
        ) -> tuple[str, BulkJobRecord, SubmittedJob | str]:
            async with semaphore:
                try:
                    submitted_job = await submit_job_from_blocks(
                        command_block=self.command_block,
                        execution_profile_block=effective_execution_profile_block(
                            job,
                            self.execution_profile_block,
                        ),
                        hpc_profile_block=effective_hpc_profile_block(
                            job,
                            self.hpc_profile_block,
                        ),
                        work_dir=job.work_dir,
                        job_key=job.job_key,
                        command_args=job.command_args,
                        registry=self.registry,
                        fugaku_no_check_directory=self.no_check_directory,
                    )
                except (QueueFullError, TemporarySubmitError) as exc:
                    return ("deferred", job, str(exc))
                except Exception as exc:
                    return ("failed", job, _exception_text(exc))

                return ("submitted", job, submitted_job)

        outcomes = await asyncio.gather(*(_attempt(job) for job in candidates))

        submitted: list[SubmittedJob] = []
        for kind, job, payload in outcomes:
            if kind == "submitted":
                submitted.append(payload)
                continue
            if kind == "deferred":
                _mark_deferred_if_needed(
                    registry=self.registry,
                    job_key=job.job_key,
                    error=str(payload),
                )
                continue

            _mark_failed_if_needed(
                registry=self.registry,
                job_key=job.job_key,
                error=str(payload),
            )

        return submitted

    async def _submit_count(self, *, submit_limit: int) -> int:
        queue_probe = await self._resolved_queue_probe()
        limits = [
            max(0, int(submit_limit)),
            _queue_available_slots(queue_probe),
            len(
                _pending_fifo_candidates(
                    self.registry,
                    limit=self.registry.count_submit_candidates(),
                )
            ),
        ]
        if self.target_active_jobs is not None:
            limits.append(
                max(0, int(self.target_active_jobs) - self.registry.count_active_jobs())
            )
        return max(0, min(limits))

    async def _resolved_queue_probe(self) -> QueueProbe:
        if self.queue_probe is not None:
            return self.queue_probe

        self.queue_probe = await _resolve_default_bulk_queue_probe(
            hpc_profile_block=self.hpc_profile_block,
            execution_profile_block=self.execution_profile_block,
            max_active_jobs=self.max_active_jobs,
            safety_margin=self.safety_margin,
            submit_mode="single",
        )
        return self.queue_probe
