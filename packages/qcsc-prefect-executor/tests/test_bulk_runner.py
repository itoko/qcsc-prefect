from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from qcsc_prefect_core.models.execution_profile import ExecutionProfile
from qcsc_prefect_core.queue import QueueCapacity
from qcsc_prefect_executor import from_blocks as mod
from qcsc_prefect_executor.bulk.exceptions import QueueFullError, TemporarySubmitError
from qcsc_prefect_executor.bulk.models import BulkJobSpec, BulkJobStatus, SubmittedJob
from qcsc_prefect_executor.bulk.registry import BulkJobRegistry


class _RegistryCapacityProbe:
    def __init__(
        self,
        registry_path: Path,
        *,
        max_active_jobs: int = 1,
        failures: int = 0,
    ) -> None:
        self.registry_path = registry_path
        self.max_active_jobs = max_active_jobs
        self.failures = failures
        self.calls = 0

    def get_capacity(self) -> QueueCapacity:
        self.calls += 1
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("probe failed")

        registry = BulkJobRegistry(self.registry_path)
        current_active_jobs = len(registry.get_active_jobs())
        return QueueCapacity(
            max_active_jobs=self.max_active_jobs,
            current_active_jobs=current_active_jobs,
            available_slots=max(0, self.max_active_jobs - current_active_jobs),
        )


def _spec(
    tmp_path: Path,
    job_key: str,
    *,
    wave_id: str | None = None,
    stage_id: str | None = None,
    priority: int = 0,
    command_args: dict[str, Any] | None = None,
    expected_outputs: list[Path] | None = None,
    execution_profile_block: str | None = None,
    hpc_profile_block: str | None = None,
) -> BulkJobSpec:
    return BulkJobSpec(
        job_key=job_key,
        work_dir=tmp_path / job_key,
        wave_id=wave_id,
        stage_id=stage_id,
        priority=priority,
        command_args=command_args or {},
        expected_outputs=expected_outputs or [],
        execution_profile_block=execution_profile_block,
        hpc_profile_block=hpc_profile_block,
    )


def _mark_status(registry: BulkJobRegistry, job_key: str, status: BulkJobStatus) -> None:
    if status == BulkJobStatus.QUEUED:
        registry.mark_queued(job_key)
    elif status == BulkJobStatus.RUNNING:
        registry.mark_running(job_key)
    elif status == BulkJobStatus.SUCCEEDED:
        registry.mark_succeeded(job_key)
    elif status == BulkJobStatus.FAILED:
        registry.mark_failed(job_key, error="failed")
    elif status == BulkJobStatus.CANCELLED:
        registry.mark_cancelled(job_key, error="cancelled")
    elif status == BulkJobStatus.UNKNOWN:
        registry.mark_unknown(job_key, error="unknown")


def _install_fake_submit_and_monitor(
    monkeypatch,
    *,
    submitted: list[str],
    monitor_calls: list[list[str]] | None = None,
    monitor_status_by_job: dict[str, BulkJobStatus] | None = None,
    submit_failures: dict[str, list[Exception]] | None = None,
    active_counts_after_submit: list[int] | None = None,
    assert_deferred_on_retry: set[str] | None = None,
    submit_block_calls: list[tuple[str, str, str]] | None = None,
    monitor_block_calls: list[tuple[str, list[str]]] | None = None,
) -> None:
    scheduler_to_job: dict[str, str] = {}
    submit_failures = submit_failures or {}
    monitor_status_by_job = monitor_status_by_job or {}
    assert_deferred_on_retry = assert_deferred_on_retry or set()

    async def fake_submit_job_from_blocks(
        *,
        work_dir: Path,
        job_key: str,
        command_block: str,
        execution_profile_block: str,
        hpc_profile_block: str,
        command_args: dict[str, Any] | None = None,
        registry: BulkJobRegistry | None = None,
        fugaku_no_check_directory: bool = False,
    ) -> SubmittedJob:
        assert fugaku_no_check_directory is False
        failures = submit_failures.get(job_key, [])
        if failures:
            raise failures.pop(0)

        if registry is not None and job_key in assert_deferred_on_retry:
            record = registry.get_job(job_key)
            assert record is not None
            assert record.status == BulkJobStatus.SUBMIT_DEFERRED
            assert_deferred_on_retry.remove(job_key)

        scheduler_job_id = f"sched-{job_key}"
        scheduler_to_job[scheduler_job_id] = job_key
        submitted.append(job_key)
        if submit_block_calls is not None:
            submit_block_calls.append((job_key, execution_profile_block, hpc_profile_block))
        if registry is not None:
            registry.mark_submitted(job_key, scheduler_job_id)
            if active_counts_after_submit is not None:
                active_counts_after_submit.append(len(registry.get_active_jobs()))
        return SubmittedJob(
            job_key=job_key,
            scheduler_job_id=scheduler_job_id,
            status=BulkJobStatus.SUBMITTED,
            work_dir=work_dir,
        )

    async def fake_monitor_jobs_many(
        *,
        hpc_profile_block: str,
        scheduler_job_ids: list[str],
        registry: BulkJobRegistry | None = None,
    ) -> dict[str, BulkJobStatus]:
        if monitor_calls is not None:
            monitor_calls.append(list(scheduler_job_ids))
        if monitor_block_calls is not None:
            monitor_block_calls.append((hpc_profile_block, list(scheduler_job_ids)))

        statuses: dict[str, BulkJobStatus] = {}
        for scheduler_job_id in scheduler_job_ids:
            job_key = scheduler_to_job.get(
                scheduler_job_id, scheduler_job_id.removeprefix("sched-")
            )
            status = monitor_status_by_job.get(job_key, BulkJobStatus.SUCCEEDED)
            statuses[scheduler_job_id] = status
            if registry is not None:
                _mark_status(registry, job_key, status)
        return statuses

    monkeypatch.setattr(mod, "submit_job_from_blocks", fake_submit_job_from_blocks)
    monkeypatch.setattr(mod, "monitor_jobs_many", fake_monitor_jobs_many)


def _run_bulk(
    *,
    tmp_path: Path,
    jobs: list[BulkJobSpec],
    queue_probe: _RegistryCapacityProbe,
    max_submit_per_refill: int = 100,
    stop_on_first_failure: bool = False,
):
    return asyncio.run(
        mod.run_jobs_from_blocks_bulk(
            jobs=jobs,
            command_block="cmd",
            execution_profile_block="exec",
            hpc_profile_block="hpc",
            registry_path=queue_probe.registry_path,
            queue_probe=queue_probe,
            max_active_jobs=queue_probe.max_active_jobs,
            safety_margin=0,
            max_submit_per_refill=max_submit_per_refill,
            poll_interval_seconds=0,
            refill_interval_seconds=0,
            stop_on_first_failure=stop_on_first_failure,
        )
    )


class _FixedCapacityProbe:
    def __init__(self, available_slots: int) -> None:
        self.available_slots = available_slots
        self.calls = 0

    def get_capacity(self) -> QueueCapacity:
        self.calls += 1
        return QueueCapacity(
            max_active_jobs=self.available_slots,
            current_active_jobs=0,
            available_slots=self.available_slots,
            raw_output="fixed capacity",
        )


class _NativeBulkSubmitRuntime:
    def __init__(
        self,
        *,
        parent_job_ids: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.parent_job_ids = parent_job_ids or ["9000", "9001", "9002", "9003"]
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.no_check_directory_calls: list[bool] = []

    async def submit_bulk(
        self,
        script_path: Path,
        bulk_count: int,
        *,
        cwd: Path | None = None,
    ) -> str:
        self.calls.append(
            {
                "script_path": script_path,
                "bulk_count": bulk_count,
                "cwd": cwd,
            }
        )
        if self.error is not None:
            raise self.error
        return self.parent_job_ids[len(self.calls) - 1]


def _native_bulk_specs(tmp_path: Path, count: int) -> list[BulkJobSpec]:
    return [
        _spec(
            tmp_path,
            f"job-{index}",
            stage_id="stage-a",
            command_args={"index": index},
            expected_outputs=[Path("done.marker")],
        )
        for index in range(count)
    ]


def _manifest_job_keys(call: dict[str, Any]) -> list[str]:
    manifest_dir = Path(call["cwd"]) / "manifests"
    return [
        json.loads((manifest_dir / f"{index}.json").read_text())["job_key"]
        for index in range(int(call["bulk_count"]))
    ]


def _install_native_bulk_fakes(
    monkeypatch,
    runtime: _NativeBulkSubmitRuntime,
    *,
    mark_pending_succeeded_after_monitor: bool = True,
) -> None:
    async def fake_resolve_submission_target(
        *,
        hpc_profile_block_name: str,
        execution_profile_block_name: str,
    ) -> mod.SubmissionTarget:
        return mod.SubmissionTarget(
            hpc_target="fugaku",
            queue_name="small",
            project="ra010014",
        )

    async def fake_prepare_job_from_blocks(
        *,
        command_block_name: str,
        execution_profile_block_name: str,
        hpc_profile_block_name: str,
        work_dir: Path,
        script_filename: str,
        user_args: list[str] | None = None,
        fugaku_job_name: str | None = None,
        execution_profile_overrides: dict[str, Any] | None = None,
    ) -> mod._PreparedBlockJob:
        profile = ExecutionProfile(
            command_key="native-bulk-command",
            num_nodes=1,
            mpiprocs=1,
            walltime="00:05:00",
            launcher="single",
            arguments=["--manifest", '"$QCSC_BULK_MANIFEST"'],
        )
        req = mod.FugakuJobRequest(
            queue_name="small",
            project="ra010014",
            executable="python",
            job_name=fugaku_job_name or "native-bulk",
        )
        return mod._PreparedBlockJob(
            submission_target=mod.SubmissionTarget(
                hpc_target="fugaku",
                queue_name="small",
                project="ra010014",
            ),
            work_dir=Path(work_dir).expanduser().resolve(),
            script_filename=mod.build_scheduler_script_filename(script_filename, "fugaku"),
            exec_profile=profile,
            req=req,
        )

    async def fake_monitor_jobs_many(
        *,
        hpc_profile_block: str,
        scheduler_job_ids: list[str],
        registry: BulkJobRegistry | None = None,
    ) -> dict[str, BulkJobStatus]:
        assert registry is not None
        records_by_scheduler_id = {
            record.effective_scheduler_job_id: record
            for record in registry.get_all_jobs()
            if record.effective_scheduler_job_id
        }
        for scheduler_job_id in scheduler_job_ids:
            record = records_by_scheduler_id[scheduler_job_id]
            registry.mark_succeeded(record.job_key)
        if mark_pending_succeeded_after_monitor:
            for record in registry.get_all_jobs():
                if record.status.is_submit_candidate:
                    registry.mark_succeeded(record.job_key)
        return {scheduler_job_id: BulkJobStatus.SUCCEEDED for scheduler_job_id in scheduler_job_ids}

    monkeypatch.setattr(mod, "resolve_submission_target", fake_resolve_submission_target)
    monkeypatch.setattr(mod, "_prepare_job_from_blocks", fake_prepare_job_from_blocks)
    def runtime_factory(*, no_check_directory: bool = False) -> _NativeBulkSubmitRuntime:
        runtime.no_check_directory_calls.append(no_check_directory)
        return runtime

    monkeypatch.setattr(mod, "FugakuPJMRuntime", runtime_factory)
    monkeypatch.setattr(mod, "monitor_jobs_many", fake_monitor_jobs_many)


def test_run_jobs_from_blocks_bulk_submits_only_up_to_queue_capacity(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    active_counts: list[int] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        active_counts_after_submit=active_counts,
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, f"job-{index}") for index in range(3)],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=1),
    )

    assert max(active_counts) == 1
    assert submitted == ["job-0", "job-1", "job-2"]
    assert result.succeeded == 3


def test_run_jobs_from_blocks_bulk_respects_max_submit_per_refill(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    monitor_calls: list[list[str]] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        monitor_calls=monitor_calls,
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, f"job-{index}") for index in range(3)],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=10),
        max_submit_per_refill=2,
    )

    assert len(monitor_calls[0]) == 2
    assert submitted == ["job-0", "job-1", "job-2"]
    assert result.succeeded == 3


def test_run_jobs_from_blocks_bulk_uses_per_job_blocks_for_single_submit(
    tmp_path: Path,
    monkeypatch,
):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    submit_block_calls: list[tuple[str, str, str]] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        submit_block_calls=submit_block_calls,
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[
            _spec(
                tmp_path,
                "job-0",
                execution_profile_block="exec-small",
                hpc_profile_block="hpc-small",
            ),
            _spec(tmp_path, "job-1"),
        ],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=2),
    )

    assert submitted == ["job-0", "job-1"]
    assert submit_block_calls == [
        ("job-0", "exec-small", "hpc-small"),
        ("job-1", "exec", "hpc"),
    ]
    assert result.succeeded == 2


def test_queue_full_marks_submit_deferred_and_retries_later(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        submit_failures={"job-1": [QueueFullError("queue full")]},
        assert_deferred_on_retry={"job-1"},
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "job-1")],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=1),
    )

    assert submitted == ["job-1"]
    assert result.succeeded == 1
    assert result.failed == 0


def test_temporary_submit_error_is_retried_later(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        submit_failures={"job-1": [TemporarySubmitError("busy")]},
        assert_deferred_on_retry={"job-1"},
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "job-1")],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=1),
    )

    assert submitted == ["job-1"]
    assert result.succeeded == 1


def test_completed_jobs_are_not_resubmitted_after_restart(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    registry = BulkJobRegistry(registry_path)
    registry.upsert_jobs([_spec(tmp_path, "done"), _spec(tmp_path, "new")])
    registry.mark_succeeded("done")
    submitted: list[str] = []
    _install_fake_submit_and_monitor(monkeypatch, submitted=submitted)

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "done"), _spec(tmp_path, "new")],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=1),
    )

    assert submitted == ["new"]
    assert result.succeeded == 2


def test_active_jobs_are_monitored_after_restart_not_resubmitted(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    registry = BulkJobRegistry(registry_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1")])
    registry.mark_submitted("job-1", "sched-job-1")
    submitted: list[str] = []
    monitor_calls: list[list[str]] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        monitor_calls=monitor_calls,
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "job-1")],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=1),
    )

    assert submitted == []
    assert monitor_calls == [["sched-job-1"]]
    assert result.succeeded == 1


def test_expected_output_existence_skips_submission(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    work_dir = tmp_path / "job-1"
    work_dir.mkdir()
    (work_dir / "done.txt").write_text("ok")
    submitted: list[str] = []
    _install_fake_submit_and_monitor(monkeypatch, submitted=submitted)

    result = asyncio.run(
        mod.run_jobs_from_blocks_bulk(
            jobs=[
                BulkJobSpec(
                    job_key="job-1",
                    work_dir=work_dir,
                    expected_outputs=[Path("done.txt")],
                )
            ],
            command_block="cmd",
            execution_profile_block="exec",
            hpc_profile_block="hpc",
            registry_path=registry_path,
            poll_interval_seconds=0,
            refill_interval_seconds=0,
        )
    )

    assert submitted == []
    assert result.succeeded == 1


def test_active_jobs_are_passed_to_monitor_many_in_batches(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    registry = BulkJobRegistry(registry_path)
    registry.upsert_jobs([_spec(tmp_path, "job-1"), _spec(tmp_path, "job-2")])
    registry.mark_submitted("job-1", "sched-job-1")
    registry.mark_submitted("job-2", "sched-job-2")
    submitted: list[str] = []
    monitor_calls: list[list[str]] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        monitor_calls=monitor_calls,
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "job-1"), _spec(tmp_path, "job-2")],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=2),
    )

    assert submitted == []
    assert monitor_calls == [["sched-job-1", "sched-job-2"]]
    assert result.succeeded == 2


def test_run_jobs_from_blocks_bulk_groups_monitoring_by_effective_hpc_block(
    tmp_path: Path,
    monkeypatch,
):
    registry_path = tmp_path / "bulk.sqlite"
    registry = BulkJobRegistry(registry_path)
    jobs = [
        _spec(tmp_path, "job-0", hpc_profile_block="hpc-a"),
        _spec(tmp_path, "job-1", hpc_profile_block="hpc-b"),
        _spec(tmp_path, "job-2"),
    ]
    registry.upsert_jobs(jobs)
    registry.mark_submitted("job-0", "sched-job-0")
    registry.mark_submitted("job-1", "sched-job-1")
    registry.mark_submitted("job-2", "sched-job-2")
    submitted: list[str] = []
    monitor_block_calls: list[tuple[str, list[str]]] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        monitor_block_calls=monitor_block_calls,
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=jobs,
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=3),
    )

    assert submitted == []
    assert dict(monitor_block_calls) == {
        "hpc-a": ["sched-job-0"],
        "hpc-b": ["sched-job-1"],
        "hpc": ["sched-job-2"],
    }
    assert result.succeeded == 3


def test_native_bulk_rejects_per_job_block_overrides(tmp_path: Path):
    registry_path = tmp_path / "bulk.sqlite"

    try:
        asyncio.run(
            mod.run_jobs_from_blocks_bulk(
                jobs=[
                    _spec(
                        tmp_path,
                        "job-0",
                        stage_id="stage-a",
                        execution_profile_block="exec-small",
                    )
                ],
                command_block="cmd",
                execution_profile_block="exec",
                hpc_profile_block="hpc",
                registry_path=registry_path,
                queue_probe=_FixedCapacityProbe(10),
                submit_mode="native_bulk",
                poll_interval_seconds=0,
                refill_interval_seconds=0,
            )
        )
    except ValueError as exc:
        assert "native_bulk" in str(exc)
        assert "per-job" in str(exc)
        assert "job-0" in str(exc)
    else:
        raise AssertionError("native bulk should reject per-job block overrides")


def test_native_bulk_jobs_are_monitored_by_scheduler_subjob_id(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    registry = BulkJobRegistry(registry_path)
    registry.upsert_jobs([_spec(tmp_path, "job-0"), _spec(tmp_path, "job-1")])
    for index in range(2):
        registry.mark_submitted(
            f"job-{index}",
            "12345",
            submit_mode="native_bulk",
            bulk_group_key="group-1",
            bulk_parent_job_id="12345",
            bulk_index=index,
            scheduler_subjob_id=f"12345[{index}]",
        )

    submitted: list[str] = []
    monitor_calls: list[list[str]] = []

    async def fake_submit_job_from_blocks(**_kwargs):
        raise AssertionError("active native bulk jobs must not be resubmitted")

    async def fake_monitor_jobs_many(
        *,
        hpc_profile_block: str,
        scheduler_job_ids: list[str],
        registry: BulkJobRegistry | None = None,
    ) -> dict[str, BulkJobStatus]:
        monitor_calls.append(list(scheduler_job_ids))
        assert registry is not None
        for job_key in ["job-0", "job-1"]:
            registry.mark_succeeded(job_key)
        return {scheduler_job_id: BulkJobStatus.SUCCEEDED for scheduler_job_id in scheduler_job_ids}

    monkeypatch.setattr(mod, "submit_job_from_blocks", fake_submit_job_from_blocks)
    monkeypatch.setattr(mod, "monitor_jobs_many", fake_monitor_jobs_many)

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "job-0"), _spec(tmp_path, "job-1")],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=2),
    )

    assert submitted == []
    assert monitor_calls == [["12345[0]", "12345[1]"]]
    assert result.succeeded == 2


def test_native_bulk_initial_submit_splits_fifo_groups_and_records_subjobs(
    tmp_path: Path,
    monkeypatch,
):
    registry_path = tmp_path / "bulk.sqlite"
    runtime = _NativeBulkSubmitRuntime(parent_job_ids=["9000", "9001"])
    _install_native_bulk_fakes(monkeypatch, runtime)

    result = asyncio.run(
        mod.run_jobs_from_blocks_bulk(
            jobs=_native_bulk_specs(tmp_path, 5),
            command_block="cmd",
            execution_profile_block="exec",
            hpc_profile_block="hpc",
            registry_path=registry_path,
            queue_probe=_FixedCapacityProbe(10),
            submit_mode="native_bulk",
            initial_submit_count=4,
            max_submit_per_refill=1,
            max_bulk_group_size=2,
            poll_interval_seconds=0,
            refill_interval_seconds=0,
        )
    )

    assert [call["bulk_count"] for call in runtime.calls] == [2, 2]
    assert [_manifest_job_keys(call) for call in runtime.calls] == [
        ["job-0", "job-1"],
        ["job-2", "job-3"],
    ]

    registry = BulkJobRegistry(registry_path)
    job_0 = registry.get_job("job-0")
    job_1 = registry.get_job("job-1")
    job_2 = registry.get_job("job-2")
    job_3 = registry.get_job("job-3")
    assert job_0 is not None
    assert job_1 is not None
    assert job_2 is not None
    assert job_3 is not None
    assert job_0.scheduler_subjob_id == "9000[0]"
    assert job_0.scheduler_job_id == "9000[0]"
    assert job_1.scheduler_subjob_id == "9000[1]"
    assert job_2.scheduler_subjob_id == "9001[0]"
    assert job_3.scheduler_subjob_id == "9001[1]"
    assert result.succeeded == 5


def test_native_bulk_passes_fugaku_no_check_directory_option(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    runtime = _NativeBulkSubmitRuntime(parent_job_ids=["9000"])
    _install_native_bulk_fakes(monkeypatch, runtime)

    result = asyncio.run(
        mod.run_jobs_from_blocks_bulk(
            jobs=_native_bulk_specs(tmp_path, 2),
            command_block="cmd",
            execution_profile_block="exec",
            hpc_profile_block="hpc",
            registry_path=registry_path,
            queue_probe=_FixedCapacityProbe(10),
            submit_mode="native_bulk",
            initial_submit_count=2,
            max_bulk_group_size=2,
            poll_interval_seconds=0,
            refill_interval_seconds=0,
            fugaku_no_check_directory=True,
        )
    )

    assert result.succeeded == 2
    assert [call["bulk_count"] for call in runtime.calls] == [2]
    assert runtime.no_check_directory_calls == [True]


def test_native_bulk_queue_full_marks_group_deferred(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    runtime = _NativeBulkSubmitRuntime(error=RuntimeError("ru-accept job limit exceeded"))
    _install_native_bulk_fakes(monkeypatch, runtime)

    result = asyncio.run(
        mod.run_jobs_from_blocks_bulk(
            jobs=_native_bulk_specs(tmp_path, 2),
            command_block="cmd",
            execution_profile_block="exec",
            hpc_profile_block="hpc",
            registry_path=registry_path,
            queue_probe=_FixedCapacityProbe(10),
            submit_mode="native_bulk",
            initial_submit_count=2,
            max_bulk_group_size=2,
            poll_interval_seconds=0,
            refill_interval_seconds=0,
        )
    )

    registry = BulkJobRegistry(registry_path)
    assert [call["bulk_count"] for call in runtime.calls] == [2]
    assert result.submit_deferred == 2
    assert result.failed == 0
    assert registry.status_counts() == {BulkJobStatus.SUBMIT_DEFERRED.value: 2}


def test_native_bulk_skips_succeeded_and_existing_expected_outputs(
    tmp_path: Path,
    monkeypatch,
):
    registry_path = tmp_path / "bulk.sqlite"
    output_dir = tmp_path / "job-0"
    output_dir.mkdir()
    (output_dir / "done.marker").write_text("ok")
    runtime = _NativeBulkSubmitRuntime(parent_job_ids=["9000"])
    _install_native_bulk_fakes(monkeypatch, runtime)

    result = asyncio.run(
        mod.run_jobs_from_blocks_bulk(
            jobs=_native_bulk_specs(tmp_path, 3),
            command_block="cmd",
            execution_profile_block="exec",
            hpc_profile_block="hpc",
            registry_path=registry_path,
            queue_probe=_FixedCapacityProbe(10),
            submit_mode="native_bulk",
            initial_submit_count=3,
            max_bulk_group_size=3,
            poll_interval_seconds=0,
            refill_interval_seconds=0,
        )
    )

    assert [call["bulk_count"] for call in runtime.calls] == [2]
    assert _manifest_job_keys(runtime.calls[0]) == ["job-1", "job-2"]
    assert result.succeeded == 3


def test_native_bulk_bootstrap_is_not_repeated_after_restart(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    registry = BulkJobRegistry(registry_path)
    registry.upsert_jobs(_native_bulk_specs(tmp_path, 3))
    registry.mark_submitted(
        "job-0",
        "8000[0]",
        submit_mode="native_bulk",
        bulk_group_key="old-group",
        bulk_parent_job_id="8000",
        bulk_index=0,
        scheduler_subjob_id="8000[0]",
    )
    runtime = _NativeBulkSubmitRuntime(parent_job_ids=["9000", "9001"])
    _install_native_bulk_fakes(
        monkeypatch,
        runtime,
        mark_pending_succeeded_after_monitor=False,
    )

    result = asyncio.run(
        mod.run_jobs_from_blocks_bulk(
            jobs=_native_bulk_specs(tmp_path, 3),
            command_block="cmd",
            execution_profile_block="exec",
            hpc_profile_block="hpc",
            registry_path=registry_path,
            queue_probe=_FixedCapacityProbe(10),
            submit_mode="native_bulk",
            initial_submit_count=4,
            max_submit_per_refill=1,
            max_bulk_group_size=4,
            poll_interval_seconds=0,
            refill_interval_seconds=0,
        )
    )

    assert [call["bulk_count"] for call in runtime.calls] == [1, 1]
    assert result.succeeded == 3


def test_native_bulk_group_size_does_not_increase_queue_allowance(
    tmp_path: Path,
    monkeypatch,
):
    registry_path = tmp_path / "bulk.sqlite"
    runtime = _NativeBulkSubmitRuntime(parent_job_ids=["9000"])
    _install_native_bulk_fakes(monkeypatch, runtime)

    result = asyncio.run(
        mod.run_jobs_from_blocks_bulk(
            jobs=_native_bulk_specs(tmp_path, 6),
            command_block="cmd",
            execution_profile_block="exec",
            hpc_profile_block="hpc",
            registry_path=registry_path,
            queue_probe=_FixedCapacityProbe(3),
            submit_mode="native_bulk",
            initial_submit_count=10,
            max_bulk_group_size=10,
            poll_interval_seconds=0,
            refill_interval_seconds=0,
        )
    )

    assert [call["bulk_count"] for call in runtime.calls] == [3]
    assert _manifest_job_keys(runtime.calls[0]) == ["job-0", "job-1", "job-2"]
    assert result.succeeded == 6


def test_native_bulk_submit_count_respects_target_active_jobs(tmp_path: Path):
    registry = BulkJobRegistry(tmp_path / "bulk.sqlite")
    registry.upsert_jobs(_native_bulk_specs(tmp_path, 5))
    for index in range(2):
        registry.mark_submitted(
            f"job-{index}",
            f"8000[{index}]",
            submit_mode="native_bulk",
            bulk_group_key="old-group",
            bulk_parent_job_id="8000",
            bulk_index=index,
            scheduler_subjob_id=f"8000[{index}]",
        )

    submit_count = mod._native_bulk_submit_count(
        registry=registry,
        queue_probe=_FixedCapacityProbe(10),
        submit_limit=10,
        target_active_jobs=3,
    )

    assert submit_count == 1


def test_all_jobs_eventually_succeeded_returns_bulk_run_result(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    _install_fake_submit_and_monitor(monkeypatch, submitted=submitted)

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "job-1"), _spec(tmp_path, "job-2")],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=2),
    )

    assert result.total_jobs == 2
    assert result.status_counts == {BulkJobStatus.SUCCEEDED.value: 2}
    assert result.succeeded == 2
    assert result.failed == 0
    assert result.cancelled == 0
    assert result.submit_deferred == 0
    assert result.unknown == 0
    assert result.registry_path == registry_path
    assert result.failed_jobs == []


def test_failed_job_returns_bulk_run_result(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        monitor_status_by_job={
            "job-ok": BulkJobStatus.SUCCEEDED,
            "job-fail": BulkJobStatus.FAILED,
        },
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "job-ok"), _spec(tmp_path, "job-fail")],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=2),
    )

    assert result.succeeded == 1
    assert result.failed == 1
    assert result.failed_jobs == ["job-fail"]


def test_stop_on_first_failure_stops_loop(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        monitor_status_by_job={
            "job-fail": BulkJobStatus.FAILED,
            "job-running": BulkJobStatus.RUNNING,
        },
    )

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "job-fail"), _spec(tmp_path, "job-running")],
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=2),
        stop_on_first_failure=True,
    )

    assert result.failed == 1
    assert result.status_counts == {
        BulkJobStatus.FAILED.value: 1,
        BulkJobStatus.RUNNING.value: 1,
    }


def test_wave_readiness_does_not_affect_submit_order(tmp_path: Path, monkeypatch):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    monitor_calls: list[list[str]] = []
    _install_fake_submit_and_monitor(
        monkeypatch,
        submitted=submitted,
        monitor_calls=monitor_calls,
    )

    jobs = [
        _spec(tmp_path, "wave-a-1", wave_id="wave-a", priority=3),
        _spec(tmp_path, "wave-b-1", wave_id="wave-b", priority=2),
        _spec(tmp_path, "wave-a-2", wave_id="wave-a", priority=1),
    ]
    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=jobs,
        queue_probe=_RegistryCapacityProbe(registry_path, max_active_jobs=2),
        max_submit_per_refill=2,
    )

    registry = BulkJobRegistry(registry_path)
    assert submitted[:2] == ["wave-a-1", "wave-b-1"]
    assert registry.is_wave_ready("wave-a") is True
    assert registry.is_wave_ready("wave-b") is True
    assert result.succeeded == 3


def test_queue_probe_failure_results_in_no_new_submissions_for_that_cycle(
    tmp_path: Path, monkeypatch
):
    registry_path = tmp_path / "bulk.sqlite"
    submitted: list[str] = []
    _install_fake_submit_and_monitor(monkeypatch, submitted=submitted)
    probe = _RegistryCapacityProbe(registry_path, max_active_jobs=1, failures=1)

    result = _run_bulk(
        tmp_path=tmp_path,
        jobs=[_spec(tmp_path, "job-1")],
        queue_probe=probe,
    )

    assert probe.calls >= 2
    assert submitted == ["job-1"]
    assert result.succeeded == 1
