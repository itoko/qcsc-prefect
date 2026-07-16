from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from qcsc_prefect_core.queue import QueueCapacity
from qcsc_prefect_executor.bulk import (
    BulkJobSpec,
    BulkJobStatus,
    BulkTickResult,
    GlobalFugakuBulkRunner,
)
from qcsc_prefect_executor.bulk import global_fugaku_runner as runner_mod
from qcsc_prefect_executor.bulk.exceptions import QueueFullError
from qcsc_prefect_executor.bulk.models import SubmittedJob
from qcsc_prefect_executor.bulk.registry import BulkJobRegistry


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


def _spec(
    tmp_path: Path,
    job_key: str,
    stage_id: str,
    *,
    expected_outputs: list[Path] | None = None,
    execution_profile_block: str | None = None,
    hpc_profile_block: str | None = None,
) -> BulkJobSpec:
    return BulkJobSpec(
        job_key=job_key,
        stage_id=stage_id,
        work_dir=tmp_path / job_key,
        command_args={"job_key": job_key},
        expected_outputs=expected_outputs or [],
        execution_profile_block=execution_profile_block,
        hpc_profile_block=hpc_profile_block,
    )


def _install_single_submit_fakes(
    monkeypatch,
    *,
    submitted: list[str],
    submit_failures: dict[str, Exception] | None = None,
    mark_monitored_succeeded: bool = False,
    no_check_directory_calls: list[bool] | None = None,
    submit_block_calls: list[tuple[str, str, str]] | None = None,
    monitor_block_calls: list[tuple[str, list[str]]] | None = None,
) -> None:
    submit_failures = submit_failures or {}

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
        if job_key in submit_failures:
            raise submit_failures[job_key]

        scheduler_job_id = f"sched-{job_key}"
        submitted.append(job_key)
        if submit_block_calls is not None:
            submit_block_calls.append((job_key, execution_profile_block, hpc_profile_block))
        if no_check_directory_calls is not None:
            no_check_directory_calls.append(fugaku_no_check_directory)
        if registry is not None:
            registry.mark_submitted(job_key, scheduler_job_id)
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
        if monitor_block_calls is not None:
            monitor_block_calls.append((hpc_profile_block, list(scheduler_job_ids)))
        if registry is not None and mark_monitored_succeeded:
            records_by_scheduler_id = {
                record.effective_scheduler_job_id: record
                for record in registry.get_all_jobs()
                if record.effective_scheduler_job_id
            }
            for scheduler_job_id in scheduler_job_ids:
                record = records_by_scheduler_id[scheduler_job_id]
                registry.mark_succeeded(record.job_key)
            return {
                scheduler_job_id: BulkJobStatus.SUCCEEDED
                for scheduler_job_id in scheduler_job_ids
            }
        return {
            scheduler_job_id: BulkJobStatus.SUBMITTED
            for scheduler_job_id in scheduler_job_ids
        }

    monkeypatch.setattr(runner_mod, "submit_job_from_blocks", fake_submit_job_from_blocks)
    monkeypatch.setattr(runner_mod, "monitor_jobs_many", fake_monitor_jobs_many)


def _runner(
    tmp_path: Path,
    *,
    queue_probe: _FixedCapacityProbe | None = None,
    initial_submit_count: int | None = 3,
    max_submit_per_refill: int = 2,
    no_check_directory: bool = False,
    submit_workers: int = 8,
) -> GlobalFugakuBulkRunner:
    return GlobalFugakuBulkRunner(
        command_block="cmd",
        execution_profile_block="exec",
        hpc_profile_block="hpc",
        registry_path=tmp_path / "bulk.sqlite",
        queue_probe=queue_probe or _FixedCapacityProbe(10),
        initial_submit_count=initial_submit_count,
        max_submit_per_refill=max_submit_per_refill,
        no_check_directory=no_check_directory,
        submit_workers=submit_workers,
    )


def test_public_bulk_api_exports_global_fugaku_bulk_runner():
    assert callable(GlobalFugakuBulkRunner)
    assert BulkTickResult.__name__ == "BulkTickResult"


def test_submit_workers_must_be_positive(tmp_path: Path):
    try:
        _runner(tmp_path, submit_workers=0)
    except ValueError as exc:
        assert "submit_workers" in str(exc)
    else:
        raise AssertionError("submit_workers=0 should fail validation")


def test_tick_submits_initial_count_then_refill_count(tmp_path: Path, monkeypatch):
    submitted: list[str] = []
    _install_single_submit_fakes(monkeypatch, submitted=submitted)
    runner = _runner(tmp_path, initial_submit_count=3, max_submit_per_refill=2)
    runner.register_jobs([_spec(tmp_path, f"qpy-{index}", "qpy") for index in range(5)])

    first = asyncio.run(runner.tick())
    second = asyncio.run(runner.tick())

    assert [job.job_key for job in first.submitted] == ["qpy-0", "qpy-1", "qpy-2"]
    assert [job.job_key for job in second.submitted] == ["qpy-3", "qpy-4"]
    assert submitted == ["qpy-0", "qpy-1", "qpy-2", "qpy-3", "qpy-4"]
    assert runner.all_submitted("qpy") is True


def test_tick_passes_default_no_check_directory_false(tmp_path: Path, monkeypatch):
    submitted: list[str] = []
    no_check_directory_calls: list[bool] = []
    _install_single_submit_fakes(
        monkeypatch,
        submitted=submitted,
        no_check_directory_calls=no_check_directory_calls,
    )
    runner = _runner(tmp_path, initial_submit_count=1, max_submit_per_refill=1)
    runner.register_jobs([_spec(tmp_path, "qpy-0", "qpy")])

    tick = asyncio.run(runner.tick())

    assert [job.job_key for job in tick.submitted] == ["qpy-0"]
    assert no_check_directory_calls == [False]


def test_tick_passes_opt_in_no_check_directory_true(tmp_path: Path, monkeypatch):
    submitted: list[str] = []
    no_check_directory_calls: list[bool] = []
    _install_single_submit_fakes(
        monkeypatch,
        submitted=submitted,
        no_check_directory_calls=no_check_directory_calls,
    )
    runner = _runner(
        tmp_path,
        initial_submit_count=1,
        max_submit_per_refill=1,
        no_check_directory=True,
    )
    runner.register_jobs([_spec(tmp_path, "qpy-0", "qpy")])

    tick = asyncio.run(runner.tick())

    assert [job.job_key for job in tick.submitted] == ["qpy-0"]
    assert no_check_directory_calls == [True]


def test_tick_uses_per_job_blocks_and_runner_defaults(tmp_path: Path, monkeypatch):
    submitted: list[str] = []
    submit_block_calls: list[tuple[str, str, str]] = []
    _install_single_submit_fakes(
        monkeypatch,
        submitted=submitted,
        submit_block_calls=submit_block_calls,
    )
    runner = _runner(
        tmp_path,
        initial_submit_count=2,
        max_submit_per_refill=2,
        submit_workers=1,
    )
    runner.register_jobs(
        [
            _spec(
                tmp_path,
                "qpy-0",
                "qpy",
                execution_profile_block="exec-small",
                hpc_profile_block="hpc-small",
            ),
            _spec(tmp_path, "qpy-1", "qpy"),
        ]
    )

    tick = asyncio.run(runner.tick())

    assert [job.job_key for job in tick.submitted] == ["qpy-0", "qpy-1"]
    assert submit_block_calls == [
        ("qpy-0", "exec-small", "hpc-small"),
        ("qpy-1", "exec", "hpc"),
    ]


def test_monitor_groups_jobs_by_effective_hpc_block(tmp_path: Path, monkeypatch):
    submitted: list[str] = []
    monitor_block_calls: list[tuple[str, list[str]]] = []
    _install_single_submit_fakes(
        monkeypatch,
        submitted=submitted,
        monitor_block_calls=monitor_block_calls,
    )
    runner = _runner(tmp_path)
    runner.register_jobs(
        [
            _spec(tmp_path, "qpy-0", "qpy", hpc_profile_block="hpc-a"),
            _spec(tmp_path, "qpy-1", "qpy", hpc_profile_block="hpc-b"),
            _spec(tmp_path, "qpy-2", "qpy"),
        ]
    )
    runner.registry.mark_submitted("qpy-0", "sched-qpy-0")
    runner.registry.mark_submitted("qpy-1", "sched-qpy-1")
    runner.registry.mark_submitted("qpy-2", "sched-qpy-2")

    monitored = asyncio.run(runner._monitor_once())

    assert monitored == {
        "sched-qpy-0": BulkJobStatus.SUBMITTED,
        "sched-qpy-1": BulkJobStatus.SUBMITTED,
        "sched-qpy-2": BulkJobStatus.SUBMITTED,
    }
    assert dict(monitor_block_calls) == {
        "hpc-a": ["sched-qpy-0"],
        "hpc-b": ["sched-qpy-1"],
        "hpc": ["sched-qpy-2"],
    }


def test_submit_workers_one_and_many_submit_same_job_set(tmp_path: Path, monkeypatch):
    def run_case(case_dir: Path, submit_workers: int) -> tuple[set[str], dict[str, int]]:
        submitted: list[str] = []
        _install_single_submit_fakes(monkeypatch, submitted=submitted)
        runner = _runner(
            case_dir,
            initial_submit_count=6,
            max_submit_per_refill=6,
            submit_workers=submit_workers,
        )
        runner.register_jobs([_spec(case_dir, f"qpy-{index}", "qpy") for index in range(6)])

        tick = asyncio.run(runner.tick())

        assert {job.job_key for job in tick.submitted} == set(submitted)
        return set(submitted), runner.status_counts("qpy")

    single_workers, single_counts = run_case(tmp_path / "single", submit_workers=1)
    many_workers, many_counts = run_case(tmp_path / "many", submit_workers=16)

    expected = {f"qpy-{index}" for index in range(6)}
    assert single_workers == expected
    assert many_workers == expected
    assert single_counts == many_counts == {BulkJobStatus.SUBMITTED.value: 6}


def test_submit_workers_caps_concurrent_submits(tmp_path: Path, monkeypatch):
    def run_case(case_dir: Path, submit_workers: int) -> int:
        stats = {"active": 0, "max_active": 0}

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
            stats["active"] += 1
            stats["max_active"] = max(stats["max_active"], stats["active"])
            try:
                await asyncio.sleep(0.01)
                scheduler_job_id = f"sched-{job_key}"
                if registry is not None:
                    registry.mark_submitted(job_key, scheduler_job_id)
                return SubmittedJob(
                    job_key=job_key,
                    scheduler_job_id=scheduler_job_id,
                    status=BulkJobStatus.SUBMITTED,
                    work_dir=work_dir,
                )
            finally:
                stats["active"] -= 1

        async def fake_monitor_jobs_many(
            *,
            hpc_profile_block: str,
            scheduler_job_ids: list[str],
            registry: BulkJobRegistry | None = None,
        ) -> dict[str, BulkJobStatus]:
            return {
                scheduler_job_id: BulkJobStatus.SUBMITTED
                for scheduler_job_id in scheduler_job_ids
            }

        monkeypatch.setattr(runner_mod, "submit_job_from_blocks", fake_submit_job_from_blocks)
        monkeypatch.setattr(runner_mod, "monitor_jobs_many", fake_monitor_jobs_many)

        runner = _runner(
            case_dir,
            initial_submit_count=8,
            max_submit_per_refill=8,
            submit_workers=submit_workers,
        )
        runner.register_jobs([_spec(case_dir, f"qpy-{index}", "qpy") for index in range(8)])

        tick = asyncio.run(runner.tick())

        assert len(tick.submitted) == 8
        assert stats["max_active"] <= submit_workers
        return stats["max_active"]

    workers_two = run_case(tmp_path / "workers-two", submit_workers=2)
    workers_four = run_case(tmp_path / "workers-four", submit_workers=4)

    assert workers_two == 2
    assert workers_four == 4
    assert workers_two < workers_four


def test_submit_failure_isolated_to_one_job(tmp_path: Path, monkeypatch):
    submitted: list[str] = []
    _install_single_submit_fakes(
        monkeypatch,
        submitted=submitted,
        submit_failures={"qpy-1": RuntimeError("boom")},
    )
    runner = _runner(
        tmp_path,
        initial_submit_count=3,
        max_submit_per_refill=3,
        submit_workers=3,
    )
    runner.register_jobs([_spec(tmp_path, f"qpy-{index}", "qpy") for index in range(3)])

    tick = asyncio.run(runner.tick())

    assert {job.job_key for job in tick.submitted} == {"qpy-0", "qpy-2"}
    assert set(submitted) == {"qpy-0", "qpy-2"}
    assert runner.registry.get_job("qpy-1").status == BulkJobStatus.FAILED
    assert runner.status_counts("qpy") == {
        BulkJobStatus.FAILED.value: 1,
        BulkJobStatus.SUBMITTED.value: 2,
    }


def test_queue_full_isolated_and_deferred_job_is_not_auto_retried(
    tmp_path: Path,
    monkeypatch,
):
    submitted: list[str] = []
    _install_single_submit_fakes(
        monkeypatch,
        submitted=submitted,
        submit_failures={"qpy-0": QueueFullError("queue full")},
    )
    runner = _runner(
        tmp_path,
        initial_submit_count=2,
        max_submit_per_refill=2,
        submit_workers=2,
    )
    runner.register_jobs([_spec(tmp_path, f"qpy-{index}", "qpy") for index in range(2)])

    first = asyncio.run(runner.tick())
    second = asyncio.run(runner.tick())

    assert [job.job_key for job in first.submitted] == ["qpy-1"]
    assert second.submitted == []
    assert submitted == ["qpy-1"]
    assert runner.all_submitted("qpy") is False
    assert runner.status_counts("qpy") == {
        BulkJobStatus.SUBMIT_DEFERRED.value: 1,
        BulkJobStatus.SUBMITTED.value: 1,
    }


def test_register_trimsqd_jobs_later_and_tick_submits_them(tmp_path: Path, monkeypatch):
    submitted: list[str] = []
    _install_single_submit_fakes(monkeypatch, submitted=submitted)
    runner = _runner(tmp_path, initial_submit_count=2, max_submit_per_refill=2)
    runner.register_jobs([_spec(tmp_path, f"qpy-{index}", "qpy") for index in range(2)])

    asyncio.run(runner.tick())
    assert runner.all_submitted("qpy") is True

    runner.register_jobs([_spec(tmp_path, f"trim-{index}", "trim_sqd") for index in range(2)])
    trim_tick = asyncio.run(runner.tick())

    assert [job.job_key for job in trim_tick.submitted] == ["trim-0", "trim-1"]
    assert runner.all_submitted("trim_sqd") is True
    assert submitted == ["qpy-0", "qpy-1", "trim-0", "trim-1"]


def test_existing_succeeded_jobs_are_skipped(tmp_path: Path, monkeypatch):
    submitted: list[str] = []
    _install_single_submit_fakes(monkeypatch, submitted=submitted)
    runner = _runner(tmp_path, initial_submit_count=3, max_submit_per_refill=2)
    done_dir = tmp_path / "qpy-0"
    done_dir.mkdir()
    (done_dir / "done.marker").write_text("ok")
    runner.register_jobs(
        [
            _spec(
                tmp_path,
                "qpy-0",
                "qpy",
                expected_outputs=[Path("done.marker")],
            ),
            _spec(tmp_path, "qpy-1", "qpy"),
        ]
    )

    tick = asyncio.run(runner.tick())

    assert [job.job_key for job in tick.submitted] == ["qpy-1"]
    assert submitted == ["qpy-1"]
    assert runner.status_counts("qpy") == {
        BulkJobStatus.SUBMITTED.value: 1,
        BulkJobStatus.SUCCEEDED.value: 1,
    }


def test_queue_capacity_caps_tick_submit_count(tmp_path: Path, monkeypatch):
    submitted: list[str] = []
    _install_single_submit_fakes(monkeypatch, submitted=submitted)
    runner = _runner(
        tmp_path,
        queue_probe=_FixedCapacityProbe(2),
        initial_submit_count=5,
        max_submit_per_refill=5,
    )
    runner.register_jobs([_spec(tmp_path, f"qpy-{index}", "qpy") for index in range(5)])

    tick = asyncio.run(runner.tick())

    assert [job.job_key for job in tick.submitted] == ["qpy-0", "qpy-1"]
    assert submitted == ["qpy-0", "qpy-1"]
