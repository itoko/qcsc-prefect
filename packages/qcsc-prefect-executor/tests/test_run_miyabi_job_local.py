from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from qcsc_prefect_adapters.miyabi.builder import MiyabiJobRequest
from qcsc_prefect_adapters.miyabi.runtime import SubmitResult
from qcsc_prefect_core.models.execution_profile import ExecutionProfile
from qcsc_prefect_executor.miyabi import run as run_mod


class _LoggerStub:
    def __init__(self) -> None:
        self.info_lines: list[str] = []
        self.error_lines: list[str] = []
        self.warning_lines: list[str] = []

    def info(self, message: str) -> None:
        self.info_lines.append(message)

    def error(self, message: str) -> None:
        self.error_lines.append(message)

    def warning(self, message: str, *args: Any) -> None:
        self.warning_lines.append(message % args if args else message)


class _RuntimeStub:
    def __init__(
        self, final_status: dict[str, Any], calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]]
    ) -> None:
        self._final_status = final_status
        self._calls = calls

    async def submit(self, script_path: Path, *, cwd: Path | None = None) -> SubmitResult:
        self._calls.append(("submit", (script_path,), {"cwd": cwd}))
        return SubmitResult(job_id="12345.miyabi", raw_output="12345.miyabi")

    async def wait_final_status(
        self,
        job_id: str,
        *,
        watch_poll_interval: float = 10.0,
        timeout_seconds: float | None = None,
        stdout_fallback_path: Path | None = None,
    ) -> dict[str, Any]:
        self._calls.append(
            (
                "wait_final_status",
                (job_id,),
                {
                    "watch_poll_interval": watch_poll_interval,
                    "timeout_seconds": timeout_seconds,
                    "stdout_fallback_path": stdout_fallback_path,
                },
            )
        )
        return self._final_status


def test_run_miyabi_job_local_mock(tmp_path: Path, monkeypatch):
    stdout_path = tmp_path / "output.out"
    stderr_path = tmp_path / "output.err"
    stdout_path.write_text("hello from stdout")
    stderr_path.write_text("warning from stderr")

    final_status = {
        "Exit_status": "0",
        "Output_Path": f"login-node:{stdout_path}",
        "Error_Path": f"login-node:{stderr_path}",
        "Job_Name": "pytest-miyabi",
        "queue": "normal",
        "Resource_List.select": "1",
    }
    runtime_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    artifact_calls: list[dict[str, Any]] = []
    logger = _LoggerStub()

    def runtime_factory() -> _RuntimeStub:
        return _RuntimeStub(final_status=final_status, calls=runtime_calls)

    async def fake_create_table_artifact(*, table: list[list[Any]], key: str) -> None:
        artifact_calls.append({"table": table, "key": key})

    monkeypatch.setattr(run_mod, "MiyabiPBSRuntime", runtime_factory)
    monkeypatch.setattr(run_mod, "get_run_logger", lambda: logger)
    monkeypatch.setattr(run_mod, "create_table_artifact", fake_create_table_artifact)

    profile = ExecutionProfile(
        command_key="hello",
        num_nodes=1,
        launcher="single",
    )
    req = MiyabiJobRequest(
        queue_name="normal",
        project="z99999",
        executable="/bin/echo",
    )

    result = asyncio.run(
        run_mod.run_miyabi_job(
            work_dir=tmp_path,
            script_filename="job.pbs",
            exec_profile=profile,
            req=req,
            watch_poll_interval=0.01,
            timeout_seconds=5,
            metrics_artifact_key="pytest-metrics",
        )
    )

    assert result.job_id == "12345.miyabi"
    assert result.exit_status == 0
    assert result.job_status["Exit_status"] == "0"
    assert (tmp_path / "job.pbs").exists()
    assert logger.info_lines[0] == f"Create Script file in {tmp_path / 'job.pbs'}"
    assert logger.info_lines[1:] == ["hello from stdout"]
    assert logger.error_lines == ["warning from stderr"]
    assert runtime_calls[0][0] == "submit"
    assert runtime_calls[1] == (
        "wait_final_status",
        ("12345.miyabi",),
        {
            "watch_poll_interval": 0.01,
            "timeout_seconds": 5,
            "stdout_fallback_path": tmp_path / "output.out",
        },
    )
    assert len(artifact_calls) == 1
    assert artifact_calls[0]["key"] == "pytest-metrics"
    assert logger.warning_lines == []


def test_run_miyabi_job_local_mock_logs_fallback_warning(tmp_path: Path, monkeypatch):
    """When qstat -fH never confirms completion and the stdout-stability fallback fires,
    run_miyabi_job must surface that (via a warning log), not swallow it silently."""
    stdout_path = tmp_path / "output.out"
    stdout_path.write_text("hello from stdout")

    final_status = {
        "job_state": "F",
        "Exit_status": "0",
        "Output_Path": str(stdout_path),
        "_fallback": "qstat -fH never reported job_id=12345.miyabi as finished",
    }
    runtime_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    logger = _LoggerStub()

    def runtime_factory() -> _RuntimeStub:
        return _RuntimeStub(final_status=final_status, calls=runtime_calls)

    async def fake_create_table_artifact(*, table: list[list[Any]], key: str) -> None:
        pass

    monkeypatch.setattr(run_mod, "MiyabiPBSRuntime", runtime_factory)
    monkeypatch.setattr(run_mod, "get_run_logger", lambda: logger)
    monkeypatch.setattr(run_mod, "create_table_artifact", fake_create_table_artifact)

    profile = ExecutionProfile(command_key="hello", num_nodes=1, launcher="single")
    req = MiyabiJobRequest(queue_name="normal", project="z99999", executable="/bin/echo")

    result = asyncio.run(
        run_mod.run_miyabi_job(
            work_dir=tmp_path,
            script_filename="job.pbs",
            exec_profile=profile,
            req=req,
            watch_poll_interval=0.01,
            timeout_seconds=5,
        )
    )

    assert result.exit_status == 0
    assert len(logger.warning_lines) == 1
    assert "12345.miyabi" in logger.warning_lines[0]
