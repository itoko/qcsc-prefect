from __future__ import annotations

import asyncio
from pathlib import Path

from qcsc_prefect_adapters.miyabi import runtime as runtime_mod


def test_submit_parses_job_id(tmp_path: Path, monkeypatch):
    calls: list[tuple[tuple[str, ...], Path | None]] = []

    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        calls.append((args, cwd))
        return "12345.miyabi\n"

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.MiyabiPBSRuntime()

    result = asyncio.run(rt.submit(tmp_path / "job.pbs", cwd=tmp_path))

    assert result.job_id == "12345.miyabi"
    assert result.raw_output == "12345.miyabi"
    assert calls == [(("qsub", str(tmp_path / "job.pbs")), tmp_path)]


def test_submit_retries_transient_qsub_failure(tmp_path: Path, monkeypatch):
    """A transient qsub failure (e.g. a brief PBS server connection blip, observed in
    production as 'qsub: cannot connect to server ... errno=15010') must not immediately
    kill the whole run -- it should retry and succeed once the server is reachable again."""
    attempts: list[int] = []

    async def flaky_run_command(*args: str, cwd: Path | None = None) -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("qsub: cannot connect to server opbs (errno=15010)")
        return "12345.miyabi\n"

    monkeypatch.setattr(runtime_mod, "run_command", flaky_run_command)
    _real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda _seconds: _real_sleep(0))
    rt = runtime_mod.MiyabiPBSRuntime()

    result = asyncio.run(
        rt.submit(tmp_path / "job.pbs", cwd=tmp_path, max_attempts=3, retry_delay_seconds=0.01)
    )

    assert result.job_id == "12345.miyabi"
    assert len(attempts) == 3


def test_submit_raises_after_exhausting_retries(tmp_path: Path, monkeypatch):
    async def always_fails(*args: str, cwd: Path | None = None) -> str:
        raise RuntimeError("qsub: cannot connect to server opbs (errno=15010)")

    monkeypatch.setattr(runtime_mod, "run_command", always_fails)
    _real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda _seconds: _real_sleep(0))
    rt = runtime_mod.MiyabiPBSRuntime()

    try:
        asyncio.run(
            rt.submit(tmp_path / "job.pbs", cwd=tmp_path, max_attempts=2, retry_delay_seconds=0.01)
        )
        assert False, "expected SubmitError"
    except runtime_mod.SubmitError as exc:
        assert "after 2 attempt(s)" in str(exc)


def test_wait_final_status_parses_qstat_output(monkeypatch):
    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        return (
            "Job Id: 12345.miyabi\n"
            "    Job_Name = test-job\n"
            "    queue = normal\n"
            "    Exit_status = 0\n"
            "    resources_used.mem = 1048576kb\n"
            "    Variable_List = A=B,\n"
            "\tC=D\n"
        )

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.MiyabiPBSRuntime()

    status = asyncio.run(
        rt.wait_final_status("12345.miyabi", watch_poll_interval=0.01, timeout_seconds=3)
    )

    assert status["Job_Name"] == "test-job"
    assert status["queue"] == "normal"
    assert status["Exit_status"] == "0"
    assert status["resources_used.mem"] == "1048576kb"
    assert status["Variable_List"] == "A=B,C=D"


def test_wait_final_status_default_fallback_grace_is_conservative():
    """Regression guard: a short default here fires during normal PBS queue-wait/module-
    load/inter-print quiet periods, not just genuine hangs -- confirmed in production, where
    a 120s default killed a job 2 minutes into a completely normal run that finished
    successfully 30 minutes later. The default must stay comfortably above any plausible
    legitimate quiet stretch (queue wait, node boot, long solver iterations)."""
    import inspect

    sig = inspect.signature(runtime_mod.MiyabiPBSRuntime.wait_final_status)
    default = sig.parameters["stdout_fallback_grace_seconds"].default
    assert default >= 900.0, (
        f"stdout_fallback_grace_seconds default is {default}s -- too short, will misfire "
        "during normal startup/quiet periods and silently discard real results"
    )


def test_wait_final_status_falls_back_to_stable_stdout(tmp_path: Path, monkeypatch):
    """qstat -fH never finding the job (e.g. purged from short-lived PBS history) must not
    spin forever if the job's own stdout file has stopped growing."""

    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        return "No matching job found.\n"

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.MiyabiPBSRuntime()

    stdout_path = tmp_path / "output.out"
    stdout_path.write_text("some solver output\n")

    status = asyncio.run(
        rt.wait_final_status(
            "12345.miyabi",
            watch_poll_interval=0.01,
            timeout_seconds=5,
            stdout_fallback_path=stdout_path,
            stdout_fallback_grace_seconds=0.05,
        )
    )

    assert status["job_state"] == "F"
    assert status["Exit_status"] == "0"
    assert status["Output_Path"] == str(stdout_path)
    assert "_fallback" in status


def test_wait_final_status_fallback_waits_for_stdout_to_stabilize(tmp_path: Path, monkeypatch):
    """A still-growing stdout file must not trigger the fallback early."""

    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        return "No matching job found.\n"

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.MiyabiPBSRuntime()

    stdout_path = tmp_path / "output.out"
    stdout_path.write_text("partial\n")

    async def grow_then_wait():
        task = asyncio.create_task(
            rt.wait_final_status(
                "12345.miyabi",
                watch_poll_interval=0.01,
                timeout_seconds=5,
                stdout_fallback_path=stdout_path,
                stdout_fallback_grace_seconds=0.05,
            )
        )
        await asyncio.sleep(0.03)
        stdout_path.write_text("partial\nmore output\n")  # still growing -> resets the clock
        return await task

    status = asyncio.run(grow_then_wait())
    assert status["_fallback"] is not None


def test_cancel_invokes_qdel(monkeypatch):
    calls: list[tuple[str, ...]] = []

    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.MiyabiPBSRuntime()

    asyncio.run(rt.cancel("12345.miyabi"))

    assert calls == [("qdel", "12345.miyabi")]
