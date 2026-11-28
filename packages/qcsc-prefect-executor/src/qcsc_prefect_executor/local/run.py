from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from prefect.artifacts import create_table_artifact
from prefect.logging import get_run_logger
from qcsc_prefect_adapters.local.runtime import (
    LocalJobRequest,
    LocalRuntime,
    build_local_command,
)
from qcsc_prefect_core.models.execution_profile import ExecutionProfile

MAX_LOG_SIZE = 10_000


def _truncate_log(text: str) -> str:
    if len(text) > MAX_LOG_SIZE:
        # QCSC_LOCAL_LOG_TAIL=N (opt-in): also keep the last N characters, so the end of a long
        # solver output (e.g. the last Davidson iterations) is not lost. Unset/0: unchanged.
        tail = int(os.environ.get("QCSC_LOCAL_LOG_TAIL", "0") or 0)
        cut = len(text) - MAX_LOG_SIZE
        if tail > 0 and cut > tail:
            return (text[:MAX_LOG_SIZE] + f"... (truncated {cut - tail} chars; last {tail} chars follow)\n"
                    + text[-tail:])
        return text[:MAX_LOG_SIZE] + f"... (truncated {cut} chars)"
    return text


@dataclass(frozen=True)
class LocalRunResult:
    """Result returned after a command exits on the local Prefect worker."""

    job_id: str
    exit_status: int
    stdout: str
    stderr: str


async def run_local_job(
    *,
    work_dir: Path,
    exec_profile: ExecutionProfile,
    req: LocalJobRequest,
    timeout_seconds: float | None = None,
    metrics_artifact_key: str = "local-job-metrics",
) -> LocalRunResult:
    """Execute a command directly without generating or submitting a job script."""

    command = build_local_command(exec_profile=exec_profile, executable=req.executable)
    work_dir.mkdir(parents=True, exist_ok=True)

    logger = get_run_logger()
    process_result = await LocalRuntime().execute(
        command,
        cwd=work_dir,
        environments=exec_profile.environments,
        timeout_seconds=timeout_seconds,
    )

    if process_result.stdout:
        logger.info(_truncate_log(process_result.stdout))
    if process_result.stderr:
        logger.error(_truncate_log(process_result.stderr))

    job_id = f"local-{process_result.pid}"
    artifact = {
        "job_id": job_id,
        "exit_status": process_result.exit_status,
        "command": shlex.join(command),
        "work_dir": str(work_dir),
    }
    await create_table_artifact(
        table=[list(artifact.keys()), list(artifact.values())],
        key=metrics_artifact_key,
    )

    return LocalRunResult(
        job_id=job_id,
        exit_status=process_result.exit_status,
        stdout=process_result.stdout,
        stderr=process_result.stderr,
    )
