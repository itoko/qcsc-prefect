from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar


class SubmitError(RuntimeError):
    """Raised when job submission fails."""


class WaitTimeout(RuntimeError):
    """Raised when waiting for final job status times out."""


class CancelError(RuntimeError):
    """Raised when job cancellation fails."""


async def run_command(*args: str, cwd: Path | None = None) -> str:
    """Run a Miyabi PBS command asynchronously and return decoded stdout.

    Args:
        *args: Command and arguments passed to
            `asyncio.create_subprocess_exec`.
        cwd: Optional working directory for the subprocess.

    Returns:
        Standard output decoded with replacement for invalid bytes.

    Raises:
        RuntimeError: If the command exits with a non-zero return code. The
            error message includes decoded stdout and stderr for diagnostics.
    """
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out_b, err_b = await proc.communicate()
    out = (out_b or b"").decode(errors="replace")
    err = (err_b or b"").decode(errors="replace")

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)} rc={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}"
        )
    return out


@dataclass(frozen=True)
class SubmitResult:
    """Submission result returned after PBS accepts a batch script.

    Attributes:
        job_id: PBS job id parsed from ``qsub`` stdout.
        raw_output: Raw, stripped stdout emitted by ``qsub``.
    """

    job_id: str
    raw_output: str


class MiyabiPBSRuntime:
    """Async runtime wrapper for Miyabi PBS scheduler commands.

    The runtime maps to the core PBS commands used on Miyabi:
    ``qsub`` for submission, ``qstat -fH`` for completed-job status, and
    ``qdel`` for cancellation. Workflow code usually calls
    `qcsc_prefect_executor.miyabi.run.run_miyabi_job` or
    `qcsc_prefect_executor.from_blocks.run_job_from_blocks` instead.
    """

    QSTAT_OUT: ClassVar[re.Pattern] = re.compile(r"Job Id: (\d+\.\w+)\n((?:[ \t]+.*(?:\n|$))*)")

    async def submit(
        self,
        script_path: Path,
        *,
        cwd: Path | None = None,
        max_attempts: int = 3,
        retry_delay_seconds: float = 15.0,
    ) -> SubmitResult:
        """Submit a PBS script with ``qsub``.

        Retries on any failed ``qsub`` invocation -- observed in production as a
        transient ``qsub: cannot connect to server ... (errno=15010)`` during a brief PBS
        server blip, which otherwise kills the entire multi-hour/multi-step run over a
        single dropped connection. The server was reachable again within minutes, so a
        short retry is enough; a real, sustained outage still surfaces as ``SubmitError``
        after ``max_attempts``.

        Args:
            script_path: Path to the PBS script file.
            cwd: Optional working directory for ``qsub`` execution.
            max_attempts: Number of ``qsub`` attempts before giving up.
            retry_delay_seconds: Delay between attempts.

        Returns:
            Parsed submission result including job id and raw output.

        Raises:
            SubmitError: If submission fails on every attempt, or a job id cannot be
                parsed from a successful-looking ``qsub`` call.
        """
        last_exc: Exception | None = None
        stdout: str | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                stdout = await run_command("qsub", str(script_path), cwd=cwd)
                break
            except Exception as e:
                last_exc = e
                if attempt < max_attempts:
                    await asyncio.sleep(retry_delay_seconds)
        if stdout is None:
            raise SubmitError(
                f"qsub failed for {script_path} after {max_attempts} attempt(s)"
            ) from last_exc

        out = stdout.strip()
        if not out:
            raise SubmitError("qsub returned empty stdout; cannot parse job id.")
        job_id = out.split()[0]
        return SubmitResult(job_id=job_id, raw_output=out)

    async def wait_final_status(
        self,
        job_id: str,
        *,
        watch_poll_interval: float = 10.0,
        timeout_seconds: float | None = None,
        stdout_fallback_path: str | Path | None = None,
        stdout_fallback_grace_seconds: float = 1800.0,
    ) -> dict[str, Any]:
        """Wait until PBS reports the job in the finished-job list.

        ``qstat -fH`` exposes completed jobs on Miyabi. This method polls that
        view, parses PBS ``key = value`` records, and preserves continuation
        lines in the returned payload.

        Args:
            job_id: PBS job id to watch.
            watch_poll_interval: Seconds to wait between ``qstat`` calls.
            timeout_seconds: Optional maximum wait time.
            stdout_fallback_path: Optional path to the job's own stdout file
                (every Miyabi job redirects to ``<work_dir>/output.out`` per
                the PBS template). Miyabi's ``qstat -fH`` history is short-
                lived and can purge a finished job before any poll observes
                it there, which otherwise makes this loop spin silently until
                ``timeout_seconds`` (confirmed: a job that completed and
                produced full output was unobservable via ``-fH`` within
                hours). If given, a job whose stdout file exists and stops
                growing for ``stdout_fallback_grace_seconds`` is treated as
                finished even though qstat no longer reports it. This is a
                best-effort completion signal, not a success signal -- the
                returned dict reports ``Exit_status=0`` and callers should
                still validate actual results (e.g. by reading expected
                output files), exactly as they must already do since qstat's
                own ``Exit_status`` is trusted without independent checking
                today.
            stdout_fallback_grace_seconds: How long the stdout file's size
                must be unchanged before the fallback fires. Must comfortably
                exceed normal PBS queue-wait + node-boot + module-load time,
                and any legitimate quiet stretch between print statements
                during a long solve (confirmed in production: this fired
                after only 120s once, ~2 minutes into a job still in normal
                startup, before it had produced any output at all -- killing
                a task that would have completed successfully 30 minutes
                later). Default is deliberately conservative (30 min): a
                false positive here silently discards real results and
                crashes the run, while a false negative just waits a bit
                longer within the existing ``timeout_seconds`` budget.

        Returns:
            Parsed final PBS status dictionary, or a minimal best-effort
            dictionary (flagged via the ``_fallback`` key) if
            ``stdout_fallback_path`` triggered instead.

        Raises:
            WaitTimeout: If ``timeout_seconds`` elapses before final status is
                observed.
            RuntimeError: If an underlying ``qstat`` command fails.
        """
        start = asyncio.get_running_loop().time()
        fallback_last_size: int | None = None
        fallback_stable_since: float | None = None
        try:
            while True:
                if timeout_seconds is not None:
                    now = asyncio.get_running_loop().time()
                    if now - start > timeout_seconds:
                        raise WaitTimeout(f"timeout waiting for job_id={job_id}")

                stdout = await run_command("qstat", "-fH", job_id)
                match = re.search(self.QSTAT_OUT, stdout)
                if match:
                    current_key = ""
                    out: dict[str, Any] = {}

                    for line in match.group(2).splitlines():
                        if len(line) == 0:
                            continue

                        # continuation lines start with tab in this qstat output
                        if line.startswith("\t"):
                            # append continuation to the previous key
                            out[current_key] += line.strip()
                        else:
                            key, val = line.split("=", 1)
                            current_key = key.strip()
                            out[current_key] = val.strip()

                    return out

                if stdout_fallback_path is not None:
                    now = asyncio.get_running_loop().time()
                    try:
                        size = os.path.getsize(stdout_fallback_path)
                    except OSError:
                        size = None
                    if size is None:
                        fallback_last_size = None
                        fallback_stable_since = None
                    elif size != fallback_last_size:
                        fallback_last_size = size
                        fallback_stable_since = now
                    elif (
                        fallback_stable_since is not None
                        and now - fallback_stable_since >= stdout_fallback_grace_seconds
                    ):
                        return {
                            "job_state": "F",
                            "Exit_status": "0",
                            "Output_Path": str(stdout_fallback_path),
                            "_fallback": (
                                f"qstat -fH never reported job_id={job_id} as finished "
                                "(likely purged from PBS history before a poll observed "
                                f"it); falling back to stdout-file stability after "
                                f"{stdout_fallback_grace_seconds:.0f}s with no growth."
                            ),
                        }

                await asyncio.sleep(watch_poll_interval)

        except asyncio.CancelledError:
            # keep same behavior: cancel => qdel
            await run_command("qdel", job_id)
            return {}

    async def cancel(self, job_id: str) -> None:
        """Cancel a PBS job using ``qdel``.

        Args:
            job_id: Target PBS job id.

        Raises:
            CancelError: If cancellation fails.
        """

        try:
            await run_command("qdel", job_id)
        except Exception as e:
            raise CancelError(f"qdel failed for job_id={job_id}") from e
