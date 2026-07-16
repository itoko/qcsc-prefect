from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qcsc_prefect_executor.bulk.models import (
    ACTIVE_BULK_JOB_STATUSES,
    SUBMIT_CANDIDATE_BULK_JOB_STATUSES,
    TERMINAL_BULK_JOB_STATUSES,
    BulkJobRecord,
    BulkJobSpec,
    BulkJobStatus,
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_values(statuses: Sequence[BulkJobStatus]) -> list[str]:
    return [status.value for status in statuses]


def _coerce_status(status: BulkJobStatus | str) -> BulkJobStatus:
    return status if isinstance(status, BulkJobStatus) else BulkJobStatus(str(status))


def _coerce_statuses(statuses: Sequence[BulkJobStatus | str]) -> list[BulkJobStatus]:
    return [_coerce_status(status) for status in statuses]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _json_loads_dict(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    value = json.loads(text)
    if isinstance(value, dict):
        return value
    return {}


def _json_loads_paths(text: str | None) -> list[Path]:
    if not text:
        return []
    value = json.loads(text)
    if not isinstance(value, list):
        return []
    return [Path(path) for path in value]


def _expected_outputs_json(paths: list[Path]) -> str:
    return _json_dumps([str(path) for path in paths])


def _outputs_are_complete(paths: list[Path], *, work_dir: Path) -> bool:
    if not paths:
        return False
    normalized = [path if path.is_absolute() else work_dir / path for path in paths]
    return all(path.exists() for path in normalized)


class BulkJobRegistry:
    """Persistent SQLite registry for bulk HPC job state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bulk_jobs (
                    job_key TEXT PRIMARY KEY,
                    wave_id TEXT,
                    target_id TEXT,
                    stage_id TEXT,
                    status TEXT NOT NULL,
                    work_dir TEXT NOT NULL,
                    scheduler_job_id TEXT,
                    submit_attempts INTEGER NOT NULL DEFAULT 0,
                    monitor_attempts INTEGER NOT NULL DEFAULT 0,
                    command_args_json TEXT,
                    expected_outputs_json TEXT,
                    execution_profile_block TEXT,
                    hpc_profile_block TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    submitted_at TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    last_error TEXT,
                    priority INTEGER NOT NULL DEFAULT 0,
                    max_submit_attempts INTEGER NOT NULL DEFAULT 5,
                    submit_mode TEXT NOT NULL DEFAULT 'single',
                    bulk_group_key TEXT,
                    bulk_parent_job_id TEXT,
                    bulk_index INTEGER,
                    scheduler_subjob_id TEXT
                )
                """
            )
            self._migrate_schema(conn)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bulk_jobs_status ON bulk_jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bulk_jobs_wave_id ON bulk_jobs(wave_id)")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_bulk_jobs_status_created_at
                ON bulk_jobs(status, created_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_bulk_jobs_stage_wave_status
                ON bulk_jobs(stage_id, wave_id, status)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_bulk_jobs_bulk_parent_job_id
                ON bulk_jobs(bulk_parent_job_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_bulk_jobs_scheduler_subjob_id
                ON bulk_jobs(scheduler_subjob_id)
                """
            )

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        existing_columns = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(bulk_jobs)").fetchall()
        }
        column_defs = {
            "stage_id": "stage_id TEXT",
            "submit_mode": "submit_mode TEXT NOT NULL DEFAULT 'single'",
            "bulk_group_key": "bulk_group_key TEXT",
            "bulk_parent_job_id": "bulk_parent_job_id TEXT",
            "bulk_index": "bulk_index INTEGER",
            "scheduler_subjob_id": "scheduler_subjob_id TEXT",
            "execution_profile_block": "execution_profile_block TEXT",
            "hpc_profile_block": "hpc_profile_block TEXT",
        }
        for column_name, column_def in column_defs.items():
            if column_name not in existing_columns:
                conn.execute(f"ALTER TABLE bulk_jobs ADD COLUMN {column_def}")

    def upsert_jobs(self, jobs: list[BulkJobSpec]) -> None:
        """Register jobs idempotently without resetting existing progress."""

        if not jobs:
            return

        with self._connect() as conn:
            for job in jobs:
                existing = conn.execute(
                    "SELECT * FROM bulk_jobs WHERE job_key = ?",
                    (job.job_key,),
                ).fetchone()
                if existing is not None and existing["status"] == BulkJobStatus.SUCCEEDED.value:
                    continue

                now = _utcnow_iso()
                command_args_json = _json_dumps(job.command_args)
                expected_outputs_json = _expected_outputs_json(job.expected_outputs)
                outputs_complete = _outputs_are_complete(
                    job.expected_outputs,
                    work_dir=job.work_dir,
                )

                if existing is None:
                    status = BulkJobStatus.SUCCEEDED if outputs_complete else BulkJobStatus.PENDING
                    conn.execute(
                        """
                        INSERT INTO bulk_jobs (
                            job_key,
                            wave_id,
                            target_id,
                            stage_id,
                            status,
                            work_dir,
                            scheduler_job_id,
                            submit_attempts,
                            monitor_attempts,
                            command_args_json,
                            expected_outputs_json,
                            execution_profile_block,
                            hpc_profile_block,
                            created_at,
                            updated_at,
                            submitted_at,
                            started_at,
                            finished_at,
                            last_error,
                            priority,
                            max_submit_attempts
                        )
                        VALUES (
                            ?, ?, ?, ?, ?, ?, NULL, 0, 0, ?, ?, ?, ?, ?, ?, ?,
                            NULL, NULL, NULL, ?, ?
                        )
                        """,
                        (
                            job.job_key,
                            job.wave_id,
                            job.target_id,
                            job.stage_id,
                            status.value,
                            str(job.work_dir),
                            command_args_json,
                            expected_outputs_json,
                            job.execution_profile_block,
                            job.hpc_profile_block,
                            now,
                            now,
                            now if status == BulkJobStatus.SUCCEEDED else None,
                            job.priority,
                            job.max_submit_attempts,
                        ),
                    )
                    continue

                if outputs_complete:
                    conn.execute(
                        """
                        UPDATE bulk_jobs
                        SET
                            wave_id = ?,
                            target_id = ?,
                            stage_id = ?,
                            status = ?,
                            work_dir = ?,
                            command_args_json = ?,
                            expected_outputs_json = ?,
                            execution_profile_block = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN ?
                                ELSE execution_profile_block
                            END,
                            hpc_profile_block = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN ?
                                ELSE hpc_profile_block
                            END,
                            updated_at = ?,
                            finished_at = COALESCE(finished_at, ?),
                            last_error = NULL,
                            priority = ?,
                            max_submit_attempts = ?
                        WHERE job_key = ?
                        """,
                        (
                            job.wave_id,
                            job.target_id,
                            job.stage_id,
                            BulkJobStatus.SUCCEEDED.value,
                            str(job.work_dir),
                            command_args_json,
                            expected_outputs_json,
                            job.execution_profile_block,
                            job.hpc_profile_block,
                            now,
                            now,
                            job.priority,
                            job.max_submit_attempts,
                            job.job_key,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE bulk_jobs
                        SET
                            wave_id = ?,
                            target_id = ?,
                            stage_id = ?,
                            work_dir = ?,
                            command_args_json = ?,
                            expected_outputs_json = ?,
                            execution_profile_block = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN ?
                                ELSE execution_profile_block
                            END,
                            hpc_profile_block = CASE
                                WHEN scheduler_job_id IS NULL AND scheduler_subjob_id IS NULL
                                THEN ?
                                ELSE hpc_profile_block
                            END,
                            updated_at = ?,
                            priority = ?,
                            max_submit_attempts = ?
                        WHERE job_key = ?
                        """,
                        (
                            job.wave_id,
                            job.target_id,
                            job.stage_id,
                            str(job.work_dir),
                            command_args_json,
                            expected_outputs_json,
                            job.execution_profile_block,
                            job.hpc_profile_block,
                            now,
                            job.priority,
                            job.max_submit_attempts,
                            job.job_key,
                        ),
                    )

    def get_active_jobs(self) -> list[BulkJobRecord]:
        statuses = _status_values(tuple(ACTIVE_BULK_JOB_STATUSES))
        return self._fetch_records(
            f"status IN ({self._placeholders(statuses)})",
            statuses,
            "submitted_at IS NULL, submitted_at, created_at, job_key",
        )

    def get_monitorable_jobs(self) -> list[BulkJobRecord]:
        statuses = _status_values(
            (
                BulkJobStatus.SUBMITTED,
                BulkJobStatus.QUEUED,
                BulkJobStatus.RUNNING,
                BulkJobStatus.UNKNOWN,
            )
        )
        return self._fetch_records(
            f"status IN ({self._placeholders(statuses)})",
            statuses,
            "submitted_at IS NULL, submitted_at, created_at, job_key",
        )

    def get_submit_candidates(self, limit: int) -> list[BulkJobRecord]:
        if limit <= 0:
            return []

        statuses = _status_values(tuple(SUBMIT_CANDIDATE_BULK_JOB_STATUSES))
        where = (
            f"status IN ({self._placeholders(statuses)}) AND submit_attempts < max_submit_attempts"
        )
        return self._fetch_records(
            where,
            [*statuses, int(limit)],
            "priority DESC, created_at, job_key LIMIT ?",
        )

    def get_submit_candidates_fifo(self, limit: int) -> list[BulkJobRecord]:
        if limit <= 0:
            return []

        statuses = _status_values(tuple(SUBMIT_CANDIDATE_BULK_JOB_STATUSES))
        return self._fetch_records(
            f"status IN ({self._placeholders(statuses)})",
            [*statuses, int(limit)],
            "created_at ASC, rowid ASC, job_key ASC LIMIT ?",
        )

    def count_submit_candidates(self) -> int:
        statuses = _status_values(tuple(SUBMIT_CANDIDATE_BULK_JOB_STATUSES))
        return self._count_records(
            f"status IN ({self._placeholders(statuses)})",
            statuses,
        )

    def count_active_jobs(self) -> int:
        statuses = _status_values(tuple(ACTIVE_BULK_JOB_STATUSES))
        return self._count_records(
            f"status IN ({self._placeholders(statuses)})",
            statuses,
        )

    def bootstrap_done(self) -> bool:
        return (
            self._count_records(
                """
                submit_attempts > 0
                OR submitted_at IS NOT NULL
                OR scheduler_job_id IS NOT NULL
                OR scheduler_subjob_id IS NOT NULL
                """,
                [],
            )
            > 0
        )

    def get_job(self, job_key: str) -> BulkJobRecord | None:
        records = self._fetch_records("job_key = ?", [job_key], "job_key")
        if not records:
            return None
        return records[0]

    def get_all_jobs(self) -> list[BulkJobRecord]:
        return self._fetch_records("1 = 1", [], "created_at, job_key")

    def mark_submitted(
        self,
        job_key: str,
        scheduler_job_id: str,
        *,
        submit_mode: str = "single",
        bulk_group_key: str | None = None,
        bulk_parent_job_id: str | None = None,
        bulk_index: int | None = None,
        scheduler_subjob_id: str | None = None,
    ) -> None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE bulk_jobs
                SET
                    status = ?,
                    scheduler_job_id = ?,
                    submit_mode = ?,
                    bulk_group_key = ?,
                    bulk_parent_job_id = ?,
                    bulk_index = ?,
                    scheduler_subjob_id = ?,
                    submit_attempts = submit_attempts + 1,
                    submitted_at = COALESCE(submitted_at, ?),
                    updated_at = ?,
                    last_error = NULL
                WHERE job_key = ? AND status != ?
                """,
                (
                    BulkJobStatus.SUBMITTED.value,
                    scheduler_job_id,
                    submit_mode,
                    bulk_group_key,
                    bulk_parent_job_id,
                    bulk_index,
                    scheduler_subjob_id,
                    now,
                    now,
                    job_key,
                    BulkJobStatus.SUCCEEDED.value,
                ),
            )

    def mark_queued(self, job_key: str) -> None:
        self._mark_status(job_key, BulkJobStatus.QUEUED)

    def mark_running(self, job_key: str) -> None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE bulk_jobs
                SET
                    status = ?,
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?
                WHERE job_key = ? AND status != ?
                """,
                (
                    BulkJobStatus.RUNNING.value,
                    now,
                    now,
                    job_key,
                    BulkJobStatus.SUCCEEDED.value,
                ),
            )

    def mark_submit_deferred(self, job_key: str, error: str | None = None) -> None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE bulk_jobs
                SET
                    status = ?,
                    submit_attempts = submit_attempts + 1,
                    updated_at = ?,
                    last_error = ?
                WHERE job_key = ? AND status != ?
                """,
                (
                    BulkJobStatus.SUBMIT_DEFERRED.value,
                    now,
                    error,
                    job_key,
                    BulkJobStatus.SUCCEEDED.value,
                ),
            )

    def mark_succeeded(self, job_key: str) -> None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE bulk_jobs
                SET
                    status = ?,
                    finished_at = COALESCE(finished_at, ?),
                    updated_at = ?,
                    last_error = NULL
                WHERE job_key = ?
                """,
                (BulkJobStatus.SUCCEEDED.value, now, now, job_key),
            )

    def mark_failed(self, job_key: str, error: str | None = None) -> None:
        self._mark_terminal(job_key, BulkJobStatus.FAILED, error=error)

    def mark_cancelled(self, job_key: str, error: str | None = None) -> None:
        self._mark_terminal(job_key, BulkJobStatus.CANCELLED, error=error)

    def mark_unknown(self, job_key: str, error: str | None = None) -> None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE bulk_jobs
                SET
                    status = ?,
                    monitor_attempts = monitor_attempts + 1,
                    updated_at = ?,
                    last_error = ?
                WHERE job_key = ? AND status != ?
                """,
                (
                    BulkJobStatus.UNKNOWN.value,
                    now,
                    error,
                    job_key,
                    BulkJobStatus.SUCCEEDED.value,
                ),
            )

    def record_monitor_attempt(self, job_key: str) -> None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE bulk_jobs
                SET monitor_attempts = monitor_attempts + 1, updated_at = ?
                WHERE job_key = ?
                """,
                (now, job_key),
            )

    def all_terminal(self) -> bool:
        statuses = _status_values(tuple(TERMINAL_BULK_JOB_STATUSES))
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM bulk_jobs
                WHERE status NOT IN ({self._placeholders(statuses)})
                """,
                statuses,
            ).fetchone()
        return int(row["count"]) == 0

    def jobs_for_wave(self, wave_id: str) -> list[BulkJobRecord]:
        return self._fetch_records(
            "wave_id = ?",
            [wave_id],
            "created_at, job_key",
        )

    def is_wave_ready(self, wave_id: str) -> bool:
        jobs = self.jobs_for_wave(wave_id)
        if not jobs:
            return False
        return all(job.status == BulkJobStatus.SUCCEEDED for job in jobs)

    def get_ready_waves(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT wave_id
                FROM bulk_jobs
                WHERE wave_id IS NOT NULL
                ORDER BY wave_id
                """
            ).fetchall()
        return [str(row["wave_id"]) for row in rows if self.is_wave_ready(str(row["wave_id"]))]

    def status_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM bulk_jobs
                GROUP BY status
                ORDER BY status
                """
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def reset_jobs_for_rerun(
        self,
        *,
        job_keys: Sequence[str] | None = None,
        statuses: Sequence[BulkJobStatus | str] | None = None,
        to_status: BulkJobStatus | str = BulkJobStatus.PENDING,
        clear_scheduler_ids: bool = True,
        clear_error: bool = False,
    ) -> int:
        target_status = _coerce_status(to_status)
        source_statuses = _coerce_statuses(
            tuple(statuses)
            if statuses is not None
            else (BulkJobStatus.FAILED, BulkJobStatus.UNKNOWN)
        )
        if not source_statuses:
            return 0

        reset_job_keys = None if job_keys is None else [str(job_key) for job_key in job_keys]
        if reset_job_keys is not None and not reset_job_keys:
            return 0

        now = _utcnow_iso()
        source_status_values = _status_values(tuple(source_statuses))
        set_clauses = [
            "status = ?",
            "submit_attempts = 0",
            "monitor_attempts = 0",
            "updated_at = ?",
        ]
        params: list[Any] = [target_status.value, now]

        if clear_scheduler_ids:
            set_clauses.extend(
                [
                    "scheduler_job_id = NULL",
                    "submitted_at = NULL",
                    "started_at = NULL",
                    "finished_at = NULL",
                    "submit_mode = 'single'",
                    "bulk_group_key = NULL",
                    "bulk_parent_job_id = NULL",
                    "bulk_index = NULL",
                    "scheduler_subjob_id = NULL",
                ]
            )
        if clear_error:
            set_clauses.append("last_error = NULL")

        where_clauses = [f"status IN ({self._placeholders(source_status_values)})"]
        params.extend(source_status_values)
        if reset_job_keys is not None:
            where_clauses.append(f"job_key IN ({self._placeholders(reset_job_keys)})")
            params.extend(reset_job_keys)

        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE bulk_jobs
                SET {", ".join(set_clauses)}
                WHERE {" AND ".join(where_clauses)}
                """,
                params,
            )
        return int(cursor.rowcount)

    def refresh_completed_jobs_from_outputs(self) -> None:
        now = _utcnow_iso()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM bulk_jobs WHERE status != ?",
                (BulkJobStatus.SUCCEEDED.value,),
            ).fetchall()
            for row in rows:
                expected_outputs = _json_loads_paths(row["expected_outputs_json"])
                work_dir = Path(str(row["work_dir"]))
                if not _outputs_are_complete(expected_outputs, work_dir=work_dir):
                    continue

                conn.execute(
                    """
                    UPDATE bulk_jobs
                    SET
                        status = ?,
                        finished_at = COALESCE(finished_at, ?),
                        updated_at = ?,
                        last_error = NULL
                    WHERE job_key = ?
                    """,
                    (
                        BulkJobStatus.SUCCEEDED.value,
                        now,
                        now,
                        row["job_key"],
                    ),
                )

    def _mark_status(self, job_key: str, status: BulkJobStatus) -> None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE bulk_jobs
                SET status = ?, updated_at = ?
                WHERE job_key = ? AND status != ?
                """,
                (
                    status.value,
                    now,
                    job_key,
                    BulkJobStatus.SUCCEEDED.value,
                ),
            )

    def _mark_terminal(
        self,
        job_key: str,
        status: BulkJobStatus,
        *,
        error: str | None,
    ) -> None:
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE bulk_jobs
                SET
                    status = ?,
                    finished_at = COALESCE(finished_at, ?),
                    updated_at = ?,
                    last_error = ?
                WHERE job_key = ? AND status != ?
                """,
                (
                    status.value,
                    now,
                    now,
                    error,
                    job_key,
                    BulkJobStatus.SUCCEEDED.value,
                ),
            )

    def _fetch_records(
        self,
        where_clause: str,
        params: Sequence[Any],
        order_clause: str,
    ) -> list[BulkJobRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM bulk_jobs
                WHERE {where_clause}
                ORDER BY {order_clause}
                """,
                list(params),
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def _count_records(self, where_clause: str, params: Sequence[Any]) -> int:
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM bulk_jobs
                WHERE {where_clause}
                """,
                list(params),
            ).fetchone()
        return int(row["count"])

    @staticmethod
    def _placeholders(values: Sequence[Any]) -> str:
        return ", ".join("?" for _ in values)


def _record_from_row(row: sqlite3.Row) -> BulkJobRecord:
    try:
        status = BulkJobStatus(str(row["status"]))
    except ValueError:
        status = BulkJobStatus.UNKNOWN

    return BulkJobRecord(
        job_key=str(row["job_key"]),
        wave_id=row["wave_id"],
        target_id=row["target_id"],
        stage_id=row["stage_id"],
        status=status,
        work_dir=Path(str(row["work_dir"])),
        scheduler_job_id=row["scheduler_job_id"],
        submit_attempts=int(row["submit_attempts"]),
        monitor_attempts=int(row["monitor_attempts"]),
        command_args=_json_loads_dict(row["command_args_json"]),
        expected_outputs=_json_loads_paths(row["expected_outputs_json"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        submitted_at=row["submitted_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        last_error=row["last_error"],
        priority=int(row["priority"]),
        max_submit_attempts=int(row["max_submit_attempts"]),
        submit_mode=str(row["submit_mode"] or "single"),
        bulk_group_key=row["bulk_group_key"],
        bulk_parent_job_id=row["bulk_parent_job_id"],
        bulk_index=None if row["bulk_index"] is None else int(row["bulk_index"]),
        scheduler_subjob_id=row["scheduler_subjob_id"],
        execution_profile_block=row["execution_profile_block"],
        hpc_profile_block=row["hpc_profile_block"],
    )
