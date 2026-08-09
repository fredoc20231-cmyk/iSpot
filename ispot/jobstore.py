"""
Persistent job store.

The MVP kept job status in a process-local dict, so a restart (or crash) lost
every in-flight and completed job. This module provides a small pluggable store
so job metadata survives restarts:

    InMemoryJobStore   the original behavior (no persistence)
    SqliteJobStore     durable, on-disk, safe for the single-server MVP

``create_job_store`` selects the backend from ISPOT_JOBS_BACKEND
("sqlite" default, or "memory"). This is the seam a Redis/Celery-backed store
would plug into for distributed execution (roadmap item 1); the API and worker
depend only on the JobStore interface, not on where jobs live.

Stdlib only (json + sqlite3), so it is fully unit-testable in CI.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any, Optional


def _jsonable_default(obj: Any):
    """Best-effort conversion of non-JSON-native values (e.g. numpy scalars).

    Avoids importing numpy: numpy scalars expose ``.item()`` and arrays expose
    ``.tolist()``. Anything else falls back to its string form.
    """
    if hasattr(obj, "item") and not hasattr(obj, "__len__"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


class JobStore:
    """Interface for job persistence."""

    def save(self, job: dict) -> None:
        raise NotImplementedError

    def get(self, job_id: str) -> Optional[dict]:
        raise NotImplementedError

    def all(self) -> list[dict]:
        raise NotImplementedError

    def delete(self, job_id: str) -> None:
        raise NotImplementedError

    def __contains__(self, job_id: str) -> bool:
        return self.get(job_id) is not None


class InMemoryJobStore(JobStore):
    """Non-persistent store (original MVP behavior)."""

    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def save(self, job: dict) -> None:
        job_id = job["job_id"]
        with self._lock:
            # Store a copy so later in-place mutation doesn't retroactively
            # change persisted state.
            self._jobs[job_id] = json.loads(json.dumps(job, default=_jsonable_default))

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def all(self) -> list[dict]:
        with self._lock:
            return [dict(j) for j in self._jobs.values()]

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)


class SqliteJobStore(JobStore):
    """Durable job store backed by SQLite (job dict serialized as JSON)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self._conn.commit()

    def save(self, job: dict) -> None:
        job_id = job["job_id"]
        payload = json.dumps(job, default=_jsonable_default)
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (job_id, data) VALUES (?, ?) "
                "ON CONFLICT(job_id) DO UPDATE SET data=excluded.data",
                (job_id, payload),
            )
            self._conn.commit()

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def all(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM jobs").fetchall()
        return [json.loads(r[0]) for r in rows]

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def create_job_store(
    db_path: Optional[str] = None, backend: Optional[str] = None
) -> JobStore:
    """Create a job store from the requested/configured backend.

    backend resolution: explicit arg -> ISPOT_JOBS_BACKEND -> "sqlite".
    """
    backend = (backend or os.environ.get("ISPOT_JOBS_BACKEND") or "sqlite").lower()
    if backend == "memory":
        return InMemoryJobStore()
    if backend == "sqlite":
        if not db_path:
            db_path = os.environ.get("ISPOT_JOBS_DB", "ispot_jobs/jobs.db")
        return SqliteJobStore(db_path)
    raise ValueError(f"Unknown job store backend: {backend!r}")
