"""
Retention cleanup for job storage.

The upload endpoint documents a 7-day retention policy, but only the
"delete on successful completion" half was implemented — abandoned uploads
(uploaded but never benchmarked, or failed) accumulated on disk forever. This
module removes job directories for jobs that never completed and are older than
the retention window.

The age/eligibility rule is a pure, unit-tested predicate; the orchestration
walks the job store and removes directories.
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from typing import Optional

DEFAULT_MAX_AGE_DAYS = 7


def is_expired(
    age_seconds: float, completed_at: Optional[str], max_age_seconds: float
) -> bool:
    """A job is expired only if it never completed AND is older than the window.

    Completed jobs are retained (their uploads are already deleted; their
    results are the deliverable).
    """
    return completed_at is None and age_seconds > max_age_seconds


def _job_age_seconds(job: dict, now: datetime) -> Optional[float]:
    created = job.get("created_at")
    if not created:
        return None
    try:
        return (now - datetime.fromisoformat(created)).total_seconds()
    except (ValueError, TypeError):
        return None


def cleanup_expired_jobs(
    jobs: dict,
    jobs_dir: str,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    now: Optional[datetime] = None,
) -> list[str]:
    """Remove directories for expired jobs. Returns the removed job IDs.

    ``jobs`` maps job_id -> job dict (with ``created_at`` / ``completed_at``).
    Only the on-disk directory is removed here; the caller is responsible for
    dropping the entries from its in-memory map and persistent store.
    """
    if now is None:
        now = datetime.now()
    max_age_seconds = max_age_days * 86400
    removed: list[str] = []
    for job_id, job in list(jobs.items()):
        age = _job_age_seconds(job, now)
        if age is None:
            continue
        if is_expired(age, job.get("completed_at"), max_age_seconds):
            shutil.rmtree(os.path.join(str(jobs_dir), job_id), ignore_errors=True)
            removed.append(job_id)
    return removed
