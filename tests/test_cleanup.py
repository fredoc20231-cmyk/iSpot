"""Unit tests for job-retention cleanup."""
from datetime import datetime, timedelta

from ispot.cleanup import is_expired, cleanup_expired_jobs

WEEK = 7 * 86400


def test_is_expired_rules():
    # Never completed + older than window -> expired.
    assert is_expired(WEEK + 1, None, WEEK) is True
    # Never completed but within window -> keep.
    assert is_expired(WEEK - 1, None, WEEK) is False
    # Completed -> always keep, regardless of age.
    assert is_expired(WEEK * 10, "2020-01-01T00:00:00", WEEK) is False


def _mkjob(job_id, created, completed=None):
    return {"job_id": job_id, "created_at": created, "completed_at": completed}


def test_cleanup_removes_only_expired_orphans(tmp_path):
    now = datetime(2026, 1, 30, 12, 0, 0)
    old = (now - timedelta(days=10)).isoformat()
    recent = (now - timedelta(days=1)).isoformat()

    jobs = {
        "old_orphan": _mkjob("old_orphan", old),                 # remove
        "old_completed": _mkjob("old_completed", old, recent),   # keep (completed)
        "recent_orphan": _mkjob("recent_orphan", recent),        # keep (young)
        "no_timestamp": {"job_id": "no_timestamp"},              # keep (no created_at)
    }
    for jid in jobs:
        (tmp_path / jid).mkdir()

    removed = cleanup_expired_jobs(jobs, str(tmp_path), max_age_days=7, now=now)

    assert removed == ["old_orphan"]
    assert not (tmp_path / "old_orphan").exists()
    assert (tmp_path / "old_completed").exists()
    assert (tmp_path / "recent_orphan").exists()
    assert (tmp_path / "no_timestamp").exists()


def test_cleanup_handles_bad_timestamp(tmp_path):
    jobs = {"weird": {"job_id": "weird", "created_at": "not-a-date"}}
    (tmp_path / "weird").mkdir()
    removed = cleanup_expired_jobs(jobs, str(tmp_path), now=datetime(2026, 1, 1))
    assert removed == []
    assert (tmp_path / "weird").exists()
