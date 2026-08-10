"""Unit tests for the persistent job store."""
import pytest

from ispot.jobstore import (
    InMemoryJobStore,
    SqliteJobStore,
    create_job_store,
    _jsonable_default,
)


def _job(job_id="j1", **extra):
    base = {"job_id": job_id, "status": "queued", "progress": 0.0}
    base.update(extra)
    return base


@pytest.mark.parametrize("make", [
    lambda tmp: InMemoryJobStore(),
    lambda tmp: SqliteJobStore(str(tmp / "jobs.db")),
])
def test_save_get_all_delete_contains(make, tmp_path):
    store = make(tmp_path)
    assert store.get("missing") is None
    assert "missing" not in store

    store.save(_job("a", status="running"))
    store.save(_job("b", status="completed"))

    assert "a" in store
    assert store.get("a")["status"] == "running"
    ids = sorted(j["job_id"] for j in store.all())
    assert ids == ["a", "b"]

    store.delete("a")
    assert "a" not in store
    assert len(store.all()) == 1


def test_save_updates_existing(tmp_path):
    store = SqliteJobStore(str(tmp_path / "jobs.db"))
    store.save(_job("a", status="running", progress=0.5))
    store.save(_job("a", status="completed", progress=1.0))
    got = store.get("a")
    assert got["status"] == "completed"
    assert got["progress"] == 1.0
    assert len(store.all()) == 1  # upsert, not duplicate


def test_sqlite_persists_across_reopen(tmp_path):
    db = str(tmp_path / "jobs.db")
    store = SqliteJobStore(db)
    store.save(_job("a", status="completed", data_profile={"n_spots": 42}))
    store.close()

    reopened = SqliteJobStore(db)
    got = reopened.get("a")
    assert got is not None
    assert got["status"] == "completed"
    assert got["data_profile"]["n_spots"] == 42


def test_inmemory_stores_a_copy(tmp_path):
    store = InMemoryJobStore()
    job = _job("a", status="running")
    store.save(job)
    job["status"] = "mutated-after-save"
    assert store.get("a")["status"] == "running"


def test_create_job_store_backends(tmp_path):
    assert isinstance(create_job_store(backend="memory"), InMemoryJobStore)
    assert isinstance(
        create_job_store(str(tmp_path / "j.db"), backend="sqlite"), SqliteJobStore
    )
    with pytest.raises(ValueError):
        create_job_store(backend="nonsense")


def test_create_job_store_env(monkeypatch):
    monkeypatch.setenv("ISPOT_JOBS_BACKEND", "memory")
    assert isinstance(create_job_store(), InMemoryJobStore)


def test_jsonable_default_handles_numpy_like():
    class FakeScalar:
        def item(self):
            return 7

    class FakeArray:
        def tolist(self):
            return [1, 2, 3]

    assert _jsonable_default(FakeScalar()) == 7
    assert _jsonable_default(FakeArray()) == [1, 2, 3]
    # Unknown objects fall back to their string form.
    obj = object()
    assert isinstance(_jsonable_default(obj), str)


def test_sqlite_serializes_numpy_like_values(tmp_path):
    class FakeBool:
        def item(self):
            return True

    store = SqliteJobStore(str(tmp_path / "jobs.db"))
    store.save(_job("a", has_ground_truth=FakeBool()))
    assert store.get("a")["has_ground_truth"] is True
