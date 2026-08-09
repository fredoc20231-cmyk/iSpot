"""Unit tests for job outcome classification."""
from ispot.job_status import classify_job_status


def _row(method, error=None, seed=42):
    return {"method": method, "seed": seed, "error": error}


def test_all_methods_succeed_is_completed():
    results = [_row("a"), _row("b"), _row("c")]
    s = classify_job_status(results)
    assert s["status"] == "completed"
    assert s["n_methods"] == 3
    assert s["n_succeeded"] == 3
    assert s["n_failed"] == 0
    assert s["failed_methods"] == []


def test_some_methods_fail_is_partial():
    results = [_row("a"), _row("b", error="boom"), _row("c")]
    s = classify_job_status(results)
    assert s["status"] == "completed_partial"
    assert s["n_succeeded"] == 2
    assert s["n_failed"] == 1
    assert s["failed_methods"] == [{"method": "b", "error": "boom"}]
    assert s["succeeded_methods"] == ["a", "c"]


def test_all_methods_fail_is_failed():
    results = [_row("a", error="x"), _row("b", error="y")]
    s = classify_job_status(results)
    assert s["status"] == "failed"
    assert s["n_succeeded"] == 0
    assert s["n_failed"] == 2


def test_empty_results_is_failed():
    s = classify_job_status([])
    assert s["status"] == "failed"
    assert s["n_methods"] == 0


def test_method_with_one_good_seed_counts_as_succeeded():
    # A stochastic method: one seed crashed, another produced a result.
    results = [_row("m", error="seed1 died", seed=1), _row("m", error=None, seed=2)]
    s = classify_job_status(results)
    assert s["status"] == "completed"
    assert s["n_succeeded"] == 1
    assert s["n_failed"] == 0


def test_first_error_message_is_reported():
    results = [_row("m", error="first"), _row("m", error="second")]
    s = classify_job_status(results)
    assert s["status"] == "failed"
    assert s["failed_methods"] == [{"method": "m", "error": "first"}]
