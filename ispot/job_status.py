"""
Job outcome classification.

A benchmark runs many (method, seed) attempts. One method failing (an R
subprocess crash, an OOM on a large sample, a missing optional dependency)
should not sink the whole job when other methods produced usable results. This
module maps the per-attempt result rows to a single job-level outcome:

    completed          every attempted method produced at least one result
    completed_partial  some methods succeeded, some failed entirely
    failed             no method produced any result

Pure Python (no third-party deps) so it is unit-testable in isolation.
"""
from __future__ import annotations

from typing import Any


def classify_job_status(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify the overall outcome of a benchmark job.

    Parameters
    ----------
    results : list of dict
        Per-(method, seed) result rows. Each row must carry a ``method`` key;
        a row is considered a failure when its ``error`` value is not ``None``.
        A method counts as succeeded if *any* of its seed runs succeeded.

    Returns
    -------
    dict with keys:
        status : "completed" | "completed_partial" | "failed"
        n_methods, n_succeeded, n_failed : int
        succeeded_methods : list[str]  (sorted)
        failed_methods : list[dict]    (sorted; {"method", "error"})
    """
    methods: dict[str, dict[str, Any]] = {}
    for row in results:
        name = row.get("method", "unknown")
        entry = methods.setdefault(name, {"ok": False, "error": None})
        if row.get("error") is None:
            entry["ok"] = True
        elif entry["error"] is None:
            entry["error"] = str(row.get("error"))

    succeeded = sorted(m for m, e in methods.items() if e["ok"])
    failed = sorted(m for m, e in methods.items() if not e["ok"])

    if not methods or not succeeded:
        status = "failed"
    elif not failed:
        status = "completed"
    else:
        status = "completed_partial"

    return {
        "status": status,
        "n_methods": len(methods),
        "n_succeeded": len(succeeded),
        "n_failed": len(failed),
        "succeeded_methods": succeeded,
        "failed_methods": [
            {"method": m, "error": methods[m]["error"]} for m in failed
        ],
    }
