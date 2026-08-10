"""Boot + HTTP smoke test: prove the FastAPI app imports and serves.

Importing ispot.api pulls in scanpy (via cluster_estimation/preprocessing),
matplotlib/reportlab (deliverables), seeds the meta-learning DB, and trains the
model — so a clean import is a real "the app starts" check. Skipped when the
stack isn't installed.
"""
import pytest

pytest.importorskip("scanpy")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


def test_api_boots_and_serves_read_endpoints():
    import ispot.api as api

    with TestClient(api.app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/methods").status_code == 200
        assert client.get("/api/platforms").status_code == 200
        # recommend works even with no job (cold-start / general ranking)
        assert client.get("/api/meta-learning/recommend").status_code == 200
        assert client.get("/api/plugins").status_code == 200

        # Availability matrix reflects the real environment: Leiden runs
        # (scanpy installed), torch-backed methods do not (torch absent here).
        r = client.get("/api/methods/availability")
        assert r.status_code == 200
        body = r.json()
        assert body["availability"]["Leiden_PCA"]["available"] is True
        assert "Leiden_PCA" in body["runnable"]
        assert body["availability"]["GraphST"]["available"] is False
        assert body["default_methods"]  # non-empty (at least Leiden_PCA)


def test_api_key_gate(monkeypatch):
    monkeypatch.setenv("ISPOT_API_KEY", "secret")
    import ispot.api as api

    with TestClient(api.app) as client:
        # No key -> rejected before the job lookup.
        r = client.post("/api/benchmark", json={"job_id": "does-not-exist"})
        assert r.status_code == 401
        # Correct key -> passes auth, then 404 for the missing job.
        r2 = client.post(
            "/api/benchmark",
            json={"job_id": "does-not-exist"},
            headers={"X-API-Key": "secret"},
        )
        assert r2.status_code == 404
