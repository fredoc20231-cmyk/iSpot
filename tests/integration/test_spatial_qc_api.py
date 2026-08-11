"""Integration test for the fast standalone SpatialQC endpoint (POST /api/qc).

Proves the run-first QC mode boots, loads real data through the loader stack,
and returns a FastQC-style PASS/WARN/FAIL report plus downloadable deliverables
— without running preprocessing/clustering. Runs in the integration CI job.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scanpy")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
ad = pytest.importorskip("anndata")

from fastapi.testclient import TestClient  # noqa: E402


def _write_h5ad(path, n_side=16, n_genes=60, seed=0):
    rng = np.random.default_rng(seed)
    coords = np.array([(x, y) for x in range(n_side) for y in range(n_side)], dtype=float)
    n = len(coords)
    # spatial depth gradient -> non-trivial Moran's I
    depth = (coords[:, 0] + 1) * 30
    X = np.vstack([rng.poisson(d / n_genes, size=n_genes) for d in depth]).astype("float32")
    adata = ad.AnnData(X)
    adata.var_names = ["MT-CO1"] + [f"g{i}" for i in range(n_genes - 1)]
    adata.obs_names = [f"s{i}" for i in range(n)]
    adata.obsm["spatial"] = coords
    adata.uns["platform"] = "Visium"
    adata.write_h5ad(path)


def test_qc_endpoint_returns_report_and_files(tmp_path, monkeypatch):
    monkeypatch.setenv("ISPOT_JOBS_DIR", str(tmp_path / "jobs"))
    import importlib
    import ispot.api as api
    importlib.reload(api)

    h5ad = tmp_path / "sample.h5ad"
    _write_h5ad(str(h5ad))

    with TestClient(api.app) as client:
        with open(h5ad, "rb") as fh:
            r = client.post(
                "/api/qc",
                files={"file": ("sample.h5ad", fh, "application/octet-stream")},
                data={"platform": "Visium", "sample_id": "sampleA"},
            )
        assert r.status_code == 200, r.text
        body = r.json()

        # Report shape
        assert body["platform"] == "Visium"
        assert body["platform_confidence"] == "explicit"
        assert body["summary"]["overall"] in ("pass", "warn", "fail")
        ids = [m["id"] for m in body["report"]["modules"]]
        assert "spatial_signal" in ids and "tissue_coverage" in ids
        assert body["qc_summary"]["sample_id"] == "sampleA"

        # Deliverables are downloadable through the standard results route.
        html = client.get(body["qc_report_html"])
        assert html.status_code == 200
        assert "SpatialQC Report" in html.text
        assert "data:image/png;base64," in html.text   # detailed report embeds figures
        assert "Spatial Signal Sanity Check" in html.text
        summ = client.get(body["qc_summary_json"])
        assert summ.status_code == 200
        assert summ.json()["platform"] == "Visium"


def test_qc_endpoint_rejects_bad_extension(tmp_path, monkeypatch):
    monkeypatch.setenv("ISPOT_JOBS_DIR", str(tmp_path / "jobs2"))
    import importlib
    import ispot.api as api
    importlib.reload(api)

    with TestClient(api.app) as client:
        r = client.post(
            "/api/qc",
            files={"file": ("bad.txt", b"not data", "text/plain")},
        )
        assert r.status_code in (400, 415, 422)
