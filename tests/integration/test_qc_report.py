"""Integration test for the QC HTML/JSON deliverable."""
import json
import os

import pytest

np = pytest.importorskip("numpy")
ad = pytest.importorskip("anndata")
pytest.importorskip("matplotlib")  # ispot.deliverables imports matplotlib


def test_generate_qc_report_writes_html_and_json(tmp_path):
    from ispot.qc import compute_qc
    from ispot.deliverables import generate_qc_report

    rng = np.random.default_rng(0)
    X = rng.multinomial(1000, [1 / 40] * 40, size=100).astype("float32")
    adata = ad.AnnData(X)
    adata.var_names = ["MT-CO1"] + [f"g{i}" for i in range(39)]
    adata.obsm["spatial"] = rng.random((100, 2))

    qc = compute_qc(adata, platform="Visium")
    html_path = generate_qc_report(qc, output_dir=str(tmp_path))

    assert os.path.exists(html_path)
    assert os.path.exists(os.path.join(str(tmp_path), "qc_report.json"))
    html = open(html_path).read()
    assert "QC Report" in html and "Modules" in html
    saved = json.load(open(os.path.join(str(tmp_path), "qc_report.json")))
    assert saved["summary"]["overall"] in ("pass", "warn", "fail")
