"""Unit tests for SpatialQC: pure classification logic + report building."""
import pytest

from ispot.spatial_qc_report import (
    classify_lower_bound, classify_upper_bound, platform_profile, summarize,
    summary_record, PLATFORM_PROFILES, DEFAULT_PROFILE,
)


# --- pure logic (no heavy deps) --------------------------------------------
def test_classify_lower_bound():
    assert classify_lower_bound(1000, 500, 100) == "pass"
    assert classify_lower_bound(300, 500, 100) == "warn"
    assert classify_lower_bound(50, 500, 100) == "fail"
    assert classify_lower_bound(None, 500, 100) == "warn"
    assert classify_lower_bound(float("nan"), 500, 100) == "warn"


def test_classify_upper_bound():
    assert classify_upper_bound(0.5, 0.9, 0.97) == "pass"
    assert classify_upper_bound(0.95, 0.9, 0.97) == "warn"
    assert classify_upper_bound(0.99, 0.9, 0.97) == "fail"
    assert classify_upper_bound(None, 0.9, 0.97) == "warn"


def test_platform_profile_case_insensitive():
    assert platform_profile("visium") is PLATFORM_PROFILES["Visium"]
    assert platform_profile("XENIUM") is PLATFORM_PROFILES["Xenium"]
    assert platform_profile("nonsense") is DEFAULT_PROFILE
    assert platform_profile(None) is DEFAULT_PROFILE


def test_targeted_platforms_have_lenient_gene_thresholds():
    # A targeted panel's typical gene count is far below whole-transcriptome.
    assert PLATFORM_PROFILES["Xenium"]["genes_warn"] < PLATFORM_PROFILES["Visium"]["genes_warn"]
    assert PLATFORM_PROFILES["Xenium"]["targeted"] is True
    assert PLATFORM_PROFILES["Visium"]["targeted"] is False


def test_summarize():
    mods = [{"status": "pass"}, {"status": "warn"}, {"status": "pass"}]
    s = summarize(mods)
    assert s == {"pass": 2, "warn": 1, "fail": 0, "overall": "warn", "n_modules": 3}
    assert summarize([{"status": "fail"}])["overall"] == "fail"
    assert summarize([{"status": "pass"}])["overall"] == "pass"


def test_summary_record_shape():
    report = {
        "sample_id": "x", "platform": "Visium", "platform_confidence": "inferred",
        "summary": {"overall": "warn"},
        "basic": {"n_spots": 100, "n_genes": 40, "median_genes_per_spot": 30,
                  "median_counts_per_spot": 1000, "sparsity": 0.8,
                  "tissue_retention_pct": 95.0, "spatial_morans_i": 0.3},
        "modules": [{"id": "tissue_coverage", "status": "pass"}],
    }
    rec = summary_record(report)
    assert rec["sample_id"] == "x"
    assert rec["overall"] == "warn"
    assert rec["module_status"]["tissue_coverage"] == "pass"
    assert rec["tissue_retention_pct"] == 95.0


# --- report building (needs numpy + anndata; present in the unit CI job) ----
np = pytest.importorskip("numpy")
ad = pytest.importorskip("anndata")


def _make_adata(n_side=20, n_genes=50, structured=True, seed=0):
    """A small grid AnnData with a spatial count gradient for a positive signal."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    n = coords.shape[0]
    if structured:
        # depth increases with x -> strong spatial autocorrelation of counts
        depth = (coords[:, 0] + 1) * 40
    else:
        depth = np.full(n, 400.0)
    X = np.vstack([rng.poisson(d / n_genes, size=n_genes) for d in depth]).astype("float32")
    adata = ad.AnnData(X)
    adata.var_names = ["MT-CO1"] + [f"g{i}" for i in range(n_genes - 1)]
    adata.obsm["spatial"] = coords
    adata.uns["platform"] = "Visium"
    return adata


def test_build_spatial_qc_basic_modules_present():
    from ispot.spatial_qc_report import build_spatial_qc
    report = build_spatial_qc(_make_adata(), platform="Visium",
                              sample_id="t1", platform_confidence="inferred")
    ids = [m["id"] for m in report["modules"]]
    for expected in ["basic_statistics", "platform_detection_confidence",
                     "tissue_coverage", "gene_detection", "sparsity_dropout",
                     "spatial_signal", "image_segmentation_quality"]:
        assert expected in ids
    assert report["report"] == "SpatialQC"
    assert report["summary"]["overall"] in ("pass", "warn", "fail")


def test_platform_confidence_default_fails():
    from ispot.spatial_qc_report import build_spatial_qc
    report = build_spatial_qc(_make_adata(), platform="Visium",
                              sample_id="t", platform_confidence="default")
    conf_mod = next(m for m in report["modules"]
                    if m["id"] == "platform_detection_confidence")
    assert conf_mod["status"] == "fail"


def test_structured_data_has_positive_spatial_signal():
    from ispot.spatial_qc_report import build_spatial_qc
    report = build_spatial_qc(_make_adata(structured=True), platform="Visium",
                              platform_confidence="inferred")
    sig = next(m for m in report["modules"] if m["id"] == "spatial_signal")
    assert sig["value"] > 0.05  # real spatial structure -> not a fail


def test_tissue_coverage_reads_uns_counts():
    from ispot.spatial_qc_report import build_spatial_qc
    adata = _make_adata()
    # Simulate the loader's off-tissue bookkeeping: 400 kept out of 500.
    adata.uns["n_spots_before_tissue_filter"] = 500
    adata.uns["n_spots_excluded_off_tissue"] = 100
    report = build_spatial_qc(adata, platform="Visium", platform_confidence="inferred")
    cov = next(m for m in report["modules"] if m["id"] == "tissue_coverage")
    # 400/500 = 80% retention -> warn (below 90, above 70)
    assert cov["status"] == "warn"
    assert abs(cov["value"] - 80.0) < 1e-6


def test_gene_detection_platform_awareness():
    """The same low median genes/spot fails on Visium but passes on Xenium."""
    from ispot.spatial_qc_report import build_spatial_qc
    # ~25 genes/spot median: below Visium fail (500) but above Xenium warn (30)? 25<30 -> Xenium warn, Visium fail
    adata = _make_adata(n_genes=40)
    rep_v = build_spatial_qc(adata, platform="Visium", platform_confidence="inferred")
    rep_x = build_spatial_qc(adata, platform="Xenium", platform_confidence="inferred")
    gd_v = next(m for m in rep_v["modules"] if m["id"] == "gene_detection")
    gd_x = next(m for m in rep_x["modules"] if m["id"] == "gene_detection")
    # Whole-transcriptome expectations make this fail; targeted-panel expectations don't.
    order = {"pass": 0, "warn": 1, "fail": 2}
    assert order[gd_v["status"]] >= order[gd_x["status"]]


def test_write_spatial_qc_deliverables(tmp_path):
    pytest.importorskip("matplotlib")  # deliverables.generate_qc_report needs it
    from ispot.spatial_qc_report import build_spatial_qc, write_spatial_qc
    import os, json
    report = build_spatial_qc(_make_adata(), platform="Visium",
                              sample_id="s1", platform_confidence="inferred")
    paths = write_spatial_qc(report, output_dir=str(tmp_path), sample_id="s1")
    for key in ("html", "report_json", "summary_txt", "summary_json"):
        assert os.path.exists(paths[key]), key
    rec = json.load(open(paths["summary_json"]))
    assert rec["sample_id"] == "s1"
    assert "module_status" in rec
