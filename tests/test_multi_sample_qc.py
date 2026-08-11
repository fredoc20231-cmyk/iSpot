"""Unit tests for the MultiQC-style multi-sample aggregator (pure Python)."""
import json

from ispot.multi_sample_qc import (
    aggregate_qc, load_summaries, render_multi_sample_html, METRICS,
)


def _rec(sid, genes, counts, sparsity, retention, morans, n_spots, overall="pass"):
    return {
        "sample_id": sid, "platform": "Visium", "platform_confidence": "inferred",
        "overall": overall,
        "median_genes_per_spot": genes, "median_counts_per_spot": counts,
        "sparsity": sparsity, "tissue_retention_pct": retention,
        "spatial_morans_i": morans, "n_spots": n_spots,
    }


def _consistent_batch(n=6):
    # A uniform batch: identical-ish good samples.
    return [_rec(f"s{i}", 4000, 8000, 0.90, 98.0, 0.40, 3600) for i in range(n)]


def test_consistent_batch_has_no_outliers():
    agg = aggregate_qc(_consistent_batch())
    assert agg["n_samples"] == 6
    assert agg["n_outlier_samples"] == 0
    assert agg["outliers"] == []
    assert agg["batch_status"] == "pass"


def test_low_retention_outlier_is_flagged():
    recs = _consistent_batch(5)
    recs.append(_rec("bad", 4000, 8000, 0.90, 40.0, 0.40, 3600))  # retention way low
    agg = aggregate_qc(recs)
    flagged = [o for o in agg["outliers"] if o["sample_id"] == "bad"]
    assert any(o["metric"] == "tissue_retention_pct" for o in flagged)
    bad = next(s for s in agg["samples"] if s["sample_id"] == "bad")
    assert bad["is_outlier"] is True
    # The z-score must be in the concerning (below-mean) direction.
    assert bad["zscores"]["tissue_retention_pct"] < 0


def test_high_sparsity_outlier_direction():
    recs = _consistent_batch(5)
    recs.append(_rec("sparse", 4000, 8000, 0.999, 98.0, 0.40, 3600))
    agg = aggregate_qc(recs)
    assert any(o["sample_id"] == "sparse" and o["metric"] == "sparsity"
               for o in agg["outliers"])


def test_fail_sample_always_surfaced_even_if_not_statistical_outlier():
    recs = _consistent_batch(5)
    # Metrics identical to the batch, but this sample self-reported overall=fail.
    recs.append(_rec("failing", 4000, 8000, 0.90, 98.0, 0.40, 3600, overall="fail"))
    agg = aggregate_qc(recs)
    failing = next(s for s in agg["samples"] if s["sample_id"] == "failing")
    assert failing["is_outlier"] is True
    assert agg["batch_status"] == "fail"


def test_single_sample_no_crash_no_outlier():
    agg = aggregate_qc([_rec("only", 4000, 8000, 0.9, 98.0, 0.4, 3600)])
    assert agg["n_samples"] == 1
    # std is 0 with one sample -> no z-scores, no false outliers.
    assert agg["n_outlier_samples"] == 0


def test_missing_metric_is_skipped_gracefully():
    recs = _consistent_batch(4)
    partial = _rec("partial", 4000, 8000, 0.9, 98.0, 0.4, 3600)
    del partial["spatial_morans_i"]  # missing metric
    recs.append(partial)
    agg = aggregate_qc(recs)
    # Missing metric must not appear in that sample's z-scores or crash.
    partial_row = next(s for s in agg["samples"] if s["sample_id"] == "partial")
    assert "spatial_morans_i" not in partial_row["zscores"]


def test_load_summaries_from_dir(tmp_path):
    d = tmp_path / "sampleA"
    d.mkdir()
    rec = _rec("A", 4000, 8000, 0.9, 98.0, 0.4, 3600)
    (d / "qc_summary.json").write_text(json.dumps(rec))
    recs = load_summaries([str(tmp_path)])
    assert len(recs) == 1 and recs[0]["sample_id"] == "A"


def test_render_multi_sample_html(tmp_path):
    recs = _consistent_batch(3)
    recs.append(_rec("bad", 4000, 8000, 0.90, 30.0, 0.40, 3600))
    agg = aggregate_qc(recs)
    path = render_multi_sample_html(agg, output_dir=str(tmp_path))
    html = open(path).read()
    assert "Multi-sample" in html
    assert "bad" in html                     # the outlier row is present
    assert "Flagged outliers" in html


def test_metrics_directions_are_valid():
    for _key, _label, direction in METRICS:
        assert direction in ("low", "high", "both")
