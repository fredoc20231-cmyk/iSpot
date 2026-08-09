"""Unit tests for Hungarian-matched metrics (ARI / NMI / F1)."""
import pytest

from ispot.metrics import compute_metrics, match_clusters_to_labels


def test_perfect_partition_relabeled_scores_one():
    # Same partition, different cluster ids -> all agreement metrics == 1.
    gt = ["A", "A", "A", "B", "B", "B"]
    pred = ["0", "0", "0", "1", "1", "1"]
    m = compute_metrics(gt, pred)
    assert m["ari"] == pytest.approx(1.0)
    assert m["nmi"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["weighted_f1"] == pytest.approx(1.0)
    assert m["n_clusters_pred"] == 2
    assert m["n_clusters_true"] == 2
    assert m["n_spots"] == 6


def test_all_expected_keys_present():
    m = compute_metrics(["A", "B"], ["0", "1"])
    for key in [
        "ari", "nmi", "macro_f1", "weighted_f1",
        "runtime", "n_spots", "n_clusters_pred", "n_clusters_true",
    ]:
        assert key in m


def test_hungarian_mapping_recovers_true_labels():
    # Swapped cluster ids must be re-aligned to the ground-truth labels.
    gt = ["A", "A", "B", "B"]
    pred = ["1", "1", "0", "0"]
    mapped = list(match_clusters_to_labels(gt, pred))
    assert mapped == ["A", "A", "B", "B"]


def test_over_clustering_marks_unmatched():
    # More predicted clusters than true -> the extra one is "unmatched".
    gt = ["A", "A", "A", "A"]
    pred = ["0", "0", "1", "1"]
    mapped = list(match_clusters_to_labels(gt, pred))
    assert "unmatched" in mapped


def test_runtime_passthrough():
    m = compute_metrics(["A", "B"], ["0", "1"], runtime_sec=12.5)
    assert m["runtime"] == pytest.approx(12.5)
