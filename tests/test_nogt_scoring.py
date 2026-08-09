"""Unit tests for the no-ground-truth composite scoring components."""
import numpy as np
import pytest

from ispot.nogt_scoring import (
    DEFAULT_WEIGHTS,
    spatial_coherence_score,
    cluster_stability_score,
    expression_separability_score,
    consensus_clustering,
    compute_nogt_score,
)


def _square_grid(n_side):
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_stability_identical_runs_is_one():
    labels = np.array(["0", "0", "1", "1", "2", "2"])
    css = cluster_stability_score([labels.copy(), labels.copy()])
    assert css == pytest.approx(1.0)


def test_stability_single_run_is_one():
    assert cluster_stability_score([np.array(["0", "1"])]) == 1.0


def test_spatial_coherence_contiguous_beats_random():
    coords = _square_grid(20)  # 400 spots
    n = coords.shape[0]

    contiguous = np.where(coords[:, 0] < 10, "0", "1")
    scs_contiguous = spatial_coherence_score(contiguous, coords, k=4)

    rng = np.random.RandomState(0)
    scattered = rng.randint(0, 2, size=n).astype(str)
    scs_scattered = spatial_coherence_score(scattered, coords, k=4)

    assert scs_contiguous > 0.7
    assert scs_contiguous > scs_scattered


def test_expression_separability_well_separated_blobs():
    rng = np.random.RandomState(0)
    a = rng.normal(0.0, 0.1, size=(50, 5))
    b = rng.normal(10.0, 0.1, size=(50, 5))
    X = np.vstack([a, b])
    labels = np.array(["0"] * 50 + ["1"] * 50)

    ess = expression_separability_score(labels, X)
    assert 0.0 <= ess <= 1.0
    assert ess > 0.7


def test_compute_nogt_score_keys_and_ranges():
    coords = _square_grid(8)  # 64 spots
    n = coords.shape[0]
    labels_a = np.where(coords[:, 0] < 4, "0", "1")
    labels_b = np.where(coords[:, 1] < 4, "0", "1")
    rng = np.random.RandomState(1)
    X = np.hstack([coords, rng.normal(size=(n, 3))])
    all_labels = {"A": labels_a, "B": labels_b}

    res = compute_nogt_score(
        labels=labels_a,
        label_runs=[labels_a, labels_a],
        coords=coords,
        X_pca=X,
        all_method_labels=all_labels,
        n_clusters=2,
    )

    for key in ["nogt_score", "scs", "css", "ess", "cas", "weights"]:
        assert key in res
    for key in ["nogt_score", "scs", "css", "ess", "cas"]:
        assert 0.0 <= res[key] <= 1.0


def test_precomputed_consensus_is_accepted():
    coords = _square_grid(8)
    n = coords.shape[0]
    labels_a = np.where(coords[:, 0] < 4, "0", "1")
    labels_b = np.where(coords[:, 1] < 4, "0", "1")
    X = np.hstack([coords, np.zeros((n, 2))])
    all_labels = {"A": labels_a, "B": labels_b}

    consensus = consensus_clustering(all_labels, 2)
    res = compute_nogt_score(
        labels=labels_a,
        label_runs=[labels_a],
        coords=coords,
        X_pca=X,
        all_method_labels=all_labels,
        n_clusters=2,
        consensus_labels=consensus,
    )
    assert 0.0 <= res["cas"] <= 1.0
