"""Unit tests for domain-map diagnostics."""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from ispot.domain_diagnostics import (  # noqa: E402
    diagnose_domains,
    spatial_label_coherence,
    _correlation_ratio,
)


def _grid(side=20):
    xs, ys = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
    return np.column_stack([xs.ravel(), ys.ravel()])


def test_correlation_ratio_perfect_and_none():
    coords = _grid()
    y = coords[:, 1]
    y_bands = (y * 5).astype(int).astype(str)          # label determined by y
    assert _correlation_ratio(y_bands, y) > 0.95
    # A label uncorrelated with x: same y-bands vs the x axis is near zero.
    assert _correlation_ratio(y_bands, coords[:, 0]) < 0.2


def test_axis_banded_flag():
    coords = _grid()
    labels = (coords[:, 1] * 5).astype(int).astype(str)   # horizontal bands
    d = diagnose_domains(coords, labels)
    assert d["flag"] == "axis-banded"
    assert d["dominant_axis"] == "y"
    assert d["axis_eta_y"] > 0.9
    assert d["spatial_coherence"] > 0.5


def test_salt_and_pepper_flag():
    coords = _grid()
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=coords.shape[0]).astype(str)  # random
    d = diagnose_domains(coords, labels)
    assert d["flag"] == "salt-and-pepper"
    assert d["spatial_coherence"] < 0.5


def test_coherent_flag_not_explained_by_one_axis():
    coords = _grid()
    # Three Voronoi regions arranged as a triangle: contiguous, and neither
    # axis alone determines the label.
    centers = np.array([[0.15, 0.2], [0.85, 0.25], [0.5, 0.9]])
    dists = np.stack([((coords - c) ** 2).sum(axis=1) for c in centers], axis=1)
    labels = dists.argmin(axis=1).astype(str)
    d = diagnose_domains(coords, labels)
    assert d["n_domains"] == 3
    assert d["spatial_coherence"] > 0.7          # contiguous regions
    assert max(d["axis_eta_x"], d["axis_eta_y"]) < 0.9   # no single-axis triviality
    assert d["flag"] == "coherent"


def test_degenerate_single_domain():
    coords = _grid()
    labels = np.array(["0"] * coords.shape[0])
    assert diagnose_domains(coords, labels)["flag"] == "degenerate"


def test_unassigned_spots_are_ignored():
    coords = _grid(side=20)
    # Two big left/right regions; drop a spatial band of spots as unassigned.
    labels = np.where(coords[:, 0] > 0.5, "1", "0").astype("<U11")
    labels[coords[:, 1] < 0.25] = "unassigned"
    d = diagnose_domains(coords, labels)
    # Classified on the assigned remainder (two coherent regions), NOT derailed
    # to a single-domain "degenerate" verdict by the unassigned spots.
    assert d["n_domains"] == 2
    assert d["flag"] != "degenerate"


def test_coherence_bounds():
    coords = _grid()
    labels = (coords[:, 1] * 5).astype(int).astype(str)
    c = spatial_label_coherence(coords, labels)
    assert 0.0 <= c <= 1.0
