"""Unit tests for density-based tissue detection (Slide-seq/Stereo-seq/DBiT-seq).

Bead/chip platforms ship no histology image and no in_tissue metadata, so the
viewer would otherwise show the raw device geometry. detect_tissue_by_expression_
density recovers the real tissue footprint from count density alone.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from ispot.tissue_segmentation import detect_tissue_by_expression_density  # noqa: E402


def _chip_with_circular_tissue(side=40, seed=0):
    """A full rectangular chip; only a centered circle carries real signal."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    cx = cy = (side - 1) / 2.0
    r = side * 0.3
    in_circle = (coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2 <= r * r
    counts = np.where(
        in_circle,
        rng.integers(800, 1500, size=coords.shape[0]),   # real tissue: high counts
        rng.integers(0, 30, size=coords.shape[0]),        # background: near-empty
    ).astype(float)
    return coords, counts, in_circle


def test_density_keeps_tissue_drops_background():
    coords, counts, in_circle = _chip_with_circular_tissue()
    on_tissue = detect_tissue_by_expression_density(coords, counts)
    assert on_tissue.dtype == bool and on_tissue.shape == (coords.shape[0],)
    # Most real-tissue spots kept; most background dropped, with clear separation.
    kept_tissue = on_tissue[in_circle].mean()
    kept_bg = on_tissue[~in_circle].mean()
    assert kept_tissue > 0.85
    assert kept_bg < 0.25
    assert kept_tissue - kept_bg > 0.6
    # And it doesn't keep everything (it actually filtered).
    assert on_tissue.sum() < coords.shape[0]


def test_density_fails_open_on_uniform_counts():
    # No discriminating signal -> keep everything (never wipe a dataset).
    coords, _, _ = _chip_with_circular_tissue()
    uniform = np.full(coords.shape[0], 500.0)
    on_tissue = detect_tissue_by_expression_density(coords, uniform)
    assert on_tissue.all()


def test_density_fails_open_on_tiny_input():
    coords = np.random.default_rng(0).random((10, 2))
    counts = np.random.default_rng(1).integers(0, 100, size=10).astype(float)
    on_tissue = detect_tissue_by_expression_density(coords, counts)
    assert on_tissue.all()
