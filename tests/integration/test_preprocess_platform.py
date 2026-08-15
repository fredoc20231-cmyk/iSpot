"""Platform-aware min_genes in preprocess().

A single min_genes=200 bar (fine for Visium) wipes out every cell on targeted,
single-cell-resolution panels (Xenium/CosMx/MERFISH), where detecting <200
genes/cell is normal — filtering to zero cells and crashing log1p downstream.
preprocess() now picks the threshold per platform.
"""
import numpy as np
import pytest

sc = pytest.importorskip("scanpy")
import anndata as ad  # noqa: E402

from ispot.preprocessing import preprocess, PLATFORM_MIN_GENES, DEFAULT_MIN_GENES  # noqa: E402


def _xenium_like(n_cells=2000, n_genes=300, seed=0):
    """Imaging-panel shape: many cells, few genes detected per cell (20-99)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_cells, n_genes), dtype="float32")
    for i in range(n_cells):
        k = int(rng.integers(20, 100))
        idx = rng.choice(n_genes, size=k, replace=False)
        X[i, idx] = rng.integers(1, 11, size=k)
    adata = ad.AnnData(X)
    adata.obsm["spatial"] = rng.random((n_cells, 2)).astype("float32")
    return adata


def test_platform_min_genes_table():
    assert PLATFORM_MIN_GENES["Visium"] == 200
    assert PLATFORM_MIN_GENES["Xenium"] == 10
    assert DEFAULT_MIN_GENES == 200
    # Unknown platform falls back to the Visium-safe default.
    assert PLATFORM_MIN_GENES.get("NoSuchPlatform", DEFAULT_MIN_GENES) == 200


def test_xenium_threshold_retains_cells_that_visium_bar_would_wipe():
    adata = _xenium_like()
    genes_per_cell = (adata.X > 0).sum(axis=1)
    # Scenario sanity: every cell is below the Visium bar but above the Xenium one.
    assert genes_per_cell.max() < 200
    assert genes_per_cell.min() >= 10

    # The old flat Visium threshold would drop ALL cells on this data.
    wiped = adata.copy()
    sc.pp.filter_cells(wiped, min_genes=200)
    assert wiped.shape[0] == 0

    # The platform-aware Xenium threshold keeps them and preprocess completes.
    out = preprocess(adata, platform="Xenium", n_top_genes=100, n_pcs=20)
    assert out.shape[0] == adata.shape[0]
    assert out.uns["qc_min_genes_used"] == 10
    assert out.uns["n_spots_before_qc"] == adata.shape[0]
    assert out.uns["n_spots_after_qc"] == out.shape[0]
    assert "X_pca" in out.obsm


def test_explicit_min_genes_overrides_platform():
    adata = _xenium_like(n_cells=500)
    out = preprocess(adata, platform="Xenium", min_genes=15, n_top_genes=100, n_pcs=20)
    assert out.uns["qc_min_genes_used"] == 15
