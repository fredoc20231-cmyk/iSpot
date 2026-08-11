"""Unit tests for the spatial feature analysis (Moran's I, SVGs, QC maps)."""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
ad = pytest.importorskip("anndata")

from ispot.spatial_features import (  # noqa: E402
    _knn_weights, morans_i_batch, gearys_c_batch, spatially_variable_genes,
    spatial_qc_maps,
)


def _grid(n_side=20):
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


def test_knn_weights_row_normalized():
    coords = _grid(10)
    W = _knn_weights(coords, k=6)
    rowsums = np.asarray(W.sum(axis=1)).ravel()
    assert np.allclose(rowsums, 1.0)


def test_morans_i_gradient_vs_noise():
    coords = _grid(20)
    n = coords.shape[0]
    W = _knn_weights(coords, k=6)
    gradient = coords[:, 0].astype(float)            # smooth spatial gradient
    constant = np.ones(n)
    rng = np.random.default_rng(0)
    noise = rng.random(n)
    M = np.column_stack([gradient, constant, noise])
    I = morans_i_batch(M, W)
    assert I[0] > 0.8            # gradient -> strong positive autocorrelation
    assert abs(I[1]) < 1e-9      # constant -> 0 (degenerate variance guard)
    assert abs(I[2]) < 0.2       # noise -> near zero


def test_gearys_c_gradient_vs_noise():
    coords = _grid(20)
    n = coords.shape[0]
    W = _knn_weights(coords, k=6)
    gradient = coords[:, 0].astype(float)
    rng = np.random.default_rng(0)
    noise = rng.random(n)
    C = gearys_c_batch(np.column_stack([gradient, noise]), W)
    assert C[0] < 0.3          # strong spatial structure -> Geary's C well below 1
    assert abs(C[1] - 1.0) < 0.3   # noise -> Geary's C near 1


def test_svg_reports_both_statistics():
    adata = _adata_with_spatial_gene()
    svg = spatially_variable_genes(adata, max_genes=40, top=3)
    top = svg["top_svgs"][0]
    assert "morans_i" in top and "gearys_c" in top
    # The engineered spatial gene: high Moran's I AND low Geary's C.
    assert top["morans_i"] > 0.8 and top["gearys_c"] < 0.5


def _adata_with_spatial_gene(n_side=20, n_genes=40, seed=0):
    rng = np.random.default_rng(seed)
    coords = _grid(n_side)
    n = coords.shape[0]
    X = rng.poisson(1.0, size=(n, n_genes)).astype("float32")
    # gene 0 follows a strong spatial gradient (high total counts too)
    X[:, 0] = (coords[:, 0] * 5 + 1).astype("float32")
    adata = ad.AnnData(X)
    adata.var_names = ["MT-CO1"] + [f"g{i}" for i in range(n_genes - 1)]
    adata.obsm["spatial"] = coords
    return adata


def test_spatially_variable_genes_ranks_gradient_first():
    adata = _adata_with_spatial_gene()
    svg = spatially_variable_genes(adata, max_genes=40, top=5, n_maps=2)
    assert svg is not None
    assert svg["top_svgs"][0]["gene"] == "MT-CO1"   # the engineered spatial gene
    assert svg["max_morans_i"] > 0.8
    assert len(svg["maps"]) == 2
    assert svg["maps"][0]["x"]                        # coordinates present


def test_spatially_variable_genes_none_without_coords():
    adata = ad.AnnData(np.ones((20, 5), dtype="float32"))
    assert spatially_variable_genes(adata) is None


def test_spatial_qc_maps_three_panels():
    maps = spatial_qc_maps(_adata_with_spatial_gene())
    assert maps is not None
    assert len(maps["panels"]) == 3
    titles = [p["title"] for p in maps["panels"]]
    assert any("counts" in t.lower() for t in titles)
    assert any("mito" in t.lower() for t in titles)
    assert maps["has_mito"] is True


def test_histology_background_none_without_image():
    from ispot.spatial_features import histology_background
    adata = _adata_with_spatial_gene()
    assert histology_background(adata) is None            # no uns['spatial'] image


def test_histology_background_encodes_image():
    pytest.importorskip("PIL")
    from ispot.spatial_features import histology_background
    adata = _adata_with_spatial_gene()
    img = (np.random.default_rng(0).random((60, 80, 3)) * 255).astype("uint8")
    adata.uns["spatial"] = {"lib": {"images": {"hires": img},
                                    "scalefactors": {"tissue_hires_scalef": 0.2}}}
    bg = histology_background(adata, max_dim=40)
    assert bg is not None
    assert bg["data_url"].startswith("data:image/png;base64,")
    assert bg["w"] <= 40 and bg["h"] <= 40                # downscaled to max_dim
    assert bg["scalef"] > 0


def test_svg_subsamples_large_data():
    # More spots than the cap -> subsampled flag set, still returns a ranking.
    adata = _adata_with_spatial_gene(n_side=40)   # 1600 spots
    svg = spatially_variable_genes(adata, max_spots=500)
    assert svg["subsampled"] is True
    assert svg["n_spots_used"] <= 500
