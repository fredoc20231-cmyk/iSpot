"""Integration tests for image-based tissue detection and the Space Ranger loader.

Gated behind importorskip so the fast unit job skips them; the integration job
installs the stack and runs them.
"""
import json
import os

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
ad = pytest.importorskip("anndata")


def test_required_optional_imports_available():
    # Task 0: these must import once requirements.txt is installed.
    pytest.importorskip("skmisc")
    from skmisc.loess import loess  # noqa: F401
    pytest.importorskip("PIL")
    from PIL import Image  # noqa: F401


def test_otsu_recovers_irregular_shape():
    # Task 2: a non-oval "plus" shape, ~0.36 area so the heuristic prefers it.
    from ispot.tissue_segmentation import detect_tissue_mask

    h = w = 100
    truth = np.zeros((h, w), dtype=bool)
    truth[40:60, :] = True   # horizontal bar
    truth[:, 40:60] = True   # vertical bar
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    img[truth] = 30

    got = detect_tissue_mask(img)
    inter = np.logical_and(got, truth).sum()
    union = np.logical_or(got, truth).sum()
    assert inter / union > 0.95


def _make_half_tissue_adata(n_side=20, hires=200, fullres=1000):
    scalef = hires / fullres
    xs = np.linspace(50, fullres - 50, n_side)
    ys = np.linspace(50, fullres - 50, n_side)
    coords = np.array([(x, y) for y in ys for x in xs], dtype=float)  # col=x, row=y
    n = coords.shape[0]
    img = np.full((hires, hires, 3), 240, dtype=np.uint8)
    img[:, : hires // 2] = 30  # left half is tissue
    adata = ad.AnnData(np.ones((n, 5), dtype="float32"))
    adata.obsm["spatial"] = coords
    adata.uns["spatial"] = {
        "s": {"images": {"hires": img}, "scalefactors": {"tissue_hires_scalef": scalef}}
    }
    return adata, coords


def test_image_detection_does_not_filter_by_default():
    # By default image-based detection must NOT drop spots (avoids removing
    # real tissue / keeping border spots); the histology image is drawn instead.
    from ispot.multiplatform_loaders import _apply_image_based_tissue_detection

    adata, coords = _make_half_tissue_adata()
    out = _apply_image_based_tissue_detection(adata)
    assert out.n_obs == coords.shape[0]  # nothing dropped
    assert "n_spots_excluded_by_image_tissue_detection" not in out.uns


def test_image_detection_filters_when_opted_in(monkeypatch):
    # With ISPOT_IMAGE_TISSUE_FILTER=1 it filters to the detected tissue half.
    monkeypatch.setenv("ISPOT_IMAGE_TISSUE_FILTER", "1")
    from ispot.multiplatform_loaders import _apply_image_based_tissue_detection

    adata, coords = _make_half_tissue_adata()
    out = _apply_image_based_tissue_detection(adata)
    expected = coords[:, 0] < 500  # x*scalef < 100  <=>  x < 500
    assert out.n_obs == int(expected.sum())
    assert out.uns.get("n_spots_excluded_by_image_tissue_detection") == int((~expected).sum())
    assert bool(np.all(np.asarray(out.obsm["spatial"])[:, 0] < 500))


def test_in_tissue_column_filters_background_spots(tmp_path):
    # A dataset carrying real off-tissue spots (in_tissue=0) and NO histology
    # image must still have those background spots filtered at load, so the
    # viewer shows only the tissue footprint.
    from ispot.multiplatform_loaders import VisiumLoader

    n = 20
    adata = ad.AnnData(np.ones((n, 5), dtype="float32"))
    adata.obsm["spatial"] = np.random.default_rng(0).random((n, 2)) * 100
    adata.obs["in_tissue"] = np.array([1] * 12 + [0] * 8)
    p = str(tmp_path / "data.h5ad")
    adata.write_h5ad(p)

    out = VisiumLoader().load(p)
    assert out.n_obs == 12
    assert out.uns.get("n_spots_excluded_off_tissue") == 8


def _write_bundle(root, all_tissue_left=True, n_side=8, n_genes=20, hires=200, fullres=1000):
    import h5py
    import scipy.sparse as sp
    from PIL import Image

    os.makedirs(root, exist_ok=True)
    spatial = os.path.join(root, "spatial")
    os.makedirs(spatial, exist_ok=True)

    n = n_side * n_side
    # genes x cells CSC matrix
    rng = np.random.default_rng(0)
    dense = rng.poisson(1.0, size=(n_genes, n)).astype("float32")
    csc = sp.csc_matrix(dense)
    barcodes = [f"BC{i}".encode() for i in range(n)]
    names = [f"g{j}".encode() for j in range(n_genes)]
    h5_path = os.path.join(root, "filtered_feature_bc_matrix.h5")
    with h5py.File(h5_path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=csc.data)
        g.create_dataset("indices", data=csc.indices)
        g.create_dataset("indptr", data=csc.indptr)
        g.create_dataset("shape", data=np.array([n_genes, n]))
        g.create_dataset("barcodes", data=np.array(barcodes))
        fg = g.create_group("features")
        fg.create_dataset("name", data=np.array(names))

    # positions in the left half (fullres) so all spots stay on tissue
    xs = np.linspace(50, fullres // 2 - 50, n_side)
    ys = np.linspace(50, fullres - 50, n_side)
    rows = []
    k = 0
    for r, yy in enumerate(ys):
        for c, xx in enumerate(xs):
            rows.append([f"BC{k}", 1, r, c, int(yy), int(xx)])  # pxl_row, pxl_col
            k += 1
    # legacy headerless tissue_positions_list.csv
    with open(os.path.join(spatial, "tissue_positions_list.csv"), "w") as fh:
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")

    with open(os.path.join(spatial, "scalefactors_json.json"), "w") as fh:
        json.dump({"tissue_hires_scalef": hires / fullres, "spot_diameter_fullres": 30}, fh)

    # ONLY the hires image (the bug scanpy.read_visium can't handle)
    img = np.full((hires, hires, 3), 240, dtype=np.uint8)
    img[:, : hires // 2] = 30  # tissue on the left
    Image.fromarray(img).save(os.path.join(spatial, "tissue_hires_image.png"))
    return h5_path


def test_space_ranger_bundle_hires_only(tmp_path):
    # Task 1: a bundle with only the hires image + legacy headerless positions
    # loads without error and exposes the histology image.
    pytest.importorskip("h5py")
    pytest.importorskip("PIL")
    from ispot.multiplatform_loaders import VisiumLoader

    h5_path = _write_bundle(str(tmp_path / "sample"))
    adata = VisiumLoader().load(h5_path)

    assert adata.n_obs > 0
    assert "spatial" in adata.obsm
    spatial_uns = adata.uns.get("spatial", {})
    assert spatial_uns and any("hires" in v.get("images", {}) for v in spatial_uns.values())
    # Default behavior no longer drops spots via image heuristic.
    assert "n_spots_excluded_by_image_tissue_detection" not in adata.uns


def test_viewer_data_includes_mask(tmp_path):
    # Task 4: the tissue mask is passed through to the viewer JSON.
    pytest.importorskip("matplotlib")
    from ispot.deliverables import generate_viewer_data

    n = 12
    adata = ad.AnnData(np.ones((n, 4), dtype="float32"))
    adata.obsm["spatial"] = np.random.default_rng(0).random((n, 2))
    adata.uns["tissue_mask_for_viewer"] = {
        "rows": ["11", "00"], "height": 2, "width": 2,
        "original_height": 200, "original_width": 200,
    }
    adata.uns["tissue_mask_scale_factor"] = 0.2

    path = generate_viewer_data(
        adata, {"good": np.array(["0"] * n)}, has_ground_truth=False, output_dir=str(tmp_path)
    )
    data = json.load(open(path))
    assert "tissue_mask" in data and data["tissue_mask_scale_factor"] == 0.2
    assert "good" in data["methods"]


def test_viewer_data_embeds_histology(tmp_path):
    # Viewer JSON carries a base64 histology image + scalefactor so the
    # frontend can align spots to the tissue (fixes dots-outside-tissue).
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    from ispot.deliverables import generate_viewer_data

    n = 12
    adata = ad.AnnData(np.ones((n, 4), dtype="float32"))
    adata.obsm["spatial"] = np.random.default_rng(0).random((n, 2)) * 100
    img = np.full((60, 80, 3), 200, dtype=np.uint8)
    adata.uns["spatial"] = {
        "s": {"images": {"hires": img},
              "scalefactors": {"tissue_hires_scalef": 0.5, "spot_diameter_fullres": 20}}
    }
    path = generate_viewer_data(
        adata, {"m": np.array(["0"] * n)}, has_ground_truth=False, output_dir=str(tmp_path)
    )
    data = json.load(open(path))
    h = data.get("histology")
    assert h and h["data_url"].startswith("data:image/png;base64,")
    assert h["width"] == 80 and h["height"] == 60
    assert h["scalef"] == 0.5 and h["spot_diameter_fullres"] == 20


def test_viewer_data_raises_on_spot_count_mismatch(tmp_path):
    # Task 4 / checklist #5: a length mismatch is a loud, traceable error.
    pytest.importorskip("matplotlib")
    from ispot.deliverables import generate_viewer_data

    n = 12
    adata = ad.AnnData(np.ones((n, 4), dtype="float32"))
    adata.obsm["spatial"] = np.random.default_rng(0).random((n, 2))
    with pytest.raises(ValueError):
        generate_viewer_data(
            adata, {"bad": np.array(["0"] * (n - 1))},
            has_ground_truth=False, output_dir=str(tmp_path),
        )
