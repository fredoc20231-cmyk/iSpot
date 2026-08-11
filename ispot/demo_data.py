"""
Bundled synthetic demo datasets for SpatialQC.

spatialGE (and Space Ranger) ship no data inside the package — their examples
download a companion dataset over the network. For a self-contained, offline,
CI-safe "try it now" experience, iSpot generates small synthetic Visium-like
samples on the fly: a healthy two-lobe section with clear spatial domains and a
synthetic H&E image (passes QC), and a degraded, shallow, near-structureless
sample (fails QC). They exercise the full SpatialQC path — tissue-space maps,
Moran's I / Geary's C spatially variable genes, and the histology overlay —
without any upload.

Pure numpy + anndata (no scanpy), deterministic, a few thousand spots — builds
in well under a second.
"""
from __future__ import annotations

import os

_DEMO_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "demo")

# Real 10x Visium Space Ranger bundles shipped in the repo. Loaded through the
# normal loader (needs h5py), so these exercise the true import path end to end.
REAL_DEMOS = {
    "visium_breast_092a": {
        "path": os.path.join(_DEMO_ROOT, "GSM6433585_092A"),
        "description": (
            "Real 10x Visium breast tumor — GSM6433585, sample 092A "
            "(Bassiouni et al.). Space Ranger bundle, ~2k in-tissue spots."),
    },
}

# Synthetic fallbacks — no external deps, always available (even in the light
# unit environment without h5py).
SYNTHETIC_DEMOS = {
    "visium_healthy_demo": (
        "Synthetic Visium-like two-lobe section: good depth, clear spatial domains, "
        "and a synthetic histology image. Should pass SpatialQC."),
    "visium_lowquality_demo": (
        "Synthetic degraded Visium sample: shallow sequencing depth, very sparse, "
        "and little spatial structure. Should raise WARN/FAIL flags."),
}

# Combined view for the API/UI. Real datasets listed first.
DEMOS = {**{k: v["description"] for k, v in REAL_DEMOS.items()}, **SYNTHETIC_DEMOS}


def list_demos() -> list[dict]:
    """Available demo datasets: [{name, description, real}].

    Real datasets whose files are actually present are listed first; a real
    dataset whose bundle is missing from the checkout is omitted.
    """
    demos = []
    for name, meta in REAL_DEMOS.items():
        if os.path.isdir(meta["path"]):
            demos.append({"name": name, "description": meta["description"], "real": True})
    for name, desc in SYNTHETIC_DEMOS.items():
        demos.append({"name": name, "description": desc, "real": False})
    return demos


def make_demo(name: str):
    """Build a demo AnnData by name.

    Real datasets load via the Space Ranger loader (requires h5py). Synthetic
    datasets are generated in-memory. Falls back to the healthy synthetic demo
    for an unknown name.
    """
    if name in REAL_DEMOS:
        from ispot.multiplatform_loaders import load_data
        return load_data(REAL_DEMOS[name]["path"], platform="Visium", sample_id=name)
    if name == "visium_lowquality_demo":
        return _build(quality="low")
    return _build(quality="high")


def _hex_grid(n_row=46, n_col=58, spacing=100.0):
    import numpy as np
    xs, ys = [], []
    for r in range(n_row):
        for c in range(n_col):
            xs.append(c * spacing + (spacing / 2.0 if (r % 2) else 0.0))
            ys.append(r * spacing * 0.866)  # hex vertical pitch
    return np.column_stack([np.array(xs, float), np.array(ys, float)])


def _two_lobe_mask(xy):
    import numpy as np
    cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
    w = xy[:, 0].max() - xy[:, 0].min()
    h = xy[:, 1].max() - xy[:, 1].min()
    lc = (cx - w * 0.22, cy)
    rc = (cx + w * 0.22, cy)
    rx, ry = w * 0.26, h * 0.44

    def inside(p, c):
        return ((p[0] - c[0]) / rx) ** 2 + ((p[1] - c[1]) / ry) ** 2 <= 1.0

    keep = np.array([inside(p, lc) or inside(p, rc) for p in xy])
    return keep, lc, rc, rx, ry


def _synthetic_histology(coords, scalef, spot_r_fullres):
    """A soft-pink H&E-like tissue footprint stamped where the spots are."""
    import numpy as np
    px = coords[:, 0] * scalef
    py = coords[:, 1] * scalef
    pad = int(spot_r_fullres * scalef * 2)
    W = int(px.max() + pad) + 1
    H = int(py.max() + pad) + 1
    img = np.full((H, W, 3), 244, dtype=np.uint8)   # light background
    r = max(2, int(spot_r_fullres * scalef * 0.8))
    pink = np.array([214, 170, 196], dtype=np.uint8)  # H&E-ish
    for cx, cy in zip(px, py):
        cxi, cyi = int(cx), int(cy)
        x0, x1 = max(0, cxi - r), min(W, cxi + r + 1)
        y0, y1 = max(0, cyi - r), min(H, cyi + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disc = (xx - cxi) ** 2 + (yy - cyi) ** 2 <= r * r
        block = img[y0:y1, x0:x1]
        block[disc] = pink
        img[y0:y1, x0:x1] = block
    return img


def _build(quality="high", seed=0):
    import numpy as np
    import anndata as ad

    rng = np.random.default_rng(seed)
    xy = _hex_grid()
    keep, lc, rc, rx, ry = _two_lobe_mask(xy)
    coords = xy[keep]
    n = coords.shape[0]

    n_genes = 200
    gene_names = ["MT-Co1", "MT-Nd1"] + [f"Gene{i}" for i in range(n_genes - 2)]

    # Per-spot domain: which lobe, and core vs ring within the nearest lobe.
    def nearest_lobe(p):
        dl = ((p[0] - lc[0]) / rx) ** 2 + ((p[1] - lc[1]) / ry) ** 2
        dr = ((p[0] - rc[0]) / rx) ** 2 + ((p[1] - rc[1]) / ry) ** 2
        return (0, dl) if dl <= dr else (1, dr)

    lobe = np.array([nearest_lobe(p)[0] for p in coords])
    radial = np.array([nearest_lobe(p)[1] for p in coords])   # 0 core .. 1 edge
    core = radial < 0.35

    if quality == "high":
        base_lam = 1.8
        X = rng.poisson(base_lam, size=(n, n_genes)).astype("float32")
        # Domain marker blocks (strong, spatially coherent).
        def boost(mask, gslice, lam):
            idx = np.where(mask)[0]
            X[np.ix_(idx, np.arange(*gslice.indices(n_genes)))] += \
                rng.poisson(lam, size=(len(idx), len(range(*gslice.indices(n_genes))))).astype("float32")
        boost(core, slice(2, 25), 9.0)                    # core markers
        boost(~core, slice(25, 48), 8.0)                  # ring markers
        boost(lobe == 0, slice(48, 70), 6.0)              # left-lobe markers
        boost(lobe == 1, slice(70, 92), 6.0)              # right-lobe markers
        # A smooth gradient gene along x.
        gx = (coords[:, 0] - coords[:, 0].min()) / (np.ptp(coords[:, 0]) or 1.0)
        X[:, 92] += (gx * 30).astype("float32")
        # Mitochondrial genes: modest, slightly higher at the edge (realistic).
        X[:, 0] += (radial * 6).astype("float32")
        X[:, 1] += (radial * 4).astype("float32")
        has_hist = True
    else:
        # Degraded: shallow, sparse, essentially no spatial domains.
        base_lam = 0.12
        X = rng.poisson(base_lam, size=(n, n_genes)).astype("float32")
        # a little uniform mito, no coherent structure
        X[:, 0] += rng.poisson(0.4, size=n).astype("float32")
        has_hist = False

    adata = ad.AnnData(X)
    adata.var_names = gene_names
    adata.obs_names = [f"spot{i}" for i in range(n)]
    adata.obsm["spatial"] = coords.astype(float)
    adata.obs["in_tissue"] = 1
    adata.uns["platform"] = "Visium"

    if has_hist:
        scalef = 0.2
        spot_r = 45.0  # full-res spot radius (px)
        img = _synthetic_histology(coords, scalef, spot_r)
        adata.uns["spatial"] = {
            "demo": {"images": {"hires": img},
                     "scalefactors": {"tissue_hires_scalef": scalef,
                                      "spot_diameter_fullres": spot_r * 2}}
        }
    return adata
