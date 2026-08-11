"""
Spatial feature analysis for SpatialQC — the parts that are genuinely *spatial*.

Generic QC (depth, sparsity, mito) is shared with single-cell. What makes a
spatial dataset worth clustering is spatial *structure*: metrics organised in
tissue space, and genes whose expression is spatially patterned. This module
provides the two spatial-specific, fast analyses that mirror a VoyagerPy /
Space Ranger EDA:

  1. Per-spot QC metrics in tissue space (total counts, genes detected,
     mitochondrial %) — the same trio VoyagerPy plots with
     ``plot_spatial_feature(["sum","detected","subsets_mito_percent"])``.
  2. Spatially variable genes (SVGs) ranked by Moran's I — the statistic Space
     Ranger reports in ``spatial_enrichment.csv``. A healthy tissue has many
     strongly spatial genes; if even the top genes are near zero there is no
     recoverable spatial structure.

Everything is a cheap linear-algebra pass (no preprocessing/PCA/clustering):
a single kNN graph, a batched Moran's I over the top genes via one sparse
matrix product, all bounded in memory by subsampling spots and capping genes so
it stays fast on Visium and tractable on large imaging datasets.
"""
from __future__ import annotations

from typing import Optional


def _knn_weights(coords, k=6):
    """Row-normalised kNN spatial weight matrix (scipy sparse csr, n x n)."""
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix

    n = coords.shape[0]
    k = int(min(k, max(1, n - 1)))
    _, idx = cKDTree(coords).query(coords, k=k + 1)
    neigh = idx[:, 1:]  # drop self
    rows = np.repeat(np.arange(n), k)
    cols = neigh.ravel()
    data = np.full(n * k, 1.0 / k, dtype=float)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def morans_i_batch(M, W):
    """Moran's I for every column of ``M`` (n x g) under weights ``W`` (n x n).

    With row-normalised weights the normalising constant n/W0 = 1, so
    I_g = sum_i dev_ig (W dev)_ig / sum_i dev_ig^2. One sparse matmul does all
    genes at once.
    """
    import numpy as np
    M = np.asarray(M, dtype=float)
    dev = M - M.mean(axis=0, keepdims=True)
    Wdev = W @ dev
    num = np.einsum("ng,ng->g", dev, Wdev)
    denom = np.einsum("ng,ng->g", dev, dev)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0, num / denom, 0.0)
    return out


def histology_background(adata, max_dim: int = 900) -> Optional[dict]:
    """Return the histology image as a base64 PNG plus the scale factor that
    maps full-res spot pixel coordinates onto it, so spatial maps can overlay
    spots on the tissue image (like a VoyagerPy/Space Ranger plot).

    Returns ``None`` when no image is available or PIL is missing. Never raises.
    """
    try:
        import base64
        import io
        import numpy as np
        from PIL import Image as PILImage

        spatial = adata.uns.get("spatial") if hasattr(adata, "uns") else None
        if not spatial:
            return None
        lib = next(iter(spatial.values()))
        images = lib.get("images", {}) or {}
        sf = lib.get("scalefactors", {}) or {}
        # Prefer hires; fall back to lowres. Use the matching scale factor.
        if "hires" in images:
            img = np.asarray(images["hires"])
            scalef = float(sf.get("tissue_hires_scalef", 1.0))
        elif "lowres" in images:
            img = np.asarray(images["lowres"])
            scalef = float(sf.get("tissue_lowres_scalef", 1.0))
        else:
            return None
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.dtype != np.uint8:
            img = (255 * (img / (img.max() or 1.0))).astype(np.uint8)

        pil = PILImage.fromarray(img[:, :, :3])
        h, w = img.shape[0], img.shape[1]
        # Downscale further to keep the embedded PNG small.
        longest = max(h, w)
        extra = (max_dim / longest) if longest > max_dim else 1.0
        if extra < 1.0:
            pil = pil.resize((max(1, int(w * extra)), max(1, int(h * extra))))
            w, h = pil.size
            scalef *= extra
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        return {"data_url": data_url, "w": int(w), "h": int(h), "scalef": float(scalef)}
    except Exception:
        return None


def gearys_c_batch(M, W):
    """Geary's C for every column of ``M`` (n x g) under weights ``W`` (n x n).

    Complements Moran's I (spatialGE's ``SThet`` reports both). With
    row-normalised weights the total weight W0 = n, so

        C_g = ((n-1) / (2n)) * sum_i [x_i^2 - 2 x_i (Wx)_i + (W x^2)_i]
                              / sum_i (x_i - xbar)^2

    C ≈ 1 means no spatial autocorrelation; C < 1 means positive spatial
    structure (nearby spots similar); C > 1 means negative autocorrelation.
    """
    import numpy as np
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    Wx = W @ M
    Wx2 = W @ (M * M)
    term = (M * M) - 2.0 * M * Wx + Wx2
    num = term.sum(axis=0)
    dev = M - M.mean(axis=0, keepdims=True)
    denom = np.einsum("ng,ng->g", dev, dev)
    factor = (n - 1) / (2.0 * n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0, factor * num / denom, 1.0)
    return out


def _subsample(n, max_spots):
    import numpy as np
    if n <= max_spots:
        return np.arange(n)
    step = int(np.ceil(n / max_spots))
    return np.arange(0, n, step)


def _densify_cols(X, cols):
    """Return X[:, cols] as a dense float ndarray (handles sparse X)."""
    import numpy as np
    sub = X[:, cols]
    sub = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
    return sub.astype(float)


def spatially_variable_genes(
    adata,
    max_genes: int = 2000,
    k: int = 6,
    max_spots: int = 12000,
    top: int = 20,
    n_maps: int = 3,
) -> Optional[dict]:
    """Rank genes by Moran's I of expression (spatial variability).

    Restricts to the ``max_genes`` most-expressed genes and (for large data)
    subsamples to ``max_spots`` spots so the whole thing is one fast pass.

    Returns ``None`` when there are no coordinates / too few spots. Otherwise a
    dict with the ranked SVG table, the fraction of tested genes that are
    strongly spatial, and per-spot expression maps for the top ``n_maps`` genes.
    """
    import numpy as np

    if "spatial" not in getattr(adata, "obsm", {}):
        return None
    coords_full = np.asarray(adata.obsm["spatial"], dtype=float)
    n = coords_full.shape[0]
    if n < 10 or adata.shape[1] < 2:
        return None

    sel = _subsample(n, max_spots)
    coords = coords_full[sel]
    X = adata.X
    Xs = X[sel]

    gene_totals = np.asarray(Xs.sum(axis=0)).ravel().astype(float)
    n_top = int(min(max_genes, (gene_totals > 0).sum()))
    if n_top < 2:
        return None
    order = np.argsort(gene_totals)[::-1][:n_top]
    M = _densify_cols(Xs, order)

    W = _knn_weights(coords, k)
    I = morans_i_batch(M, W)
    C = gearys_c_batch(M, W)   # Geary's C — spatialGE SThet's second statistic

    names = [str(adata.var_names[i]) for i in order]
    ranking = sorted(
        ({"gene": names[j], "morans_i": float(I[j]), "gearys_c": float(C[j]),
          "total_counts": float(gene_totals[order[j]])} for j in range(len(order))),
        key=lambda r: r["morans_i"], reverse=True,
    )

    strong = float(np.mean(I >= 0.25)) if I.size else 0.0
    max_i = float(np.max(I)) if I.size else 0.0

    # Spatial expression maps for the top few SVGs (on the subsampled coords).
    maps = []
    for r in ranking[:n_maps]:
        gi = names.index(r["gene"])
        vals = M[:, gi]
        maps.append({
            "title": f"{r['gene']} (I={r['morans_i']:.2f})",
            "x": [float(v) for v in coords[:, 0]],
            "y": [float(v) for v in coords[:, 1]],
            "values": [float(v) for v in vals],
            "clabel": "expression",
        })

    return {
        "n_genes_tested": int(n_top),
        "n_spots_used": int(len(sel)),
        "subsampled": bool(len(sel) < n),
        "max_morans_i": max_i,
        "strong_svg_fraction": strong,
        "top_svgs": ranking[:top],
        "maps": maps,
    }


def spatial_qc_maps(adata, max_spots: int = 6000) -> Optional[dict]:
    """Per-spot QC metrics rendered in tissue space (VoyagerPy's QC trio).

    Returns downsampled spatial coordinates plus total counts, genes detected,
    and mitochondrial % per spot — for side-by-side spatial maps. ``None`` if
    there are no coordinates.
    """
    import numpy as np

    if "spatial" not in getattr(adata, "obsm", {}):
        return None
    coords_full = np.asarray(adata.obsm["spatial"], dtype=float)
    n = coords_full.shape[0]
    if n == 0:
        return None

    sel = _subsample(n, max_spots)
    coords = coords_full[sel]
    X = adata.X
    Xs = X[sel]

    counts = np.asarray(Xs.sum(axis=1)).ravel().astype(float)
    genes = np.asarray((Xs > 0).sum(axis=1)).ravel().astype(float)

    var_upper = [str(g).upper() for g in adata.var_names]
    mito_mask = np.array([g.startswith("MT-") or g.startswith("MT.") for g in var_upper])
    if mito_mask.any():
        mito_counts = np.asarray(Xs[:, mito_mask].sum(axis=1)).ravel().astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            mito_pct = np.where(counts > 0, 100.0 * mito_counts / counts, 0.0)
    else:
        mito_pct = np.zeros_like(counts)

    x = [float(v) for v in coords[:, 0]]
    y = [float(v) for v in coords[:, 1]]
    panels = [
        {"title": "Total counts (sum)", "x": x, "y": y,
         "values": [float(v) for v in counts], "clabel": "counts"},
        {"title": "Genes detected", "x": x, "y": y,
         "values": [float(v) for v in genes], "clabel": "genes"},
        {"title": "Mitochondrial %", "x": x, "y": y,
         "values": [float(v) for v in mito_pct], "clabel": "mito %"},
    ]
    return {"panels": panels, "has_mito": bool(mito_mask.any()),
            "n_spots_used": int(len(sel)), "subsampled": bool(len(sel) < n)}
