"""
Quality control for spatial-omics data — a FastQC-style report.

FastQC (https://github.com/s-andrews/fastqc) runs a battery of modules over
high-throughput *sequencing reads* and flags each PASS / WARN / FAIL with a
plot and a short interpretation. iSpot does the analogous thing for a *spatial
transcriptomics* dataset (spots x genes + spatial coordinates), so an analyst
can scan data quality before trusting downstream clustering.

FastQC module  ->  iSpot spatial-omics analog
  Basic Statistics              -> Basic statistics (spots, genes, depth, sparsity)
  Per-sequence quality scores   -> Per-spot library size (counts) distribution
  Sequence length distribution  -> Genes-per-spot (complexity) distribution
  Per-base N content            -> Low-count / empty spots
  Adapter content               -> Mitochondrial & ribosomal content
  Sequence duplication levels   -> Gene detection rate
  Overrepresented sequences     -> Overrepresented (top) genes
  Per-base sequence quality     -> Spatial counts heatmap (capture gradients)
  (new, spatial-specific)       -> Spatial autocorrelation of counts

Each module returns {id, name, status, value, thresholds, message, plot}, where
``plot`` carries the data needed to render the module's figure. Pure
numpy/scipy/anndata (no scanpy), computed on the raw uploaded counts.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from ispot import __version__ as ISPOT_VERSION
except Exception:  # pragma: no cover
    ISPOT_VERSION = "unknown"

# Per-module thresholds: (warn, fail) with direction.
QC_THRESHOLDS = {
    "sequencing_depth": {"warn": 500, "fail": 100, "higher_is_worse": False},
    "genes_per_spot": {"warn": 250, "fail": 100, "higher_is_worse": False},
    "low_count_spots_frac": {"warn": 0.10, "fail": 0.30, "higher_is_worse": True},
    "high_mito_spots_frac": {"warn": 0.10, "fail": 0.30, "higher_is_worse": True},
    "ribo_fraction": {"warn": 0.50, "fail": 0.70, "higher_is_worse": True},
    "gene_undetected_frac": {"warn": 0.50, "fail": 0.80, "higher_is_worse": True},
    "top_gene_frac": {"warn": 0.20, "fail": 0.50, "higher_is_worse": True},
    "spot_count": {"warn": 500, "fail": 100, "higher_is_worse": False},
    "spatial_autocorr": {"warn": 0.20, "fail": 0.05, "higher_is_worse": False},
}

MIN_COUNTS_PER_SPOT = 100     # a spot below this is "low count"
HIGH_MITO_FRACTION = 0.20     # a spot above this mito fraction is "high mito"


def classify(value, warn, fail, higher_is_worse=True) -> str:
    """Return 'pass' | 'warn' | 'fail' for a value against thresholds."""
    if value is None or value != value:  # None or NaN
        return "warn"
    if higher_is_worse:
        return "fail" if value >= fail else ("warn" if value >= warn else "pass")
    return "fail" if value <= fail else ("warn" if value <= warn else "pass")


def _status(module_id, value):
    t = QC_THRESHOLDS[module_id]
    return classify(value, t["warn"], t["fail"], t["higher_is_worse"])


def _d1(x) -> np.ndarray:
    return np.asarray(x).ravel()


def _histogram(values, bins=40, log=False):
    v = _d1(values).astype(float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"bin_edges": [], "counts": [], "log": log}
    if log:
        v = np.log10(v + 1.0)
    counts, edges = np.histogram(v, bins=bins)
    return {"bin_edges": [float(e) for e in edges],
            "counts": [int(c) for c in counts], "log": bool(log)}


def _spatial_grid(values, coords, g=40):
    coords = np.asarray(coords, float)
    x, y = coords[:, 0], coords[:, 1]
    xr = (x.max() - x.min()) or 1.0
    yr = (y.max() - y.min()) or 1.0
    gx = np.clip(((x - x.min()) / xr * g).astype(int), 0, g - 1)
    gy = np.clip(((y - y.min()) / yr * g).astype(int), 0, g - 1)
    sums = np.zeros((g, g))
    cnts = np.zeros((g, g))
    v = _d1(values).astype(float)
    for xi, yi, val in zip(gx, gy, v):
        sums[yi, xi] += val
        cnts[yi, xi] += 1
    grid = [[None if cnts[i, j] == 0 else float(sums[i, j] / cnts[i, j])
             for j in range(g)] for i in range(g)]
    return {"grid": grid, "g": g,
            "extent": [float(x.min()), float(x.max()), float(y.min()), float(y.max())]}


def _spatial_autocorr(values, coords, k=6):
    """Moran's I of a per-spot value using a row-normalized kNN graph."""
    from scipy.spatial import cKDTree
    v = _d1(values).astype(float)
    coords = np.asarray(coords, float)
    n = len(v)
    if n < 10:
        return None
    k = min(k, n - 1)
    _, idx = cKDTree(coords).query(coords, k=k + 1)
    neigh = idx[:, 1:]
    dev = v - v.mean()
    denom = float(np.sum(dev ** 2))
    if denom == 0:
        return 0.0
    nb_mean = dev[neigh].mean(axis=1)  # row-normalized weights => mean over neighbors
    return float(np.sum(dev * nb_mean) / denom)


def compute_qc(adata, platform: Optional[str] = None, sample_id: Optional[str] = None) -> dict:
    """Compute the FastQC-style QC report from an AnnData of raw counts."""
    X = adata.X
    n_spots, n_genes = int(adata.shape[0]), int(adata.shape[1])

    counts_per_spot = _d1(X.sum(axis=1)).astype(float)
    genes_per_spot = _d1((X > 0).sum(axis=1)).astype(float)
    gene_totals = _d1(X.sum(axis=0)).astype(float)
    total_counts = float(counts_per_spot.sum())

    var_upper = [str(g).upper() for g in adata.var_names]
    mito_mask = np.array([g.startswith("MT-") or g.startswith("MT.") for g in var_upper])
    ribo_mask = np.array([g.startswith("RPS") or g.startswith("RPL") for g in var_upper])

    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(counts_per_spot > 0, counts_per_spot, np.nan)
        mito_counts = _d1(X[:, mito_mask].sum(axis=1)).astype(float) if mito_mask.any() else np.zeros(n_spots)
        ribo_counts = _d1(X[:, ribo_mask].sum(axis=1)).astype(float) if ribo_mask.any() else np.zeros(n_spots)
        mito_frac = mito_counts / safe
        ribo_frac = ribo_counts / safe

    median_counts = float(np.median(counts_per_spot)) if n_spots else 0.0
    median_genes = float(np.median(genes_per_spot)) if n_spots else 0.0
    low_count_frac = float(np.mean(counts_per_spot < MIN_COUNTS_PER_SPOT)) if n_spots else 1.0
    high_mito_frac = float(np.nanmean(mito_frac > HIGH_MITO_FRACTION)) if n_spots else 0.0
    median_mito = float(np.nanmedian(mito_frac)) if n_spots else 0.0
    median_ribo = float(np.nanmedian(ribo_frac)) if n_spots else 0.0
    undetected_frac = float(np.mean(gene_totals <= 0)) if n_genes else 1.0
    top_gene_frac = float(gene_totals.max() / total_counts) if total_counts > 0 else 0.0
    sparsity = 1.0 - (float((X > 0).sum()) / (n_spots * n_genes)) if n_spots and n_genes else 1.0

    # Top overrepresented genes
    order = np.argsort(gene_totals)[::-1][:20]
    top_genes = [{"gene": str(adata.var_names[i]),
                  "total": float(gene_totals[i]),
                  "pct": (float(gene_totals[i] / total_counts) if total_counts > 0 else 0.0)}
                 for i in order]

    # Spatial modules (need coordinates)
    coords = np.asarray(adata.obsm["spatial"]) if "spatial" in adata.obsm else None
    spatial_grid = _spatial_grid(counts_per_spot, coords) if coords is not None else None
    autocorr = _spatial_autocorr(counts_per_spot, coords) if coords is not None else None

    in_tissue_frac = None
    if "in_tissue" in adata.obs.columns:
        try:
            in_tissue_frac = float(np.mean(np.asarray(adata.obs["in_tissue"]).astype(float) == 1))
        except Exception:
            in_tissue_frac = None

    basic = {
        "sample_id": sample_id or adata.obs.get("sample_id", ["-"])[0] if n_spots else "-",
        "platform": platform or adata.uns.get("platform"),
        "n_spots": n_spots,
        "n_genes": n_genes,
        "total_counts": total_counts,
        "median_counts_per_spot": round(median_counts, 1),
        "median_genes_per_spot": round(median_genes, 1),
        "sparsity": round(sparsity, 4),
        "median_mito_fraction": round(median_mito, 4),
        "median_ribo_fraction": round(median_ribo, 4),
        "n_mito_genes": int(mito_mask.sum()),
        "in_tissue_fraction": in_tissue_frac,
    }

    modules = []

    def add(mid, name, status, value, message, plot=None):
        entry = {"id": mid, "name": name, "status": status,
                 "value": None if value is None else float(value), "message": message}
        if mid in QC_THRESHOLDS:
            entry["thresholds"] = {"warn": QC_THRESHOLDS[mid]["warn"],
                                   "fail": QC_THRESHOLDS[mid]["fail"]}
        if plot is not None:
            entry["plot"] = plot
        modules.append(entry)

    add("basic_statistics", "Basic statistics", "pass", None,
        f"{n_spots} spots x {n_genes} genes.", {"kind": "table", "rows": basic})

    add("sequencing_depth", "Per-spot library size (counts)",
        _status("sequencing_depth", median_counts), median_counts,
        f"Median {median_counts:.0f} counts per spot.",
        {"kind": "hist", "xlabel": "counts per spot", **_histogram(counts_per_spot)})

    add("genes_per_spot", "Genes detected per spot",
        _status("genes_per_spot", median_genes), median_genes,
        f"Median {median_genes:.0f} genes per spot.",
        {"kind": "hist", "xlabel": "genes per spot", **_histogram(genes_per_spot)})

    add("low_count_spots_frac", "Low-count / empty spots",
        _status("low_count_spots_frac", low_count_frac), low_count_frac,
        f"{low_count_frac*100:.1f}% of spots have < {MIN_COUNTS_PER_SPOT} counts.")

    add("high_mito_spots_frac", "Mitochondrial content",
        _status("high_mito_spots_frac", high_mito_frac), high_mito_frac,
        f"{high_mito_frac*100:.1f}% of spots exceed {HIGH_MITO_FRACTION*100:.0f}% mito "
        f"(median {median_mito*100:.1f}%).",
        {"kind": "hist", "xlabel": "mito fraction", **_histogram(mito_frac)})

    add("ribo_fraction", "Ribosomal content",
        _status("ribo_fraction", median_ribo), median_ribo,
        f"Median ribosomal fraction {median_ribo*100:.1f}%.",
        {"kind": "hist", "xlabel": "ribo fraction", **_histogram(ribo_frac)})

    add("gene_undetected_frac", "Gene detection rate",
        _status("gene_undetected_frac", undetected_frac), undetected_frac,
        f"{undetected_frac*100:.1f}% of genes are never detected.",
        {"kind": "hist", "xlabel": "log10(gene total counts + 1)",
         **_histogram(gene_totals, log=True)})

    add("top_gene_frac", "Overrepresented genes",
        _status("top_gene_frac", top_gene_frac), top_gene_frac,
        f"Top gene is {top_gene_frac*100:.1f}% of all counts.",
        {"kind": "table", "columns": ["gene", "total", "pct"], "top_genes": top_genes})

    add("spot_count", "Spot count (statistical power)",
        _status("spot_count", float(n_spots)), float(n_spots), f"{n_spots} spots.")

    if spatial_grid is not None:
        add("spatial_counts_map", "Spatial counts distribution", "pass", None,
            "Total counts per spatial bin — look for capture gradients or edge effects.",
            {"kind": "heatmap", **spatial_grid})

    if autocorr is not None:
        add("spatial_autocorr", "Spatial autocorrelation of counts",
            _status("spatial_autocorr", autocorr), autocorr,
            f"Moran's I of per-spot counts = {autocorr:.3f} "
            f"(higher = more spatial structure; near 0 suggests noise / non-tissue).")

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for m in modules:
        counts[m["status"]] += 1
    overall = "fail" if counts["fail"] else ("warn" if counts["warn"] else "pass")

    return {
        "version": ISPOT_VERSION,
        "basic": basic,
        "modules": modules,
        "summary": {**counts, "overall": overall, "n_modules": len(modules)},
    }
