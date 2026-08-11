"""
Quality control for spatial transcriptomics data — a FastQC-style report.

FastQC runs a set of modules over sequencing reads and flags each PASS / WARN /
FAIL. This does the analogous thing for an ST dataset (raw counts + spatial
coordinates), producing per-module verdicts an analyst can scan before trusting
downstream clustering:

  - Basic statistics (spots, genes, depth, sparsity)
  - Sequencing depth per spot (library size)
  - Genes detected per spot
  - Low-count spots (empty/near-empty capture locations)
  - Mitochondrial fraction (dissociation / dying-cell stress)
  - Ribosomal fraction
  - Gene detection rate (fraction of genes ever observed)
  - Overrepresented genes (a single gene dominating counts — contamination)
  - Spot count (statistical power)
  - Tissue coverage (fraction on tissue, when annotated)

Pure numpy / scipy / anndata (no scanpy), so it runs on raw uploaded data
before preprocessing and is unit-testable without the heavy stack.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Thresholds per module: (warn, fail). ``higher_is_worse`` controls direction.
QC_THRESHOLDS = {
    "sequencing_depth": {"warn": 500, "fail": 100, "higher_is_worse": False},
    "genes_per_spot": {"warn": 250, "fail": 100, "higher_is_worse": False},
    "low_count_spots_frac": {"warn": 0.10, "fail": 0.30, "higher_is_worse": True},
    "high_mito_spots_frac": {"warn": 0.10, "fail": 0.30, "higher_is_worse": True},
    "gene_undetected_frac": {"warn": 0.50, "fail": 0.80, "higher_is_worse": True},
    "top_gene_frac": {"warn": 0.20, "fail": 0.50, "higher_is_worse": True},
    "spot_count": {"warn": 500, "fail": 100, "higher_is_worse": False},
}

# A spot is "low count" below this many total counts.
MIN_COUNTS_PER_SPOT = 100
# A spot is "high mito" above this mitochondrial fraction.
HIGH_MITO_FRACTION = 0.20


def classify(value: float, warn: float, fail: float, higher_is_worse: bool = True) -> str:
    """Return 'pass' | 'warn' | 'fail' for a value against thresholds."""
    if value is None or value != value:  # None or NaN
        return "warn"
    if higher_is_worse:
        if value >= fail:
            return "fail"
        if value >= warn:
            return "warn"
        return "pass"
    else:
        if value <= fail:
            return "fail"
        if value <= warn:
            return "warn"
        return "pass"


def _module(module_id, name, value, message):
    t = QC_THRESHOLDS[module_id]
    status = classify(value, t["warn"], t["fail"], t["higher_is_worse"])
    return {
        "id": module_id,
        "name": name,
        "status": status,
        "value": None if value is None else float(value),
        "thresholds": {"warn": t["warn"], "fail": t["fail"]},
        "message": message,
    }


def _dense_1d(x) -> np.ndarray:
    return np.asarray(x).ravel()


def compute_qc(adata, platform: Optional[str] = None) -> dict:
    """Compute a FastQC-style QC report from an AnnData of raw counts.

    Parameters
    ----------
    adata : AnnData with raw counts in .X and gene symbols in var_names.
    platform : optional platform label for the report.

    Returns
    -------
    dict: {basic, modules: [...], summary: {...}}
    """
    X = adata.X
    n_spots, n_genes = int(adata.shape[0]), int(adata.shape[1])

    counts_per_spot = _dense_1d(X.sum(axis=1)).astype(float)
    genes_per_spot = _dense_1d((X > 0).sum(axis=1)).astype(float)
    gene_totals = _dense_1d(X.sum(axis=0)).astype(float)
    total_counts = float(counts_per_spot.sum())

    var_upper = np.array([str(g).upper() for g in adata.var_names])
    mito_mask = np.array([g.startswith("MT-") or g.startswith("MT.") for g in var_upper])
    ribo_mask = np.array([g.startswith("RPS") or g.startswith("RPL") for g in var_upper])

    with np.errstate(divide="ignore", invalid="ignore"):
        safe_counts = np.where(counts_per_spot > 0, counts_per_spot, np.nan)
        mito_counts = _dense_1d(X[:, mito_mask].sum(axis=1)).astype(float) if mito_mask.any() else np.zeros(n_spots)
        ribo_counts = _dense_1d(X[:, ribo_mask].sum(axis=1)).astype(float) if ribo_mask.any() else np.zeros(n_spots)
        mito_frac = mito_counts / safe_counts
        ribo_frac = ribo_counts / safe_counts

    median_counts = float(np.median(counts_per_spot)) if n_spots else 0.0
    median_genes = float(np.median(genes_per_spot)) if n_spots else 0.0
    low_count_frac = float(np.mean(counts_per_spot < MIN_COUNTS_PER_SPOT)) if n_spots else 1.0
    high_mito_frac = float(np.nanmean(mito_frac > HIGH_MITO_FRACTION)) if n_spots else 0.0
    undetected_frac = float(np.mean(gene_totals <= 0)) if n_genes else 1.0
    top_gene_frac = float(gene_totals.max() / total_counts) if total_counts > 0 else 0.0
    median_mito = float(np.nanmedian(mito_frac)) if n_spots else 0.0
    median_ribo = float(np.nanmedian(ribo_frac)) if n_spots else 0.0
    sparsity = 1.0 - (float((X > 0).sum()) / (n_spots * n_genes)) if n_spots and n_genes else 1.0

    in_tissue_frac = None
    if "in_tissue" in adata.obs.columns:
        try:
            iv = np.asarray(adata.obs["in_tissue"]).astype(float)
            in_tissue_frac = float(np.mean(iv == 1))
        except Exception:
            in_tissue_frac = None

    basic = {
        "n_spots": n_spots,
        "n_genes": n_genes,
        "platform": platform or adata.uns.get("platform"),
        "total_counts": total_counts,
        "median_counts_per_spot": median_counts,
        "median_genes_per_spot": median_genes,
        "sparsity": round(sparsity, 4),
        "median_mito_fraction": round(median_mito, 4),
        "median_ribo_fraction": round(median_ribo, 4),
        "n_mito_genes": int(mito_mask.sum()),
        "in_tissue_fraction": in_tissue_frac,
    }

    modules = [
        _module("sequencing_depth", "Sequencing depth (median counts/spot)",
                median_counts, f"Median {median_counts:.0f} counts per spot."),
        _module("genes_per_spot", "Genes detected per spot (median)",
                median_genes, f"Median {median_genes:.0f} genes per spot."),
        _module("low_count_spots_frac", "Low-count spots",
                low_count_frac, f"{low_count_frac*100:.1f}% of spots have < {MIN_COUNTS_PER_SPOT} counts."),
        _module("high_mito_spots_frac", "High-mitochondrial spots",
                high_mito_frac, f"{high_mito_frac*100:.1f}% of spots exceed {HIGH_MITO_FRACTION*100:.0f}% mito "
                                f"(median mito {median_mito*100:.1f}%)."),
        _module("gene_undetected_frac", "Gene detection",
                undetected_frac, f"{undetected_frac*100:.1f}% of genes are never detected."),
        _module("top_gene_frac", "Overrepresented genes",
                top_gene_frac, f"Top gene is {top_gene_frac*100:.1f}% of all counts."),
        _module("spot_count", "Spot count (statistical power)",
                float(n_spots), f"{n_spots} spots."),
    ]

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for m in modules:
        counts[m["status"]] += 1
    overall = "fail" if counts["fail"] else ("warn" if counts["warn"] else "pass")

    return {
        "basic": basic,
        "modules": modules,
        "summary": {**counts, "overall": overall, "n_modules": len(modules)},
    }
