"""
SpatialQC — the FastQC parallel for spatial-omics data.

FastQC (https://github.com/s-andrews/fastqc) is not valued for sophistication;
it is valued because it is *fast*, *universal*, *zero-configuration*, and run as
the very first step of every sequencing workflow — a standardized traffic-light
report that answers "is this data even worth analysing" before anyone commits to
a heavy pipeline. SpatialQC is the same idea for spatial transcriptomics: a
separate, fast, always-run-first mode that runs only

    load_data()  ->  profile_data()  ->  tissue detection

and never touches preprocessing, HVG selection, PCA, or the 12-method clustering
benchmark. It answers "is this dataset worth clustering" in seconds.

Each module reports PASS / WARN / FAIL against a **platform-aware** threshold
(a targeted Xenium panel and a whole-transcriptome Visium slide have wildly
different "normal" gene counts and sparsity), mirroring FastQC's module list:

    FastQC module              ->  SpatialQC module
    Basic Statistics           ->  Basic Statistics (informational)
    (new)                      ->  Platform Detection Confidence
    Per-base N content         ->  Tissue Coverage (off-tissue / retention)
    Sequence length dist.      ->  Gene Detection Distribution
    Sequence duplication       ->  Sparsity / Dropout
    Per-base sequence quality  ->  Spatial Signal Sanity Check (Moran's I)
    (new, image-specific)      ->  Image / Segmentation Quality

The report dict is emitted in the SAME shape as :func:`ispot.qc.compute_qc`
(``{version, basic, modules, summary}``) so it renders through the existing
``deliverables.generate_qc_report`` HTML/JSON writer unchanged. A flat
:func:`summary_record` is also produced for MultiQC-style cross-sample
aggregation (see ``ispot.multi_sample_qc``).

Threshold/classification logic is pure Python (no numpy) so it is unit-testable
in isolation; the metric computations reuse ``ispot.qc`` and ``ispot.profiling``.
"""
from __future__ import annotations

from typing import Optional

try:
    from ispot import __version__ as ISPOT_VERSION
except Exception:  # pragma: no cover
    ISPOT_VERSION = "unknown"


# ---------------------------------------------------------------------------
# Platform-aware thresholds
#
# "Typical" median genes-per-spot and matrix sparsity differ by orders of
# magnitude between whole-transcriptome capture (Visium/Stereo-seq/DBiT) and
# targeted imaging panels (Xenium/CosMx/MERFISH), so a single fixed cutoff would
# false-fail every panel dataset. These are deliberately permissive lower
# bounds — SpatialQC flags "this is unusually sparse/shallow *for this
# platform*", not "this differs from Visium".
# ---------------------------------------------------------------------------
PLATFORM_PROFILES: dict[str, dict] = {
    # whole-transcriptome capture
    "Visium":      {"genes_warn": 1000, "genes_fail": 500, "sparsity_warn": 0.90, "sparsity_fail": 0.97, "targeted": False},
    "Stereo-seq":  {"genes_warn": 500,  "genes_fail": 200, "sparsity_warn": 0.95, "sparsity_fail": 0.99, "targeted": False},
    "DBiT-seq":    {"genes_warn": 500,  "genes_fail": 200, "sparsity_warn": 0.95, "sparsity_fail": 0.99, "targeted": False},
    "Slide-seqV2": {"genes_warn": 100,  "genes_fail": 50,  "sparsity_warn": 0.97, "sparsity_fail": 0.99, "targeted": False},
    # targeted imaging panels (naturally few genes, very sparse)
    "MERFISH":     {"genes_warn": 20,   "genes_fail": 10,  "sparsity_warn": 0.985, "sparsity_fail": 0.998, "targeted": True},
    "CosMx":       {"genes_warn": 50,   "genes_fail": 20,  "sparsity_warn": 0.985, "sparsity_fail": 0.998, "targeted": True},
    "Xenium":      {"genes_warn": 30,   "genes_fail": 15,  "sparsity_warn": 0.985, "sparsity_fail": 0.998, "targeted": True},
}
DEFAULT_PROFILE = {"genes_warn": 200, "genes_fail": 100, "sparsity_warn": 0.95, "sparsity_fail": 0.99, "targeted": False}

# Tissue retention (fraction of spots kept after off-tissue filtering), percent.
TISSUE_RETENTION_WARN = 90.0
TISSUE_RETENTION_FAIL = 70.0

# Moran's I of per-spot total counts. Near zero => no spatial structure at all,
# which for real tissue means corrupted data or misaligned coordinates.
SPATIAL_SIGNAL_WARN = 0.20
SPATIAL_SIGNAL_FAIL = 0.05


def platform_profile(platform: Optional[str]) -> dict:
    """Case-insensitive platform threshold lookup, falling back to a default."""
    if platform:
        for name, prof in PLATFORM_PROFILES.items():
            if name.lower() == str(platform).lower():
                return prof
    return DEFAULT_PROFILE


# ---------------------------------------------------------------------------
# Pure classification helpers (no numpy) — unit-testable in isolation
# ---------------------------------------------------------------------------
def classify_lower_bound(value, warn, fail) -> str:
    """PASS when ``value`` is comfortably high; lower is worse.

    pass if value >= warn, warn if value >= fail, else fail. None/NaN -> warn.
    """
    if value is None or value != value:
        return "warn"
    if value >= warn:
        return "pass"
    if value >= fail:
        return "warn"
    return "fail"


def classify_upper_bound(value, warn, fail) -> str:
    """PASS when ``value`` is comfortably low; higher is worse.

    pass if value <= warn, warn if value <= fail, else fail. None/NaN -> warn.
    """
    if value is None or value != value:
        return "warn"
    if value <= warn:
        return "pass"
    if value <= fail:
        return "warn"
    return "fail"


def summarize(modules: list[dict]) -> dict:
    """Roll module statuses up into an overall PASS/WARN/FAIL summary."""
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for m in modules:
        counts[m.get("status", "warn")] = counts.get(m.get("status", "warn"), 0) + 1
    overall = "fail" if counts["fail"] else ("warn" if counts["warn"] else "pass")
    return {**counts, "overall": overall, "n_modules": len(modules)}


def _num(x):
    """Coerce to float or None (never raises)."""
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------
def build_spatial_qc(
    adata,
    platform: Optional[str] = None,
    sample_id: Optional[str] = None,
    platform_confidence: Optional[str] = None,
) -> dict:
    """Build the SpatialQC report from a freshly-loaded (un-preprocessed) AnnData.

    Only cheap operations: a data profile, per-spot count/gene reductions, one
    Moran's I on total counts, and a read of the tissue-filter bookkeeping the
    loader already attached to ``adata.uns``. No preprocessing/HVG/PCA/clustering.

    Parameters
    ----------
    adata : AnnData
        Raw counts with ``.obsm['spatial']``, as returned by ``load_data``.
    platform, sample_id : str, optional
    platform_confidence : {"explicit","inferred","default"}, optional
        How the platform was determined (see ``detect_platform_with_confidence``).

    Returns
    -------
    dict with ``{version, basic, modules, summary, sample_id, platform,
    platform_confidence}`` — same shape as ``ispot.qc.compute_qc`` plus the
    SpatialQC-specific top-level fields.
    """
    import numpy as np
    from ispot import qc as _qc
    from ispot.profiling import profile_data

    platform = platform or (adata.uns.get("platform") if hasattr(adata, "uns") else None)
    prof = platform_profile(platform)

    # --- data profile (fast; reused from the meta-learning feature extractor)
    try:
        feats = profile_data(adata, platform=platform).to_dict()
    except Exception:
        feats = {}

    n_spots = int(adata.shape[0])
    n_genes = int(adata.shape[1])

    X = adata.X
    counts_per_spot = _qc._d1(X.sum(axis=1)).astype(float)
    genes_per_spot = _qc._d1((X > 0).sum(axis=1)).astype(float)
    median_genes = _num(feats.get("median_genes_per_spot"))
    if median_genes is None:
        median_genes = float(np.median(genes_per_spot)) if n_spots else 0.0
    median_counts = _num(feats.get("median_counts_per_spot"))
    if median_counts is None:
        median_counts = float(np.median(counts_per_spot)) if n_spots else 0.0
    sparsity = _num(feats.get("sparsity"))
    if sparsity is None:
        sparsity = 1.0 - (float((X > 0).sum()) / (n_spots * n_genes)) if n_spots and n_genes else 1.0

    coords = np.asarray(adata.obsm["spatial"]) if "spatial" in getattr(adata, "obsm", {}) else None
    morans_i = _qc._spatial_autocorr(counts_per_spot, coords) if coords is not None else None

    # --- tissue-filter bookkeeping (attached by the loader; see multiplatform_loaders)
    uns = adata.uns if hasattr(adata, "uns") else {}
    n_before = _num(uns.get("n_spots_before_tissue_filter")) or float(n_spots)
    n_excluded_off = int(uns.get("n_spots_excluded_off_tissue", 0) or 0)
    n_excluded_img = int(uns.get("n_spots_excluded_by_image_tissue_detection", 0) or 0)
    retention_pct = (100.0 * n_spots / n_before) if n_before else 100.0
    tissue_detection_error = uns.get("tissue_image_detection_error")
    has_histology = bool(feats.get("has_histology", "img" in uns or "images" in uns or "spatial" in uns))

    modules: list[dict] = []

    # 1. Basic Statistics (informational)
    basic_rows = {
        "sample_id": sample_id or "-",
        "platform": platform or "unknown",
        "tissue_type": feats.get("tissue_type") or "-",
        "n_spots": n_spots,
        "n_genes": n_genes,
        "median_counts_per_spot": round(median_counts, 1),
        "median_genes_per_spot": round(median_genes, 1),
        "spatial_layout": feats.get("spatial_layout") or "-",
        "spot_diameter": round(_num(feats.get("spot_diameter_um")) or 0.0, 2),
        "has_histology": has_histology,
    }
    modules.append({
        "id": "basic_statistics", "name": "Basic Statistics", "status": "pass",
        "value": None, "message": f"{n_spots} spots × {n_genes} genes on {platform or 'unknown'}.",
        "plot": {"kind": "table", "rows": basic_rows},
    })

    # 2. Platform Detection Confidence — FAIL if the platform was only a default guess
    conf = (platform_confidence or "inferred").lower()
    conf_status = "fail" if conf == "default" else "pass"
    conf_msg = {
        "explicit": "Platform was specified explicitly.",
        "inferred": "Platform inferred from a positive structural signal.",
        "default": "Platform could NOT be detected — a default guess was used. "
                   "All platform-aware thresholds below may be wrong; set the "
                   "platform explicitly and re-run.",
    }.get(conf, f"Platform confidence: {conf}.")
    modules.append({
        "id": "platform_detection_confidence", "name": "Platform Detection Confidence",
        "status": conf_status, "value": None,
        "message": f"[{conf}] {conf_msg}",
    })

    # 3. Tissue Coverage — retention after off-tissue filtering
    cov_status = classify_lower_bound(retention_pct, TISSUE_RETENTION_WARN, TISSUE_RETENTION_FAIL)
    cov_msg = (f"{retention_pct:.1f}% of capture spots retained as on-tissue "
               f"({n_excluded_off} excluded by metadata")
    cov_msg += (f", {n_excluded_img} by image segmentation)." if n_excluded_img
                else ").")
    if n_excluded_off == 0 and n_excluded_img == 0:
        cov_msg = ("No off-tissue filtering was applied (no in_tissue metadata or "
                   "image mask); every capture spot was kept.")
    modules.append({
        "id": "tissue_coverage", "name": "Tissue Coverage", "status": cov_status,
        "value": round(retention_pct, 2), "message": cov_msg,
        "thresholds": {"warn": TISSUE_RETENTION_WARN, "fail": TISSUE_RETENTION_FAIL},
        "plot": {"kind": "table", "rows": {
            "spots_on_tissue": n_spots,
            "spots_before_filter": int(n_before),
            "excluded_off_tissue_metadata": n_excluded_off,
            "excluded_image_segmentation": n_excluded_img,
            "retention_pct": round(retention_pct, 2),
        }},
    })

    # 4. Gene Detection Distribution — median genes/spot vs platform-typical
    genes_status = classify_lower_bound(median_genes, prof["genes_warn"], prof["genes_fail"])
    modules.append({
        "id": "gene_detection", "name": "Gene Detection Distribution",
        "status": genes_status, "value": round(median_genes, 1),
        "message": (f"Median {median_genes:.0f} genes/spot "
                    f"(platform-typical ≥ {prof['genes_warn']} for {platform or 'unknown'})."),
        "thresholds": {"warn": prof["genes_warn"], "fail": prof["genes_fail"]},
        "plot": {"kind": "hist", "xlabel": "genes per spot", **_qc._histogram(genes_per_spot)},
    })

    # 5. Sparsity / Dropout — platform-aware upper bound
    sparsity_status = classify_upper_bound(sparsity, prof["sparsity_warn"], prof["sparsity_fail"])
    modules.append({
        "id": "sparsity_dropout", "name": "Sparsity / Dropout",
        "status": sparsity_status, "value": round(sparsity, 4),
        "message": (f"Matrix is {sparsity*100:.1f}% zeros "
                    f"(platform-typical ≤ {prof['sparsity_warn']*100:.0f}%"
                    + (", targeted panel" if prof["targeted"] else "") + ")."),
        "thresholds": {"warn": prof["sparsity_warn"], "fail": prof["sparsity_fail"]},
    })

    # 6. Spatial Signal Sanity Check — Moran's I on total counts
    if morans_i is not None:
        sig_status = classify_lower_bound(morans_i, SPATIAL_SIGNAL_WARN, SPATIAL_SIGNAL_FAIL)
        modules.append({
            "id": "spatial_signal", "name": "Spatial Signal Sanity Check",
            "status": sig_status, "value": round(morans_i, 4),
            "message": (f"Moran's I of per-spot total counts = {morans_i:.3f}. "
                        "Near zero means no spatial structure at all — data may be "
                        "corrupted or coordinates misaligned."),
            "thresholds": {"warn": SPATIAL_SIGNAL_WARN, "fail": SPATIAL_SIGNAL_FAIL},
        })

    # 7. Image / Segmentation Quality
    if tissue_detection_error:
        seg_status, seg_msg = "warn", (
            f"Image-based tissue detection failed ({tissue_detection_error}); "
            "fell back to metadata-only tissue calls.")
    elif not has_histology:
        seg_status, seg_msg = "warn", (
            "No histology image available — tissue calls rely on metadata only; "
            "image-based segmentation quality could not be assessed.")
    else:
        seg_status, seg_msg = "pass", (
            "Histology image present and tissue segmentation ran without error.")
    modules.append({
        "id": "image_segmentation_quality", "name": "Image / Segmentation Quality",
        "status": seg_status, "value": None, "message": seg_msg,
    })

    basic = {
        "sample_id": sample_id or "-",
        "platform": platform or "unknown",
        "platform_confidence": conf,
        "n_spots": n_spots,
        "n_genes": n_genes,
        "median_counts_per_spot": round(median_counts, 1),
        "median_genes_per_spot": round(median_genes, 1),
        "sparsity": round(sparsity, 4),
        "tissue_retention_pct": round(retention_pct, 2),
        "spatial_morans_i": None if morans_i is None else round(morans_i, 4),
    }

    return {
        "version": ISPOT_VERSION,
        "report": "SpatialQC",
        "sample_id": sample_id or "-",
        "platform": platform or "unknown",
        "platform_confidence": conf,
        "basic": basic,
        "modules": modules,
        "summary": summarize(modules),
    }


def summary_record(report: dict) -> dict:
    """Flatten a SpatialQC report into one aggregatable row (MultiQC-style).

    This is the machine-readable ``*_qc_summary.json`` payload. One record per
    sample; ``ispot.multi_sample_qc.aggregate_qc`` consumes a list of these to
    flag outlier samples across a study/batch.
    """
    basic = report.get("basic", {})
    statuses = {m["id"]: m["status"] for m in report.get("modules", [])}
    return {
        "sample_id": report.get("sample_id", "-"),
        "platform": report.get("platform", "unknown"),
        "platform_confidence": report.get("platform_confidence", "inferred"),
        "overall": report.get("summary", {}).get("overall", "warn"),
        "n_spots": basic.get("n_spots"),
        "n_genes": basic.get("n_genes"),
        "median_genes_per_spot": basic.get("median_genes_per_spot"),
        "median_counts_per_spot": basic.get("median_counts_per_spot"),
        "sparsity": basic.get("sparsity"),
        "tissue_retention_pct": basic.get("tissue_retention_pct"),
        "spatial_morans_i": basic.get("spatial_morans_i"),
        "module_status": statuses,
    }


def write_spatial_qc(report: dict, output_dir: str, sample_id: Optional[str] = None) -> dict:
    """Write the SpatialQC deliverables to ``output_dir``.

    Produces:
      - ``qc_report.html``  self-contained FastQC-shaped report (base64 plots)
      - ``qc_report.json``  full report dict
      - ``qc_summary.txt``  FastQC-style STATUS\\tmodule lines
      - ``qc_summary.json`` flat, MultiQC-aggregatable per-sample record

    Returns a dict of written paths.
    """
    import json
    import os
    from ispot.deliverables import generate_qc_report

    os.makedirs(output_dir, exist_ok=True)
    html_path = generate_qc_report(report, output_dir=output_dir)

    summary_path = os.path.join(output_dir, "qc_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary_record(report), fh, indent=2)

    return {
        "html": html_path,
        "report_json": os.path.join(output_dir, "qc_report.json"),
        "summary_txt": os.path.join(output_dir, "qc_summary.txt"),
        "summary_json": summary_path,
    }
