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

# Library size (total counts per spot) — generic depth expectations.
LIBSIZE_WARN = 500
LIBSIZE_FAIL = 100

# Mitochondrial fraction (median across spots) — higher is worse (stress/lysis).
MITO_WARN = 0.10
MITO_FAIL = 0.20

# Per-module FastQC-style interpretation: what it measures, and how to read it.
# Rendered verbatim in the HTML so a core-facility analyst needs no external docs.
EXPLANATIONS: dict[str, str] = {
    "basic_statistics": (
        "Summary of the dataset as loaded, before any preprocessing. Confirm the "
        "spot and gene counts, platform, and tissue type match what you submitted "
        "— a surprise here (e.g. 10x fewer spots than expected) usually means the "
        "wrong file or a truncated upload."),
    "platform_detection_confidence": (
        "Every threshold in this report is platform-aware. If the platform was only "
        "a default guess (FAIL), those thresholds may be wrong for your data — set "
        "the platform explicitly and re-run. 'Explicit' or 'inferred' means the "
        "thresholds below are trustworthy."),
    "library_size": (
        "Distribution of total UMI/transcript counts per spot — the spatial analog "
        "of FastQC's per-sequence quality. Shallow libraries (mass of the "
        "distribution near zero) limit every downstream method. Dashed lines are "
        "the WARN/FAIL cutoffs; the healthy range is to their right."),
    "gene_detection": (
        "Distribution of the number of distinct genes detected per spot "
        "(transcriptional complexity). Compared against what is typical for this "
        "platform: whole-transcriptome capture (Visium) should reach thousands, "
        "targeted panels (Xenium/CosMx/MERFISH) only tens-hundreds by design."),
    "sparsity_dropout": (
        "How empty the counts matrix is, shown as the per-gene detection rate "
        "(fraction of spots in which each gene is seen). Extreme sparsity beyond "
        "the platform norm indicates dropout/degradation that will destabilise "
        "clustering. Targeted panels are legitimately very sparse."),
    "mitochondrial_content": (
        "Fraction of counts from mitochondrial genes per spot — the analog of "
        "FastQC's adapter/contamination module. High mitochondrial content flags "
        "cell stress, lysis, or low-quality capture."),
    "spatial_signal": (
        "The critical sanity check unique to spatial data: Moran's I of total "
        "counts measures whether nearby spots resemble each other. Real tissue "
        "always has structure (positive I). A value near zero means no spatial "
        "signal at all — corrupted counts or coordinates misaligned to the tissue, "
        "and clustering results would be meaningless. The map shows counts in "
        "spatial position; look for a coherent tissue footprint, not noise."),
    "tissue_coverage": (
        "How many capture spots were retained as on-tissue after metadata/image "
        "filtering. Low retention means much of the capture area is background — "
        "keeping it drags off-tissue spots into clustering and the viewer. This is "
        "the spatial analog of FastQC's per-base N-content / adapter trimming."),
    "image_segmentation_quality": (
        "Whether a histology image was available and tissue segmentation succeeded. "
        "Without an image, tissue calls rely on metadata alone and the real tissue "
        "boundary cannot be verified."),
}


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

    # Extra arrays for the detailed per-module figures.
    gene_detected_in = _qc._d1((X > 0).sum(axis=0)).astype(float)
    gene_detection_rate = (gene_detected_in / n_spots) if n_spots else gene_detected_in
    var_upper = [str(g).upper() for g in adata.var_names] if n_genes else []
    mito_mask = np.array([g.startswith("MT-") or g.startswith("MT.") for g in var_upper]) \
        if n_genes else np.zeros(0, dtype=bool)
    has_mito = bool(mito_mask.any())
    mito_frac = None
    median_mito = 0.0
    if has_mito:
        with np.errstate(divide="ignore", invalid="ignore"):
            safe = np.where(counts_per_spot > 0, counts_per_spot, np.nan)
            mito_counts = _qc._d1(X[:, mito_mask].sum(axis=1)).astype(float)
            mito_frac = mito_counts / safe
        median_mito = float(np.nanmedian(mito_frac)) if n_spots else 0.0

    # Downsampled spatial scatter (cap points so the report JSON stays small).
    scatter = None
    if coords is not None and n_spots:
        step = int(np.ceil(n_spots / 6000)) if n_spots > 6000 else 1
        sel = np.arange(0, n_spots, step)
        scatter = {
            "kind": "scatter",
            "x": [float(v) for v in coords[sel, 0]],
            "y": [float(v) for v in coords[sel, 1]],
            "values": [float(v) for v in counts_per_spot[sel]],
            "clabel": "total counts",
        }

    modules: list[dict] = []

    def add(module: dict, figure=None):
        """Attach the FastQC-style explanation (and optional figure) and record it."""
        if figure is not None:
            module["figure"] = figure
        module["explanation"] = EXPLANATIONS.get(module["id"], "")
        modules.append(module)

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
    add({"id": "basic_statistics", "name": "Basic Statistics", "status": "pass",
         "value": None,
         "message": f"{n_spots} spots x {n_genes} genes on {platform or 'unknown'}."},
        figure={"kind": "table", "rows": basic_rows})

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
    add({"id": "platform_detection_confidence", "name": "Platform Detection Confidence",
         "status": conf_status, "value": None, "message": f"[{conf}] {conf_msg}"})

    # 3. Library size (sequencing depth) — total counts per spot
    lib_status = classify_lower_bound(median_counts, LIBSIZE_WARN, LIBSIZE_FAIL)
    add({"id": "library_size", "name": "Library Size (sequencing depth)",
         "status": lib_status, "value": round(median_counts, 1),
         "message": f"Median {median_counts:.0f} counts per spot.",
         "thresholds": {"warn": LIBSIZE_WARN, "fail": LIBSIZE_FAIL}},
        figure={"kind": "hist", "xlabel": "total counts per spot", "direction": "low",
                "warn": LIBSIZE_WARN, "fail": LIBSIZE_FAIL,
                "hist": _qc._histogram(counts_per_spot)})

    # 4. Gene Detection Distribution — median genes/spot vs platform-typical
    genes_status = classify_lower_bound(median_genes, prof["genes_warn"], prof["genes_fail"])
    add({"id": "gene_detection", "name": "Gene Detection Distribution",
         "status": genes_status, "value": round(median_genes, 1),
         "message": (f"Median {median_genes:.0f} genes/spot "
                     f"(platform-typical >= {prof['genes_warn']} for {platform or 'unknown'})."),
         "thresholds": {"warn": prof["genes_warn"], "fail": prof["genes_fail"]}},
        figure={"kind": "hist", "xlabel": "genes per spot", "direction": "low",
                "warn": prof["genes_warn"], "fail": prof["genes_fail"],
                "hist": _qc._histogram(genes_per_spot)})

    # 5. Sparsity / Dropout — platform-aware upper bound; figure = per-gene detection rate
    sparsity_status = classify_upper_bound(sparsity, prof["sparsity_warn"], prof["sparsity_fail"])
    add({"id": "sparsity_dropout", "name": "Sparsity / Dropout",
         "status": sparsity_status, "value": round(sparsity, 4),
         "message": (f"Matrix is {sparsity*100:.1f}% zeros "
                     f"(platform-typical <= {prof['sparsity_warn']*100:.0f}%"
                     + (", targeted panel" if prof["targeted"] else "") + ")."),
         "thresholds": {"warn": prof["sparsity_warn"], "fail": prof["sparsity_fail"]}},
        figure={"kind": "hist", "xlabel": "per-gene detection rate (fraction of spots)",
                "hist": _qc._histogram(gene_detection_rate)})

    # 6. Mitochondrial content (only when mito genes are present)
    if has_mito:
        mito_status = classify_upper_bound(median_mito, MITO_WARN, MITO_FAIL)
        add({"id": "mitochondrial_content", "name": "Mitochondrial Content",
             "status": mito_status, "value": round(median_mito, 4),
             "message": (f"Median mitochondrial fraction {median_mito*100:.1f}% "
                         f"across {int(mito_mask.sum())} MT gene(s)."),
             "thresholds": {"warn": MITO_WARN, "fail": MITO_FAIL}},
            figure={"kind": "hist", "xlabel": "mitochondrial fraction per spot",
                    "direction": "high", "warn": MITO_WARN, "fail": MITO_FAIL,
                    "hist": _qc._histogram(mito_frac)})

    # 7. Spatial Signal Sanity Check — Moran's I on total counts + spatial map
    if morans_i is not None:
        sig_status = classify_lower_bound(morans_i, SPATIAL_SIGNAL_WARN, SPATIAL_SIGNAL_FAIL)
        add({"id": "spatial_signal", "name": "Spatial Signal Sanity Check",
             "status": sig_status, "value": round(morans_i, 4),
             "message": (f"Moran's I of per-spot total counts = {morans_i:.3f}. "
                         "Near zero means no spatial structure at all — data may be "
                         "corrupted or coordinates misaligned."),
             "thresholds": {"warn": SPATIAL_SIGNAL_WARN, "fail": SPATIAL_SIGNAL_FAIL}},
            figure=scatter)

    # 8. Tissue Coverage — retention after off-tissue filtering
    cov_status = classify_lower_bound(retention_pct, TISSUE_RETENTION_WARN, TISSUE_RETENTION_FAIL)
    cov_msg = (f"{retention_pct:.1f}% of capture spots retained as on-tissue "
               f"({n_excluded_off} excluded by metadata")
    cov_msg += (f", {n_excluded_img} by image segmentation)." if n_excluded_img else ").")
    n_excluded_total = int(max(n_before, n_spots) - n_spots)
    if n_excluded_off == 0 and n_excluded_img == 0:
        cov_msg = ("No off-tissue filtering was applied (no in_tissue metadata or "
                   "image mask); every capture spot was kept.")
    add({"id": "tissue_coverage", "name": "Tissue Coverage", "status": cov_status,
         "value": round(retention_pct, 2), "message": cov_msg,
         "thresholds": {"warn": TISSUE_RETENTION_WARN, "fail": TISSUE_RETENTION_FAIL}},
        figure={"kind": "bar", "ylabel": "spots",
                "labels": ["on-tissue", "excluded"],
                "values": [n_spots, n_excluded_total],
                "colors": ["#3fae49", "#d64545"],
                "table": {
                    "spots_on_tissue": n_spots,
                    "spots_before_filter": int(n_before),
                    "excluded_off_tissue_metadata": n_excluded_off,
                    "excluded_image_segmentation": n_excluded_img,
                    "retention_pct": round(retention_pct, 2),
                }})

    # 9. Image / Segmentation Quality
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
    add({"id": "image_segmentation_quality", "name": "Image / Segmentation Quality",
         "status": seg_status, "value": None, "message": seg_msg})

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
      - ``qc_report.html``  self-contained, detailed FastQC-style report
                            (per-module figures + written interpretation + tables)
      - ``qc_report.json``  full report dict
      - ``qc_summary.txt``  FastQC-style STATUS\\tmodule lines
      - ``qc_summary.json`` flat, MultiQC-aggregatable per-sample record

    Returns a dict of written paths.
    """
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "qc_report.json"), "w") as fh:
        json.dump(report, fh, default=_json_default)

    with open(os.path.join(output_dir, "qc_summary.txt"), "w") as fh:
        for m in report.get("modules", []):
            fh.write(f"{m['status'].upper()}\t{m['name']}\n")

    html_path = render_spatial_qc_html(report, output_dir)

    summary_path = os.path.join(output_dir, "qc_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary_record(report), fh, indent=2)

    return {
        "html": html_path,
        "report_json": os.path.join(output_dir, "qc_report.json"),
        "summary_txt": os.path.join(output_dir, "qc_summary.txt"),
        "summary_json": summary_path,
    }


# ---------------------------------------------------------------------------
# Detailed HTML rendering — a self-contained, FastQC-style report
# ---------------------------------------------------------------------------
_STATUS_COLORS = {"pass": "#3fae49", "warn": "#e0a800", "fail": "#d64545"}
_STATUS_ICON = {"pass": "&#10003;", "warn": "!", "fail": "&#10007;"}


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _fig_to_base64(fig) -> str:
    import base64
    import io
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _render_figure(fig_spec: dict) -> str:
    """Render one module figure spec to an <img> or HTML table string."""
    if not fig_spec:
        return ""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kind = fig_spec.get("kind")

    if kind == "table":
        rows = fig_spec.get("rows", {})
        body = "".join(
            f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in rows.items())
        return (f"<table class='mini'><tr><th>metric</th><th>value</th></tr>{body}</table>")

    if kind == "hist":
        h = fig_spec.get("hist", {})
        edges = h.get("bin_edges") or []
        counts = h.get("counts") or []
        if len(edges) < 2 or not counts:
            return ""
        edges = np.asarray(edges, float)
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges)
        fig, ax = plt.subplots(figsize=(5.0, 2.7))
        ax.bar(centers, counts, width=widths, align="center",
               color="#2f7fd0", edgecolor="none")
        warn, fail = fig_spec.get("warn"), fig_spec.get("fail")
        direction = fig_spec.get("direction")
        lo, hi = float(edges[0]), float(edges[-1])
        if warn is not None and lo <= warn <= hi:
            ax.axvline(warn, color="#e0a800", ls="--", lw=1.3, label=f"warn={warn:g}")
        if fail is not None and lo <= fail <= hi:
            ax.axvline(fail, color="#d64545", ls="--", lw=1.3, label=f"fail={fail:g}")
        # shade the healthy region
        if direction == "low" and warn is not None:
            ax.axvspan(min(warn, hi), hi, color="#3fae49", alpha=0.07)
        elif direction == "high" and warn is not None:
            ax.axvspan(lo, min(warn, hi), color="#3fae49", alpha=0.07)
        ax.set_xlabel(fig_spec.get("xlabel", ""))
        ax.set_ylabel("spots" if "spot" in fig_spec.get("xlabel", "") else "count")
        ax.spines[["top", "right"]].set_visible(False)
        if warn is not None or fail is not None:
            ax.legend(fontsize=7, frameon=False)
        return f'<img alt="histogram" src="{_fig_to_base64(fig)}">'

    if kind == "bar":
        labels = fig_spec.get("labels", [])
        values = fig_spec.get("values", [])
        colors = fig_spec.get("colors") or ["#2f7fd0"] * len(labels)
        fig, ax = plt.subplots(figsize=(3.6, 2.7))
        ax.bar(range(len(values)), values, color=colors, edgecolor="none")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(fig_spec.get("ylabel", ""))
        ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(values):
            ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)
        img = f'<img alt="bar" src="{_fig_to_base64(fig)}">'
        table = fig_spec.get("table")
        if table:
            body = "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>"
                           for k, v in table.items())
            img += (f"<table class='mini'><tr><th>metric</th><th>value</th></tr>"
                    f"{body}</table>")
        return img

    if kind == "scatter":
        xs = np.asarray(fig_spec.get("x") or [], float)
        ys = np.asarray(fig_spec.get("y") or [], float)
        vals = np.asarray(fig_spec.get("values") or [], float)
        if xs.size == 0:
            return ""
        fig, ax = plt.subplots(figsize=(4.2, 3.9))
        sc = ax.scatter(xs, ys, c=vals, s=6, cmap="viridis", edgecolors="none")
        ax.set_aspect("equal")
        ax.invert_yaxis()  # image/tissue convention: y grows downward
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("spatial x"); ax.set_ylabel("spatial y")
        fig.colorbar(sc, ax=ax, shrink=0.8, label=fig_spec.get("clabel", ""))
        return f'<img alt="spatial map" src="{_fig_to_base64(fig)}">'

    return ""


def render_spatial_qc_html(report: dict, output_dir: str,
                           filename: str = "qc_report.html") -> str:
    """Render the detailed, self-contained SpatialQC HTML report.

    FastQC-style layout: a header with the overall verdict and a summary table,
    a left traffic-light navigation panel, and one section per module carrying a
    status badge, the metric value against its thresholds, a written
    interpretation, and the module's figure/table. No external assets.
    """
    import os

    modules = report.get("modules", [])
    summary = report.get("summary", {})
    overall = summary.get("overall", "warn")
    basic = report.get("basic", {})
    version = report.get("version", "")

    nav = "".join(
        f'<a href="#{m["id"]}" class="nav-item">'
        f'<span class="dot" style="background:{_STATUS_COLORS.get(m["status"], "#888")}">'
        f'{_STATUS_ICON.get(m["status"], "?")}</span>{_esc(m["name"])}</a>'
        for m in modules
    )

    sections = ""
    for m in modules:
        st = m.get("status", "warn")
        thr = ""
        if "thresholds" in m and m.get("value") is not None:
            thr = (f'<span class="thr">value = {m["value"]:.4g} '
                   f'(warn {m["thresholds"]["warn"]}, fail {m["thresholds"]["fail"]})</span>')
        expl = m.get("explanation", "")
        expl_html = f'<p class="expl">{_esc(expl)}</p>' if expl else ""
        figure = _render_figure(m.get("figure")) if m.get("figure") else ""
        sections += (
            f'<section id="{m["id"]}" class="module">'
            f'<h3><span class="badge" style="background:{_STATUS_COLORS.get(st, "#888")}">'
            f'{_STATUS_ICON.get(st, "?")} {st.upper()}</span> {_esc(m["name"])}</h3>'
            f'<p class="msg">{_esc(m.get("message", ""))} {thr}</p>'
            f'{expl_html}'
            f'<div class="plot">{figure}</div></section>'
        )

    summary_table = "".join(
        f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in {
            "Sample": basic.get("sample_id", "-"),
            "Platform": f'{basic.get("platform", "unknown")} ({report.get("platform_confidence","?")})',
            "Spots": basic.get("n_spots", "?"),
            "Genes": basic.get("n_genes", "?"),
            "Median counts/spot": basic.get("median_counts_per_spot", "?"),
            "Median genes/spot": basic.get("median_genes_per_spot", "?"),
            "Sparsity": basic.get("sparsity", "?"),
            "Tissue retention %": basic.get("tissue_retention_pct", "?"),
            "Spatial Moran's I": basic.get("spatial_morans_i", "?"),
        }.items())

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>SpatialQC Report — {_esc(basic.get('sample_id','-'))}</title>
<style>
 body{{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1a1a1a}}
 header{{background:#0b1f3a;color:#fff;padding:18px 26px}}
 header h1{{margin:0;font-size:20px}} header .sub{{opacity:.85;font-size:12px;margin-top:4px}}
 .overall{{display:inline-block;padding:3px 11px;border-radius:5px;font-weight:700;margin-left:8px;
   background:{_STATUS_COLORS.get(overall,'#888')}}}
 .layout{{display:flex;align-items:flex-start}}
 nav{{width:290px;flex:0 0 290px;border-right:1px solid #eee;padding:12px 0;position:sticky;top:0;
   max-height:100vh;overflow:auto}}
 .nav-item{{display:flex;align-items:center;gap:8px;padding:6px 16px;font-size:13px;color:#222;text-decoration:none}}
 .nav-item:hover{{background:#f5f7fa}}
 .dot{{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:50%;
   color:#fff;font-size:11px;font-weight:700}}
 main{{flex:1;padding:14px 26px;max-width:820px}}
 .module{{border-bottom:1px solid #eee;padding:16px 0}}
 .module h3{{font-size:15.5px;margin:0 0 6px;display:flex;align-items:center;gap:8px}}
 .badge{{color:#fff;font-size:11px;font-weight:700;padding:2px 8px;border-radius:4px}}
 .msg{{font-size:13px;color:#333;margin:0 0 6px}} .thr{{color:#888;margin-left:6px}}
 .expl{{font-size:12.5px;color:#555;background:#f7f9fc;border-left:3px solid #cdd8e6;
   padding:8px 12px;margin:0 0 10px;line-height:1.5}}
 table.mini,table.summary{{border-collapse:collapse;font-size:12.5px;margin-top:6px}}
 table.mini td,table.mini th,table.summary td,table.summary th{{border:1px solid #e6e6e6;padding:4px 9px;text-align:left}}
 table.summary th{{background:#f4f6f9}}
 img{{max-width:100%;height:auto;margin-top:6px}}
 .intro{{font-size:12.5px;color:#555;margin:0 0 10px}}
</style></head><body>
<header>
 <h1>SpatialQC Report <span class="overall">{overall.upper()}</span></h1>
 <div class="sub">Fast, run-first quality control for spatial transcriptomics &middot;
  {summary.get('pass',0)} pass / {summary.get('warn',0)} warn / {summary.get('fail',0)} fail
  &middot; iSpot {_esc(version)}</div>
</header>
<div class="layout">
 <nav>{nav}</nav>
 <main>
  <p class="intro">SpatialQC runs before clustering — load, profile, and tissue detection only —
   to answer "is this dataset worth clustering?". All thresholds are platform-aware; metrics are
   computed on the raw uploaded counts. Below: a dataset summary, then one module per quality
   dimension with a figure and how to read it.</p>
  <table class="summary"><tr><th>property</th><th>value</th></tr>{summary_table}</table>
  {sections}
 </main>
</div>
</body></html>
"""
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path
