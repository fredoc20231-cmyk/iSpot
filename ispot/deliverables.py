"""
Deliverable generators for the iSpot platform.

Produces four output types:
  1. Ranking table (CSV + HTML)
  2. Publication figures (PNG, 300 DPI)
  3. Interactive viewer data (JSON for web frontend)
  4. Written report (PDF)

Section 1.7 of the platform plan.
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import anndata as ad
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rcParams = matplotlib.rcParams


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Phylo color palette
PHYLO_COLORS = [
    "#0279EE", "#FF9400", "#75A025", "#FD9BED", "#E9ED4C",
    "#000000", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f",
    "#8c564b", "#aec7e8", "#ffbb78", "#2ca02c", "#d62728",
    "#1f77b4", "#9467bd", "#ff7f0e",
]

# Font defaults
rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
rcParams["svg.fonttype"] = "none"
rcParams["figure.dpi"] = 300


# ---------------------------------------------------------------------------
# 1. Ranking Table
# ---------------------------------------------------------------------------

def generate_ranking_table(
    results: pd.DataFrame,
    has_ground_truth: bool,
    output_dir: str,
    weights: dict | None = None,
) -> str:
    """Generate a ranking table from benchmark results.

    Parameters
    ----------
    results : pd.DataFrame
        One row per (method, seed) with columns:
        - method, ari (or nogt_score), macro_f1, weighted_f1, runtime,
        - scs, css, ess, cas (if no GT)
        - n_spots, n_clusters_pred
    has_ground_truth : bool
        If True, rank by ARI. If False, rank by NoGTScore.
    output_dir : str
        Directory to save the table.
    weights : dict, optional
        No-GT component weights (for display).

    Returns
    -------
    str: path to the saved CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Aggregate across seeds
    if has_ground_truth:
        score_col = "ari"
        agg = results.groupby("method").agg(
            score_mean=(score_col, "mean"),
            score_std=(score_col, "std"),
            macro_f1_mean=("macro_f1", "mean"),
            weighted_f1_mean=("weighted_f1", "mean"),
            runtime_mean=("runtime", "mean"),
            n_seeds=("seed", "count"),
        ).reset_index()
        agg = agg.sort_values("score_mean", ascending=False)
        agg["rank"] = range(1, len(agg) + 1)

        # Recommendation column
        agg["recommendation"] = ""
        if len(agg) > 0:
            agg.iloc[0, agg.columns.get_loc("recommendation")] = "Best overall"
        if len(agg) > 1:
            # Best speed/accuracy tradeoff
            best_speed = agg.loc[agg["runtime_mean"].idxmin()]
            agg.loc[agg["runtime_mean"].idxmin(), "recommendation"] = "Best speed/accuracy"
        # Rename columns
        agg = agg.rename(columns={
            "score_mean": "ARI (mean)",
            "score_std": "ARI (std)",
            "macro_f1_mean": "Macro F1",
            "weighted_f1_mean": "Weighted F1",
            "runtime_mean": "Runtime (s)",
            "n_seeds": "n seeds",
        })
    else:
        score_col = "nogt_score"
        # Aggregate all component scores — only use columns that exist
        all_agg_cols = {
            "nogt_score": "mean", "scs": "mean", "css": "mean",
            "ess": "mean", "cas": "mean", "runtime": "mean",
            "seed": "count",
        }
        agg_cols = {k: v for k, v in all_agg_cols.items() if k in results.columns}
        agg = results.groupby("method").agg(agg_cols).reset_index()
        # Build column names based on what was actually aggregated
        col_names = ["method"]
        for k in all_agg_cols:
            if k in agg_cols:
                col_names.append({
                    "nogt_score": "NoGTScore", "scs": "SCS", "css": "CSS",
                    "ess": "ESS", "cas": "CAS", "runtime": "Runtime (s)",
                    "seed": "n seeds",
                }[k])
        agg.columns = col_names
        sort_col = "NoGTScore" if "NoGTScore" in agg.columns else agg.columns[1]
        agg = agg.sort_values(sort_col, ascending=False)
        agg["rank"] = range(1, len(agg) + 1)
        agg["recommendation"] = ""
        if len(agg) > 0:
            agg.iloc[0, agg.columns.get_loc("recommendation")] = "Best overall"
        if len(agg) > 1:
            agg.loc[agg["Runtime (s)"].idxmin(), "recommendation"] = "Best speed/accuracy"

    # Reorder columns
    cols = ["rank", "method"] + [c for c in agg.columns if c not in ["rank", "method", "recommendation"]] + ["recommendation"]
    agg = agg[cols]

    # Save CSV
    csv_path = os.path.join(output_dir, "ranking_table.csv")
    agg.to_csv(csv_path, index=False, float_format="%.4f")

    # Save HTML
    html_path = os.path.join(output_dir, "ranking_table.html")
    agg.to_html(html_path, index=False, float_format="%.4f", classes="ranking-table")

    return csv_path


# ---------------------------------------------------------------------------
# 2. Publication Figures
# ---------------------------------------------------------------------------

def generate_figures(
    results: pd.DataFrame,
    adata: ad.AnnData,
    method_labels: dict[str, np.ndarray],
    has_ground_truth: bool,
    output_dir: str,
) -> list[str]:
    """Generate publication figures (PNG, 300 DPI).

    Parameters
    ----------
    results : pd.DataFrame
        Benchmark results (one row per method-seed).
    adata : AnnData
        Preprocessed data with spatial coordinates.
    method_labels : dict[str, np.ndarray]
        Method name -> cluster labels (from seed=42 run).
    has_ground_truth : bool
    output_dir : str

    Returns
    -------
    list[str]: paths to saved figure files.
    """
    os.makedirs(output_dir, exist_ok=True)
    figures = []
    coords = np.array(adata.obsm["spatial"])

    # --- Figure 1: Spatial cluster maps ---
    fig = _plot_spatial_cluster_maps(method_labels, coords, has_ground_truth, adata)
    path = os.path.join(output_dir, "fig1_spatial_clusters.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    # --- Figure 2: Metric bar chart ---
    fig = _plot_metric_bar_chart(results, has_ground_truth)
    if fig is not None:
        path = os.path.join(output_dir, "fig2_metric_bars.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures.append(path)

    # --- Figure 3: Runtime vs accuracy scatter ---
    fig = _plot_runtime_vs_accuracy(results, has_ground_truth)
    if fig is not None:
        path = os.path.join(output_dir, "fig3_runtime_vs_accuracy.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures.append(path)

    # --- Figure 4: Stability heatmap (if multiple seeds) ---
    if results["seed"].nunique() > 1:
        fig = _plot_stability_heatmap(results)
        path = os.path.join(output_dir, "fig4_stability_heatmap.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures.append(path)

    # --- Figure 5: No-GT component scores radar (if no GT) ---
    if not has_ground_truth and "scs" in results.columns:
        fig = _plot_component_radar(results)
        path = os.path.join(output_dir, "fig5_component_radar.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures.append(path)

    return figures


def _plot_spatial_cluster_maps(method_labels, coords, has_gt, adata, max_methods=8):
    """Side-by-side spatial cluster maps per method."""
    methods = list(method_labels.keys())[:max_methods]
    n = len(methods)
    if has_gt:
        n += 1  # Add ground truth panel

    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n == 1:
        axes = np.array([[axes]])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes = np.atleast_2d(axes)

    for idx, method in enumerate(methods):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        labels = np.array(method_labels[method]).astype(str)
        unique_labels = np.unique(labels)
        colors = PHYLO_COLORS[:len(unique_labels)]
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      c=[colors[i]], s=3, alpha=0.7, rasterized=True)
        ax.set_title(method, fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")

    # Ground truth panel
    if has_gt:
        idx = len(methods)
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        gt_mask = adata.obs["has_ground_truth"].values.astype(bool)
        gt = adata.obs.loc[gt_mask, "ground_truth"].values.astype(str)
        unique_gt = np.unique(gt)
        colors = PHYLO_COLORS[:len(unique_gt)]
        for i, label in enumerate(unique_gt):
            mask = gt == label
            ax.scatter(coords[gt_mask][mask, 0], coords[gt_mask][mask, 1],
                      c=[colors[i]], s=3, alpha=0.7, rasterized=True)
        ax.set_title("Ground Truth", fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")

    # Hide unused axes
    for idx in range(n, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis("off")

    fig.suptitle("Spatial Cluster Maps", fontsize=14, y=1.01)
    return fig


def _plot_metric_bar_chart(results, has_gt):
    """Bar chart of scores per method with error bars."""
    score_col = "ari" if has_gt else "nogt_score"
    label = "ARI" if has_gt else "No-GT Score"

    # Guard: if score column doesn't exist, skip this figure
    if score_col not in results.columns:
        return None
    # Drop rows with no score
    results = results.dropna(subset=[score_col])
    if len(results) == 0:
        return None

    agg = results.groupby("method")[score_col].agg(["mean", "std"]).reset_index()
    agg = agg.sort_values("mean", ascending=True)
    agg["std"] = agg["std"].fillna(0)

    fig, ax = plt.subplots(figsize=(8, max(4, len(agg) * 0.4)))
    colors = PHYLO_COLORS[:len(agg)]
    ax.barh(agg["method"], agg["mean"], xerr=agg["std"],
            color=colors, capsize=3, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(label)
    ax.set_title(f"{label} by Method")
    ax.set_xlim(0, max(agg["mean"] + agg["std"]) * 1.1)
    return fig


def _plot_runtime_vs_accuracy(results, has_gt):
    """Runtime vs accuracy scatter plot."""
    score_col = "ari" if has_gt else "nogt_score"
    label = "ARI" if has_gt else "No-GT Score"

    # Guard: if score column doesn't exist, skip this figure
    if score_col not in results.columns:
        return None
    results = results.dropna(subset=[score_col])
    if len(results) == 0:
        return None

    agg = results.groupby("method").agg(
        score_mean=(score_col, "mean"),
        runtime_mean=("runtime", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, row in agg.iterrows():
        color = PHYLO_COLORS[i % len(PHYLO_COLORS)]
        ax.scatter(row["runtime_mean"], row["score_mean"],
                  c=[color], s=100, edgecolors="black", linewidth=0.5, zorder=5)
        ax.annotate(row["method"], (row["runtime_mean"], row["score_mean"]),
                   fontsize=8, xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Runtime (s, log scale)")
    ax.set_ylabel(label)
    ax.set_xscale("log")
    ax.set_title("Runtime vs. Accuracy")
    ax.set_ylim(0, max(agg["score_mean"]) * 1.15)
    return fig


def _plot_stability_heatmap(results):
    """Pairwise ARI across seeds for each method."""
    from sklearn.metrics import adjusted_rand_score

    methods = results["method"].unique()
    seeds = sorted(results["seed"].unique())

    if len(seeds) < 2:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "Need multiple seeds for stability", ha="center", va="center")
        ax.axis("off")
        return fig

    # This requires the actual labels per seed, which we don't have in the
    # results DataFrame. We'll show the stability score (CSS) per method instead.
    if "css" in results.columns:
        agg = results.groupby("method")["css"].first().sort_values(ascending=True)
    else:
        # Compute from ARI variance as a proxy
        agg = results.groupby("method")["ari"].agg(lambda x: 1 - x.std() if len(x) > 1 else 1.0)
        agg = agg.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(6, max(3, len(agg) * 0.4)))
    colors = plt.cm.RdYlGn(agg.values)
    ax.barh(agg.index, agg.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Cluster Stability Score")
    ax.set_title("Stability Across Seeds")
    ax.set_xlim(0, 1)
    return fig


def _plot_component_radar(results):
    """Radar chart of No-GT component scores per method."""
    methods = results["method"].unique()
    components = ["scs", "css", "ess", "cas"]
    comp_labels = ["Spatial\nCoherence", "Cluster\nStability", "Expression\nSeparability", "Consensus\nAlignment"]

    # Aggregate
    agg = results.groupby("method")[components].mean()

    n_comp = len(components)
    angles = np.linspace(0, 2 * np.pi, n_comp, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, method in enumerate(methods):
        values = agg.loc[method].tolist()
        values += values[:1]
        color = PHYLO_COLORS[i % len(PHYLO_COLORS)]
        ax.plot(angles, values, "o-", linewidth=1.5, label=method, color=color, markersize=4)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), comp_labels)
    ax.set_ylim(0, 1)
    ax.set_title("No-GT Component Scores", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    return fig


# ---------------------------------------------------------------------------
# 3. Interactive Viewer Data
# ---------------------------------------------------------------------------

def _embed_histology(viewer_data: dict, adata: ad.AnnData, max_dim: int = 1200) -> None:
    """Embed a downscaled histology image + scalefactor into the viewer data.

    Enables the frontend to draw the real tissue image and place spots at
    ``coord * scalef`` in image-pixel space — the canonical Visium alignment,
    which is what makes spots land ON the tissue rather than in a raw-coordinate
    bounding box. No-op when no histology image is present.
    """
    spatial = adata.uns.get("spatial")
    if not spatial:
        return
    for sample in spatial.values():
        if not isinstance(sample, dict):
            continue
        images = sample.get("images", {}) or {}
        sfs = sample.get("scalefactors", {}) or {}
        for res_key, sf_key in (("hires", "tissue_hires_scalef"),
                                 ("lowres", "tissue_lowres_scalef")):
            if res_key not in images or sf_key not in sfs:
                continue
            import base64
            import io

            from PIL import Image

            arr = np.asarray(images[res_key])
            if arr.dtype != np.uint8:
                a = arr.astype(float)
                arr = (a / a.max() * 255).astype(np.uint8) if a.max() > 0 else a.astype(np.uint8)
            img = Image.fromarray(arr)
            scalef = float(sfs[sf_key])
            w, h = img.size
            shrink = min(1.0, max_dim / max(w, h))
            if shrink < 1.0:
                img = img.resize((max(1, int(w * shrink)), max(1, int(h * shrink))))
                scalef *= shrink  # coord*scalef now maps into the resized image
                w, h = img.size
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            spot_d = sfs.get("spot_diameter_fullres")
            viewer_data["histology"] = {
                "data_url": f"data:image/png;base64,{b64}",
                "width": int(w),
                "height": int(h),
                "scalef": scalef,
                "spot_diameter_fullres": float(spot_d) if spot_d else None,
            }
            return


def generate_viewer_data(
    adata: ad.AnnData,
    method_labels: dict[str, np.ndarray],
    has_ground_truth: bool,
    output_dir: str,
) -> str:
    """Generate JSON data for the interactive web viewer.

    Produces a JSON file with:
    - Spatial coordinates for all spots
    - Cluster labels per method
    - Ground truth labels (if available)
    - Gene expression data (top HVGs for hover display)

    Parameters
    ----------
    adata : AnnData
    method_labels : dict[str, np.ndarray]
    has_ground_truth : bool
    output_dir : str

    Returns
    -------
    str: path to the JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    coords = np.array(adata.obsm["spatial"])

    # Select top 50 HVGs for hover display
    if "highly_variable" in adata.var.columns:
        hvgs = adata.var_names[adata.var["highly_variable"]][:50].tolist()
    else:
        hvgs = adata.var_names[:50].tolist()

    # Get expression matrix for HVGs
    hvg_idx = [list(adata.var_names).index(g) for g in hvgs if g in adata.var_names]
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    expr_subset = X[:, hvg_idx]

    # Build viewer data
    viewer_data = {
        "n_spots": int(adata.shape[0]),
        "n_methods": len(method_labels),
        "has_ground_truth": has_ground_truth,
        "genes": hvgs,
        "spots": [],
        "methods": {},
    }

    # Real tissue outline for the viewer (from image-based detection at load).
    if "tissue_mask_for_viewer" in adata.uns:
        viewer_data["tissue_mask"] = adata.uns["tissue_mask_for_viewer"]
        viewer_data["tissue_mask_scale_factor"] = adata.uns.get("tissue_mask_scale_factor")

    # Embed the histology image so the viewer can draw it and align spots to it.
    try:
        _embed_histology(viewer_data, adata)
    except Exception as e:  # pragma: no cover - defensive
        viewer_data["histology_error"] = str(e)

    # Spots (coordinates + expression for hover)
    for i in range(adata.shape[0]):
        spot = {
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "expression": {hvgs[j]: float(expr_subset[i, j]) for j in range(len(hvgs))},
        }
        if has_ground_truth and bool(adata.obs["has_ground_truth"].iloc[i]):
            spot["ground_truth"] = str(adata.obs["ground_truth"].iloc[i])
        viewer_data["spots"].append(spot)

    # Method labels. A method whose label count doesn't match the viewer's spot
    # count is a real upstream misalignment (e.g. preprocessing dropped spots
    # that the label array still counts). Fail loudly with a traceable error
    # rather than silently truncating/padding, which would garble the rendered
    # shape. This must be fixed upstream, not hidden here.
    n_spots = adata.shape[0]
    for method, labels in method_labels.items():
        if len(labels) != n_spots:
            raise ValueError(
                f"Method '{method}' produced {len(labels)} labels but the "
                f"viewer's adata has {n_spots} spots — spot-count mismatch "
                f"between preprocessing and label generation must be fixed "
                f"upstream, not silently truncated/padded here."
            )
        viewer_data["methods"][method] = [str(l) for l in labels]

    # Save as JSON (may be large; consider chunking for very large datasets)
    json_path = os.path.join(output_dir, "viewer_data.json")
    with open(json_path, "w") as f:
        json.dump(viewer_data, f, default=_json_default)

    return json_path


# ---------------------------------------------------------------------------
# 4. Written Report (PDF)
# ---------------------------------------------------------------------------

_QC_COLORS = {"pass": "#75A025", "warn": "#E0A800", "fail": "#D62728"}
_QC_ICON = {"pass": "&#10003;", "warn": "!", "fail": "&#10007;"}  # tick / bang / cross


def _qc_esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fig_to_base64(fig) -> str:
    import base64
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _qc_render_plot(plot: dict) -> str:
    """Render a module's plot dict to an <img> (or HTML table) string."""
    kind = plot.get("kind")
    if kind == "hist":
        edges = plot.get("bin_edges") or []
        counts = plot.get("counts") or []
        if len(edges) < 2 or not counts:
            return ""
        edges = np.asarray(edges, float)
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges)
        fig, ax = plt.subplots(figsize=(4.2, 2.4))
        ax.bar(centers, counts, width=widths, align="center", color="#0279EE", edgecolor="none")
        ax.set_xlabel(plot.get("xlabel", ""))
        ax.set_ylabel("spots" if "spot" in plot.get("xlabel", "") else "count")
        ax.spines[["top", "right"]].set_visible(False)
        return f'<img alt="histogram" src="{_fig_to_base64(fig)}">'
    if kind == "heatmap":
        grid = plot.get("grid") or []
        arr = np.array([[np.nan if v is None else v for v in row] for row in grid], dtype=float)
        if arr.size == 0:
            return ""
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        im = ax.imshow(arr, origin="lower", cmap="viridis", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8, label="mean counts")
        return f'<img alt="spatial heatmap" src="{_fig_to_base64(fig)}">'
    if kind == "table" and "top_genes" in plot:
        rows = "".join(
            f"<tr><td>{_qc_esc(g['gene'])}</td><td>{g['total']:.0f}</td>"
            f"<td>{g['pct']*100:.2f}%</td></tr>"
            for g in plot["top_genes"]
        )
        return ("<table class='mini'><tr><th>gene</th><th>total counts</th>"
                f"<th>% of counts</th></tr>{rows}</table>")
    if kind == "table" and "rows" in plot:
        rows = "".join(
            f"<tr><td>{_qc_esc(k)}</td><td>{_qc_esc(v)}</td></tr>"
            for k, v in plot["rows"].items()
        )
        return f"<table class='mini'><tr><th>metric</th><th>value</th></tr>{rows}</table>"
    return ""


def generate_qc_report(qc: dict, output_dir: str) -> str:
    """Write the QC report as JSON + summary.txt + a FastQC-style HTML page.

    The HTML mirrors FastQC: a left summary panel of PASS/WARN/FAIL modules and
    a main column of per-module sections, each with its plot and interpretation.
    Returns the path to the HTML file.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "qc_report.json"), "w") as f:
        json.dump(qc, f, default=_json_default)

    modules = qc.get("modules", [])
    summary = qc.get("summary", {})
    overall = summary.get("overall", "warn")
    basic = qc.get("basic", {})

    # FastQC-style summary.txt (status<TAB>module)
    with open(os.path.join(output_dir, "qc_summary.txt"), "w") as f:
        for m in modules:
            f.write(f"{m['status'].upper()}\t{m['name']}\n")

    # Left summary nav + right module sections.
    nav = "".join(
        f'<a href="#{m["id"]}" class="nav-item">'
        f'<span class="dot" style="background:{_QC_COLORS.get(m["status"], "#888")}">'
        f'{_QC_ICON.get(m["status"], "?")}</span>{_qc_esc(m["name"])}</a>'
        for m in modules
    )

    sections = ""
    for m in modules:
        st = m.get("status", "warn")
        body = _qc_render_plot(m["plot"]) if m.get("plot") else ""
        thr = ""
        if "thresholds" in m and m.get("value") is not None:
            thr = (f'<span class="thr">value={m["value"]:.4g} '
                   f'(warn {m["thresholds"]["warn"]}, fail {m["thresholds"]["fail"]})</span>')
        sections += (
            f'<section id="{m["id"]}" class="module">'
            f'<h3><span class="badge" style="background:{_QC_COLORS.get(st, "#888")}">'
            f'{_QC_ICON.get(st, "?")} {st.upper()}</span> {_qc_esc(m["name"])}</h3>'
            f'<p class="msg">{_qc_esc(m.get("message", ""))} {thr}</p>'
            f'<div class="plot">{body}</div></section>'
        )

    version = qc.get("version", "")
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>iSpot QC Report</title>
<style>
 :root{{--pass:#75A025;--warn:#E0A800;--fail:#D62728}}
 body{{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,sans-serif;color:#1a1a1a}}
 header{{background:#111;color:#fff;padding:16px 24px}}
 header h1{{margin:0;font-size:18px}} header .sub{{opacity:.8;font-size:12px;margin-top:4px}}
 .overall{{display:inline-block;padding:3px 10px;border-radius:5px;font-weight:700;margin-left:8px;
   background:{_QC_COLORS.get(overall,'#888')}}}
 .layout{{display:flex;align-items:flex-start}}
 nav{{width:280px;flex:0 0 280px;border-right:1px solid #eee;padding:12px 0;position:sticky;top:0}}
 .nav-item{{display:flex;align-items:center;gap:8px;padding:6px 16px;font-size:13px;color:#222;text-decoration:none}}
 .nav-item:hover{{background:#f5f5f5}}
 .dot{{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:50%;
   color:#fff;font-size:11px;font-weight:700}}
 main{{flex:1;padding:12px 24px;max-width:760px}}
 .module{{border-bottom:1px solid #eee;padding:14px 0}}
 .module h3{{font-size:15px;margin:0 0 6px;display:flex;align-items:center;gap:8px}}
 .badge{{color:#fff;font-size:11px;font-weight:700;padding:2px 8px;border-radius:4px}}
 .msg{{font-size:13px;color:#444;margin:0 0 8px}} .thr{{color:#888;margin-left:6px}}
 table.mini{{border-collapse:collapse;font-size:12px}}
 table.mini td,table.mini th{{border:1px solid #e6e6e6;padding:4px 8px;text-align:left}}
 img{{max-width:100%;height:auto}}
</style></head><body>
<header>
  <h1>iSpot QC Report <span class="overall">{overall.upper()}</span></h1>
  <div class="sub">Spatial-omics quality control &middot; {_qc_esc(basic.get('platform') or 'unknown platform')}
   &middot; {basic.get('n_spots','?')} spots &times; {basic.get('n_genes','?')} genes
   &middot; {summary.get('pass',0)} pass / {summary.get('warn',0)} warn / {summary.get('fail',0)} fail
   &middot; iSpot {_qc_esc(version)}</div>
</header>
<div class="layout">
  <nav>{nav}</nav>
  <main>
    <p style="font-size:12px;color:#666">QC metrics are advisory, computed on the raw uploaded data
     before preprocessing. Thresholds are heuristic defaults for spatial transcriptomics.</p>
    {sections}
  </main>
</div>
</body></html>
"""
    html_path = os.path.join(output_dir, "qc_report.html")
    with open(html_path, "w") as f:
        f.write(html)
    return html_path


def generate_report(
    results: pd.DataFrame,
    ranking_table_path: str,
    figure_paths: list[str],
    has_ground_truth: bool,
    data_profile: dict,
    n_clusters: int,
    output_path: str,
    statistical_results: pd.DataFrame | None = None,
) -> str:
    """Generate a PDF report summarizing the benchmark.

    Parameters
    ----------
    results : pd.DataFrame
        Full benchmark results.
    ranking_table_path : str
        Path to the ranking table CSV.
    figure_paths : list[str]
        Paths to figure PNG files.
    has_ground_truth : bool
    data_profile : dict
        DataFeatureVector as dict.
    n_clusters : int
        Number of clusters used.
    output_path : str
        Path for the PDF file.
    statistical_results : pd.DataFrame, optional
        Pairwise statistical comparison results.

    Returns
    -------
    str: path to the PDF file.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
        PageBreak,
    )
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        rightMargin=0.75 * inch, leftMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"],
        fontSize=18, spaceAfter=12, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading", parent=styles["Heading2"],
        fontSize=14, spaceBefore=12, spaceAfter=6,
    ))

    story = []

    # Title
    story.append(Paragraph("iSpot Benchmark Report", styles["ReportTitle"]))
    story.append(Spacer(1, 0.2 * inch))

    # Executive Summary
    story.append(Paragraph("Executive Summary", styles["SectionHeading"]))
    ranking = pd.read_csv(ranking_table_path)
    best_method = ranking.iloc[0]["method"]
    best_score = ranking.iloc[0].get("ARI (mean)", ranking.iloc[0].get("NoGTScore", 0))

    score_label = "ARI" if has_ground_truth else "No-GT composite score"
    summary_text = (
        f"This report summarizes a benchmark of {len(ranking)} clustering methods "
        f"on a {data_profile.get('platform', 'unknown')} dataset with "
        f"{data_profile.get('n_spots', 'unknown')} spots and "
        f"{data_profile.get('n_genes', 'unknown')} genes. "
        f"The estimated number of clusters was {n_clusters}. "
        f"<b>{best_method}</b> achieved the highest {score_label} of {best_score:.4f}."
    )
    if not has_ground_truth:
        summary_text += (
            " Note: no ground-truth annotations were provided. "
            "The No-GT composite score is a proxy based on spatial coherence, "
            "cluster stability, expression separability, and consensus alignment. "
            "It should be interpreted as a relative ranking, not an absolute quality measure."
        )
    story.append(Paragraph(summary_text, styles["Normal"]))
    story.append(Spacer(1, 0.15 * inch))

    # Data Profile
    story.append(Paragraph("Data Profile", styles["SectionHeading"]))
    profile_rows = [
        ["Platform", str(data_profile.get("platform", "unknown"))],
        ["Spots", str(data_profile.get("n_spots", "unknown"))],
        ["Genes", str(data_profile.get("n_genes", "unknown"))],
        ["Sparsity", f"{data_profile.get('sparsity', 0):.4f}"],
        ["Spatial layout", str(data_profile.get("spatial_layout", "unknown"))],
        ["Clusters (K)", str(n_clusters)],
        ["Ground truth", "Available" if has_ground_truth else "Not available"],
    ]
    profile_table = Table(profile_rows, colWidths=[2 * inch, 3 * inch])
    profile_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ECE9E2")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#000000")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(profile_table)
    story.append(Spacer(1, 0.15 * inch))

    # Ranking Table
    story.append(Paragraph("Method Ranking", styles["SectionHeading"]))
    ranking_display = ranking.copy()
    # Format numeric columns
    for col in ranking_display.columns:
        if ranking_display[col].dtype in [np.float64, np.float32]:
            ranking_display[col] = ranking_display[col].apply(lambda x: f"{x:.4f}")
    ranking_data = [ranking_display.columns.tolist()] + ranking_display.values.tolist()
    ranking_table_pdf = Table(ranking_data, repeatRows=1)
    ranking_table_pdf.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0279EE")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAF9F3")]),
    ]))
    story.append(ranking_table_pdf)
    story.append(Spacer(1, 0.15 * inch))

    # Figures
    story.append(PageBreak())
    story.append(Paragraph("Figures", styles["SectionHeading"]))
    for fig_path in figure_paths:
        if os.path.exists(fig_path):
            img = Image(fig_path, width=6 * inch, height=4.5 * inch)
            story.append(img)
            story.append(Paragraph(
                os.path.basename(fig_path).replace("_", " ").replace(".png", ""),
                styles["Italic"],
            ))
            story.append(Spacer(1, 0.1 * inch))

    # Statistical Comparison
    if statistical_results is not None and len(statistical_results) > 0:
        story.append(PageBreak())
        story.append(Paragraph("Statistical Comparison", styles["SectionHeading"]))
        story.append(Paragraph(
            "Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction. "
            "Only significant comparisons (p &lt; 0.05) are shown.",
            styles["Normal"],
        ))
        sig = statistical_results[statistical_results.get("significant", False)] if "significant" in statistical_results.columns else statistical_results
        if len(sig) > 0:
            sig_data = [sig.columns.tolist()] + sig.head(20).values.tolist()
            sig_table = Table(sig_data, repeatRows=1)
            sig_table.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0279EE")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ]))
            story.append(sig_table)

    # Limitations
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Limitations", styles["SectionHeading"]))
    limitations = (
        "1. The benchmark evaluates clustering quality using "
        + ("ARI against ground-truth annotations." if has_ground_truth else "a composite proxy score (no ground truth available).")
        + "<br/>2. Results may vary with different preprocessing parameters or cluster counts.<br/>"
        "3. Runtime measurements reflect CPU-only execution and may differ on GPU-enabled hardware.<br/>"
        "4. The benchmark does not assess biological interpretability of the resulting clusters."
    )
    story.append(Paragraph(limitations, styles["Normal"]))

    # Reproducibility
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Reproducibility", styles["SectionHeading"]))
    repro_text = (
        f"Methods were run with {results['seed'].nunique()} seed(s). "
        f"Preprocessing: filter (min_genes=200, min_cells=3), "
        f"normalize to 1e4, log1p, HVG selection (3000 genes), PCA (50 components). "
        f"Cluster count: {n_clusters}. "
        f"Platform: iSpot v1.0."
    )
    story.append(Paragraph(repro_text, styles["Normal"]))

    doc.build(story)
    return output_path
