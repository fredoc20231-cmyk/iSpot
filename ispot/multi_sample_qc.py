"""
Multi-sample SpatialQC aggregation — the MultiQC parallel for spatial-omics.

MultiQC (https://multiqc.info) became indispensable not by adding new metrics
but by *aggregating* per-sample FastQC outputs into one view, so a core facility
running dozens of samples sees batch structure and outliers at a glance instead
of opening N reports. This module does the same for SpatialQC: it consumes the
flat ``qc_summary.json`` records emitted by ``spatial_qc_report.summary_record``
and produces one combined report that flags samples inconsistent with the batch
(e.g. one Visium slide with unusually low tissue retention among an
otherwise-uniform run).

Deliberately pure Python (``statistics`` only, no numpy) so the aggregation and
outlier logic is trivially unit-testable and runs anywhere.
"""
from __future__ import annotations

import json
import os
import statistics
from typing import Iterable, Optional

# Numeric per-sample metrics we track for batch consistency. Each entry is
# (key, human label, direction) where direction indicates which tail is "bad":
#   "low"  -> unusually LOW values are the concern (retention, genes, signal)
#   "both" -> either tail is worth flagging (spot count, gene count)
#   "high" -> unusually HIGH values are the concern (sparsity)
METRICS = [
    ("median_genes_per_spot", "Median genes/spot", "low"),
    ("median_counts_per_spot", "Median counts/spot", "low"),
    ("sparsity", "Sparsity", "high"),
    ("tissue_retention_pct", "Tissue retention %", "low"),
    ("spatial_morans_i", "Spatial Moran's I", "low"),
    ("n_spots", "Spot count", "both"),
]

DEFAULT_Z_THRESHOLD = 2.0


def load_summaries(paths: Iterable[str]) -> list[dict]:
    """Load ``qc_summary.json`` records from file paths or directories.

    A path may be a summary JSON file, or a directory that (recursively) contains
    ``qc_summary.json`` files. Unreadable files are skipped.
    """
    records: list[dict] = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in files:
                    if f.endswith("qc_summary.json"):
                        rec = _read_json(os.path.join(root, f))
                        if rec is not None:
                            records.append(rec)
        else:
            rec = _read_json(p)
            if rec is not None:
                records.append(rec)
    return records


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def _finite(x):
    try:
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    except (TypeError, ValueError):
        return None


def aggregate_qc(records: list[dict], z_threshold: float = DEFAULT_Z_THRESHOLD) -> dict:
    """Aggregate per-sample SpatialQC records into a batch report.

    For each numeric metric, computes mean/std/min/max across samples and a
    per-sample z-score; a sample is an outlier on that metric when its z-score
    exceeds ``z_threshold`` in the *concerning* direction. Samples whose own
    SpatialQC ``overall`` is ``fail`` are always surfaced.

    Returns a dict with per-metric stats, per-sample rows (each carrying its
    z-scores and a flag), the flagged outliers, and batch status counts.
    """
    n = len(records)
    metrics_out: dict[str, dict] = {}
    # Precompute mean/std per metric.
    for key, label, direction in METRICS:
        values = [(_finite(r.get(key))) for r in records]
        present = [v for v in values if v is not None]
        if len(present) >= 1:
            mean = statistics.fmean(present)
            std = statistics.pstdev(present) if len(present) > 1 else 0.0
            metrics_out[key] = {
                "label": label, "direction": direction,
                "mean": mean, "std": std,
                "min": min(present), "max": max(present),
                "n_present": len(present),
            }
        else:
            metrics_out[key] = {
                "label": label, "direction": direction,
                "mean": None, "std": None, "min": None, "max": None, "n_present": 0,
            }

    samples: list[dict] = []
    outliers: list[dict] = []
    status_counts = {"pass": 0, "warn": 0, "fail": 0}

    for r in records:
        sid = r.get("sample_id", "-")
        overall = r.get("overall", "warn")
        status_counts[overall] = status_counts.get(overall, 0) + 1

        zscores: dict[str, float] = {}
        flags: list[str] = []
        for key, label, direction in METRICS:
            stats = metrics_out[key]
            v = _finite(r.get(key))
            if v is None or stats["std"] is None or stats["std"] == 0:
                continue
            z = (v - stats["mean"]) / stats["std"]
            zscores[key] = round(z, 3)
            concerning = (
                (direction == "low" and z <= -z_threshold) or
                (direction == "high" and z >= z_threshold) or
                (direction == "both" and abs(z) >= z_threshold)
            )
            if concerning:
                flags.append(key)
                outliers.append({
                    "sample_id": sid, "metric": key, "label": label,
                    "value": v, "z": round(z, 3), "batch_mean": round(stats["mean"], 4),
                    "reason": f"{label} = {v:g} is {abs(z):.1f}σ "
                              f"{'below' if z < 0 else 'above'} the batch mean "
                              f"({stats['mean']:g}).",
                })

        samples.append({
            "sample_id": sid,
            "platform": r.get("platform", "unknown"),
            "platform_confidence": r.get("platform_confidence"),
            "overall": overall,
            "metrics": {k: _finite(r.get(k)) for k, _l, _d in METRICS},
            "zscores": zscores,
            "flagged_metrics": flags,
            "is_outlier": bool(flags) or overall == "fail",
        })

    n_outlier_samples = sum(1 for s in samples if s["is_outlier"])
    return {
        "report": "SpatialQC-MultiSample",
        "n_samples": n,
        "z_threshold": z_threshold,
        "metrics": metrics_out,
        "samples": samples,
        "outliers": outliers,
        "n_outlier_samples": n_outlier_samples,
        "status_counts": status_counts,
        "batch_status": "fail" if status_counts.get("fail") else (
            "warn" if (n_outlier_samples or status_counts.get("warn")) else "pass"),
    }


# ---------------------------------------------------------------------------
# HTML rendering (self-contained, no external assets)
# ---------------------------------------------------------------------------
_COLORS = {"pass": "#3fae49", "warn": "#e8a33d", "fail": "#d64545"}
_ICON = {"pass": "✓", "warn": "!", "fail": "✕"}


def _fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def render_multi_sample_html(agg: dict, output_dir: str,
                             filename: str = "multi_sample_qc.html") -> str:
    """Render the aggregated batch report as one self-contained HTML file."""
    os.makedirs(output_dir, exist_ok=True)
    metrics = agg.get("metrics", {})
    metric_keys = [k for k, _l, _d in METRICS]
    sc = agg.get("status_counts", {})

    head = f"""<!doctype html><html><head><meta charset="utf-8">
<title>SpatialQC — Multi-sample report</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;color:#1a1a1a}}
 header{{background:#0b1f3a;color:#fff;padding:18px 26px}}
 header h1{{margin:0;font-size:19px}} header p{{margin:4px 0 0;color:#9fb3cc;font-size:13px}}
 .wrap{{padding:22px 26px}}
 .pills span{{display:inline-block;padding:4px 12px;border-radius:14px;color:#fff;font-size:12px;margin-right:8px}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:14px}}
 th,td{{border:1px solid #e2e2e2;padding:6px 9px;text-align:right}}
 th:first-child,td:first-child{{text-align:left}}
 th{{background:#f4f6f9}}
 .st{{font-weight:600;text-align:center;color:#fff;border-radius:4px;padding:2px 6px}}
 tr.outlier td{{background:#fff6f6}}
 .flag{{background:#ffd9d9;font-weight:600}}
 .muted{{color:#777}} h2{{font-size:15px;margin:24px 0 6px}}
 .out li{{margin:3px 0;font-size:13px}}
</style></head><body>
<header><h1>SpatialQC — Multi-sample report</h1>
<p>{agg.get('n_samples',0)} samples · {agg.get('n_outlier_samples',0)} flagged · batch status
<b style="color:{_COLORS.get(agg.get('batch_status','warn'))}">{agg.get('batch_status','warn').upper()}</b></p></header>
<div class="wrap">
<div class="pills">
 <span style="background:{_COLORS['pass']}">PASS {sc.get('pass',0)}</span>
 <span style="background:{_COLORS['warn']}">WARN {sc.get('warn',0)}</span>
 <span style="background:{_COLORS['fail']}">FAIL {sc.get('fail',0)}</span>
</div>
"""

    # Per-sample table
    header_cells = "".join(
        f"<th>{metrics.get(k, {}).get('label', k)}</th>" for k in metric_keys)
    rows = []
    for s in agg.get("samples", []):
        cls = " class=\"outlier\"" if s.get("is_outlier") else ""
        cells = []
        for k in metric_keys:
            flagged = k in s.get("flagged_metrics", [])
            cell_cls = ' class="flag"' if flagged else ""
            cells.append(f"<td{cell_cls}>{_fmt(s['metrics'].get(k))}</td>")
        st = s.get("overall", "warn")
        rows.append(
            f"<tr{cls}><td>{s.get('sample_id')}</td>"
            f"<td>{s.get('platform')}</td>"
            f"<td><span class='st' style='background:{_COLORS.get(st)}'>{_ICON.get(st,'?')} {st}</span></td>"
            + "".join(cells) + "</tr>")
    table = (f"<h2>Per-sample metrics</h2><table><tr><th>Sample</th><th>Platform</th>"
             f"<th>Status</th>{header_cells}</tr>{''.join(rows)}</table>")

    # Outliers list
    outs = agg.get("outliers", [])
    if outs:
        items = "".join(f"<li><b>{o['sample_id']}</b>: {o['reason']}</li>" for o in outs)
        outliers_html = f"<h2>Flagged outliers ({len(outs)})</h2><ul class='out'>{items}</ul>"
    else:
        outliers_html = "<h2>Flagged outliers</h2><p class='muted'>None — batch is consistent.</p>"

    html = head + table + outliers_html + "</div></body></html>"
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path
