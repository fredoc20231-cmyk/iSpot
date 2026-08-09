"""Helper for method runners to handle no-ground-truth cases.

When ground truth is not available, compute_metrics fails because the
GT-masked arrays are empty. This helper wraps the metrics computation
and returns None for metrics while preserving labels.
"""
import numpy as np
from ispot.metrics import compute_metrics


def safe_compute_metrics(adata, cluster_key, runtime, gt_key="ground_truth", valid_key="has_ground_truth"):
    """Compute metrics if GT is available, otherwise return None metrics.

    Always returns a dict with labels populated.
    """
    labels = adata.obs[cluster_key].values.astype(str)

    if valid_key in adata.obs.columns and adata.obs[valid_key].any():
        mask = adata.obs[valid_key].values.astype(bool)
        gt = adata.obs.loc[mask, gt_key].values
        pred = adata.obs.loc[mask, cluster_key].values
        m = compute_metrics(gt, pred, runtime)
    else:
        # No ground truth — return None metrics but valid labels
        m = {
            "ari": None,
            "macro_f1": None,
            "weighted_f1": None,
            "runtime": float(runtime) if runtime is not None else None,
            "n_spots": int(adata.shape[0]),
            "n_clusters_pred": int(adata.obs[cluster_key].nunique()),
            "n_clusters_true": None,
        }

    m["labels"] = labels
    return m
