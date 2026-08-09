"""
Evaluation metrics for spatial transcriptomics clustering benchmark.
Computes ARI, macro F1, weighted F1, and runtime.

Uses Hungarian label matching (scipy.optimize.linear_sum_assignment) to align
predicted cluster labels to ground truth labels before computing F1, exactly
as the author's benchmarking code does. This is the standard approach for
unsupervised clustering evaluation where cluster IDs are arbitrary.

This module is ported from the verified st_benchmark/metrics.py implementation.
The Hungarian alignment bug was caught and fixed here — do not bypass
ispot.metrics.evaluate() by computing F1 directly.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, f1_score
from scipy.optimize import linear_sum_assignment


def match_clusters_to_labels(true_labels, pred_labels):
    """Align predicted cluster labels to ground truth via Hungarian assignment.

    Uses scipy.optimize.linear_sum_assignment on a cost matrix of negative
    overlap counts, matching each predicted cluster to its best-overlapping
    true label. Unmatched predicted clusters get the label "unmatched".
    """
    true_labels = pd.Series(true_labels).astype(str)
    pred_labels = pd.Series(pred_labels).astype(str)

    true_classes = true_labels.unique()
    pred_classes = pred_labels.unique()

    cost_matrix = np.zeros((len(pred_classes), len(true_classes)))
    for i, p in enumerate(pred_classes):
        for j, t in enumerate(true_classes):
            overlap = ((pred_labels == p) & (true_labels == t)).sum()
            cost_matrix[i, j] = -overlap

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {pred_classes[r]: true_classes[c] for r, c in zip(row_ind, col_ind)}

    for p in pred_classes:
        if p not in mapping:
            mapping[p] = "unmatched"

    pred_mapped = pred_labels.map(mapping)
    return pred_mapped


def compute_metrics(ground_truth, predicted, runtime_sec=None):
    """Compute clustering evaluation metrics.

    Filters out NaN/None predictions before computing metrics.
    Uses Hungarian label matching for F1 (matching author's code).

    Returns
    -------
    dict with keys: ari, macro_f1, weighted_f1, runtime, n_spots,
                    n_clusters_pred, n_clusters_true
    """
    gt = np.array(ground_truth)
    pred = np.array(predicted)

    valid_mask = pd.notna(pred) & (pred != "nan") & (pred != "None")
    gt = gt[valid_mask]
    pred = pred[valid_mask]

    assert len(gt) == len(pred), f"Length mismatch: gt={len(gt)}, pred={len(pred)}"
    assert len(gt) > 0, "Empty labels after filtering"

    gt = np.array([str(x) for x in gt])
    pred = np.array([str(x) for x in pred])

    ari = adjusted_rand_score(gt, pred)

    pred_mapped = match_clusters_to_labels(gt, pred)
    macro_f1 = f1_score(gt, pred_mapped, average="macro", zero_division=0)
    weighted_f1 = f1_score(gt, pred_mapped, average="weighted", zero_division=0)

    return {
        "ari": float(ari),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "runtime": float(runtime_sec) if runtime_sec is not None else None,
        "n_spots": int(len(gt)),
        "n_clusters_pred": int(len(np.unique(pred))),
        "n_clusters_true": int(len(np.unique(gt))),
    }


def res_search_fixed_clus(adata, fixed_clus_count, low=0.01, high=2.5, guess=1,
                           increment=0.01, max_iter=50, cluster_key="leiden"):
    """Binary search for leiden resolution giving exactly fixed_clus_count clusters."""
    import scanpy as sc
    res = guess
    for _ in range(max_iter):
        sc.tl.leiden(adata, random_state=0, resolution=res, key_added=cluster_key)
        count_unique = len(pd.DataFrame(adata.obs[cluster_key])[cluster_key].unique())

        if count_unique == fixed_clus_count:
            return res

        if count_unique < fixed_clus_count:
            low = res
        else:
            high = res

        if high - low < increment:
            break

        res = (low + high) / 2

    return res


def evaluate(adata, cluster_key, gt_key="ground_truth", valid_key="has_ground_truth",
             runtime_sec=None, drop_na=True):
    """Evaluate clustering results stored in adata.obs.

    Filters to spots with valid ground truth before computing metrics.
    This is the canonical entry point — always use this, never compute F1
    directly, to ensure Hungarian label alignment is applied.
    """
    if drop_na:
        mask = adata.obs[valid_key].values.astype(bool)
    else:
        mask = np.ones(len(adata), dtype=bool)

    gt = adata.obs.loc[mask, gt_key].values
    pred = adata.obs.loc[mask, cluster_key].values
    return compute_metrics(gt, pred, runtime_sec)
