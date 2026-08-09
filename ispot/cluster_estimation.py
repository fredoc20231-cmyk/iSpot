"""
Cluster count auto-estimation.

When the user doesn't specify the expected number of clusters K, we estimate
it using the "knee" (elbow) of the spatial coherence curve as Leiden
resolution increases. Spatial coherence (Moran's I) increases sharply as we
go from 1 cluster (no spatial structure) to a few clusters (spatial domains
emerge), then plateaus once the true domain count is reached. The knee of
this curve is the optimal K.

This is analogous to the elbow method in k-means but uses spatial coherence
instead of within-cluster sum of squares.

Section 1.4.2 of the platform plan.
"""
from __future__ import annotations

import numpy as np
import anndata as ad
import scanpy as sc
from sklearn.metrics import silhouette_score

from ispot.nogt_scoring import spatial_coherence_score


def _find_knee(ks: np.ndarray, scores: np.ndarray) -> int:
    """Find the knee (elbow) point in a monotonically decreasing curve.

    For spatial coherence vs. K, the curve decreases sharply at first
    (going from 2 to a few clusters breaks up large regions) then
    plateaus (adding more clusters beyond the true K doesn't reduce
    coherence much). The knee is where the decline transitions from
    steep to gentle.

    Method: compute the discrete derivative (differences between
    consecutive K values). The knee is the last K where the derivative
    is steeper than the average derivative — i.e., the point where
    further increases in K yield diminishing losses in coherence.

    Parameters
    ----------
    ks : np.ndarray
        Cluster counts (x-axis), sorted ascending.
    scores : np.ndarray
        Scores (y-axis), expected to be decreasing with K.

    Returns
    -------
    int: optimal K at the knee point.
    """
    if len(ks) < 3:
        return int(ks[len(ks) // 2])  # middle value as fallback

    # Compute differences (negative for decreasing curve)
    diffs = np.diff(scores)  # score[k+1] - score[k], should be negative

    # The knee is where the magnitude of the drop transitions from
    # large to small. Find the point where the cumulative drop
    # accounts for most of the total drop.
    total_drop = abs(scores[-1] - scores[0])
    if total_drop < 1e-6:
        return int(ks[len(ks) // 2])

    # Cumulative drop as fraction of total
    cum_drop = np.cumsum(np.abs(diffs))
    cum_frac = cum_drop / total_drop

    # The knee is the last K where we haven't yet captured 70% of the
    # total drop — i.e., the point where most of the coherence loss
    # has happened and further K increases are in the plateau region.
    # We use 70% as the threshold (captures the "elbow" region).
    threshold = 0.70
    knee_idx = 0
    for i, frac in enumerate(cum_frac):
        if frac < threshold:
            knee_idx = i
        else:
            break

    # knee_idx is the index in the diffs array; the corresponding K is ks[knee_idx + 1]
    # (the K at which we've just passed the threshold)
    return int(ks[knee_idx + 1])


def estimate_n_clusters(
    adata: ad.AnnData,
    resolutions: list[float] | None = None,
    k_range: tuple[int, int] = (2, 20),
    sample_size: int = 10000,
    random_state: int = 42,
    coords: np.ndarray | None = None,
) -> dict:
    """Estimate the optimal number of clusters via spatial coherence knee detection.

    Runs Leiden clustering at multiple resolutions. For each resulting K,
    computes spatial coherence (Moran's I of cluster indicators). Finds the
    knee of the spatial coherence vs. K curve — the point where adding more
    clusters stops providing substantial spatial coherence gains.

    Parameters
    ----------
    adata : AnnData
        Preprocessed data with .obsm['X_pca'] and neighbors graph.
    resolutions : list of float, optional
        Leiden resolutions to try. Default: fine-grained from 0.1 to 3.0.
    k_range : tuple[int, int]
        Valid range for cluster count (min, max).
    sample_size : int
        Subsample for silhouette computation (reported but not used for K selection).
    random_state : int
    coords : np.ndarray, optional
        Spatial coordinates. If None, uses adata.obsm['spatial'].

    Returns
    -------
    dict with keys:
        - n_clusters: estimated optimal K
        - spatial_coherence: SCS at optimal K
        - silhouette: silhouette score at optimal K (for reference)
        - all_results: list of per-K results
        - sensitivity: dict with results at K-1, K, K+1
    """
    if resolutions is None:
        # Fine-grained resolutions to get good K coverage
        resolutions = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                       0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

    if "X_pca" not in adata.obsm:
        raise ValueError("adata must have .obsm['X_pca'] — run preprocessing first")
    if "connectivities" not in adata.obsp:
        raise ValueError("adata must have neighbors graph — run preprocessing first")

    if coords is None:
        coords = np.array(adata.obsm["spatial"])

    X_pca = adata.obsm["X_pca"]
    n = adata.shape[0]

    use_idx = None
    if n > sample_size:
        rng = np.random.RandomState(random_state)
        use_idx = rng.choice(n, sample_size, replace=False)

    results = []
    per_k = {}  # k -> best result

    for res in resolutions:
        key = f"_est_{res}"
        sc.tl.leiden(adata, resolution=res, key_added=key, random_state=random_state)
        labels = adata.obs[key].values.astype(str)
        k = len(np.unique(labels))

        if k < k_range[0] or k > k_range[1]:
            continue

        # Spatial coherence (primary metric for knee detection)
        scs = spatial_coherence_score(labels, coords, k=6)

        # Silhouette (reported for reference)
        if use_idx is not None:
            sil = silhouette_score(X_pca[use_idx], labels[use_idx], metric="euclidean")
        else:
            try:
                sil = silhouette_score(X_pca, labels, metric="euclidean")
            except Exception:
                sil = 0.0

        result = {
            "n_clusters": k,
            "resolution": res,
            "spatial_coherence": float(scs),
            "silhouette": float(sil),
        }
        results.append(result)

        # Keep best SCS per K (in case multiple resolutions give same K)
        if k not in per_k or scs > per_k[k]["spatial_coherence"]:
            per_k[k] = result

    if not results:
        return {
            "n_clusters": 7,
            "spatial_coherence": 0.0,
            "silhouette": 0.0,
            "all_results": [],
            "sensitivity": {},
            "warning": "No valid cluster counts found in range; defaulting to 7.",
        }

    # Build sorted K vs SCS curve for knee detection
    sorted_ks = sorted(per_k.keys())
    scs_curve = np.array([per_k[k]["spatial_coherence"] for k in sorted_ks])
    ks_array = np.array(sorted_ks)

    # Find knee
    best_k = _find_knee(ks_array, scs_curve)

    # Sensitivity analysis
    sensitivity = {}
    for delta in [-1, 0, 1]:
        k_test = best_k + delta
        if k_test in per_k:
            label = f"K{delta:+d}" if delta != 0 else "K"
            sensitivity[label] = {
                "n_clusters": k_test,
                "spatial_coherence": per_k[k_test]["spatial_coherence"],
                "silhouette": per_k[k_test]["silhouette"],
            }

    # Clean up
    for res in resolutions:
        key = f"_est_{res}"
        if key in adata.obs.columns:
            del adata.obs[key]

    best_result = per_k[best_k]
    return {
        "n_clusters": int(best_k),
        "spatial_coherence": float(best_result["spatial_coherence"]),
        "silhouette": float(best_result["silhouette"]),
        "all_results": results,
        "sensitivity": sensitivity,
    }
