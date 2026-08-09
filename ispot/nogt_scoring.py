"""
No-ground-truth scoring engine.

When users have no annotations, we cannot compute ARI. Instead we use a
multi-criteria composite score from four proxy metrics:

  A. Spatial Coherence Score (SCS) — weight 0.35
     Moran's I of cluster indicator variables. Measures whether clusters
     form contiguous spatial regions.

  B. Cluster Stability Score (CSS) — weight 0.25
     Mean pairwise ARI across random seeds. Measures reproducibility.

  C. Expression Separability Score (ESS) — weight 0.20
     Silhouette score in PCA space. Measures cluster separability.

  D. Consensus Alignment Score (CAS) — weight 0.20
     ARI between a method's labels and the consensus of all methods.

Composite: NoGTScore = 0.35*SCS + 0.25*CSS + 0.20*ESS + 0.20*CAS

Section 1.5.2 of the platform plan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# A. Spatial Coherence Score (SCS)
# ---------------------------------------------------------------------------

def _build_spatial_weights(coords: np.ndarray, k: int = 6) -> np.ndarray:
    """Build a row-normalized kNN spatial weights matrix.

    Parameters
    ----------
    coords : np.ndarray, shape (n_spots, 2)
    k : int
        Number of nearest neighbors. Default 6 (hexagonal grid).

    Returns
    -------
    np.ndarray, shape (n_spots, n_spots)
        Row-normalized weights matrix (sparse-friendly in future).
    """
    n = len(coords)
    k = min(k, n - 1)
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k + 1)  # +1 because self is included

    # Build dense weights matrix (OK for Visium-scale ~50k spots)
    # For larger datasets, use scipy.sparse
    if n > 20000:
        return _build_spatial_weights_sparse(coords, k, indices)

    W = np.zeros((n, n))
    for i in range(n):
        neighbors = indices[i, 1:]  # exclude self
        W[i, neighbors] = 1.0
    # Row-normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W = W / row_sums
    return W


def _build_spatial_weights_sparse(coords: np.ndarray, k: int, indices: np.ndarray):
    """Sparse version for large datasets."""
    import scipy.sparse as sp
    n = len(coords)
    rows = []
    cols = []
    vals = []
    for i in range(n):
        neighbors = indices[i, 1:]
        for j in neighbors:
            rows.append(i)
            cols.append(j)
            vals.append(1.0)
    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    # Row-normalize
    row_sums = np.array(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    D = sp.diags(1.0 / row_sums)
    W = D @ W
    return W


def _morans_i(values: np.ndarray, W: np.ndarray) -> float:
    """Compute Moran's I for a variable.

    Moran's I = (n / S0) * (sum_i sum_j w_ij * (x_i - x_bar) * (x_j - x_bar))
              / (sum_i (x_i - x_bar)^2)

    where S0 = sum of all weights.

    Parameters
    ----------
    values : np.ndarray, shape (n_spots,)
        Binary indicator (1 if spot in cluster c, 0 otherwise).
    W : np.ndarray or sparse matrix
        Row-normalized spatial weights matrix.

    Returns
    -------
    float: Moran's I, typically in [-1, 1]. Positive = spatially clustered.
    """
    n = len(values)
    x = values.astype(float)
    x_bar = np.mean(x)
    dev = x - x_bar

    # S0 = sum of all weights
    if hasattr(W, "sum"):
        S0 = W.sum()
    else:
        S0 = np.sum(W)

    if S0 == 0 or np.sum(dev ** 2) == 0:
        return 0.0

    # Numerator: sum_i sum_j w_ij * dev_i * dev_j
    if hasattr(W, "dot"):
        # Sparse matrix
        Wdev = W.dot(dev)
        numerator = n * np.sum(dev * Wdev)
    else:
        Wdev = W @ dev
        numerator = n * np.sum(dev * Wdev)

    denominator = S0 * np.sum(dev ** 2)
    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def spatial_coherence_score(
    labels: np.ndarray,
    coords: np.ndarray,
    k: int = 6,
) -> float:
    """Compute the Spatial Coherence Score (SCS).

    For each cluster, compute Moran's I of the binary indicator variable.
    Weight by cluster size. Normalize to [0, 1].

    Parameters
    ----------
    labels : np.ndarray, shape (n_spots,)
        Predicted cluster labels.
    coords : np.ndarray, shape (n_spots, 2)
        Spatial coordinates.
    k : int
        Number of spatial neighbors.

    Returns
    -------
    float: SCS in [0, 1]. Higher = more spatially coherent clusters.
    """
    labels = np.array(labels).astype(str)
    n = len(labels)
    if n < 10:
        return 0.5  # too few spots for meaningful spatial statistics

    W = _build_spatial_weights(coords, k=k)

    clusters = np.unique(labels)
    if len(clusters) <= 1:
        return 0.5  # degenerate: all one cluster

    weighted_morans = []
    weights = []

    for c in clusters:
        binary = (labels == c).astype(float)
        if binary.sum() < 2 or binary.sum() > n - 2:
            # Skip clusters that are too small or too large
            continue
        mi = _morans_i(binary, W)
        weighted_morans.append(mi)
        weights.append(binary.sum())

    if not weighted_morans:
        return 0.5

    # Weighted mean of Moran's I
    weights = np.array(weights, dtype=float)
    weighted_morans = np.array(weighted_morans)
    mean_morans = np.average(weighted_morans, weights=weights)

    # Normalize from [-1, 1] to [0, 1]
    scs = (mean_morans + 1.0) / 2.0
    return float(np.clip(scs, 0.0, 1.0))


# ---------------------------------------------------------------------------
# B. Cluster Stability Score (CSS)
# ---------------------------------------------------------------------------

def cluster_stability_score(label_runs: list[np.ndarray]) -> float:
    """Compute the Cluster Stability Score (CSS).

    Mean pairwise ARI between clustering results from different seeds.

    Parameters
    ----------
    label_runs : list of np.ndarray
        List of label arrays, one per seed. Each shape (n_spots,).

    Returns
    -------
    float: CSS in [0, 1]. Higher = more stable across seeds.
    """
    if len(label_runs) < 2:
        return 1.0  # single run: trivially stable (but not informative)

    aris = []
    for i in range(len(label_runs)):
        for j in range(i + 1, len(label_runs)):
            ari = adjusted_rand_score(
                np.array(label_runs[i]).astype(str),
                np.array(label_runs[j]).astype(str),
            )
            aris.append(ari)

    css = float(np.mean(aris))
    return max(0.0, css)  # clip negative ARI to 0


# ---------------------------------------------------------------------------
# C. Expression Separability Score (ESS)
# ---------------------------------------------------------------------------

def expression_separability_score(
    labels: np.ndarray,
    X_pca: np.ndarray,
    sample_size: int = 10000,
    random_state: int = 42,
) -> float:
    """Compute the Expression Separability Score (ESS).

    Silhouette score in PCA space, normalized to [0, 1].

    Parameters
    ----------
    labels : np.ndarray, shape (n_spots,)
    X_pca : np.ndarray, shape (n_spots, n_pcs)
        PCA representation.
    sample_size : int
        Subsample for large datasets (silhouette is O(n^2)).

    Returns
    -------
    float: ESS in [0, 1]. Higher = more separable clusters.
    """
    labels = np.array(labels).astype(str)
    n = len(labels)

    if len(np.unique(labels)) <= 1:
        return 0.5  # degenerate

    # Subsample for efficiency
    if n > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, sample_size, replace=False)
        labels = labels[idx]
        X_pca = X_pca[idx]

    try:
        sil = silhouette_score(X_pca, labels, metric="euclidean")
    except Exception:
        return 0.5

    # Normalize from [-1, 1] to [0, 1]
    ess = (sil + 1.0) / 2.0
    return float(np.clip(ess, 0.0, 1.0))


# ---------------------------------------------------------------------------
# D. Consensus Alignment Score (CAS)
# ---------------------------------------------------------------------------

def consensus_clustering(
    label_runs: dict[str, np.ndarray],
    n_clusters: int,
) -> np.ndarray:
    """Build consensus labels from multiple methods via co-association matrix.

    Parameters
    ----------
    label_runs : dict[str, np.ndarray]
        Method name -> label array. All arrays must have same length.
    n_clusters : int
        Number of clusters for spectral clustering of the co-association matrix.

    Returns
    -------
    np.ndarray: consensus cluster labels.
    """
    from sklearn.cluster import SpectralClustering

    methods = list(label_runs.keys())
    n = len(label_runs[methods[0]])
    M = len(methods)

    # Build co-association matrix
    # C[i,j] = fraction of methods that assign i and j to the same cluster
    labels_matrix = np.zeros((n, M), dtype=int)
    for m_idx, method in enumerate(methods):
        labels = np.array(label_runs[method]).astype(str)
        # Map string labels to integers
        unique_labels = {label: idx for idx, label in enumerate(np.unique(labels))}
        labels_matrix[:, m_idx] = [unique_labels[l] for l in labels]

    # Compute co-association in batches for memory efficiency
    if n > 5000:
        return _consensus_batched(labels_matrix, M, n_clusters)

    C = np.zeros((n, n))
    for m in range(M):
        same = (labels_matrix[:, m:m+1] == labels_matrix[:, m:m+1].T)
        C += same.astype(float)
    C /= M

    # Spectral clustering on the co-association matrix
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=42,
        assign_labels="kmeans",
    )
    consensus_labels = sc.fit_predict(C)
    return consensus_labels


def _consensus_batched(labels_matrix: np.ndarray, M: int, n_clusters: int) -> np.ndarray:
    """Batched consensus for large datasets."""
    from sklearn.cluster import SpectralClustering
    n = labels_matrix.shape[0]

    # Build sparse co-association matrix
    import scipy.sparse as sp
    # For each method, build a sparse same-cluster matrix
    C = sp.csr_matrix((n, n), dtype=float)
    for m in range(M):
        labels = labels_matrix[:, m]
        # For each pair in the same cluster, add 1/M
        for cluster_id in np.unique(labels):
            members = np.where(labels == cluster_id)[0]
            if len(members) < 2:
                continue
            # Create row/col indices for all pairs
            rows, cols = np.meshgrid(members, members, indexing="ij")
            C = C + sp.csr_matrix(
                (np.full(len(rows.ravel()), 1.0 / M),
                 (rows.ravel(), cols.ravel())),
                shape=(n, n),
            )

    # Spectral clustering
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=42,
        assign_labels="kmeans",
    )
    consensus_labels = sc.fit_predict(C.toarray())
    return consensus_labels


def consensus_alignment_score(
    method_labels: np.ndarray,
    all_method_labels: dict[str, np.ndarray],
    n_clusters: int,
) -> float:
    """Compute the Consensus Alignment Score (CAS) for a single method.

    Parameters
    ----------
    method_labels : np.ndarray
        Labels from the method being scored.
    all_method_labels : dict[str, np.ndarray]
        All methods' labels (including the one being scored).
    n_clusters : int

    Returns
    -------
    float: CAS in [0, 1]. Higher = more aligned with consensus.
    """
    consensus = consensus_clustering(all_method_labels, n_clusters)
    ari = adjusted_rand_score(
        np.array(method_labels).astype(str),
        consensus.astype(str),
    )
    return max(0.0, float(ari))


# ---------------------------------------------------------------------------
# Composite No-GT Score
# ---------------------------------------------------------------------------

# Default weights (Section 1.5.2 of the plan)
DEFAULT_WEIGHTS = {
    "scs": 0.35,
    "css": 0.25,
    "ess": 0.20,
    "cas": 0.20,
}


def compute_nogt_score(
    labels: np.ndarray,
    label_runs: list[np.ndarray],
    coords: np.ndarray,
    X_pca: np.ndarray,
    all_method_labels: dict[str, np.ndarray],
    n_clusters: int,
    weights: dict[str, float] | None = None,
    k_spatial: int = 6,
) -> dict:
    """Compute the composite No-GT score and all component scores.

    Parameters
    ----------
    labels : np.ndarray
        Labels from the primary seed run (seed=42).
    label_runs : list[np.ndarray]
        Labels from all seed runs (for stability).
    coords : np.ndarray, shape (n_spots, 2)
        Spatial coordinates.
    X_pca : np.ndarray
        PCA representation for silhouette.
    all_method_labels : dict[str, np.ndarray]
        All methods' labels (for consensus).
    n_clusters : int
    weights : dict, optional
        Component weights. Defaults to DEFAULT_WEIGHTS.
    k_spatial : int
        Number of spatial neighbors for Moran's I.

    Returns
    -------
    dict with keys: nogt_score, scs, css, ess, cas, weights
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    scs = spatial_coherence_score(labels, coords, k=k_spatial)
    css = cluster_stability_score(label_runs)
    ess = expression_separability_score(labels, X_pca)
    cas = consensus_alignment_score(labels, all_method_labels, n_clusters)

    nogt = (
        weights["scs"] * scs
        + weights["css"] * css
        + weights["ess"] * ess
        + weights["cas"] * cas
    )

    return {
        "nogt_score": float(nogt),
        "scs": float(scs),
        "css": float(css),
        "ess": float(ess),
        "cas": float(cas),
        "weights": dict(weights),
    }
