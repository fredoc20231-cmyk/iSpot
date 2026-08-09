"""
Data profiling: extract a DataFeatureVector from an AnnData.

This module characterizes a spatial transcriptomics dataset before any
method runs. The feature vector feeds the meta-learning engine and
determines compute dispatch.

Section 1.2 of the platform plan.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree


@dataclass
class DataFeatureVector:
    """Feature vector characterizing a spatial transcriptomics dataset.

    Used by the meta-learning engine to predict method performance.
    """
    n_spots: int
    n_genes: int
    sparsity: float
    median_genes_per_spot: float
    median_counts_per_spot: float
    spatial_layout: str          # "hexagonal" | "square" | "random"
    spot_diameter_um: float
    has_histology: bool
    platform: str
    tissue_type: Optional[str]
    n_expected_clusters: Optional[int]
    spatial_extent: float        # max pairwise distance
    coordinate_density: float    # spots per unit area

    def to_dict(self) -> dict:
        return asdict(self)

    def to_feature_array(self, feature_order: list[str] | None = None) -> np.ndarray:
        """Return numeric features as a 1D array for ML models.

        Categorical fields (spatial_layout, platform, tissue_type) are
        one-hot encoded by the caller; this returns only the numeric fields.
        """
        numeric_fields = [
            "n_spots", "n_genes", "sparsity", "median_genes_per_spot",
            "median_counts_per_spot", "spot_diameter_um", "has_histology",
            "n_expected_clusters", "spatial_extent", "coordinate_density",
        ]
        if feature_order is not None:
            numeric_fields = feature_order
        return np.array([float(getattr(self, f) or 0.0) for f in numeric_fields])


def detect_spatial_layout(coords: np.ndarray) -> str:
    """Auto-detect spatial layout from spot coordinates.

    Parameters
    ----------
    coords : np.ndarray, shape (n_spots, 2)
        Spatial coordinates.

    Returns
    -------
    str: "hexagonal", "square", or "random"

    Method:
        Compute kNN distances (k=2) for all spots. On a hexagonal grid,
        the nearest-neighbor distance is constant and the second-nearest
        is ~1.73x (sqrt(3)). On a square grid, both nearest and
        second-nearest are the same distance. On random layouts, distances
        are broadly distributed.
    """
    if len(coords) < 10:
        return "random"

    tree = cKDTree(coords)
    # Query 3 nearest neighbors (excluding self)
    dists, _ = tree.query(coords, k=3)
    nn1 = dists[:, 1]  # nearest neighbor
    nn2 = dists[:, 2]  # second nearest

    # Coefficient of variation of nearest-neighbor distance
    cv_nn1 = np.std(nn1) / (np.mean(nn1) + 1e-10)

    if cv_nn1 < 0.05:
        # Very regular grid — distinguish hexagonal from square
        ratio = np.median(nn2) / (np.median(nn1) + 1e-10)
        if 1.6 < ratio < 1.9:
            return "hexagonal"
        elif 0.95 < ratio < 1.1:
            return "square"
        else:
            return "square"  # default for regular grids
    elif cv_nn1 < 0.3:
        return "square"  # semi-regular
    else:
        return "random"


def estimate_spot_diameter(coords: np.ndarray, layout: str) -> float:
    """Estimate spot diameter / center-to-center spacing in coordinate units.

    For grid layouts, this is the nearest-neighbor distance.
    For random layouts, this is the median nearest-neighbor distance.
    """
    if len(coords) < 2:
        return 0.0
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    nn1 = dists[:, 1]
    return float(np.median(nn1))


def compute_coordinate_density(coords: np.ndarray) -> float:
    """Compute spots per unit area.

    Uses the convex hull area as the spatial extent.
    """
    if len(coords) < 3:
        return 0.0
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(coords)
        area = hull.volume  # 2D: volume = area
        if area > 0:
            return float(len(coords) / area)
    except Exception:
        pass
    return 0.0


def compute_sparsity(adata: ad.AnnData) -> float:
    """Compute the fraction of zeros in the expression matrix."""
    X = adata.X
    if hasattr(X, "toarray"):
        # Sparse matrix
        n_elements = X.shape[0] * X.shape[1]
        n_nonzero = X.nnz
        return 1.0 - (n_nonzero / n_elements)
    else:
        return float(np.mean(X == 0))


def profile_data(
    adata: ad.AnnData,
    platform: str | None = None,
    tissue_type: str | None = None,
    n_expected_clusters: int | None = None,
) -> DataFeatureVector:
    """Extract a DataFeatureVector from an AnnData.

    Parameters
    ----------
    adata : AnnData
        Raw or preprocessed data. Must have .obsm['spatial'].
    platform : str, optional
        Platform name. If None, read from adata.uns['platform'] or "unknown".
    tissue_type : str, optional
        Tissue type. If None, read from adata.uns.get('tissue_type').
    n_expected_clusters : int, optional
        Expected number of clusters. If None, read from adata.uns.

    Returns
    -------
    DataFeatureVector
    """
    coords = np.array(adata.obsm["spatial"])

    # Spatial layout detection
    layout = detect_spatial_layout(coords)
    # Override from metadata if available
    if "spatial_layout" in adata.uns:
        layout = adata.uns["spatial_layout"]

    # Spot diameter
    spot_diameter = estimate_spot_diameter(coords, layout)

    # Spatial extent (max pairwise distance — subsample for large datasets)
    if len(coords) > 5000:
        idx = np.random.choice(len(coords), 5000, replace=False)
        extent = float(np.max(pdist(coords[idx])))
    else:
        extent = float(np.max(pdist(coords)))

    # Coordinate density
    density = compute_coordinate_density(coords)

    # Expression stats
    sparsity = compute_sparsity(adata)

    # Genes per spot (non-zero genes)
    X = adata.X
    if hasattr(X, "toarray"):
        genes_per_spot = np.array((X > 0).sum(axis=1)).ravel()
        counts_per_spot = np.array(X.sum(axis=1)).ravel()
    else:
        genes_per_spot = (np.asarray(X) > 0).sum(axis=1)
        counts_per_spot = np.asarray(X).sum(axis=1)

    median_genes = float(np.median(genes_per_spot))
    median_counts = float(np.median(counts_per_spot))

    # Platform
    if platform is None:
        platform = adata.uns.get("platform", "unknown")

    # Tissue type
    if tissue_type is None:
        tissue_type = adata.uns.get("tissue_type", None)

    # Expected clusters
    if n_expected_clusters is None:
        n_expected_clusters = adata.uns.get("n_expected_clusters", None)

    # Histology
    has_histology = "img" in adata.uns or "images" in adata.uns

    return DataFeatureVector(
        n_spots=int(adata.shape[0]),
        n_genes=int(adata.shape[1]),
        sparsity=float(sparsity),
        median_genes_per_spot=median_genes,
        median_counts_per_spot=median_counts,
        spatial_layout=layout,
        spot_diameter_um=float(spot_diameter),
        has_histology=bool(has_histology),
        platform=str(platform),
        tissue_type=tissue_type,
        n_expected_clusters=n_expected_clusters,
        spatial_extent=extent,
        coordinate_density=density,
    )
