"""
Standardized preprocessing for all spatial transcriptomics datasets.

Every dataset, once loaded, goes through this identical pipeline:
  1. Filter cells (platform-aware min_genes threshold) and genes (min_cells=3)
  2. Normalize total to 1e4
  3. Log1p
  4. Highly variable gene selection (seurat_v3, n_top_genes=3000)
  5. PCA (50 components on HVGs)
  6. Neighbors graph (15 neighbors on PCA)

This guarantees every method sees identically-preprocessed data.
Original counts preserved in .layers['counts'].
"""
import scanpy as sc

# A single min_genes=200 threshold was previously applied to every platform.
# That's a reasonable bar for Visium, where each "spot" aggregates several
# cells and typically detects thousands of genes -- but imaging-based,
# single-cell-resolution platforms (Xenium, CosMx, MERFISH) use targeted
# panels of roughly 100-1,000 genes total, where detecting well under 200
# genes per cell is a normal, expected result of the technology, not a
# quality problem. Applying the Visium threshold there silently discards
# the majority of cells before any clustering method even runs (and can
# filter to ZERO cells, crashing log1p on an empty array).
PLATFORM_MIN_GENES = {
    "Visium": 200,
    "Slide-seqV2": 100,
    "Stereo-seq": 100,
    "Xenium": 10,
    "CosMx": 10,
    "MERFISH": 10,
    "DBiT-seq": 50,
}
DEFAULT_MIN_GENES = 200  # fallback for an unrecognized platform name


def preprocess(adata, platform: str = "Visium", n_top_genes=3000, n_pcs=50, min_genes=None, min_cells=3):
    """Standard preprocessing pipeline. Returns a copy.

    Parameters
    ----------
    adata : AnnData
        Raw unprocessed data with .X = counts
    platform : str
        Platform name (e.g. "Visium", "Xenium", "MERFISH"). Used to pick a
        platform-appropriate min_genes threshold unless one is given
        explicitly. See PLATFORM_MIN_GENES for the per-platform defaults.
    n_top_genes : int
        Number of highly variable genes to select
    n_pcs : int
        Number of PCA components
    min_genes : int, optional
        Minimum genes per spot. If not given, uses the platform-appropriate
        default from PLATFORM_MIN_GENES rather than a single fixed value.
    min_cells : int
        Minimum cells per gene

    Returns
    -------
    AnnData with .X = log-normalized, .layers['counts'] = raw,
    .obsm['X_pca'] = PCA, neighbors graph computed. adata.uns contains
    'n_spots_before_qc'/'n_spots_after_qc'/'qc_min_genes_used' so callers can
    surface QC retention to the user instead of it happening silently.
    """
    if min_genes is None:
        min_genes = PLATFORM_MIN_GENES.get(platform, DEFAULT_MIN_GENES)

    adata = adata.copy()
    adata.layers["counts"] = adata.X.copy()

    n_before = adata.shape[0]
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata.uns["n_spots_before_qc"] = int(n_before)
    adata.uns["n_spots_after_qc"] = int(adata.shape[0])
    adata.uns["qc_min_genes_used"] = int(min_genes)

    sc.pp.filter_genes(adata, min_cells=min_cells)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top_genes, flavor="seurat_v3", layer="counts"
    )

    sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=True)
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15)

    return adata


def preprocess_author_style(adata, slide_id=None):
    """Author's preprocessing: filter in_tissue, normalize, log1p.

    Used by the 7-method runners which follow the author's exact pipeline.
    Does NOT do HVG/PCA — each method does its own feature selection.
    """
    if "in_tissue" in adata.obs.columns:
        adata = adata[adata.obs["in_tissue"] == 1].copy()
    adata.var_names_make_unique()
    if slide_id is not None:
        adata.obs["slide_id"] = slide_id
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata
