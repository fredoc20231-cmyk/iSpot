"""
Standardized preprocessing for all spatial transcriptomics datasets.

Every dataset, once loaded, goes through this identical pipeline:
  1. Filter cells (min_genes=200) and genes (min_cells=3)
  2. Normalize total to 1e4
  3. Log1p
  4. Highly variable gene selection (seurat_v3, n_top_genes=3000)
  5. PCA (50 components on HVGs)
  6. Neighbors graph (15 neighbors on PCA)

This guarantees every method sees identically-preprocessed data.
Original counts preserved in .layers['counts'].
"""
import scanpy as sc


def preprocess(adata, n_top_genes=3000, n_pcs=50, min_genes=200, min_cells=3):
    """Standard preprocessing pipeline. Returns a copy.

    Parameters
    ----------
    adata : AnnData
        Raw unprocessed data with .X = counts
    n_top_genes : int
        Number of highly variable genes to select
    n_pcs : int
        Number of PCA components
    min_genes : int
        Minimum genes per spot
    min_cells : int
        Minimum cells per gene

    Returns
    -------
    AnnData with .X = log-normalized, .layers['counts'] = raw,
    .obsm['X_pca'] = PCA, neighbors graph computed
    """
    adata = adata.copy()
    adata.layers["counts"] = adata.X.copy()

    sc.pp.filter_cells(adata, min_genes=min_genes)
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
