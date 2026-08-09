"""
GraphST baseline: graph-contrastive learning for spatial domain identification.

Already wired and working — ported from st_benchmark/run_baselines_v3.py
and benchmark_runner_py.py patterns.
"""
import time
import numpy as np
import scanpy as sc
from ispot.metrics import compute_metrics, res_search_fixed_clus
from ispot.methods._nogt_helper import safe_compute_metrics


def run(adata, n_clusters, seed=42, **kwargs):
    """Run GraphST with leiden clustering and resolution search."""
    import GraphST
    adata = adata.copy()

    # GraphST preprocessing
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)

    # Build spatial graph
    GraphST.spatial_construct(adata, k=7)

    t0 = time.time()
    GraphST.train_graphst(adata, n_epochs=500, random_seed=seed)
    runtime = time.time() - t0

    sc.pp.neighbors(adata, use_rep="GraphST_emb", n_neighbors=15)
    eval_resolution = res_search_fixed_clus(adata, n_clusters)
    sc.tl.leiden(adata, resolution=eval_resolution, key_added="graphst_leiden")

    m = safe_compute_metrics(adata, "graphst_leiden", runtime)
    m["resolution"] = float(eval_resolution)
    m["labels"] = adata.obs["graphst_leiden"].values.astype(str)
    return m
