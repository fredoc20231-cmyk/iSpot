"""
SpaGCN baseline: graph convolutional network using spatial and expression info.
Expression-only mode (no histology image).

Ported from st_benchmark/run_baselines_v3.py.
"""
import time
import numpy as np
import scanpy as sc
from ispot.metrics import compute_metrics
from ispot.methods._nogt_helper import safe_compute_metrics


def run(adata, n_clusters, seed=42, p=0.5, **kwargs):
    """Run SpaGCN in expression-only mode."""
    import SpaGCN
    from ispot.preprocessing import PLATFORM_MIN_GENES, DEFAULT_MIN_GENES

    adata = adata.copy()
    adata.layers["counts"] = adata.X.copy()
    min_genes = PLATFORM_MIN_GENES.get(kwargs.get("platform", "Visium"), DEFAULT_MIN_GENES)
    SpaGCN.prefilter_cells(adata, min_genes=min_genes)
    SpaGCN.prefilter_genes(adata, min_cells=3)
    SpaGCN.prefilter_specialgenes(adata)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if hasattr(adata.X, 'toarray'):
        adata.X = adata.X.toarray()

    x = adata.obsm["spatial"][:, 0]
    y = adata.obsm["spatial"][:, 1]
    adj = SpaGCN.calculate_adj_matrix(x=x, y=y, histology=False)
    l = SpaGCN.search_l(p, adj, start=0.01, end=1000, tol=1e-5, max_run=100)
    res = max(0.1, n_clusters * 0.15)

    t0 = time.time()
    clf = SpaGCN.SpaGCN()
    clf.set_l(l)
    clf.train(adata, adj, init_spa=True, init="louvain", res=res, tol=5e-3, lr=0.05, max_epochs=20)
    y_pred, _ = clf.predict()
    runtime = time.time() - t0

    try:
        refined = SpaGCN.refine(sample_id=adata.obs_names, pred=y_pred, dis=adj)
    except Exception:
        refined = y_pred

    adata.obs["spagcn_pred"] = refined
    m = safe_compute_metrics(adata, "spagcn_pred", runtime)
    m["l"] = float(l)
    m["res"] = float(res)
    m["labels"] = adata.obs["spagcn_pred"].values.astype(str)
    return m
