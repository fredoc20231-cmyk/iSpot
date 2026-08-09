"""
STMSGAL: spatial transcriptomics multi-sample graph attention learning.

Ported from st_benchmark/benchmark_runner_py.py.
Model code imported from /workspace/models/STMSGAL (CC0 LICENSE).
"""
import time
import sys
import os
import numpy as np
import scanpy as sc
from ispot.metrics import compute_metrics, res_search_fixed_clus
from ispot.methods._nogt_helper import safe_compute_metrics
from ispot.loaders import convert_ensg_to_symbol

_STMSGAL_PATH = os.environ.get("STMSGAL_PATH", "/workspace/models/STMSGAL")
if _STMSGAL_PATH not in sys.path:
    sys.path.insert(0, _STMSGAL_PATH)


def _get_median_nn_distance(coords, k=6):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    return np.median(distances[:, -1])


def _rescale_spatial(adata, ref_dist, k=6):
    coords = adata.obsm["spatial"].astype(float)
    cur = _get_median_nn_distance(coords, k)
    scale = ref_dist / cur
    adata.obsm["spatial"] = coords * scale
    return adata


def run(adata, n_clusters, seed=42, n_epochs=100, dataset="DLPFC", **kwargs):
    """Run STMSGAL with spatial coord rescaling for non-DLPFC datasets."""
    import STMSGAL

    # Rescale spatial coords for non-DLPFC to match DLPFC median NN distance
    if dataset != "DLPFC":
        ref_adata = sc.read_h5ad("/mnt/shared-workspace/shared/data/dlpfc_h5ad/151507.h5ad") \
            if os.path.exists("/mnt/shared-workspace/shared/data/dlpfc_h5ad/151507.h5ad") \
            else sc.read_h5ad("/workspace/dlpfc_h5/151507.h5")
        ref_dist = _get_median_nn_distance(ref_adata.obsm["spatial"])
        adata = _rescale_spatial(adata, ref_dist)

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
    STMSGAL.Cal_Spatial_Net(adata, rad_cutoff=300)
    STMSGAL.Stats_Spatial_Net(adata)

    t0 = time.time()
    adata, pred_dsc = STMSGAL.train_STMSGAL(
        adata, alpha=0.7, pre_resolution=0.2,
        n_epochs=n_epochs, save_attention=True, save_loss=False,
        n_cluster=n_clusters, cost_ssc_coef=0.1
    )
    runtime = time.time() - t0

    sc.pp.neighbors(adata, use_rep="STMSGAL")
    eval_resolution = res_search_fixed_clus(adata, n_clusters)
    sc.tl.leiden(adata, resolution=eval_resolution)

    m = safe_compute_metrics(adata, "leiden", runtime)
    m["resolution"] = float(eval_resolution)
    m["labels"] = adata.obs["leiden"].values.astype(str)
    return m
