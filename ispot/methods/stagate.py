"""
STAGATE baseline: spatial-aware autoencoder for domain detection.
Uses TF1-compatible version with mclust clustering.

Ported from st_benchmark/run_baselines_v3.py.
"""
import time
import scanpy as sc
from ispot.metrics import compute_metrics
from ispot.methods._nogt_helper import safe_compute_metrics
from ispot.preprocessing import preprocess


def run(adata, n_clusters, seed=42, n_epochs=500, lr=0.0001, **kwargs):
    """Run STAGATE with mclust clustering."""
    import STAGATE
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    adata = preprocess(adata, platform=kwargs.get("platform", "Visium"))
    STAGATE.Cal_Spatial_Net(adata, k_cutoff=15, model="KNN")

    t0 = time.time()
    STAGATE.train_STAGATE(adata, n_epochs=n_epochs, lr=lr, random_seed=seed)
    runtime = time.time() - t0

    sc.pp.neighbors(adata, n_neighbors=15, use_rep="STAGATE")
    cluster_col = "mclust"
    try:
        STAGATE.mclust_R(adata, num_cluster=n_clusters, used_obsm="STAGATE")
    except Exception as e:
        sc.tl.leiden(adata, resolution=0.5, key_added="stagate_leiden", random_state=seed)
        cluster_col = "stagate_leiden"

    m = safe_compute_metrics(adata, cluster_col, runtime)
    return m
