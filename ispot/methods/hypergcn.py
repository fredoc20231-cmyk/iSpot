"""
HyperGCN: hypergraph Laplacian graph convolutional network.

Ported from st_benchmark/benchmark_runner_py.py.
The NaN-loss issue was fixed by:
  1. Filtering zero-variance genes before scaling (prevents NaN from std=0)
  2. Replacing remaining NaN/inf with 0 after scaling (safety net)
These fixes are in the run() function below.

Model code is imported from a local clone at /workspace/models/HyperGCN
(upstream repo has no LICENSE file, so code is not vendored).
"""
import time
import os
import sys
import numpy as np
import scanpy as sc
from ispot.metrics import compute_metrics, res_search_fixed_clus
from ispot.methods._nogt_helper import safe_compute_metrics

# Add model repo to path
_HYPERGCN_PATH = os.environ.get("HYPERGCN_PATH", "/workspace/models/HyperGCN")
if _HYPERGCN_PATH not in sys.path:
    sys.path.insert(0, _HYPERGCN_PATH)


def run(adata, n_clusters, seed=42, n_epochs=200, **kwargs):
    """Run HyperGCN with NaN-loss fix.

    The NaN overflow in the reparameterization function is prevented by:
    - Filtering zero-variance genes before sc.pp.scale (std=0 → NaN)
    - np.nan_to_num safety net after scaling
    """
    import torch
    from sklearn.decomposition import PCA
    from sklearn.neighbors import kneighbors_graph
    from hpLapGCN import hpLapGCN

    device = "cpu"
    k = 50
    cell_feat_dim = 300

    class Params:
        pass
    params = Params()
    params.device = device
    params.k = k
    params.epochs = n_epochs
    params.feat_hidden1 = 20
    params.feat_hidden2 = 11
    params.gcn_hidden1 = 20
    params.gcn_hidden2 = 11
    params.p_drop = 0.2
    params.using_dec = True
    params.using_mask = False
    params.feat_w = 10
    params.clu = 0.1
    params.gcn_lr = 0.01
    params.gcn_decay = 0.01
    params.dec_interval = 20
    params.dec_tol = 0.0
    params.dec_cluster_n = n_clusters

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)

    # NaN FIX: Filter zero-variance genes before scaling
    gene_std = X.std(axis=0)
    nonzero_var = gene_std > 0
    X = X[:, nonzero_var]
    X_scaled = sc.pp.scale(X, zero_center=True, max_value=10, copy=True)
    # NaN FIX: Replace any remaining NaN/inf with 0
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=10.0, neginf=-10.0)

    n_components = min(cell_feat_dim, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    adata_X = pca.fit_transform(X_scaled)
    params.cell_feat_dim = adata_X.shape[1]

    spatial_co = adata.obsm["spatial"]
    adj = kneighbors_graph(spatial_co, k, mode="connectivity",
                           metric="euclidean", include_self=True, n_jobs=-1)
    adj_hp = torch.tensor(adj.toarray().astype(np.float32))
    graph_dict = {"spatial": spatial_co, "adj_norm": adj_hp}

    t0 = time.time()
    sedr_net = hpLapGCN(adata_X, graph_dict, params)
    if params.using_dec:
        sedr_net.train_with_dec()
    else:
        sedr_net.train_without_dec()
    runtime = time.time() - t0

    sedr_feat, _, _, _ = sedr_net.process()
    adata.obsm["HyperGCN"] = sedr_feat
    sc.pp.neighbors(adata, use_rep="HyperGCN", n_neighbors=20)

    eval_resolution = res_search_fixed_clus(adata, n_clusters)
    sc.tl.leiden(adata, resolution=eval_resolution)

    m = safe_compute_metrics(adata, "leiden", runtime)
    m["resolution"] = float(eval_resolution)
    m["labels"] = adata.obs["leiden"].values.astype(str)
    return m
