"""
SCOIGET: spatial copy-number inference and clustering.

Ported from st_benchmark/benchmark_runner_py.py.
Model code imported from /workspace/models/SCOIGET (has LICENSE).
This is the slowest method — runtime scales with spot count.
"""
import time
import os
import sys
import numpy as np
import scanpy as sc
from ispot.metrics import compute_metrics
from ispot.methods._nogt_helper import safe_compute_metrics
from ispot.loaders import convert_ensg_to_symbol

_SCOIGET_PATH = os.environ.get("SCOIGET_PATH", "/workspace/models/SCOIGET")
if _SCOIGET_PATH not in sys.path:
    sys.path.insert(0, _SCOIGET_PATH)


def _ensure_gene_ids(adata, species, release=98):
    """Map gene symbols to Ensembl IDs via pyensembl."""
    from pyensembl import EnsemblRelease
    if "gene_ids" in adata.var.columns:
        return adata

    sample_names = adata.var_names[:10].tolist()
    if all(n.startswith("ENSG") or n.startswith("ENSMUSG") for n in sample_names if n):
        adata.var["gene_ids"] = list(adata.var_names)
        return adata

    data = EnsemblRelease(release, species=species)

    def symbol_to_ensembl_id(symbol):
        try:
            ids = data.gene_ids_of_gene_name(symbol)
            return ids[0] if ids else None
        except ValueError:
            return None

    gene_ids = [symbol_to_ensembl_id(sym) for sym in adata.var_names]
    adata.var["gene_ids"] = gene_ids
    return adata


def run(adata, n_clusters, seed=42, dataset="DLPFC", slide_id="unknown", **kwargs):
    """Run SCOIGET two-stage training and clustering."""
    import torch
    from sklearn.decomposition import PCA
    from scoiget.cnv_utils import (
        add_genomic_locations, gene_binning_from_adata,
        perform_clustering, find_normal_cluster, compute_pseudo_copy
    )
    from scoiget.graph_utils import (
        get_x_bin_data_torch, construct_spatial_knn_graph,
        compute_edge_weights_and_probabilities
    )
    from scoiget.train_utils import prepare_data, train_scoiget
    from scoiget.cluster_utils import clustering

    output_dir = f"/workspace/scoiget_output/{slide_id}"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    species = "mouse" if dataset == "MOSTA" else "human"

    # Convert Ensembl IDs to gene symbols if needed (DLPFC)
    if dataset == "DLPFC":
        symbols, keep_mask = convert_ensg_to_symbol(adata.var_names)
        adata = adata[:, keep_mask].copy()
        adata.var_names = symbols
        adata.var_names_make_unique()

    adata = _ensure_gene_ids(adata, species)
    adata = add_genomic_locations(adata)
    adata, _ = gene_binning_from_adata(adata, 10)
    adata = get_x_bin_data_torch(adata, bin_size=10)
    adata_binned = adata.uns["binned_data"]
    X_bin = adata_binned.obsm["X_bin"]
    adata.obsm["feat"] = X_bin
    construct_spatial_knn_graph(adata, n_neighbors=5)
    compute_edge_weights_and_probabilities(adata, use_norm_x=False)

    # Stage 1
    data = prepare_data(adata, use_norm_x=False)
    model_stage1 = train_scoiget(
        data, original_dim=data.x.shape[1], intermediate_dim=128, latent_dim=32,
        max_cp=15, kl_weights=0.1, epochs=100, lr=0.001, lambda_smooth=0.1,
        use_mini_batch=False, dropout=0.2, hmm_states=3, gnn_heads=8,
        device=device, save_path=output_dir
    )

    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.edge_attr = data.edge_attr.to(device)
    model_stage1 = model_stage1.to(device)
    with torch.no_grad():
        z_mean, z_var, latent_z = model_stage1.z_encoder(data.x, data.edge_index)
        reconstructed_features = model_stage1.decoder(latent_z)
        pseudo_copy_number, _ = model_stage1.encoder(
            [data.x, reconstructed_features], data.edge_index
        )

    pseudo_copy_number = pseudo_copy_number.detach().cpu().numpy()
    scaling_factor = pseudo_copy_number.mean()
    adata.obsm["norm_x"] = pseudo_copy_number / scaling_factor

    # Stage 2
    compute_edge_weights_and_probabilities(adata, use_norm_x=True)
    data_norm = prepare_data(adata, use_norm_x=True)

    t0 = time.time()
    scoiget_model = train_scoiget(
        data_norm, original_dim=data_norm.x.shape[1], intermediate_dim=128, latent_dim=32,
        max_cp=15, kl_weights=0.5, epochs=100, lr=0.001, lambda_smooth=1,
        use_mini_batch=False, validation_split=0.2, dropout=0.2, hmm_states=3, gnn_heads=8,
        device=device
    )
    runtime = time.time() - t0

    scoiget_model.eval()
    data_norm.x = data_norm.x.to(device)
    data_norm.edge_index = data_norm.edge_index.to(device)
    data_norm.edge_attr = data_norm.edge_attr.to(device)
    scoiget_model = scoiget_model.to(device)
    with torch.no_grad():
        z_mean, z_var, latent_z = scoiget_model.z_encoder(data_norm.x, data_norm.edge_index)
        reconstructed_features = scoiget_model.decoder(latent_z)
        copy_number_profile, _ = scoiget_model.encoder(
            [data_norm.x, reconstructed_features], data_norm.edge_index
        )

    adata.obsm["latent"] = latent_z.cpu().numpy()
    adata.obsm["copy_number"] = copy_number_profile.cpu().numpy()

    pca = PCA(svd_solver="arpack")
    adata.obsm["X_pca"] = pca.fit_transform(adata.obsm["copy_number"])

    adata = clustering(adata, method="leiden", auto_choose=False, n_clusters=n_clusters,
                       refinement=False, use_rep="copy_number")

    m = safe_compute_metrics(adata, "leiden", runtime)
    m["labels"] = adata.obs["leiden"].values.astype(str)
    return m
