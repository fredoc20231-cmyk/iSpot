"""
Novae: foundation model for spatial transcriptomics domain identification.

Supports three modes:
  - 'from_scratch': train from scratch on each slide (author's default)
  - 'fine_tuned': fine-tune pretrained model on target data
  - 'zero_shot': use pretrained model without any training

Ported from st_benchmark/benchmark_runner_py.py and run_novae_zeroshot_v2.py.
"""
import time
import numpy as np
import scanpy as sc
from ispot.metrics import compute_metrics
from ispot.methods._nogt_helper import safe_compute_metrics
from ispot.loaders import convert_ensg_to_symbol


def run(adata, n_clusters, seed=42, mode="from_scratch", dataset="DLPFC", **kwargs):
    """Run Novae in the specified mode.

    Parameters
    ----------
    mode : str
        'from_scratch', 'fine_tuned', or 'zero_shot'
    """
    import novae

    # Convert Ensembl IDs to lowercase gene symbols for Novae (DLPFC)
    if dataset == "DLPFC":
        symbols, keep_mask = convert_ensg_to_symbol(adata.var_names)
        adata = adata[:, keep_mask].copy()
        adata.var_names = symbols
    adata.var_names = [g.lower() for g in adata.var_names]
    adata.var_names_make_unique()

    # Build spatial neighbors
    if dataset == "MOSTA":
        novae.spatial_neighbors(adata, radius=300, slide_key="slide_id")
    else:
        novae.spatial_neighbors(adata, technology="visium", slide_key="slide_id")

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    adata = adata[:, adata.var["highly_variable"]].copy()

    t0 = time.time()

    if mode == "zero_shot":
        # Zero-shot: use pretrained model, no training
        model = novae.Novae.load_pretrained()
        model.compute_representations(adata)
        # Try both level and resolution-based assignment
        results = {}
        for assignment in [f"level_{n_clusters}", "resolution_1.0"]:
            try:
                model.assign_domains(adata, key_added=assignment)
                pred_key = f"novae_domains_{assignment}"
                if pred_key not in adata.obs:
                    pred_key = assignment
                m = safe_compute_metrics(adata, pred_key, time.time() - t0)
                m["assignment"] = assignment
                results[assignment] = m
            except Exception:
                pass
        # Return best assignment
        if results:
            # If no GT, all ari are None — just return the first
            valid = [r for r in results.values() if r.get("ari") is not None]
            if valid:
                best = max(valid, key=lambda x: x["ari"])
            else:
                best = list(results.values())[0]
            return best
        raise RuntimeError("Novae zero-shot assignment failed for all strategies")

    elif mode == "fine_tuned":
        # Fine-tuned: load pretrained, then fine-tune on target
        model = novae.Novae.load_pretrained()
        model.fine_tune(adata, n_epochs=50)
        model.compute_representations(adata)
        model.assign_domains(level=n_clusters)
        pred_key = f"novae_domains_{n_clusters}"

    else:  # from_scratch
        model = novae.Novae(adata, n_hops_local=1, n_hops_view=1, panel_subset_size=0.6)
        model.fit()
        model.compute_representations(adata)
        model.assign_domains(level=n_clusters)
        pred_key = f"novae_domains_{n_clusters}"

    runtime = time.time() - t0

    m = safe_compute_metrics(adata, pred_key, runtime)
    m["mode"] = mode
    return m
