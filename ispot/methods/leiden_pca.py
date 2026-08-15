"""
Leiden/PCA baseline: plain Leiden clustering on PCA representation.
Binary search for resolution giving target cluster count.

Fully working — ported from st_benchmark/run_baselines_v3.py.
"""
import time
import numpy as np
import scanpy as sc
from ispot.metrics import compute_metrics
from ispot.methods._nogt_helper import safe_compute_metrics


def run(adata, n_clusters, seed=42, **kwargs):
    """Run Leiden on PCA with binary search for target cluster count.

    Returns dict with ari, macro_f1, weighted_f1, runtime, n_spots, etc.
    """
    from ispot.preprocessing import preprocess

    adata = preprocess(adata, platform=kwargs.get("platform", "Visium"))

    def res_search(adata, target, low=0.01, high=2.5, max_iter=50):
        res = 1.0
        for _ in range(max_iter):
            sc.tl.leiden(adata, random_state=seed, resolution=res, key_added="leiden_baseline")
            n = adata.obs["leiden_baseline"].nunique()
            if n == target:
                return res
            if n < target:
                low = res
            else:
                high = res
            if high - low < 0.01:
                break
            res = (low + high) / 2
        return res

    t0 = time.time()
    res = res_search(adata, n_clusters)
    runtime = time.time() - t0

    m = safe_compute_metrics(adata, "leiden_baseline", runtime)
    m["resolution"] = float(res)
    m["labels"] = adata.obs["leiden_baseline"].values.astype(str)
    return m
