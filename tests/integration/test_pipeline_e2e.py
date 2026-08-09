"""End-to-end pipeline smoke test on a synthetic dataset.

Exercises the real scientific path — preprocess -> Leiden/PCA -> no-GT scoring
-> cluster estimation -> ranking deliverable — with the actual scanpy/leiden
stack. Skipped automatically when the heavy stack isn't installed (so the fast
unit CI job ignores it); runs in the integration CI job.
"""
import os

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scanpy")
pytest.importorskip("leidenalg")
pytest.importorskip("skmisc")  # scanpy seurat_v3 HVG dependency
ad = pytest.importorskip("anndata")


def make_synthetic(seed=0, n_side=18, n_genes=3500):
    """A small Visium-like dataset with 4 spatial-quadrant ground-truth domains."""
    rng = np.random.default_rng(seed)
    coords = np.array(
        [(x, y) for x in range(n_side) for y in range(n_side)], dtype=float
    )
    n = len(coords)
    gt = (
        np.where(coords[:, 0] < n_side / 2, 0, 1)
        + 2 * np.where(coords[:, 1] < n_side / 2, 0, 1)
    ).astype(str)

    # Baseline counts, dense enough that every spot passes min_genes=200.
    X = rng.poisson(1.0, size=(n, n_genes)).astype("float32")
    # Cluster-specific marker blocks so structure is recoverable.
    for c in range(4):
        idx = np.where(gt == str(c))[0]
        block = slice(c * 200, (c + 1) * 200)
        X[idx, block] += rng.poisson(5.0, size=(len(idx), 200)).astype("float32")

    adata = ad.AnnData(X)
    adata.obs_names = [f"s{i}" for i in range(n)]
    adata.var_names = [f"g{j}" for j in range(n_genes)]
    adata.obsm["spatial"] = coords
    adata.obs["ground_truth"] = gt
    adata.obs["has_ground_truth"] = True
    adata.obs["sample_id"] = "synthetic"
    adata.uns["platform"] = "Visium"
    return adata


def test_pipeline_end_to_end(tmp_path):
    import pandas as pd

    from ispot.methods import leiden_pca
    from ispot.preprocessing import preprocess
    from ispot.nogt_scoring import compute_nogt_score
    from ispot.cluster_estimation import estimate_n_clusters
    from ispot.deliverables import generate_ranking_table

    adata = make_synthetic()
    n = adata.n_obs

    # 1. Method run (preprocesses internally) with ground truth -> real metrics.
    m = leiden_pca.run(adata.copy(), n_clusters=4, seed=42)
    labels = np.array(m["labels"]).astype(str)
    assert len(labels) == n
    assert m["ari"] is not None and -1.0 <= m["ari"] <= 1.0

    # 2. Preprocess once for PCA / coords used by scoring + estimation.
    pp = preprocess(adata.copy())
    assert pp.n_obs == n  # dense synthetic data passes the cell filter
    coords = np.asarray(adata.obsm["spatial"])
    X_pca = pp.obsm["X_pca"]

    # 3. No-GT composite score is well-formed.
    nogt = compute_nogt_score(
        labels=labels, label_runs=[labels], coords=coords, X_pca=X_pca,
        all_method_labels={"Leiden_PCA": labels}, n_clusters=4,
    )
    for key in ["nogt_score", "scs", "css", "ess", "cas"]:
        assert 0.0 <= nogt[key] <= 1.0

    # 4. Cluster-count estimation returns a sane K.
    est = estimate_n_clusters(pp)
    assert isinstance(est["n_clusters"], int) and est["n_clusters"] >= 2

    # 5. Ranking deliverable is produced.
    df = pd.DataFrame([{
        "method": "Leiden_PCA", "seed": 42, "ari": m["ari"],
        "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
        "runtime": m.get("runtime") or 0.1, "n_spots": n,
        "n_clusters_pred": m.get("n_clusters_pred"),
    }])
    path = generate_ranking_table(df, has_ground_truth=True, output_dir=str(tmp_path))
    assert os.path.exists(path)
