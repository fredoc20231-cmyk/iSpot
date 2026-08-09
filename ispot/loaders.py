"""
Data loaders for spatial transcriptomics datasets.

Each loader returns a raw (unprocessed) AnnData with:
  - .X = counts (sparse or dense)
  - .obsm['spatial'] = spatial coordinates (float array, n_spots x 2)
  - .obs['ground_truth'] = ground truth labels
  - .obs['has_ground_truth'] = bool mask (True for annotated spots)
  - .obs['sample_id'] = sample identifier
  - .obs['in_tissue'] = 1 for in-tissue spots (if applicable)

All preprocessing happens through ispot.preprocessing.preprocess(),
never inside a loader.
"""
import os
import gzip
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import h5py

# Data paths — check shared-workspace first, then workspace
DLPFC_DIR = "/mnt/shared-workspace/shared/data/dlpfc_h5ad"
HER2_DIR = "/mnt/shared-workspace/shared/data/her2st/data"
MOSTA_DIR = "/mnt/shared-workspace/shared/data"

# Fallback paths
DLPFC_DIR_FALLBACK = "/workspace/dlpfc_h5"
HER2_DIR_FALLBACK = "/workspace/her2st/data"

# Cluster counts from author's constants.py/constants.R
CLUSTER_COUNTS = {
    "DLPFC": {
        "151507": 7, "151508": 7, "151509": 7, "151510": 7,
        "151669": 5, "151670": 5, "151671": 5, "151672": 5,
        "151673": 7, "151674": 7, "151675": 7, "151676": 7,
    },
    "HER2+": {
        "A1": 6, "B1": 5, "C1": 4, "D1": 4,
        "E1": 4, "F1": 4, "G2": 7, "H1": 7,
    },
    "MOSTA": {
        "E9.5_E2S1": 14, "E9.5_E2S2": 13, "E9.5_E2S3": 13, "E9.5_E2S4": 13,
    },
}

DATASET_SLIDES = {
    "DLPFC": list(CLUSTER_COUNTS["DLPFC"].keys()),
    "HER2+": list(CLUSTER_COUNTS["HER2+"].keys()),
    "MOSTA": list(CLUSTER_COUNTS["MOSTA"].keys()),
}

# Cache for Ensembl-to-symbol mapping
_ENSG_MAPPING = None


def _get_ensg_mapping():
    global _ENSG_MAPPING
    if _ENSG_MAPPING is None:
        for path in ["/workspace/ensg_to_symbol.json",
                     "/mnt/shared-workspace/shared/ensg_to_symbol.json"]:
            if os.path.exists(path):
                with open(path) as f:
                    _ENSG_MAPPING = json.load(f)
                break
        else:
            _ENSG_MAPPING = {}
    return _ENSG_MAPPING


def convert_ensg_to_symbol(gene_names):
    """Convert Ensembl IDs to gene symbols, lowercased for Novae compatibility.

    Unmapped genes are dropped. Returns (filtered_gene_names, keep_mask).
    """
    mapping = _get_ensg_mapping()
    symbols = []
    keep = []
    for g in gene_names:
        sym = mapping.get(g, None)
        if sym is not None:
            symbols.append(sym.lower())
            keep.append(True)
        else:
            keep.append(False)
    return symbols, np.array(keep)


def _find_path(primary, fallback):
    """Return the first existing path."""
    if os.path.exists(primary):
        return primary
    if os.path.exists(fallback):
        return fallback
    return primary  # let it fail with a clear error


def load_dlpfc(slide_id):
    """Load a single DLPFC slide from HDF5 as AnnData.

    Source: spatialLIBD (Bioconductor ExperimentHub), 12 slides.
    Ground truth: layer_guess (Layer1-6, WM, NA for background).
    """
    # Try .h5ad first (shared-workspace), then .h5 (workspace fallback)
    h5ad_path = os.path.join(DLPFC_DIR, f"{slide_id}.h5ad")
    h5_path = os.path.join(DLPFC_DIR_FALLBACK, f"{slide_id}.h5")

    if os.path.exists(h5ad_path):
        adata = ad.read_h5ad(h5ad_path)
        # Ensure required fields
        if "spatial" not in adata.obsm:
            adata.obsm["spatial"] = adata.obs[["array_col", "array_row"]].values.astype(float)
        if "ground_truth" not in adata.obs and "layer_guess" in adata.obs:
            adata.obs["ground_truth"] = adata.obs["layer_guess"]
        adata.obs["has_ground_truth"] = [
            (g is not None) and (str(g) != "") and (str(g).lower() != "nan") and (str(g) != "NA")
            for g in adata.obs["ground_truth"]
        ]
        adata.obs["sample_id"] = slide_id
        adata.obs["in_tissue"] = 1
        return adata

    # Fall back to .h5 format (custom HDF5 with sparse matrix)
    with h5py.File(h5_path, "r") as f:
        data = f["counts_data"][:]
        indices = f["counts_indices"][:]
        indptr = f["counts_indptr"][:]
        dims = f["counts_dims"][:]
        n_genes, n_spots = int(dims[0]), int(dims[1])

        counts = sp.csc_matrix((data, indices, indptr), shape=(n_genes, n_spots))
        counts = counts.T.tocsr()

        gene_names = [s.decode("utf-8") if isinstance(s, bytes) else s for s in f["gene_names"][:]]
        barcodes = [s.decode("utf-8") if isinstance(s, bytes) else s for s in f["barcodes"][:]]
        spatial = f["spatial"][:].T

        gt_raw = f["ground_truth"][:]
        gt = [s.decode("utf-8") if isinstance(s, bytes) else s for s in gt_raw]

        sample_id_val = f["sample_id"][0]
        if isinstance(sample_id_val, bytes):
            sample_id_val = sample_id_val.decode("utf-8")

    adata = ad.AnnData(X=counts)
    adata.var_names = gene_names
    adata.obs_names = barcodes
    adata.obsm["spatial"] = spatial.astype(float)
    adata.obs["ground_truth"] = gt
    adata.obs["sample_id"] = sample_id_val
    adata.obs["has_ground_truth"] = [
        (g is not None) and (str(g) != "") and (str(g).lower() != "nan") and (str(g) != "NA")
        for g in gt
    ]
    adata.obs["in_tissue"] = 1

    return adata


def load_her2(slide_id):
    """Load a single HER2+ breast cancer slide as AnnData.

    Source: Zenodo doi.org/10.5281/zenodo.4751624.
    Ground truth: pathologist annotation (includes 'undetermined' as valid cluster).
    Uses 8 annotated slides: A1, B1, C1, D1, E1, F1, G2, H1.
    """
    data_dir = _find_path(HER2_DIR, HER2_DIR_FALLBACK)

    cnt_file = os.path.join(data_dir, "ST-cnts", f"{slide_id}.tsv.gz")
    with gzip.open(cnt_file, "rt") as f:
        header = f.readline().strip().split("\t")
        genes = header[1:]
        spots, data = [], []
        for line in f:
            parts = line.strip().split("\t")
            spots.append(parts[0])
            row = [int(x) for x in parts[1:]]
            if len(row) > len(genes):
                row = row[:len(genes)]
            elif len(row) < len(genes):
                row = row + [0] * (len(genes) - len(row))
            data.append(row)

    counts = sp.csr_matrix(np.array(data, dtype=np.float32))

    sel_file = os.path.join(data_dir, "ST-spotfiles", f"{slide_id}_selection.tsv")
    sel = pd.read_csv(sel_file, sep="\t")

    lbl_file = os.path.join(data_dir, "ST-pat", "lbl", f"{slide_id}_labeled_coordinates.tsv")
    lbl = pd.read_csv(lbl_file, sep="\t")

    merged = lbl.merge(
        sel, left_on=["x", "y"], right_on=["new_x", "new_y"], how="inner", suffixes=("_lbl", "_sel")
    )
    merged["spot_id"] = merged.apply(lambda r: f"{int(r['x_sel'])}x{int(r['y_sel'])}", axis=1)

    label_map = dict(zip(merged["spot_id"], merged["label"]))
    coord_map = dict(zip(merged["spot_id"], merged[["pixel_x_sel", "pixel_y_sel"]].values))

    sel_spots = sel.apply(lambda r: f"{int(r['x'])}x{int(r['y'])}", axis=1)
    for i, sid in enumerate(sel_spots):
        if sid not in coord_map:
            coord_map[sid] = [sel.iloc[i]["pixel_x"], sel.iloc[i]["pixel_y"]]

    adata = ad.AnnData(X=counts)
    adata.var_names = genes
    adata.obs_names = spots
    adata.obs["sample_id"] = slide_id

    spatial = np.array([coord_map.get(s, [0.0, 0.0]) for s in spots])
    adata.obsm["spatial"] = spatial.astype(float)

    gt = [label_map.get(s, "NA") for s in spots]
    adata.obs["ground_truth"] = gt
    adata.obs["has_ground_truth"] = [
        g != "NA" and str(g).lower() != "nan" and str(g) != "" for g in gt
    ]
    adata.obs["in_tissue"] = 1

    return adata


def load_mosta(slide_id):
    """Load a single MOSTA Stereo-seq slide as AnnData.

    Source: CNGB / Chen et al. 2022, E9.5 stage, 4 slides.
    Ground truth: anatomical region annotation.
    """
    path = os.path.join(MOSTA_DIR, f"{slide_id}.MOSTA.h5ad")
    if not os.path.exists(path):
        path = os.path.join("/workspace", f"{slide_id}.MOSTA.h5ad")

    adata = ad.read_h5ad(path)
    adata.obs["ground_truth"] = adata.obs["annotation"]
    adata.obs["has_ground_truth"] = adata.obs["ground_truth"].notna()
    adata.obs["sample_id"] = slide_id
    adata.obs["in_tissue"] = 1

    return adata


def load_slideseqv2():
    """Load squidpy Slide-seqV2 mouse cerebellum/hippocampus dataset.

    Used for Novae zero-shot generalizability testing (different platform).
    41,786 spots, 14 cell-type clusters, spatial coords in obsm['spatial'].
    """
    import squidpy as sq
    adata = sq.datasets.slideseqv2()
    adata.obs["ground_truth"] = adata.obs["cluster"]
    adata.obs["has_ground_truth"] = adata.obs["ground_truth"].notna()
    adata.obs["sample_id"] = "slideseqv2_cerebellum"
    adata.obs["in_tissue"] = 1
    # Ensure spatial is in obsm
    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = adata.obs[["x", "y"]].values.astype(float)
    return adata


def load_sample(dataset, slide_id):
    """Dispatch to the appropriate loader based on dataset name."""
    if dataset == "DLPFC":
        return load_dlpfc(slide_id)
    elif dataset == "HER2+":
        return load_her2(slide_id)
    elif dataset == "MOSTA":
        return load_mosta(slide_id)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_n_clusters(dataset, slide_id):
    """Get the author-specified target cluster count for a slide."""
    return CLUSTER_COUNTS[dataset][slide_id]


def qc_check(adata, dataset, slide_id):
    """Run QC checks on a loaded dataset. Returns dict of check results.

    Does not silently drop — reports all issues.
    """
    n_spots = adata.shape[0]
    n_genes = adata.shape[1]
    gt_valid = adata.obs["has_ground_truth"].sum()
    gt_missing = len(adata) - gt_valid
    gt_missing_pct = gt_missing / len(adata) * 100 if len(adata) > 0 else 0
    n_gt_clusters = adata.obs.loc[adata.obs["has_ground_truth"], "ground_truth"].nunique()

    # Check for all-zero genes/spots
    if hasattr(adata.X, 'toarray'):
        X_dense = adata.X.toarray()
    else:
        X_dense = np.array(adata.X)
    all_zero_genes = (X_dense.sum(axis=0) == 0).sum()
    all_zero_spots = (X_dense.sum(axis=1) == 0).sum()

    # Expected ranges from Table 2
    expected = {
        "DLPFC": {"spots": (3000, 5000), "clusters": (5, 7)},
        "HER2+": {"spots": (150, 750), "clusters": (4, 7)},
        "MOSTA": {"spots": (4000, 6000), "clusters": (13, 14)},
    }

    checks = {
        "dataset": dataset,
        "slide_id": slide_id,
        "n_spots": n_spots,
        "n_genes": n_genes,
        "gt_valid": int(gt_valid),
        "gt_missing": int(gt_missing),
        "gt_missing_pct": round(gt_missing_pct, 1),
        "n_gt_clusters": int(n_gt_clusters),
        "all_zero_genes": int(all_zero_genes),
        "all_zero_spots": int(all_zero_spots),
        "spatial_present": "spatial" in adata.obsm,
    }

    if dataset in expected:
        exp = expected[dataset]
        checks["spots_in_range"] = exp["spots"][0] <= n_spots <= exp["spots"][1]
        checks["clusters_in_range"] = exp["clusters"][0] <= n_gt_clusters <= exp["clusters"][1]
        checks["gt_missing_ok"] = gt_missing_pct <= 20.0

    return checks
