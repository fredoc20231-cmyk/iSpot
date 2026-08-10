"""
Multi-platform data loaders for spatial transcriptomics data.

Each loader normalizes a platform-specific input into a standard AnnData:
  - .X = counts (sparse or dense)
  - .obsm['spatial'] = 2D coordinates (n_spots x 2)
  - .obs['ground_truth'] = labels if provided, else None
  - .obs['has_ground_truth'] = bool mask
  - .obs['sample_id'] = sample identifier
  - .obs['in_tissue'] = 1 for in-tissue spots (if applicable)
  - .uns['platform'] = platform name
  - .uns['spatial_layout'] = auto-detected

Section 1.1 of the platform plan.

Phase 1: Visium (already in ispot.loaders)
Phase 3: Slide-seq, MERFISH, CosMx, Xenium, Stereo-seq, DBiT-seq
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import anndata as ad
from typing import Optional

from ispot.profiling import detect_spatial_layout


# ---------------------------------------------------------------------------
# Base loader interface
# ---------------------------------------------------------------------------

def _find_histology_image_and_scale(adata):
    """Return (image_array, scale_factor) for the best available histology
    image in adata.uns['spatial'], or (None, None)."""
    spatial_uns = adata.uns.get("spatial")
    if not spatial_uns:
        return None, None
    for sample_key, sample_data in spatial_uns.items():
        images = sample_data.get("images", {}) if isinstance(sample_data, dict) else {}
        scalefactors = sample_data.get("scalefactors", {}) if isinstance(sample_data, dict) else {}
        for res_key, sf_key in (("hires", "tissue_hires_scalef"),
                                ("lowres", "tissue_lowres_scalef")):
            if res_key in images and sf_key in scalefactors:
                return np.asarray(images[res_key]), float(scalefactors[sf_key])
    return None, None


def _apply_image_based_tissue_detection(adata):
    """Optionally drop off-tissue spots using image-derived tissue detection.

    Image-based Otsu segmentation is only a heuristic: an imperfect mask can
    drop real tissue spots (viewer shows less than the full section) or keep
    background spots near the border (dots outside the tissue). So by default
    this does NOT filter — the authoritative ``in_tissue`` annotation handles
    filtering, and the histology image is drawn in the viewer so the full
    section is visible with spots overlaid via the scalefactor. Set
    ``ISPOT_IMAGE_TISSUE_FILTER=1`` to opt back into hard image-based filtering.

    No-op (returns adata unchanged) when no histology image is available.
    """
    image, scale_factor = _find_histology_image_and_scale(adata)
    if image is None:
        return adata

    filter_on = os.environ.get("ISPOT_IMAGE_TISSUE_FILTER", "").strip() in ("1", "true", "True")
    if not filter_on:
        return adata  # image is drawn in the viewer; do not drop spots by heuristic

    from ispot.tissue_segmentation import detect_tissue_mask, spots_in_tissue_mask
    tissue_mask = detect_tissue_mask(image)
    coords = np.array(adata.obsm["spatial"])
    on_tissue = spots_in_tissue_mask(coords, tissue_mask, scale_factor)
    n_excluded = int((~on_tissue).sum())
    if n_excluded > 0:
        adata = adata[on_tissue].copy()
        adata.uns["n_spots_excluded_by_image_tissue_detection"] = n_excluded
    return adata


class BaseLoader:
    """Base class for platform-specific data loaders.

    Subclasses implement _load_raw() which returns an AnnData with
    at minimum .X and .obsm['spatial']. The base class handles
    normalization of metadata fields.
    """

    platform_name: str = "unknown"

    def load(
        self,
        path: str,
        sample_id: str | None = None,
        ground_truth_col: str | None = None,
        **kwargs,
    ) -> ad.AnnData:
        """Load and normalize a spatial transcriptomics dataset.

        Parameters
        ----------
        path : str
            Path to the data file or directory.
        sample_id : str, optional
            Sample identifier. If None, derived from path.
        ground_truth_col : str, optional
            Column name in the data containing ground truth annotations.
            If None, no ground truth is set.

        Returns
        -------
        AnnData with standardized fields.
        """
        adata = self._load_raw(path, **kwargs)

        # Ensure spatial coordinates exist
        if "spatial" not in adata.obsm:
            raise ValueError(f"{self.platform_name} loader did not produce spatial coordinates")

        # Set sample_id
        if sample_id is None:
            sample_id = os.path.basename(path).split(".")[0]
        adata.obs["sample_id"] = pd.Categorical([sample_id] * adata.shape[0])

        # Set ground truth
        if ground_truth_col is not None and ground_truth_col in adata.obs.columns:
            adata.obs["ground_truth"] = adata.obs[ground_truth_col].astype(str)
            adata.obs["has_ground_truth"] = adata.obs["ground_truth"].notna() & (
                adata.obs["ground_truth"] != "nan"
            ) & (adata.obs["ground_truth"] != "")
        elif "ground_truth" in adata.obs.columns and "has_ground_truth" in adata.obs.columns:
            # File already has ground truth columns — preserve them
            gt = adata.obs["ground_truth"]
            if gt.dtype == "category":
                gt = gt.astype(str)
            adata.obs["ground_truth"] = gt
            # Ensure has_ground_truth is a proper boolean mask
            if adata.obs["has_ground_truth"].dtype != bool:
                adata.obs["has_ground_truth"] = adata.obs["has_ground_truth"].astype(bool)
            # Also handle "NA" strings as no-GT
            adata.obs.loc[adata.obs["ground_truth"] == "NA", "has_ground_truth"] = False
        else:
            adata.obs["ground_truth"] = None
            adata.obs["has_ground_truth"] = False

        # Set in_tissue (default: all in tissue) and, critically, actually
        # FILTER on it. Previously this column was read and preserved but never
        # used, so a raw/unfiltered Visium export (or any dataset with real
        # off-tissue background spots marked in_tissue=0) carried every
        # background spot straight into clustering and the viewer — dots
        # scattered across the whole rectangular capture array instead of the
        # actual tissue footprint.
        if "in_tissue" not in adata.obs.columns:
            adata.obs["in_tissue"] = 1
        else:
            in_tissue_vals = (
                pd.to_numeric(adata.obs["in_tissue"], errors="coerce")
                .fillna(1).astype(int)
            )
            n_off_tissue = int((in_tissue_vals == 0).sum())
            if n_off_tissue > 0:
                n_before = adata.shape[0]
                adata = adata[in_tissue_vals.values == 1].copy()
                adata.uns["n_spots_excluded_off_tissue"] = n_off_tissue
                adata.uns["n_spots_before_tissue_filter"] = int(n_before)

        # Image-based tissue detection: if a histology image is available,
        # exclude off-tissue capture spots and record the real tissue outline
        # for the viewer. Best-effort — never fail the load on a detection error.
        try:
            adata = _apply_image_based_tissue_detection(adata)
        except Exception as e:
            adata.uns["tissue_image_detection_error"] = str(e)

        # Set platform metadata
        adata.uns["platform"] = self.platform_name

        # Auto-detect spatial layout (on the possibly-filtered coordinates)
        coords = np.array(adata.obsm["spatial"])
        adata.uns["spatial_layout"] = detect_spatial_layout(coords)

        return adata

    def _load_raw(self, path: str, **kwargs) -> ad.AnnData:
        """Platform-specific loading logic. Override in subclasses."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Visium loader (wraps existing ispot.loaders)
# ---------------------------------------------------------------------------

def _load_space_ranger_bundle(matrix_h5_path: str, spatial_dir: str) -> ad.AnnData:
    """Self-contained Space Ranger loader.

    Tolerant of only one of the hires/lowres images being present (many public
    datasets ship only hires), of both tissue_positions formats (v1 headerless
    ``tissue_positions_list.csv`` and v2 ``tissue_positions.csv`` with header),
    and of CellRanger v2/v3 feature naming. scanpy.read_visium requires BOTH
    images or raises, which this avoids.
    """
    import json

    import h5py
    import scipy.sparse as sp

    with h5py.File(matrix_h5_path, "r") as f:
        group = f["matrix"]
        data = group["data"][:]
        indices = group["indices"][:]
        indptr = group["indptr"][:]
        shape = group["shape"][:]
        barcodes = [b.decode() if isinstance(b, bytes) else b for b in group["barcodes"][:]]
        feature_group = group["features"] if "features" in group else group
        name_key = "name" if "name" in feature_group else "gene_names"  # v3 vs v2
        gene_names = [g.decode() if isinstance(g, bytes) else g for g in feature_group[name_key][:]]
    X = sp.csc_matrix((data, indices, indptr), shape=tuple(shape)).T.tocsr()

    adata = ad.AnnData(X=X)
    adata.obs_names = barcodes
    adata.var_names = gene_names
    adata.var_names_make_unique()

    positions_path = None
    for candidate in ("tissue_positions.csv", "tissue_positions_list.csv"):
        p = os.path.join(spatial_dir, candidate)
        if os.path.exists(p):
            positions_path = p
            break
    if positions_path is None:
        raise ValueError(
            f"No tissue_positions.csv or tissue_positions_list.csv found in {spatial_dir}"
        )

    pos_cols = ["barcode", "in_tissue", "array_row", "array_col",
                "pxl_row_in_fullres", "pxl_col_in_fullres"]
    with open(positions_path) as fh:
        first_line = fh.readline().strip().split(",")
    has_header = not (len(first_line) > 1 and first_line[1] in ("0", "1"))
    positions = pd.read_csv(
        positions_path, header=0 if has_header else None,
        names=None if has_header else pos_cols,
    )
    if has_header:
        positions.columns = pos_cols[: len(positions.columns)]
    positions = positions.set_index("barcode").reindex(adata.obs_names)

    adata.obs["in_tissue"] = positions["in_tissue"].fillna(1).astype(int).values
    adata.obsm["spatial"] = (
        positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].fillna(0).values.astype(float)
    )

    scalefactors_path = os.path.join(spatial_dir, "scalefactors_json.json")
    scalefactors = {}
    if os.path.exists(scalefactors_path):
        with open(scalefactors_path) as f:
            scalefactors = json.load(f)

    images = {}
    for res_key, filename in (("hires", "tissue_hires_image.png"),
                              ("lowres", "tissue_lowres_image.png")):
        img_path = os.path.join(spatial_dir, filename)
        if os.path.exists(img_path):
            from PIL import Image as PILImage
            images[res_key] = np.array(PILImage.open(img_path).convert("RGB"))

    library_id = os.path.basename(os.path.dirname(spatial_dir)) or "sample"
    if images or scalefactors:
        adata.uns["spatial"] = {library_id: {"images": images, "scalefactors": scalefactors}}
    return adata


class VisiumLoader(BaseLoader):
    """Loader for 10x Visium data.

    Accepts:
    - .h5ad files (pre-processed AnnData)
    - Space Ranger output: filtered_feature_bc_matrix.h5 (+ sibling spatial/)
    - Space Ranger output directory containing the h5 file
    """

    platform_name = "Visium"

    def _load_raw(self, path: str, **kwargs) -> ad.AnnData:
        if path.endswith(".h5ad"):
            return ad.read_h5ad(path)
        elif path.endswith(".h5"):
            spatial_dir = os.path.join(os.path.dirname(path), "spatial")
            if os.path.isdir(spatial_dir):
                return _load_space_ranger_bundle(matrix_h5_path=path, spatial_dir=spatial_dir)
            raise ValueError(
                f"Found {path} but no 'spatial/' directory alongside it "
                f"(expected tissue_positions.csv, scalefactors_json.json, and "
                f"at least one tissue image)."
            )
        elif os.path.isdir(path):
            # Directory containing Space Ranger output.
            h5_files = [f for f in os.listdir(path) if f.endswith(".h5")]
            spatial_dir = os.path.join(path, "spatial")
            if h5_files and os.path.isdir(spatial_dir):
                return _load_space_ranger_bundle(
                    matrix_h5_path=os.path.join(path, h5_files[0]), spatial_dir=spatial_dir,
                )
            if h5_files:
                # No spatial/ dir — fall back to scanpy (needs both images).
                import scanpy as sc
                return sc.read_visium(path, count_file=h5_files[0])
            h5ad_files = [f for f in os.listdir(path) if f.endswith(".h5ad")]
            if h5ad_files:
                return ad.read_h5ad(os.path.join(path, h5ad_files[0]))
        raise ValueError(f"Could not load Visium data from {path}")


# ---------------------------------------------------------------------------
# Slide-seqV2 loader
# ---------------------------------------------------------------------------

class SlideSeqLoader(BaseLoader):
    """Loader for Slide-seqV2 data.

    Accepts .h5ad files with bead coordinates in .obsm['spatial'].
    Beads are irregularly spaced (random layout).
    """

    platform_name = "Slide-seqV2"

    def _load_raw(self, path: str, **kwargs) -> ad.AnnData:
        if path.endswith(".h5ad"):
            adata = ad.read_h5ad(path)
            # Slide-seq beads may have coordinates in different obsm keys
            if "spatial" not in adata.obsm:
                for key in ["X_spatial", "coordinates", "bead_xy"]:
                    if key in adata.obsm:
                        adata.obsm["spatial"] = adata.obsm[key]
                        break
            return adata
        raise ValueError(f"Slide-seqV2 loader requires .h5ad file, got: {path}")


# ---------------------------------------------------------------------------
# MERFISH loader
# ---------------------------------------------------------------------------

class MERFISHLoader(BaseLoader):
    """Loader for MERFISH data.

    Accepts:
    - .h5ad files
    - .csv files with columns: x, y, gene, count (long format)
    - .csv files with gene columns (wide format) + x, y coordinate columns

    MERFISH produces single-cell resolution data. If n_cells > 50,000,
    coordinates are binned into pseudo-spots for compatibility with
    spot-based methods.
    """

    platform_name = "MERFISH"

    def _load_raw(self, path: str, bin_size: int | None = None, **kwargs) -> ad.AnnData:
        if path.endswith(".h5ad"):
            adata = ad.read_h5ad(path)
            if "spatial" not in adata.obsm:
                for key in ["X_spatial", "coordinates", "xy"]:
                    if key in adata.obsm:
                        adata.obsm["spatial"] = adata.obsm[key]
                        break
            return self._maybe_bin(adata, bin_size)

        elif path.endswith(".csv"):
            df = pd.read_csv(path)
            return self._parse_merfish_csv(df, bin_size)

        raise ValueError(f"MERFISH loader requires .h5ad or .csv, got: {path}")

    def _parse_merfish_csv(self, df: pd.DataFrame, bin_size: int | None) -> ad.AnnData:
        """Parse MERFISH CSV into AnnData.

        Handles both long format (x, y, gene, count) and wide format
        (x, y, gene1, gene2, ...).
        """
        coord_cols = self._find_coord_columns(df)

        if "gene" in df.columns and "count" in df.columns:
            # Long format: pivot to wide
            wide = df.pivot_table(
                index=[coord_cols[0], coord_cols[1]],
                columns="gene", values="count", fill_value=0,
            )
            coords = wide.index.to_frame().values.astype(float)
            adata = ad.AnnData(X=wide.values.astype(float))
            adata.obsm["spatial"] = coords
            adata.var_names = wide.columns
        else:
            # Wide format: x, y, gene1, gene2, ...
            gene_cols = [c for c in df.columns if c not in coord_cols]
            coords = df[coord_cols].values.astype(float)
            adata = ad.AnnData(X=df[gene_cols].values.astype(float))
            adata.obsm["spatial"] = coords
            adata.var_names = gene_cols

        adata.obs_names = [f"cell_{i}" for i in range(adata.shape[0])]
        return self._maybe_bin(adata, bin_size)

    def _find_coord_columns(self, df: pd.DataFrame) -> list[str]:
        """Find x, y coordinate columns in a DataFrame."""
        candidates = [
            (["x", "y"]), (["X", "Y"]), (["x_coord", "y_coord"]),
            (["centroid_x", "centroid_y"]), (["global_x", "global_y"]),
        ]
        for pair in candidates:
            if all(c in df.columns for c in pair):
                return pair
        raise ValueError("Could not find coordinate columns in MERFISH CSV")

    def _maybe_bin(self, adata: ad.AnnData, bin_size: int | None) -> ad.AnnData:
        """Bin single-cell data into pseudo-spots if too many cells."""
        n = adata.shape[0]
        if bin_size is not None or n > 50000:
            return self._bin_to_pseudo_spots(adata, bin_size or 50)
        return adata

    def _bin_to_pseudo_spots(self, adata: ad.AnnData, bin_size: float) -> ad.AnnData:
        """Bin single-cell coordinates into pseudo-spots.

        Sums counts within each spatial bin. Bin centers become new spot
        coordinates.
        """
        coords = np.array(adata.obsm["spatial"])
        # Create bin indices
        x_bins = (coords[:, 0] / bin_size).astype(int)
        y_bins = (coords[:, 1] / bin_size).astype(int)
        bin_ids = x_bins * 100000 + y_bins

        unique_bins, inverse = np.unique(bin_ids, return_inverse=True)

        # Aggregate counts per bin
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        binned = np.zeros((len(unique_bins), X.shape[1]))
        bin_coords = np.zeros((len(unique_bins), 2))
        bin_counts = np.zeros(len(unique_bins))

        for i, bin_id in enumerate(unique_bins):
            mask = bin_ids == bin_id
            binned[i] = X[mask].sum(axis=0)
            bin_coords[i] = coords[mask].mean(axis=0)
            bin_counts[i] = mask.sum()

        new_adata = ad.AnnData(X=binned)
        new_adata.obsm["spatial"] = bin_coords
        new_adata.var_names = adata.var_names
        new_adata.obs_names = [f"bin_{i}" for i in range(len(unique_bins))]
        new_adata.obs["n_cells_per_bin"] = bin_counts
        new_adata.uns["binning"] = {"bin_size": bin_size, "original_n_cells": n}

        return new_adata


# ---------------------------------------------------------------------------
# CosMx loader
# ---------------------------------------------------------------------------

class CosMxLoader(BaseLoader):
    """Loader for NanoString CosMx data.

    Accepts .csv files from NanoString's CosMx output.
    Typically has columns: x_local_px, y_local_px, fov, cell_id, and gene columns.
    """

    platform_name = "CosMx"

    def _load_raw(self, path: str, bin_size: int | None = None, **kwargs) -> ad.AnnData:
        if path.endswith(".h5ad"):
            adata = ad.read_h5ad(path)
            return adata
        elif path.endswith(".csv"):
            df = pd.read_csv(path)

            # Find coordinate columns
            coord_cols = None
            for pair in [("x_local_px", "y_local_px"), ("x", "y"), ("Center_X", "Center_Y")]:
                if all(c in df.columns for c in pair):
                    coord_cols = pair
                    break
            if coord_cols is None:
                raise ValueError("Could not find CosMx coordinate columns")

            # Find gene columns (non-metadata columns)
            meta_cols = set(coord_cols + ["fov", "cell_id", "cell_ID", "slide_id",
                           "fov_x", "fov_y", "Area", "Nucleus_Area"])
            gene_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype in [np.float64, np.int64]]

            coords = df[coord_cols].values.astype(float)
            adata = ad.AnnData(X=df[gene_cols].values.astype(float))
            adata.obsm["spatial"] = coords
            adata.var_names = gene_cols
            adata.obs_names = [f"cell_{i}" for i in range(adata.shape[0])]

            if "fov" in df.columns:
                adata.obs["fov"] = df["fov"].values

            return adata
        raise ValueError(f"CosMx loader requires .csv or .h5ad, got: {path}")


# ---------------------------------------------------------------------------
# Xenium loader
# ---------------------------------------------------------------------------

class XeniumLoader(BaseLoader):
    """Loader for 10x Xenium data.

    Accepts:
    - .h5ad files
    - Xenium output directory (containing cells.parquet and cell_metadata.csv)
    """

    platform_name = "Xenium"

    def _load_raw(self, path: str, bin_size: int | None = None, **kwargs) -> ad.AnnData:
        if path.endswith(".h5ad"):
            return ad.read_h5ad(path)

        if os.path.isdir(path):
            # Xenium output directory
            # Look for cells.parquet or cell_feature_matrix.h5
            h5_path = os.path.join(path, "cell_feature_matrix.h5")
            meta_path = os.path.join(path, "cells.parquet")

            if os.path.exists(h5_path) and os.path.exists(meta_path):
                import scanpy as sc
                adata = sc.read_10x_h5(h5_path)

                # Read coordinates
                meta = pd.read_parquet(meta_path)
                coord_cols = None
                for pair in [("x_centroid", "y_centroid"), ("x", "y")]:
                    if all(c in meta.columns for c in pair):
                        coord_cols = pair
                        break
                if coord_cols:
                    adata.obsm["spatial"] = meta[coord_cols].values.astype(float)

                return adata

        raise ValueError(f"Xenium loader requires .h5ad or output directory, got: {path}")


# ---------------------------------------------------------------------------
# Stereo-seq loader
# ---------------------------------------------------------------------------

class StereoSeqLoader(BaseLoader):
    """Loader for Stereo-seq data.

    Accepts .h5ad files with sub-cellular coordinates.
    Bins into pseudo-spots if n_cells > 50,000.
    """

    platform_name = "Stereo-seq"

    def _load_raw(self, path: str, bin_size: int | None = None, **kwargs) -> ad.AnnData:
        if path.endswith(".h5ad"):
            adata = ad.read_h5ad(path)
            if "spatial" not in adata.obsm:
                for key in ["X_spatial", "coordinates", "spatial_coordinator"]:
                    if key in adata.obsm:
                        adata.obsm["spatial"] = adata.obsm[key]
                        break

            # Stereo-seq has very high resolution; bin if needed
            n = adata.shape[0]
            if bin_size is not None or n > 50000:
                return self._bin(adata, bin_size or 100)
            return adata

        raise ValueError(f"Stereo-seq loader requires .h5ad, got: {path}")

    def _bin(self, adata: ad.AnnData, bin_size: float) -> ad.AnnData:
        """Bin sub-cellular data into pseudo-spots."""
        coords = np.array(adata.obsm["spatial"])
        x_bins = (coords[:, 0] / bin_size).astype(int)
        y_bins = (coords[:, 1] / bin_size).astype(int)
        bin_ids = x_bins * 1000000 + y_bins

        unique_bins = np.unique(bin_ids)
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()

        binned = np.zeros((len(unique_bins), X.shape[1]))
        bin_coords = np.zeros((len(unique_bins), 2))

        for i, bin_id in enumerate(unique_bins):
            mask = bin_ids == bin_id
            binned[i] = X[mask].sum(axis=0)
            bin_coords[i] = coords[mask].mean(axis=0)

        new_adata = ad.AnnData(X=binned)
        new_adata.obsm["spatial"] = bin_coords
        new_adata.var_names = adata.var_names
        new_adata.obs_names = [f"bin_{i}" for i in range(len(unique_bins))]
        new_adata.uns["binning"] = {"bin_size": bin_size, "original_n": adata.shape[0]}
        return new_adata


# ---------------------------------------------------------------------------
# DBiT-seq loader
# ---------------------------------------------------------------------------

class DBiTLoader(BaseLoader):
    """Loader for DBiT-seq data.

    Accepts .h5ad files. DBiT-seq uses a microfluidic chip with regular
    grid spots, similar to Visium but potentially different spot sizes.
    """

    platform_name = "DBiT-seq"

    def _load_raw(self, path: str, **kwargs) -> ad.AnnData:
        if path.endswith(".h5ad"):
            adata = ad.read_h5ad(path)
            if "spatial" not in adata.obsm:
                for key in ["X_spatial", "coordinates"]:
                    if key in adata.obsm:
                        adata.obsm["spatial"] = adata.obsm[key]
                        break
            return adata
        raise ValueError(f"DBiT-seq loader requires .h5ad, got: {path}")


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------

LOADER_REGISTRY = {
    "Visium": VisiumLoader,
    "Slide-seqV2": SlideSeqLoader,
    "MERFISH": MERFISHLoader,
    "CosMx": CosMxLoader,
    "Xenium": XeniumLoader,
    "Stereo-seq": StereoSeqLoader,
    "DBiT-seq": DBiTLoader,
}


def get_loader(platform: str) -> BaseLoader:
    """Get a loader instance for a platform.

    Parameters
    ----------
    platform : str
        Platform name (case-insensitive).

    Returns
    -------
    BaseLoader subclass instance.
    """
    # Case-insensitive matching
    for key, cls in LOADER_REGISTRY.items():
        if key.lower() == platform.lower():
            return cls()
    raise ValueError(
        f"Unknown platform: {platform}. Available: {list(LOADER_REGISTRY.keys())}"
    )


def load_data(
    path: str,
    platform: str | None = None,
    sample_id: str | None = None,
    ground_truth_col: str | None = None,
    **kwargs,
) -> ad.AnnData:
    """Load spatial transcriptomics data from any supported platform.

    If platform is None, attempts auto-detection from file structure.

    Parameters
    ----------
    path : str
        Path to data file or directory.
    platform : str, optional
        Platform name. If None, auto-detected.
    sample_id : str, optional
    ground_truth_col : str, optional
    **kwargs : platform-specific options (e.g., bin_size for MERFISH).

    Returns
    -------
    AnnData with standardized fields.
    """
    if platform is None:
        platform = auto_detect_platform(path)

    loader = get_loader(platform)
    return loader.load(path, sample_id=sample_id, ground_truth_col=ground_truth_col, **kwargs)


def auto_detect_platform(path: str) -> str:
    """Auto-detect the spatial transcriptomics platform from file structure.

    Heuristics:
    - .h5 with "filtered_feature_bc_matrix" → Visium
    - Directory with "cell_feature_matrix.h5" + "cells.parquet" → Xenium
    - .csv with "x_local_px" column → CosMx
    - .h5ad with "platform" in .uns → use that
    - .h5ad → default to Visium (most common)
    """
    if path.endswith(".h5ad"):
        try:
            adata = ad.read_h5ad(path, backed="r")
            if "platform" in adata.uns:
                platform = str(adata.uns["platform"])
                adata.file.close()
                return platform
            adata.file.close()
        except Exception:
            pass
        return "Visium"  # default for .h5ad

    if path.endswith(".h5"):
        return "Visium"

    if path.endswith(".csv"):
        # Peek at columns
        try:
            df = pd.read_csv(path, nrows=2)
            if "x_local_px" in df.columns:
                return "CosMx"
        except Exception:
            pass
        return "MERFISH"  # default for CSV

    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "cell_feature_matrix.h5")):
            return "Xenium"
        if any(f.endswith(".h5") for f in os.listdir(path)):
            return "Visium"

    return "Visium"  # fallback
