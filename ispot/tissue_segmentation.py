"""Tissue boundary detection from histology image pixels.

No new dependencies: numpy + scipy.ndimage only. Used to exclude off-tissue
capture spots and to render the real tissue outline in the viewer (instead of
a bounding box / oval).
"""
from __future__ import annotations

import numpy as np


def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
    img = image.astype(float)
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 3 and img.shape[-1] >= 3:
        return 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    return img if img.ndim == 2 else img[..., 0]


def _otsu_threshold(gray: np.ndarray, n_bins: int = 256) -> float:
    hist, bin_edges = np.histogram(gray.ravel(), bins=n_bins, range=(0.0, 1.0))
    hist = hist.astype(float)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return 0.5
    sum_all = np.sum(hist * bin_centers)
    sum_bg, weight_bg = 0.0, 0.0
    best_idx, best_between_var = 0, -1.0
    for i in range(n_bins):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += hist[i] * bin_centers[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        # >= is deliberate: ties across an empty gap between two clean
        # clusters must advance to the LAST tied bin (middle of the gap),
        # not the first (edge of the lower cluster) — using > here silently
        # produces an empty foreground class and 0 detected tissue.
        if between_var >= best_between_var:
            best_between_var = between_var
            best_idx = i
    return float(bin_edges[min(best_idx + 1, n_bins)])


def detect_tissue_mask(image: np.ndarray, min_component_fraction: float = 0.01) -> np.ndarray:
    from scipy import ndimage
    gray = _rgb_to_gray(image)
    thresh = _otsu_threshold(gray)
    candidates = []
    for mask in (gray < thresh, gray >= thresh):
        cleaned = ndimage.binary_fill_holes(mask)
        labeled, n_components = ndimage.label(cleaned)
        if n_components == 0:
            candidates.append((0, cleaned))
            continue
        sizes = ndimage.sum(cleaned, labeled, range(1, n_components + 1))
        largest_size = sizes.max()
        keep_labels = {i + 1 for i, s in enumerate(sizes) if s >= min_component_fraction * largest_size}
        final_mask = np.isin(labeled, list(keep_labels))
        candidates.append((final_mask.sum(), final_mask))
    h, w = gray.shape
    total_px = h * w
    best_mask, best_score = None, -1.0
    for area, mask in candidates:
        frac = area / total_px
        score = 1.0 - abs(frac - 0.35) if 0.02 <= frac <= 0.9 else -abs(frac - 0.35)
        if score > best_score:
            best_score, best_mask = score, mask
    return best_mask if best_mask is not None else np.ones_like(gray, dtype=bool)


def spots_in_tissue_mask(spatial_coords: np.ndarray, tissue_mask: np.ndarray, scale_factor: float) -> np.ndarray:
    h, w = tissue_mask.shape
    img_coords = spatial_coords * scale_factor
    cols = np.clip(img_coords[:, 0].astype(int), 0, w - 1)
    rows = np.clip(img_coords[:, 1].astype(int), 0, h - 1)
    return tissue_mask[rows, cols]


def detect_tissue_by_expression_density(
    coords: np.ndarray,
    total_counts: np.ndarray,
    grid_bins: int = 100,
    min_component_fraction: float = 0.02,
) -> np.ndarray:
    """Detect real tissue coverage from expression density alone, for
    platforms with no histology image and no in_tissue metadata (Slide-seq,
    Stereo-seq, DBiT-seq). Without this, every raw bead/spot on the device
    is treated as "on tissue," so the viewer shows the device's raw
    geometry (a round Slide-seq puck, a rectangular Stereo-seq/DBiT-seq
    chip) instead of the actual, usually smaller and irregular, tissue
    footprint placed on it.

    Real tissue-covered spots have substantially higher total UMI/count
    density than background/ambient spots picking up only stray RNA — a
    standard heuristic in the Slide-seq/HDST literature. This bins spots onto
    a coarse 2D grid, applies the same Otsu thresholding used for image-based
    detection (reused here on count density instead of pixel intensity), and
    keeps only the largest spatially-contiguous high-density region — so
    isolated high-count outlier spots don't get treated as separate "tissue
    islands," and the result is a real, contiguous tissue shape.

    Fails open: on too-few spots or no discriminating signal it returns all
    True (keep everything), so it can never wipe out a dataset.

    Returns
    -------
    np.ndarray: boolean array, shape (n_spots,). True = on real tissue.
    """
    from scipy import ndimage

    coords = np.asarray(coords, dtype=float)
    total_counts = np.asarray(total_counts, dtype=float).ravel()
    n = len(coords)
    if n < 20:
        return np.ones(n, dtype=bool)  # too few spots to estimate density

    x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
    x_max, y_max = coords[:, 0].max(), coords[:, 1].max()
    x_range = max(x_max - x_min, 1e-9)
    y_range = max(y_max - y_min, 1e-9)

    x_bins = np.clip(((coords[:, 0] - x_min) / x_range * (grid_bins - 1)).astype(int), 0, grid_bins - 1)
    y_bins = np.clip(((coords[:, 1] - y_min) / y_range * (grid_bins - 1)).astype(int), 0, grid_bins - 1)

    # Mean log1p-count density per occupied grid cell (occupancy-normalized so
    # one very high-count spot doesn't beat a cell full of moderate spots).
    density_grid = np.zeros((grid_bins, grid_bins))
    occupancy_grid = np.zeros((grid_bins, grid_bins))
    np.add.at(density_grid, (y_bins, x_bins), np.log1p(total_counts))
    np.add.at(occupancy_grid, (y_bins, x_bins), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_density_grid = np.where(occupancy_grid > 0, density_grid / np.maximum(occupancy_grid, 1), 0.0)

    finite_vals = mean_density_grid[occupancy_grid > 0]
    if len(finite_vals) == 0 or finite_vals.max() == finite_vals.min():
        return np.ones(n, dtype=bool)  # no discriminating signal — don't filter
    normalized = (mean_density_grid - finite_vals.min()) / (finite_vals.max() - finite_vals.min())

    thresh = _otsu_threshold(normalized)
    high_density_mask = (normalized >= thresh) & (occupancy_grid > 0)

    # Keep only the largest spatially-contiguous high-density region(s).
    labeled, n_components = ndimage.label(high_density_mask)
    if n_components == 0:
        return np.ones(n, dtype=bool)
    sizes = ndimage.sum(high_density_mask, labeled, range(1, n_components + 1))
    largest_size = sizes.max()
    keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_component_fraction * largest_size]
    final_grid_mask = np.isin(labeled, keep_labels)

    return final_grid_mask[y_bins, x_bins]


def downsample_mask_for_viewer(mask: np.ndarray, max_dim: int = 150) -> dict:
    from scipy import ndimage
    h, w = mask.shape
    scale = min(1.0, max_dim / max(h, w))
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    downsampled = ndimage.zoom(mask.astype(float), (new_h / h, new_w / w), order=1) > 0.5
    rows = ["".join("1" if v else "0" for v in row) for row in downsampled]
    return {"rows": rows, "height": new_h, "width": new_w, "original_height": h, "original_width": w}
