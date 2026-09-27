"""Vegetation and tree-canopy mapping from RGB (ExG) or multispectral (NDVI) imagery.

With an optional height model (nDSM / CHM = height above ground, e.g. DSM minus DTM
from DJI Terra) vegetation is split into low vegetation (1) and trees (2).
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import rasterio
from scipy import ndimage

from . import indices
from .io import align_to, iter_windows, mask_profile, overview_shape, read_bands, same_grid

LOW_VEG, TREE = 1, 2


def estimate_threshold(path: str, band_map: dict, index_name: str) -> float:
    with rasterio.open(path) as src:
        bands = read_bands(src, band_map, out_shape=overview_shape(src, 2048))
    vals = indices.compute_index(index_name, bands)
    t = indices.otsu(vals)
    # sensible floors so bare/urban scenes don't get half-labelled as vegetation
    floor = 0.2 if index_name == "NDVI" else 0.05
    return max(t, floor)


def prepare_height(height_path: str | None, ref_path: str, workdir: str) -> str | None:
    if not height_path:
        return None
    if same_grid(height_path, ref_path):
        return height_path
    out = os.path.join(workdir, "height_aligned.tif")
    return align_to(height_path, ref_path, out)


def vegetation_mask(path: str, band_map: dict, out_path: str, threshold: float | None = None,
                    height_path: str | None = None, tree_min_height: float = 2.5,
                    min_pixels: int = 20, tile: int = 2048, progress=None) -> dict:
    """Write a uint8 mask (0 none, 1 low veg, 2 tree) and return stats."""
    index_name = indices.vegetation_index_name(band_map)
    if threshold is None:
        threshold = estimate_threshold(path, band_map, index_name)
    workdir = os.path.dirname(os.path.abspath(out_path)) or tempfile.gettempdir()
    hpath = prepare_height(height_path, path, workdir)
    counts = np.zeros(3, dtype=np.int64)
    with rasterio.open(path) as src, rasterio.open(out_path, "w", **mask_profile(src)) as dst:
        hsrc = rasterio.open(hpath) if hpath else None
        wins = list(iter_windows(src.width, src.height, tile, 0))
        for i, win in enumerate(wins):
            b = read_bands(src, band_map, window=win)
            idx = indices.compute_index(index_name, b)
            veg = np.nan_to_num(idx, nan=-9) > threshold
            veg = ndimage.binary_opening(veg, iterations=1)
            veg = _remove_small(veg, min_pixels)
            out = veg.astype(np.uint8) * LOW_VEG
            if hsrc is not None:
                h = hsrc.read(1, window=win).astype("float32")
                if hsrc.nodata is not None:
                    h[h == hsrc.nodata] = np.nan
                out[veg & (np.nan_to_num(h) >= tree_min_height)] = TREE
            dst.write(out, 1, window=win)
            counts += np.bincount(out.ravel(), minlength=3)[:3]
            if progress:
                progress((i + 1) / len(wins))
        if hsrc is not None:
            hsrc.close()
        px_area = abs(src.res[0] * src.res[1])
    total = int(counts.sum())
    counts = [int(c) for c in counts]
    return {"index": index_name, "threshold": round(float(threshold), 4),
            "low_veg_pct": round(100 * counts[1] / total, 2) if total else 0,
            "tree_pct": round(100 * counts[2] / total, 2) if total else 0,
            "veg_pct": round(100 * (counts[1] + counts[2]) / total, 2) if total else 0,
            "pixel_area_crs_units": px_area, "mask": out_path}


def _remove_small(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1 or not mask.any():
        return mask
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[lab]
