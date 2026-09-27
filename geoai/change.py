"""Change detection between two dates.

Raster change
  index : difference of a spectral index (NDVI for multispectral / Sentinel / Landsat,
          ExG for RGB drone). Output 1 = decrease (e.g. vegetation loss), 2 = increase.
  cva   : Change Vector Analysis magnitude over all bands after radiometric
          normalisation. Output 1 = changed.
  height: difference of two nDSMs/DSMs (drone). 1 = height loss (demolition, tree
          cut), 2 = height gain (new structure / growth).
Thresholds default to Otsu on the difference image; you can override them.

Object change
  footprint_change compares building footprints from two dates -> new / demolished /
  modified / unchanged.
"""
from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
import rasterio
from scipy import ndimage

from . import indices
from .io import align_to, iter_windows, mask_profile, overview_shape, read_bands, same_grid

DECREASE, INCREASE, CHANGED = 1, 2, 1


def _align(path1, path2, workdir, name="t2_aligned.tif"):
    if same_grid(path1, path2):
        return path2
    return align_to(path2, path1, os.path.join(workdir, name))


def _norm_params(src1, src2, band_map):
    """Mean/std matching of date 2 to date 1 per band (relative radiometric normalisation)."""
    shp = overview_shape(src1, 1500)
    b1 = read_bands(src1, band_map, out_shape=shp)
    b2 = read_bands(src2, band_map, out_shape=shp)
    params = {}
    for k in b1:
        v = np.isfinite(b1[k]) & np.isfinite(b2[k])
        m1, s1 = np.mean(b1[k][v]), np.std(b1[k][v]) or 1
        m2, s2 = np.mean(b2[k][v]), np.std(b2[k][v]) or 1
        params[k] = (m1, s1, m2, s2)
    return params


def _apply_norm(b2, params):
    return {k: (v - params[k][2]) / params[k][3] * params[k][1] + params[k][0] for k, v in b2.items()}


def raster_change(path1: str, path2: str, band_map: dict, out_path: str, method: str = "index",
                  index_name: str | None = None, threshold: float | None = None,
                  normalize: bool = True, min_pixels: int = 50, tile: int = 2048,
                  progress=None) -> dict:
    workdir = os.path.dirname(os.path.abspath(out_path))
    p2 = _align(path1, path2, workdir)
    if method == "index" and index_name is None:
        index_name = indices.vegetation_index_name(band_map)

    with rasterio.open(path1) as s1, rasterio.open(p2) as s2:
        params = _norm_params(s1, s2, band_map) if (normalize and method != "height") else None
        stds = {k: v[1] for k, v in params.items()} if params else None

        def diff(win=None, out_shape=None):
            if method == "height":
                a = s1.read(1, window=win, out_shape=out_shape).astype("float32")
                b = s2.read(1, window=win, out_shape=out_shape).astype("float32")
                for arr, s in ((a, s1), (b, s2)):
                    if s.nodata is not None:
                        arr[arr == s.nodata] = np.nan
                return b - a
            b1 = read_bands(s1, band_map, window=win, out_shape=out_shape)
            b2 = read_bands(s2, band_map, window=win, out_shape=out_shape)
            if params:
                b2 = _apply_norm(b2, params)
            if method == "index":
                return indices.compute_index(index_name, b2) - indices.compute_index(index_name, b1)
            # CVA: euclidean magnitude of standardised band differences
            sq = sum(((b2[k] - b1[k]) / (stds[k] if stds else 1)) ** 2 for k in b1)
            return np.sqrt(sq)

        if threshold is None:
            d = diff(out_shape=overview_shape(s1, 2048))
            if method == "height":
                threshold = 2.5
            else:
                threshold = indices.otsu(np.abs(d) if method == "index" else d)
                # guard against Otsu splitting pure noise in near-identical scenes
                if method == "index":
                    threshold = max(threshold, 3 * float(np.nanstd(d)) if np.isfinite(d).any() else 0)
        counts = np.zeros(3, np.int64)
        with rasterio.open(out_path, "w", **mask_profile(s1)) as dst:
            wins = list(iter_windows(s1.width, s1.height, tile, 0))
            for i, win in enumerate(wins):
                d = np.nan_to_num(diff(win))
                out = np.zeros(d.shape, np.uint8)
                if method == "cva":
                    out[_clean(d > threshold, min_pixels)] = CHANGED
                else:
                    out[_clean(d < -threshold, min_pixels)] = DECREASE
                    out[_clean(d > threshold, min_pixels)] = INCREASE
                dst.write(out, 1, window=win)
                counts += np.bincount(out.ravel(), minlength=3)[:3]
                if progress:
                    progress((i + 1) / len(wins))
    tot = int(counts.sum()) or 1
    res = {"method": method, "index": index_name, "threshold": round(float(threshold), 4),
           "mask": out_path}
    if method == "cva":
        res["changed_pct"] = round(100 * float(counts[1]) / tot, 2)
    else:
        res["decrease_pct"] = round(100 * float(counts[1]) / tot, 2)
        res["increase_pct"] = round(100 * float(counts[2]) / tot, 2)
    return res


def _clean(mask, min_pixels):
    mask = ndimage.binary_opening(mask, iterations=1)
    if min_pixels <= 1 or not mask.any():
        return mask
    lab, n = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[lab]


def footprint_change(before: gpd.GeoDataFrame, after: gpd.GeoDataFrame, iou_match: float = 0.3,
                     modified_ratio: float = 0.25) -> gpd.GeoDataFrame:
    """Classify buildings as new / demolished / modified / unchanged between two dates."""
    if after.crs != before.crs:
        after = after.to_crs(before.crs)
    b = before.reset_index(drop=True)
    a = after.reset_index(drop=True)
    matched_b, rows = set(), []
    sidx = b.sindex
    for i, ga in enumerate(a.geometry):
        best_j, best_iou = None, 0.0
        for j in sidx.query(ga, predicate="intersects"):
            gb = b.geometry.iloc[j]
            inter = ga.intersection(gb).area
            iou = inter / (ga.area + gb.area - inter)
            if iou > best_iou:
                best_j, best_iou = int(j), iou
        if best_j is None or best_iou < iou_match:
            rows.append(("new", ga, None, ga.area, round(best_iou, 3)))
        else:
            matched_b.add(best_j)
            gb = b.geometry.iloc[best_j]
            ratio = abs(ga.area - gb.area) / max(gb.area, 1e-9)
            status = "modified" if ratio > modified_ratio or best_iou < 0.6 else "unchanged"
            rows.append((status, ga, gb.area, ga.area, round(best_iou, 3)))
    for j, gb in enumerate(b.geometry):
        if j not in matched_b:
            rows.append(("demolished", gb, gb.area, None, 0.0))
    out = gpd.GeoDataFrame({"status": [r[0] for r in rows], "iou": [r[4] for r in rows]},
                           geometry=[r[1] for r in rows], crs=before.crs)
    from .vector import add_metrics
    return add_metrics(out)
