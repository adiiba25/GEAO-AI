"""Building footprint extraction and regularisation.

Two routes to a building mask:
  1. Height route (drone): nDSM >= min_height AND not vegetation. Very reliable for
     drone surveys where DJI Terra (or similar) gives a DSM and DTM.
  2. Model route (drone or high-res satellite): a segmentation mask from detect.py
     (zero-shot LangSAM, or your own trained YOLO-seg / U-Net weights).
Either mask is polygonised and regularised into clean footprints.
"""
from __future__ import annotations

import math
import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from scipy import ndimage

from . import indices
from .io import align_to, iter_windows, mask_profile, read_bands, same_grid
from .vector import add_metrics, metric_crs, polygonize_mask


def building_mask_from_height(path: str, band_map: dict, height_path: str, out_path: str,
                              min_height: float = 2.5, max_height: float = 80.0,
                              veg_threshold: float | None = None, tile: int = 2048,
                              progress=None) -> str:
    from .vegetation import estimate_threshold
    index_name = indices.vegetation_index_name(band_map)
    if veg_threshold is None:
        veg_threshold = estimate_threshold(path, band_map, index_name)
    if not same_grid(height_path, path):
        height_path = align_to(height_path, path,
                               os.path.join(os.path.dirname(os.path.abspath(out_path)), "height_aligned.tif"))
    with rasterio.open(path) as src, rasterio.open(height_path) as hs, \
            rasterio.open(out_path, "w", **mask_profile(src)) as dst:
        wins = list(iter_windows(src.width, src.height, tile, 0))
        for i, win in enumerate(wins):
            b = read_bands(src, band_map, window=win)
            veg = np.nan_to_num(indices.compute_index(index_name, b), nan=-9) > veg_threshold
            h = hs.read(1, window=win).astype("float32")
            if hs.nodata is not None:
                h[h == hs.nodata] = np.nan
            h = np.nan_to_num(h)
            bld = (h >= min_height) & (h <= max_height) & ~veg
            bld = ndimage.binary_opening(bld, iterations=2)
            bld = ndimage.binary_fill_holes(bld)
            dst.write(bld.astype(np.uint8), 1, window=win)
            if progress:
                progress((i + 1) / len(wins))
    return out_path


def regularize(gdf: gpd.GeoDataFrame, simplify_m: float = 0.5,
               rect_threshold: float = 0.80) -> gpd.GeoDataFrame:
    """Square-up footprints: near-rectangular shapes -> minimum rotated rectangle,
    others -> simplified outline. Works in a metric CRS then returns original CRS."""
    if gdf.empty:
        return gdf
    crs = gdf.crs
    m = gdf.to_crs(metric_crs(gdf))
    geoms, rect, orient = [], [], []
    for g in m.geometry:
        g = g.buffer(0)
        if g.is_empty:
            geoms.append(g); rect.append(0); orient.append(0)
            continue
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        g = type(g)(g.exterior)  # drop interior courtyards/holes from noise
        mrr = g.minimum_rotated_rectangle
        r = g.area / mrr.area if mrr.area else 0
        out = mrr if r >= rect_threshold else g.simplify(simplify_m, preserve_topology=True)
        geoms.append(out)
        rect.append(round(r, 3))
        orient.append(round(_orientation(mrr), 1))
    m = m.set_geometry(geoms)
    m["rectangularity"] = rect
    m["orient_deg"] = orient
    return m.to_crs(crs)


def _orientation(poly) -> float:
    xs, ys = poly.exterior.coords.xy
    best = 0.0
    longest = 0.0
    for i in range(len(xs) - 1):
        dx, dy = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
        L = math.hypot(dx, dy)
        if L > longest:
            longest, best = L, math.degrees(math.atan2(dy, dx)) % 180
    return best


def footprints_from_mask(mask_path: str, out_path: str | None = None, min_area_m2: float = 12.0,
                         max_area_m2: float = 50_000, simplify_m: float = 0.5,
                         height_path: str | None = None) -> gpd.GeoDataFrame:
    gdf = polygonize_mask(mask_path, min_area_m2=min_area_m2)
    if gdf.empty:
        return gdf
    gdf = gdf[gdf["area_m2"] <= max_area_m2]
    gdf = regularize(gdf, simplify_m=simplify_m)
    gdf = add_metrics(gdf)
    gdf = gdf[gdf["area_m2"] >= min_area_m2].reset_index(drop=True)
    if height_path:
        gdf["height_m"] = zonal_mean(height_path, gdf)
    gdf.insert(0, "bldg_id", [f"B{i + 1:06d}" for i in range(len(gdf))])
    gdf = gdf.drop(columns=[c for c in ("value",) if c in gdf.columns])
    if out_path:
        from .vector import save_vector
        save_vector(gdf, out_path, layer="building_footprints")
    return gdf


def zonal_mean(raster_path: str, gdf: gpd.GeoDataFrame, band: int = 1) -> list:
    vals = []
    with rasterio.open(raster_path) as src:
        g = gdf.to_crs(src.crs)
        for geom in g.geometry:
            try:
                arr, _ = rio_mask(src, [geom], crop=True, indexes=band, filled=False)
                v = arr.compressed()
                v = v[np.isfinite(v)]
                vals.append(round(float(np.median(v)), 2) if v.size else None)
            except Exception:
                vals.append(None)
    return vals
