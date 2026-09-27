"""Vector helpers: polygonise masks tile-by-tile, metric areas, saving."""
from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import box, shape
from shapely.ops import unary_union

from .io import iter_windows


def polygonize_mask(mask_path: str, value_field: str = "value", tile: int = 4096,
                    min_area_m2: float = 0.0, keep_values: list[int] | None = None) -> gpd.GeoDataFrame:
    """Polygonise a (large) integer mask raster, merging pieces cut at tile edges.

    Zero / nodata pixels are ignored. Each distinct non-zero value becomes its own class.
    """
    records, edge_pieces = [], {}
    with rasterio.open(mask_path) as src:
        crs = src.crs
        for win in iter_windows(src.width, src.height, tile, 0):
            arr = src.read(1, window=win)
            if not arr.any():
                continue
            tr = src.window_transform(win)
            win_box = box(*rasterio.windows.bounds(win, src.transform))
            inner = win_box.buffer(-abs(src.res[0]) * 0.5)
            for geom, val in shapes(arr, mask=arr > 0, transform=tr):
                val = int(val)
                if keep_values and val not in keep_values:
                    continue
                g = shape(geom)
                if not g.within(inner):  # touches tile edge -> merge later
                    edge_pieces.setdefault(val, []).append(g)
                else:
                    records.append((val, g))
    for val, pieces in edge_pieces.items():
        merged = unary_union(pieces)
        for g in getattr(merged, "geoms", [merged]):
            records.append((val, g))
    gdf = gpd.GeoDataFrame({value_field: [r[0] for r in records]},
                           geometry=[r[1] for r in records], crs=crs)
    if len(gdf):
        gdf = add_metrics(gdf)
        if min_area_m2 > 0:
            gdf = gdf[gdf["area_m2"] >= min_area_m2].reset_index(drop=True)
    return gdf


def metric_crs(gdf: gpd.GeoDataFrame):
    if gdf.crs is None:
        return None
    if gdf.crs.is_projected:
        return gdf.crs
    return gdf.estimate_utm_crs()


def add_metrics(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        gdf["area_m2"] = []
        return gdf
    m = gdf.to_crs(metric_crs(gdf)) if gdf.crs is not None else gdf
    gdf = gdf.copy()
    gdf["area_m2"] = m.geometry.area.round(2).values
    gdf["perim_m"] = m.geometry.length.round(2).values
    return gdf


def save_vector(gdf: gpd.GeoDataFrame, path: str, layer: str | None = None) -> str:
    """Save as GeoPackage (default, opens in ArcGIS Pro/QGIS), GeoJSON or Shapefile."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".geojson":
        gdf.to_crs(4326).to_file(path, driver="GeoJSON")
    elif ext == ".shp":
        gdf.to_file(path)
    else:
        gdf.to_file(path, layer=layer or os.path.splitext(os.path.basename(path))[0], driver="GPKG")
    return path


def raster_footprint(path: str) -> gpd.GeoDataFrame:
    """Valid-data footprint of a raster (approximate, from an overview)."""
    from rasterio.enums import Resampling
    with rasterio.open(path) as src:
        scale = max(src.width, src.height) / 1024
        scale = max(scale, 1)
        h, w = int(src.height / scale), int(src.width / scale)
        m = src.dataset_mask(out_shape=(h, w), resampling=Resampling.nearest)
        arr = src.read(1, out_shape=(h, w), resampling=Resampling.nearest)
        valid = (m > 0) & (arr != 0) if src.nodata is None else (m > 0)
        tr = src.transform * src.transform.scale(src.width / w, src.height / h)
        geoms = [shape(g) for g, v in shapes(valid.astype(np.uint8), mask=valid, transform=tr)]
        crs = src.crs
    if not geoms:
        with rasterio.open(path) as src:
            geoms = [box(*src.bounds)]
    aoi = unary_union(geoms).buffer(0)
    return gpd.GeoDataFrame(geometry=[aoi], crs=crs)
