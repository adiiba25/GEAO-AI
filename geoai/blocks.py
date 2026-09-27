"""Urban blocks: areas enclosed by the road network.

Road sources
  osm       download from OpenStreetMap for the imagery footprint (needs osmnx + internet)
  file      your own road centrelines (GPKG/SHP/GeoJSON), optionally with a width column
  mask      a detected road mask raster (from detect.segment_raster)
Blocks = AOI minus buffered roads, exploded, with building / vegetation stats attached.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

from .vector import add_metrics, metric_crs, polygonize_mask

# default carriageway widths (m) by OSM highway class
ROAD_WIDTHS = {"motorway": 16, "trunk": 14, "primary": 12, "secondary": 10, "tertiary": 8,
               "unclassified": 6, "residential": 6, "service": 4, "living_street": 5,
               "track": 3, "path": 1.5, "footway": 1.5}
DRIVABLE = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
            "residential", "service", "living_street", "motorway_link", "trunk_link",
            "primary_link", "secondary_link", "tertiary_link", "track"}


def roads_from_osm(aoi: gpd.GeoDataFrame, drivable_only: bool = True) -> gpd.GeoDataFrame:
    import osmnx as ox
    poly = aoi.to_crs(4326).union_all().buffer(0.001)
    feats = ox.features_from_polygon(poly, tags={"highway": True})
    feats = feats[feats.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    feats["highway"] = feats["highway"].apply(lambda v: v[0] if isinstance(v, list) else v)
    if drivable_only:
        feats = feats[feats["highway"].isin(DRIVABLE)]
    return feats[["highway", "geometry"]].reset_index(drop=True)


def roads_from_file(path: str) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def buffer_roads(roads: gpd.GeoDataFrame, default_width: float = 6.0,
                 width_field: str | None = "width_m") -> gpd.GeoDataFrame:
    crs = metric_crs(roads)
    r = roads.to_crs(crs)
    widths = []
    for _, row in r.iterrows():
        w = None
        if width_field and width_field in r.columns and pd.notna(row.get(width_field)):
            try:
                w = float(row[width_field])
            except (TypeError, ValueError):
                w = None
        if w is None and "highway" in r.columns:
            hw = str(row["highway"]).replace("_link", "")
            w = ROAD_WIDTHS.get(hw)
        widths.append(w or default_width)
    r = r.set_geometry([g.buffer(w / 2, cap_style="flat") for g, w in zip(r.geometry, widths)])
    return r


def make_blocks(aoi: gpd.GeoDataFrame, road_polys: gpd.GeoDataFrame, min_area_m2: float = 200,
                buildings: gpd.GeoDataFrame | None = None,
                vegetation: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    crs = metric_crs(aoi)
    a = aoi.to_crs(crs).union_all()
    roads = road_polys.to_crs(crs).union_all() if len(road_polys) else None
    area = a.difference(roads) if roads is not None else a
    parts = [g for g in getattr(area, "geoms", [area]) if g.area >= min_area_m2]
    # number blocks north-to-south, west-to-east
    order = np.lexsort(([g.centroid.x for g in parts], [-round(g.centroid.y, -1) for g in parts]))
    blocks = gpd.GeoDataFrame(geometry=[parts[i] for i in order], crs=crs)
    blocks.insert(0, "block_id", [f"BLK{i + 1:04d}" for i in range(len(blocks))])
    blocks = add_metrics(blocks)
    blocks["area_ha"] = (blocks["area_m2"] / 10_000).round(3)
    if buildings is not None and len(buildings):
        b = buildings.to_crs(crs)
        cent = b.copy()
        cent["geometry"] = b.geometry.representative_point()
        cent["_barea"] = b.geometry.area
        j = gpd.sjoin(cent[["_barea", "geometry"]], blocks[["block_id", "geometry"]],
                      predicate="within", how="inner")
        agg = j.groupby("block_id").agg(bldg_count=("_barea", "size"), bldg_area_m2=("_barea", "sum"))
        blocks = blocks.merge(agg, on="block_id", how="left")
        blocks[["bldg_count", "bldg_area_m2"]] = blocks[["bldg_count", "bldg_area_m2"]].fillna(0)
        blocks["bldg_count"] = blocks["bldg_count"].astype(int)
        blocks["bldg_cover_pct"] = (100 * blocks["bldg_area_m2"] / blocks["area_m2"]).round(1)
        blocks["bldg_per_ha"] = (blocks["bldg_count"] / blocks["area_ha"]).round(1)
    if vegetation is not None and len(vegetation):
        v = vegetation.to_crs(crs).union_all()
        blocks["veg_cover_pct"] = [round(100 * g.intersection(v).area / g.area, 1) for g in blocks.geometry]
    return blocks.to_crs(aoi.crs)


def road_polys_from_mask(mask_path: str, min_area_m2: float = 50) -> gpd.GeoDataFrame:
    return polygonize_mask(mask_path, min_area_m2=min_area_m2)
