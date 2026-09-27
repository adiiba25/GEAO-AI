"""GeoAI Studio command line - batch processing (e.g. on a Vast.ai GPU box).

Examples
  python cli.py info ortho.tif
  python cli.py vegetation ortho.tif --ndsm ndsm.tif -o out/
  python cli.py footprints ortho.tif --ndsm ndsm.tif -o out/
  python cli.py segment ortho.tif --target building --backend langsam -o out/
  python cli.py vehicles ortho.tif --weights yolo26m-obb.pt -o out/
  python cli.py change t1.tif t2.tif --method index -o out/
  python cli.py change t1_ndsm.tif t2_ndsm.tif --method height -o out/
  python cli.py footprint-change out/buildings_2024.gpkg out/buildings_2026.gpkg -o out/
  python cli.py blocks ortho.tif --roads osm --buildings out/buildings.gpkg -o out/
Use --preset to pick the band order (default RGB). Run  python cli.py presets  to list.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import geopandas as gpd


def _bm(args):
    from geoai import indices
    names = list(indices.PRESETS)
    p = args.preset
    if p.isdigit():
        p = names[int(p)]
    match = [n for n in names if n.lower().startswith(p.lower())] or [p]
    return dict(indices.PRESETS[match[0]])


def _rgb(bm):
    return (bm.get("red", 1), bm.get("green", 2), bm.get("blue", 3))


def _o(args, name):
    os.makedirs(args.out, exist_ok=True)
    return os.path.join(args.out, name)


def main(argv=None):
    ap = argparse.ArgumentParser(description="GeoAI Studio CLI", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, image=True):
        if image:
            p.add_argument("image")
        p.add_argument("-o", "--out", default="outputs")
        p.add_argument("--preset", default="RGB", help="band preset name prefix or number")
        return p

    sub.add_parser("presets")
    common(sub.add_parser("info"))
    p = common(sub.add_parser("vegetation")); p.add_argument("--ndsm"); p.add_argument("--threshold", type=float)
    p.add_argument("--tree-height", type=float, default=2.5); p.add_argument("--min-area", type=float, default=4)
    p = common(sub.add_parser("footprints")); p.add_argument("--ndsm", required=True)
    p.add_argument("--min-height", type=float, default=2.5); p.add_argument("--min-area", type=float, default=12)
    p = common(sub.add_parser("segment")); p.add_argument("--target", default="building")
    p.add_argument("--backend", default="langsam", choices=["langsam", "yolo-seg"]); p.add_argument("--weights")
    p.add_argument("--gsd", type=float, default=0.3); p.add_argument("--ndsm")
    p = common(sub.add_parser("vehicles")); p.add_argument("--weights", default="yolo26n-obb.pt", help="yolo26{n,s,m,l,x}-obb.pt, yolo11*-obb.pt or custom .pt")
    p.add_argument("--conf", type=float, default=0.25); p.add_argument("--gsd", type=float, default=0.25)
    p.add_argument("--all-classes", action="store_true"); p.add_argument("--device")
    p = common(sub.add_parser("change"), image=False); p.add_argument("image1"); p.add_argument("image2")
    p.add_argument("--method", default="index", choices=["index", "cva", "height"])
    p.add_argument("--threshold", type=float); p.add_argument("--min-area", type=float, default=5)
    p = common(sub.add_parser("footprint-change"), image=False); p.add_argument("before"); p.add_argument("after")
    p = common(sub.add_parser("blocks")); p.add_argument("--roads", default="osm", help="osm | path to lines | path to road mask .tif")
    p.add_argument("--buildings"); p.add_argument("--vegetation"); p.add_argument("--road-width", type=float, default=6)
    p.add_argument("--min-area", type=float, default=200)

    a = ap.parse_args(argv)
    from geoai import blocks, change, footprints, indices, vegetation
    from geoai import io as gio
    from geoai.vector import polygonize_mask, raster_footprint, save_vector

    if a.cmd == "presets":
        for i, n in enumerate(indices.PRESETS):
            print(i, n, indices.PRESETS[n])
        return
    if a.cmd == "info":
        print(gio.raster_info(a.image).summary()); return

    bm = _bm(a)
    if a.cmd == "vegetation":
        s = vegetation.vegetation_mask(a.image, bm, _o(a, "vegetation_mask.tif"), a.threshold, a.ndsm, a.tree_height)
        g = polygonize_mask(s["mask"], "class_id", min_area_m2=a.min_area)
        g["class"] = g["class_id"].map({1: "low vegetation", 2: "tree"})
        save_vector(g, _o(a, "vegetation.gpkg")); print(json.dumps(s, indent=1, default=str), len(g), "polygons")
    elif a.cmd == "footprints":
        mp = footprints.building_mask_from_height(a.image, bm, a.ndsm, _o(a, "building_mask.tif"), a.min_height)
        g = footprints.footprints_from_mask(mp, _o(a, "buildings.gpkg"), a.min_area, height_path=a.ndsm)
        print(len(g), "buildings ->", _o(a, "buildings.gpkg"))
    elif a.cmd == "segment":
        from geoai.detect import make_segmenter, segment_raster
        seg = make_segmenter(a.backend, a.target, a.weights)
        mp = segment_raster(a.image, _o(a, f"{a.target}_mask.tif"), seg, _rgb(bm), a.gsd)
        if a.target == "building":
            g = footprints.footprints_from_mask(mp, _o(a, "buildings_ai.gpkg"), height_path=a.ndsm)
        else:
            g = polygonize_mask(mp, min_area_m2=20); save_vector(g, _o(a, f"{a.target}s_ai.gpkg"))
        print(len(g), a.target, "polygons")
    elif a.cmd == "vehicles":
        from geoai.detect import detect_objects
        g = detect_objects(a.image, _rgb(bm), a.weights, None if a.all_classes else {"small vehicle", "large vehicle"},
                           a.conf, a.gsd, device=a.device)
        save_vector(g, _o(a, "objects.gpkg")); print(g["class"].value_counts() if len(g) else "no detections")
    elif a.cmd == "change":
        s = change.raster_change(a.image1, a.image2, bm, _o(a, f"change_{a.method}.tif"), a.method, threshold=a.threshold)
        g = polygonize_mask(s["mask"], "code", min_area_m2=a.min_area)
        g["status"] = g["code"].map({1: "changed"} if a.method == "cva" else {1: "loss", 2: "gain"})
        save_vector(g, _o(a, f"change_{a.method}.gpkg")); print(json.dumps(s, indent=1), len(g), "change polygons")
    elif a.cmd == "footprint-change":
        g = change.footprint_change(gpd.read_file(a.before), gpd.read_file(a.after))
        save_vector(g, _o(a, "building_change.gpkg")); print(g.status.value_counts())
    elif a.cmd == "blocks":
        aoi = raster_footprint(a.image)
        if a.roads == "osm":
            roads = blocks.buffer_roads(blocks.roads_from_osm(aoi), a.road_width)
        elif a.roads.lower().endswith((".tif", ".tiff")):
            roads = blocks.road_polys_from_mask(a.roads)
        else:
            roads = blocks.buffer_roads(blocks.roads_from_file(a.roads), a.road_width)
        bld = gpd.read_file(a.buildings) if a.buildings else None
        veg = gpd.read_file(a.vegetation) if a.vegetation else None
        g = blocks.make_blocks(aoi, roads, a.min_area, bld, veg)
        save_vector(g, _o(a, "blocks.gpkg")); print(len(g), "blocks ->", _o(a, "blocks.gpkg"))


if __name__ == "__main__":
    sys.exit(main())
