"""Deep-learning object detection and segmentation over large georeferenced rasters.

Everything runs tile-by-tile, resampling each tile to the ground resolution the model
was trained at (target_gsd), so the same code works on 3 cm drone orthos and 50 cm
satellite scenes. Heavy dependencies (torch, ultralytics, segment-geospatial) are
optional and imported lazily - install requirements-ai.txt to enable this module.

Backends
--------
vehicles  : Ultralytics YOLO-OBB pretrained on DOTA (aerial) - small/large vehicle,
            plus plane, ship, storage tank ... out of the box, no training needed.
buildings : "langsam"  zero-shot text prompt ("building") via segment-geospatial
roads       "yolo-seg" your own trained Ultralytics segmentation weights
                        (recommended for production; e.g. trained on Open Cities AI
                        / SpaceNet or on your own digitised Kampala tiles)
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from .io import gsd_metres, mask_profile, stretch_to_uint8

DOTA_VEHICLE_CLASSES = {"small vehicle", "large vehicle"}


# ---------------------------------------------------------------- tiling helper
def _tiles(src, target_gsd: float, tile_px: int, overlap_px: int):
    """Yield (window, out_shape, scale) where out_shape is the model-resolution tile size."""
    src_gsd = gsd_metres(src)
    scale = max(target_gsd / src_gsd, 1e-6)          # >1 means downsample
    win_px = int(round(tile_px * scale))
    ov = int(round(overlap_px * scale))
    step = max(1, win_px - ov)
    for row in range(0, src.height, step):
        for col in range(0, src.width, step):
            w, h = min(win_px, src.width - col), min(win_px, src.height - row)
            out_shape = (max(1, int(round(h / scale))), max(1, int(round(w / scale))))
            yield Window(col, row, w, h), out_shape, scale
            if col + w >= src.width:
                break
        if row + min(win_px, src.height - row) >= src.height:
            break


def _read_rgb(src, rgb, window, out_shape, stretch_stats):
    arr = src.read(list(rgb), window=window, out_shape=(3, *out_shape),
                   resampling=Resampling.bilinear).astype("float32")
    lo, hi = stretch_stats
    img = np.clip((arr - lo[:, None, None]) / (hi - lo)[:, None, None] * 255, 0, 255).astype(np.uint8)
    valid = np.any(arr != 0, axis=0)
    return img.transpose(1, 2, 0), valid


def _global_stretch(src, rgb):
    """Percentile stretch from an overview so tiles get consistent contrast."""
    scale = max(1, max(src.width, src.height) / 2048)
    ov = src.read(list(rgb), out_shape=(3, int(src.height / scale), int(src.width / scale)),
                  resampling=Resampling.average).astype("float32")
    lo, hi = [], []
    for b in ov:
        v = b[b != 0]
        if v.size == 0:
            v = b.ravel()
        l, h = np.percentile(v, [1, 99])
        lo.append(l); hi.append(max(h, l + 1))
    if src.dtypes[0] == "uint8":   # already display-ready 8-bit (drone ortho)
        return np.zeros(3, "float32"), np.full(3, 255, "float32")
    return np.array(lo, "float32"), np.array(hi, "float32")


def _px_to_world(tr, pts_xy):
    return [tr * (float(x), float(y)) for x, y in pts_xy]


# ---------------------------------------------------------------- vehicles (OBB)
def detect_objects(path: str, rgb=(1, 2, 3), weights: str = "yolo11n-obb.pt",
                   classes: set[str] | None = DOTA_VEHICLE_CLASSES, conf: float = 0.25,
                   target_gsd: float = 0.25, tile_px: int = 1024, overlap_px: int = 128,
                   iou_nms: float = 0.4, device: str | None = None, progress=None) -> gpd.GeoDataFrame:
    """Oriented-box object detection. Default: vehicles with DOTA-pretrained YOLO11-OBB.

    target_gsd ~0.2-0.3 m matches DOTA; don't feed 3 cm drone pixels unresampled.
    Pass classes=None to keep every DOTA class (planes, ships, tanks, pools, ...).
    """
    from ultralytics import YOLO
    model = YOLO(weights)
    recs = []
    with rasterio.open(path) as src:
        stats = _global_stretch(src, rgb)
        tiles = list(_tiles(src, target_gsd, tile_px, overlap_px))
        for i, (win, oshape, scale) in enumerate(tiles):
            img, valid = _read_rgb(src, rgb, win, oshape, stats)
            if valid.mean() < 0.05:
                continue
            tr = src.window_transform(win) * rasterio.Affine.scale(win.width / oshape[1], win.height / oshape[0])
            res = model.predict(img[:, :, ::-1], imgsz=tile_px, conf=conf, device=device, verbose=False)[0]
            if res.obb is None or len(res.obb) == 0:
                continue
            quads = res.obb.xyxyxyxy.cpu().numpy()
            cls = res.obb.cls.cpu().numpy().astype(int)
            cf = res.obb.conf.cpu().numpy()
            for q, c, s in zip(quads, cls, cf):
                name = res.names[int(c)]
                if classes and name not in classes:
                    continue
                recs.append({"class": name, "conf": round(float(s), 3),
                             "geometry": Polygon(_px_to_world(tr, q))})
            if progress:
                progress((i + 1) / len(tiles))
        crs = src.crs
    gdf = gpd.GeoDataFrame(recs, geometry="geometry", crs=crs) if recs else \
        gpd.GeoDataFrame({"class": [], "conf": []}, geometry=[], crs=crs)
    return nms_polygons(gdf, iou_nms)


def nms_polygons(gdf: gpd.GeoDataFrame, iou_thr: float = 0.4) -> gpd.GeoDataFrame:
    """Remove duplicate detections from overlapping tiles (polygon NMS)."""
    if len(gdf) < 2:
        return gdf.reset_index(drop=True)
    gdf = gdf.sort_values("conf", ascending=False).reset_index(drop=True)
    geoms = list(gdf.geometry)
    tree = STRtree(geoms)
    keep, dropped = [], set()
    for i, g in enumerate(geoms):
        if i in dropped:
            continue
        keep.append(i)
        for j in tree.query(g):
            j = int(j)
            if j <= i or j in dropped:
                continue
            inter = g.intersection(geoms[j]).area
            union = g.area + geoms[j].area - inter
            if union and inter / union > iou_thr:
                dropped.add(j)
    return gdf.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------- segmentation
class _LangSAMBackend:
    def __init__(self, prompt, box_threshold=0.24, text_threshold=0.24):
        from samgeo.text_sam import LangSAM
        self.model, self.prompt = LangSAM(), prompt
        self.bt, self.tt = box_threshold, text_threshold

    def __call__(self, img):
        out = self.model.predict(img, self.prompt, box_threshold=self.bt,
                                 text_threshold=self.tt, return_results=True)
        masks = out[0] if isinstance(out, tuple) else getattr(self.model, "masks", None)
        m = np.zeros(img.shape[:2], bool)
        if masks is None or len(masks) == 0:
            return m
        masks = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
        return np.any(masks.astype(bool), axis=0).reshape(m.shape)


class _YoloSegBackend:
    def __init__(self, weights, conf=0.3, classes=None, device=None, imgsz=1024):
        from ultralytics import YOLO
        self.model, self.conf, self.device, self.imgsz = YOLO(weights), conf, device, imgsz
        self.class_ids = None
        if classes:
            names = self.model.names
            self.class_ids = [k for k, v in names.items() if v in classes]

    def __call__(self, img):
        from skimage.draw import polygon as draw_poly
        m = np.zeros(img.shape[:2], bool)
        r = self.model.predict(img[:, :, ::-1], imgsz=self.imgsz, conf=self.conf,
                               classes=self.class_ids, device=self.device, verbose=False)[0]
        if r.masks is None:
            return m
        for poly in r.masks.xy:
            if len(poly) >= 3:
                rr, cc = draw_poly(poly[:, 1], poly[:, 0], m.shape)
                m[rr, cc] = True
        return m


def make_segmenter(backend: str, prompt: str = "building", weights: str | None = None,
                   classes=None, conf: float = 0.3, device=None, tile_px: int = 1024):
    if backend == "langsam":
        return _LangSAMBackend(prompt)
    if backend == "yolo-seg":
        if not weights:
            raise ValueError("yolo-seg backend needs trained segmentation weights (.pt)")
        return _YoloSegBackend(weights, conf, classes, device, tile_px)
    raise ValueError(f"unknown backend {backend}")


def segment_raster(path: str, out_mask: str, segmenter, rgb=(1, 2, 3), target_gsd: float = 0.3,
                   tile_px: int = 1024, overlap_px: int = 96, progress=None) -> str:
    """Run a segmenter over the raster and write a uint8 mask at the source resolution.
    Overlapping tile predictions are OR-ed together."""
    with rasterio.open(path) as src:
        stats = _global_stretch(src, rgb)
        prof = mask_profile(src)
        with rasterio.open(out_mask, "w", **prof):
            pass
        tiles = list(_tiles(src, target_gsd, tile_px, overlap_px))
        with rasterio.open(out_mask, "r+") as dst:
            for i, (win, oshape, _) in enumerate(tiles):
                img, valid = _read_rgb(src, rgb, win, oshape, stats)
                if valid.mean() < 0.05:
                    continue
                pred = segmenter(img) & valid
                if pred.any():
                    full = _resize_nearest(pred, (int(win.height), int(win.width)))
                    cur = dst.read(1, window=win).astype(bool)
                    dst.write((cur | full).astype(np.uint8), 1, window=win)
                if progress:
                    progress((i + 1) / len(tiles))
    return out_mask


def _resize_nearest(a: np.ndarray, shape) -> np.ndarray:
    ri = (np.arange(shape[0]) * a.shape[0] / shape[0]).astype(int).clip(0, a.shape[0] - 1)
    ci = (np.arange(shape[1]) * a.shape[1] / shape[1]).astype(int).clip(0, a.shape[1] - 1)
    return a[ri][:, ci]


def ai_available() -> dict:
    out = {}
    for mod in ("torch", "ultralytics", "samgeo"):
        try:
            __import__(mod)
            out[mod] = True
        except Exception:
            out[mod] = False
    try:
        import torch
        out["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        out["cuda"] = False
    return out
