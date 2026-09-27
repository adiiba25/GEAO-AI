"""Raster I/O helpers: info, tiling windows, band reading, alignment and previews."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window


@dataclass
class RasterInfo:
    path: str
    width: int
    height: int
    count: int
    dtype: str
    crs: str
    res: tuple
    bounds: tuple
    nodata: float | None
    gsd_m: float  # approx ground sample distance in metres

    def summary(self) -> str:
        return (f"{self.width} x {self.height} px, {self.count} band(s), {self.dtype}, "
                f"CRS {self.crs}, GSD ~{self.gsd_m:.3f} m")


def gsd_metres(src) -> float:
    """Approximate pixel size in metres, also for geographic (degree) CRSs."""
    xres = abs(src.res[0])
    if src.crs is not None and src.crs.is_geographic:
        lat = (src.bounds.top + src.bounds.bottom) / 2
        return xres * 111_320 * math.cos(math.radians(lat))
    unit = 1.0
    try:
        unit = src.crs.linear_units_factor[1] if src.crs else 1.0
    except Exception:
        pass
    return xres * unit


def raster_info(path: str) -> RasterInfo:
    with rasterio.open(path) as src:
        return RasterInfo(path, src.width, src.height, src.count, src.dtypes[0],
                          src.crs.to_string() if src.crs else "None", src.res,
                          tuple(src.bounds), src.nodata, gsd_metres(src))


def iter_windows(width: int, height: int, tile: int = 1024, overlap: int = 0) -> Iterator[Window]:
    """Yield windows covering the raster. With overlap>0 windows overlap by that many px."""
    step = max(1, tile - overlap)
    for row in range(0, height, step):
        for col in range(0, width, step):
            w = min(tile, width - col)
            h = min(tile, height - row)
            yield Window(col, row, w, h)
            if col + w >= width:
                break
        if row + min(tile, height - row) >= height:
            break


def read_bands(src, band_map: dict, window: Window | None = None,
               out_shape: tuple | None = None) -> dict:
    """Read named bands (e.g. {'red': 1, 'nir': 4}) as float32 with nodata -> NaN."""
    out = {}
    nodata = src.nodata
    for name, idx in band_map.items():
        if idx is None or idx > src.count:
            continue
        kw = {"window": window}
        if out_shape is not None:
            kw.update(out_shape=out_shape, resampling=Resampling.average)
        arr = src.read(idx, **kw).astype("float32")
        if nodata is not None:
            arr[arr == nodata] = np.nan
        out[name] = arr
    # treat all-zero pixels (common ortho background) as nodata
    if out and nodata is None:
        stack = np.stack(list(out.values()))
        zero = np.all(stack == 0, axis=0)
        for k in out:
            out[k][zero] = np.nan
    return out


def overview_shape(src, max_px: int = 2048) -> tuple:
    scale = max(src.width, src.height) / max_px
    scale = max(scale, 1.0)
    return int(src.height / scale), int(src.width / scale)


def same_grid(path_a: str, path_b: str) -> bool:
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        return (a.crs == b.crs and a.transform.almost_equals(b.transform)
                and a.width == b.width and a.height == b.height)


def align_to(src_path: str, ref_path: str, out_path: str,
             resampling: Resampling = Resampling.bilinear) -> str:
    """Reproject/resample src onto the exact grid of ref (CRS, transform, size)."""
    with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
        profile = ref.profile.copy()
        profile.update(count=src.count, dtype=src.dtypes[0], nodata=src.nodata,
                       compress="deflate", tiled=True, blockxsize=512, blockysize=512,
                       BIGTIFF="IF_SAFER")
        with rasterio.open(out_path, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                reproject(rasterio.band(src, b), rasterio.band(dst, b),
                          dst_transform=ref.transform, dst_crs=ref.crs,
                          resampling=resampling, num_threads=4)
    return out_path


def mask_profile(src, dtype="uint8", nodata=0) -> dict:
    p = src.profile.copy()
    p.update(count=1, dtype=dtype, nodata=nodata, compress="deflate", tiled=True,
             blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER", photometric=None)
    p.pop("photometric", None)
    return p


def stretch_to_uint8(arr: np.ndarray, p_low=2, p_high=98) -> np.ndarray:
    """Percentile stretch per band (arr: bands x h x w) -> uint8."""
    out = np.zeros(arr.shape, dtype=np.uint8)
    for i in range(arr.shape[0]):
        band = arr[i].astype("float32")
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, [p_low, p_high])
        if hi <= lo:
            hi = lo + 1
        out[i] = np.clip((np.nan_to_num(band, nan=lo) - lo) / (hi - lo) * 255, 0, 255)
    return out


def preview_rgb(path: str, rgb=(1, 2, 3), max_px: int = 1600):
    """Return (uint8 HxWx3 image in EPSG:4326, [[south, west], [north, east]]) for web maps."""
    with rasterio.open(path) as src:
        bands = [b for b in rgb if b <= src.count] or [1]
        while len(bands) < 3:
            bands.append(bands[-1])
        with WarpedVRT(src, crs="EPSG:4326", resampling=Resampling.bilinear) as vrt:
            shape = overview_shape(vrt, max_px)
            data = vrt.read(bands, out_shape=(3, *shape), resampling=Resampling.average,
                            masked=True).astype("float32").filled(np.nan)
            b = vrt.bounds
    zero = np.all(np.nan_to_num(data) == 0, axis=0)
    data[:, zero] = np.nan
    img = stretch_to_uint8(data)
    alpha = np.where(np.all(np.isfinite(data), axis=0), 255, 0).astype(np.uint8)
    rgba = np.dstack([img.transpose(1, 2, 0), alpha])
    return rgba, [[b.bottom, b.left], [b.top, b.right]]


def raster_bounds_4326(path: str) -> tuple:
    with rasterio.open(path) as src:
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds)
