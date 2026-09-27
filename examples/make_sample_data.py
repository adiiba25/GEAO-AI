"""Generate a synthetic drone survey (two dates) to try the app without real data.

Creates in ./sample_data:
  ortho_2024.tif / ortho_2026.tif   3-band RGB orthomosaics, 0.10 m GSD, UTM 36N
  ndsm_2024.tif  / ndsm_2026.tif    height above ground (m)
  roads.gpkg                        road centrelines for block creation
Between dates: 4 new buildings, 2 demolished, a patch of trees cleared.
"""
import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString
from skimage.draw import disk, polygon

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sample_data")
GSD, N = 0.10, 3000                     # 300 m x 300 m
X0, Y0 = 453_000.0, 36_300.0            # near Kampala, EPSG:32636
CRS = "EPSG:32636"
rng = np.random.default_rng(42)
ROADS_PX = [(0, 1450, 1550), (1, 1450, 1550), (0, 400, 470), (1, 2500, 2570)]  # (axis, start, end)


def rotated_rect(cy, cx, h, w, ang):
    a = np.radians(ang)
    pts = np.array([[-h / 2, -w / 2], [-h / 2, w / 2], [h / 2, w / 2], [h / 2, -w / 2]])
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    p = pts @ R.T + [cy, cx]
    return polygon(p[:, 0], p[:, 1], (N, N))


def on_road(cy, cx, pad=90):
    for axis, s, e in ROADS_PX:
        v = cy if axis == 0 else cx
        if s - pad < v < e + pad:
            return True
    return False


def make_buildings(n, existing=()):
    out = []
    while len(out) < n:
        cy, cx = rng.integers(150, N - 150, 2)
        if on_road(cy, cx) or any(abs(cy - b[0]) < 170 and abs(cx - b[1]) < 170 for b in list(out) + list(existing)):
            continue
        out.append((cy, cx, rng.integers(70, 160), rng.integers(80, 200), rng.uniform(-25, 25),
                    rng.uniform(3, 9), rng.choice(3)))
    return out


def make_trees(n, blds):
    out = []
    while len(out) < n:
        cy, cx = rng.integers(60, N - 60, 2)
        if on_road(cy, cx, 40) or any(abs(cy - b[0]) < 130 and abs(cx - b[1]) < 130 for b in blds):
            continue
        out.append((cy, cx, rng.integers(25, 60), rng.uniform(5, 14)))
    return out


def render(blds, trees):
    img = np.zeros((3, N, N), np.float32)
    img[:] = np.array([150, 115, 80])[:, None, None]                 # bare soil
    img += rng.normal(0, 8, img.shape)
    gy, gx = np.mgrid[0:N, 0:N]
    grass = (np.sin(gy / 180) + np.cos(gx / 230)) > 0.9
    img[:, grass] = np.array([95, 140, 60])[:, None] + rng.normal(0, 6, (3, grass.sum()))
    h = np.zeros((N, N), np.float32) + rng.normal(0, 0.1, (N, N)).astype(np.float32)
    h[grass] += 0.3
    for axis, s, e in ROADS_PX:
        sl = (slice(s, e), slice(None)) if axis == 0 else (slice(None), slice(s, e))
        img[(slice(None),) + sl] = 105 + rng.normal(0, 4, img[(slice(None),) + sl].shape)
        h[sl] = 0
    roof = {0: (170, 60, 45), 1: (185, 185, 190), 2: (120, 90, 70)}   # tile, iron sheet, brown
    for cy, cx, hh, ww, ang, bh, rc in blds:
        rr, cc = rotated_rect(cy, cx, hh, ww, ang)
        img[:, rr, cc] = np.array(roof[rc])[:, None] + rng.normal(0, 5, (3, rr.size))
        h[rr, cc] = bh
    for cy, cx, r, th in trees:
        rr, cc = disk((cy, cx), r, shape=(N, N))
        img[:, rr, cc] = np.array([45, 100, 35])[:, None] + rng.normal(0, 10, (3, rr.size))
        d = np.hypot(rr - cy, cc - cx) / r
        h[rr, cc] = np.maximum(h[rr, cc], th * (1 - 0.6 * d ** 2))
    for axis, s, e in ROADS_PX:                                        # a few cars
        for _ in range(6):
            p = rng.integers(100, N - 100)
            mid = (s + e) // 2 + rng.integers(-20, 20)
            cy, cx = (mid, p) if axis == 0 else (p, mid)
            ch, cw = (18, 45) if axis == 0 else (45, 18)
            col = rng.choice([[230, 230, 235], [30, 30, 35], [180, 30, 30], [40, 60, 150]])
            img[:, cy - ch // 2:cy + ch // 2, cx - cw // 2:cx + cw // 2] = np.array(col)[:, None, None]
            h[cy - ch // 2:cy + ch // 2, cx - cw // 2:cx + cw // 2] = 1.5
    return np.clip(img, 1, 255).astype(np.uint8), h


def write(path, arr, dtype):
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", width=N, height=N, count=arr.shape[0], dtype=dtype,
                       crs=CRS, transform=from_origin(X0, Y0, GSD, GSD), compress="deflate",
                       tiled=True, blockxsize=512, blockysize=512) as dst:
        dst.write(arr.astype(dtype))


def main():
    os.makedirs(OUT, exist_ok=True)
    blds = make_buildings(30)
    trees = make_trees(45, blds)
    img1, h1 = render(blds, trees)
    blds2 = blds[2:] + make_buildings(4, existing=blds)                          # 2 demolished, 4 new
    trees2 = [t for t in trees if not (t[0] < 1400 and t[1] < 1400)]    # clearing NW quadrant
    img2, h2 = render(blds2, trees2)
    write(os.path.join(OUT, "ortho_2024.tif"), img1, "uint8")
    write(os.path.join(OUT, "ortho_2026.tif"), img2, "uint8")
    write(os.path.join(OUT, "ndsm_2024.tif"), h1, "float32")
    write(os.path.join(OUT, "ndsm_2026.tif"), h2, "float32")
    lines = []
    for axis, s, e in ROADS_PX:
        m = (s + e) / 2 * GSD
        lines.append(LineString([(X0, Y0 - m), (X0 + N * GSD, Y0 - m)]) if axis == 0 else
                     LineString([(X0 + m, Y0), (X0 + m, Y0 - N * GSD)]))
    widths = [(e - s) * GSD for _, s, e in ROADS_PX]
    gpd.GeoDataFrame({"highway": ["primary", "primary", "residential", "residential"],
                      "width_m": widths}, geometry=lines, crs=CRS).to_file(
        os.path.join(OUT, "roads.gpkg"), driver="GPKG")
    print(f"Sample data written to {os.path.abspath(OUT)}: {len(blds)} -> {len(blds2)} buildings, "
          f"{len(trees)} -> {len(trees2)} trees")


if __name__ == "__main__":
    main()
