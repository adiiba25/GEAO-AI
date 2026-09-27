"""Band presets and spectral indices."""
from __future__ import annotations

import numpy as np

# 1-based band indices for common products. Users can also pass a custom mapping.
PRESETS: dict[str, dict] = {
    "RGB (drone ortho / basemap)": {"red": 1, "green": 2, "blue": 3},
    "Multispectral drone (R,G,B,RE,NIR)": {"red": 1, "green": 2, "blue": 3, "rededge": 4, "nir": 5},
    "4-band B,G,R,NIR (Planet, Pleiades, SkySat)": {"blue": 1, "green": 2, "red": 3, "nir": 4},
    "4-band R,G,B,NIR (Maxar/WorldView pan-sharp)": {"red": 1, "green": 2, "blue": 3, "nir": 4},
    "Sentinel-2 stack B2,B3,B4,B8": {"blue": 1, "green": 2, "red": 3, "nir": 4},
    "Sentinel-2 10+20m stack B2,B3,B4,B8,B11,B12": {"blue": 1, "green": 2, "red": 3, "nir": 4,
                                                     "swir1": 5, "swir2": 6},
    "Landsat 8/9 SR stack SR_B1..SR_B7": {"blue": 2, "green": 3, "red": 4, "nir": 5,
                                          "swir1": 6, "swir2": 7},
}


def _safe_ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out


def ndvi(b):  # vegetation, needs NIR
    return _safe_ratio(b["nir"] - b["red"], b["nir"] + b["red"])


def exg(b):  # Excess Green on chromatic coordinates - vegetation from plain RGB
    total = b["red"] + b["green"] + b["blue"]
    r, g, bl = (_safe_ratio(b[k], total) for k in ("red", "green", "blue"))
    return 2 * g - r - bl


def ndwi(b):  # open water (McFeeters)
    return _safe_ratio(b["green"] - b["nir"], b["green"] + b["nir"])


def ndbi(b):  # built-up, needs SWIR
    return _safe_ratio(b["swir1"] - b["nir"], b["swir1"] + b["nir"])


def brightness(b):
    keys = [k for k in ("red", "green", "blue") if k in b]
    return np.nanmean(np.stack([b[k] for k in keys]), axis=0)


INDEX_FUNCS = {"NDVI": (ndvi, ("nir", "red")), "ExG": (exg, ("red", "green", "blue")),
               "NDWI": (ndwi, ("green", "nir")), "NDBI": (ndbi, ("swir1", "nir"))}


def available_indices(band_map: dict) -> list[str]:
    return [k for k, (_, need) in INDEX_FUNCS.items() if all(n in band_map for n in need)]


def vegetation_index_name(band_map: dict) -> str:
    return "NDVI" if "nir" in band_map else "ExG"


def compute_index(name: str, bands: dict) -> np.ndarray:
    return INDEX_FUNCS[name][0](bands)


def otsu(values: np.ndarray) -> float:
    from skimage.filters import threshold_otsu
    v = values[np.isfinite(values)]
    if v.size < 10:
        return 0.0
    if v.size > 2_000_000:
        v = np.random.default_rng(0).choice(v, 2_000_000, replace=False)
    return float(threshold_otsu(v))
