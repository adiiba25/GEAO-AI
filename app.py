"""GeoAI Studio - web map app for drone & satellite imagery.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io as _io
import json
import os
import time
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

from geoai import blocks, change, footprints, indices, vegetation
from geoai import io as gio
from geoai.detect import ai_available
from geoai.vector import polygonize_mask, raster_footprint, save_vector

st.set_page_config(page_title="GeoAI Studio", page_icon="🛰️", layout="wide")

STYLES = {
    "trees": {"color": "#1b7a2e", "fillColor": "#2ea043", "fillOpacity": 0.45, "weight": 1},
    "vegetation": {"color": "#6aa84f", "fillColor": "#93c47d", "fillOpacity": 0.35, "weight": 0.5},
    "buildings": {"color": "#c0392b", "fillColor": "#e74c3c", "fillOpacity": 0.35, "weight": 1.5},
    "vehicles": {"color": "#f1c40f", "fillColor": "#f1c40f", "fillOpacity": 0.6, "weight": 1.5},
    "objects": {"color": "#8e44ad", "fillColor": "#9b59b6", "fillOpacity": 0.5, "weight": 1.5},
    "roads": {"color": "#555", "fillColor": "#888", "fillOpacity": 0.5, "weight": 1},
    "blocks": {"color": "#2c3e50", "fillColor": "#3498db", "fillOpacity": 0.12, "weight": 2, "dashArray": "6,4"},
}
CHANGE_COLORS = {"new": "#27ae60", "demolished": "#c0392b", "modified": "#e67e22", "unchanged": "#95a5a6",
                 "loss": "#c0392b", "gain": "#27ae60", "changed": "#e67e22"}

ss = st.session_state
ss.setdefault("layers", {})        # name -> dict(gdf, kind, path)
ss.setdefault("stats", {})


# ------------------------------------------------------------------ helpers
def workspace() -> Path:
    p = Path(ss.get("workdir", "outputs")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_upload(upl, sub="uploads") -> str | None:
    if upl is None:
        return None
    d = workspace() / sub
    d.mkdir(exist_ok=True)
    out = d / upl.name
    if not out.exists() or out.stat().st_size != upl.size:
        with open(out, "wb") as f:
            f.write(upl.getbuffer())
    return str(out)


def raster_input(label, key, help_=None, optional=False):
    mode = st.radio(label, ["Path on disk", "Upload"], horizontal=True, key=f"{key}_mode",
                    help=help_)
    if mode == "Upload":
        return save_upload(st.file_uploader(label, type=["tif", "tiff", "vrt", "jp2", "img"],
                                            key=f"{key}_up", label_visibility="collapsed"))
    p = st.text_input(label, key=f"{key}_path", label_visibility="collapsed",
                      placeholder=("optional - " if optional else "") + r"e.g. C:\Drone\ortho.tif")
    p = p.strip().strip('"')
    if p and not os.path.exists(p):
        st.error("File not found")
        return None
    return p or None


@st.cache_data(show_spinner=False)
def cached_preview(path, rgb, mtime):
    rgba, bounds = gio.preview_rgb(path, rgb)
    buf = _io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return buf.getvalue(), bounds


def add_layer(name, gdf, kind, path=None):
    ss.layers[name] = {"gdf": gdf, "kind": kind, "path": path}


def progress_bar(text):
    bar = st.progress(0.0, text=text)
    return lambda f: bar.progress(min(1.0, float(f)), text=text)


def out_path(name):
    return str(workspace() / name)


def band_map_ui():
    preset = st.selectbox("Sensor / band order", list(indices.PRESETS), key="preset")
    bm = dict(indices.PRESETS[preset])
    with st.expander("Custom band numbers"):
        cols = st.columns(3)
        for i, k in enumerate(["red", "green", "blue", "nir", "rededge", "swir1"]):
            v = cols[i % 3].number_input(k, 0, 20, int(bm.get(k, 0)), key=f"b_{k}",
                                         help="0 = not available")
            if v:
                bm[k] = int(v)
            else:
                bm.pop(k, None)
    return bm


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🛰️ GeoAI Studio")
    st.caption("Drone & satellite imagery → trees, objects, footprints, change, blocks")
    ss.workdir = st.text_input("Output folder", ss.get("workdir", "outputs"))
    st.subheader("Imagery")
    img1 = raster_input("Image (date 1)", "img1", "GeoTIFF orthomosaic or satellite scene")
    img2 = raster_input("Image (date 2) for change", "img2", optional=True)
    with st.expander("Height models (drone DSM–DTM / nDSM)"):
        st.caption("Height above ground in metres. Splits trees from grass and gives "
                   "reliable building footprints without AI.")
        h1 = raster_input("nDSM date 1", "h1", optional=True)
        h2 = raster_input("nDSM date 2", "h2", optional=True)
    band_map = band_map_ui()
    rgb = (band_map.get("red", 1), band_map.get("green", 2), band_map.get("blue", 3))
    ai = ai_available()
    st.subheader("AI engine")
    st.write(("🟢 " if ai["ultralytics"] else "⚪ ") + "YOLO (vehicles / custom models)")
    st.write(("🟢 " if ai["samgeo"] else "⚪ ") + "LangSAM zero-shot segmentation")
    st.write(("🟢 GPU (CUDA)" if ai["cuda"] else "⚪ CPU only"))
    if st.button("Clear all results"):
        ss.layers, ss.stats = {}, {}

if not img1:
    st.header("Load imagery to begin")
    st.markdown(
        "- **Drone**: DJI Terra / Pix4D / ODM orthomosaic GeoTIFF (+ optional DSM/DTM-derived nDSM)\n"
        "- **High-res satellite**: Maxar, Planet, Airbus, Pleiades (0.3–3 m)\n"
        "- **Sentinel-2 / Landsat**: stacked GeoTIFF – best for vegetation and land change\n\n"
        "No data yet? Run `python examples/make_sample_data.py` and use the files in `sample_data/`.")
    st.stop()

info = gio.raster_info(img1)
st.caption(f"**{os.path.basename(img1)}** — {info.summary()}")
coarse = info.gsd_m > 5

left, right = st.columns([2, 3], gap="large")

# ------------------------------------------------------------------ processing tabs
with left:
    t_veg, t_obj, t_bld, t_chg, t_blk, t_exp = st.tabs(
        ["🌳 Vegetation", "🚗 Objects", "🏠 Footprints", "🔁 Change", "🧱 Blocks", "⬇️ Export"])

    # ---------------- vegetation / trees
    with t_veg:
        idx = indices.vegetation_index_name(band_map)
        st.write(f"Uses **{idx}** ({'multispectral' if idx == 'NDVI' else 'RGB-only'}).")
        auto = st.checkbox("Automatic threshold (Otsu)", True, key="veg_auto")
        thr = None if auto else st.slider("Threshold", -0.5, 1.0, 0.3 if idx == "NDVI" else 0.08, 0.01)
        tree_h = st.number_input("Tree min height (m)", 1.0, 20.0, 2.5, 0.5, disabled=not h1)
        min_area = st.number_input("Min polygon area (m²)", 0.0, 10000.0, 4.0 if not coarse else 900.0)
        if st.button("Map vegetation", type="primary"):
            mp = out_path("vegetation_mask.tif")
            with st.spinner("Classifying vegetation…"):
                s = vegetation.vegetation_mask(img1, band_map, mp, thr, h1, tree_h,
                                               progress=progress_bar("Vegetation"))
                g = polygonize_mask(mp, "class_id", min_area_m2=min_area)
                g["class"] = g["class_id"].map({1: "low vegetation", 2: "tree"})
            ss.stats["vegetation"] = s
            if h1:
                add_layer("Trees", g[g.class_id == 2], "trees", save_vector(g[g.class_id == 2], out_path("trees.gpkg")))
                add_layer("Low vegetation", g[g.class_id == 1], "vegetation")
            else:
                add_layer("Vegetation", g, "vegetation", save_vector(g, out_path("vegetation.gpkg")))
        if "vegetation" in ss.stats:
            s = ss.stats["vegetation"]
            c = st.columns(3)
            c[0].metric("Vegetation", f"{s['veg_pct']}%")
            c[1].metric("Tree canopy", f"{s['tree_pct']}%" if h1 else "needs nDSM")
            c[2].metric("Threshold", s["threshold"])

    # ---------------- object detection
    with t_obj:
        if coarse:
            st.warning(f"GSD ~{info.gsd_m:.0f} m is too coarse for individual objects. "
                       "Use Vegetation, Change and Blocks for Sentinel/Landsat.")
        st.markdown("**Vehicles & aerial objects** — YOLO26-OBB pretrained on DOTA aerial imagery (no training).")
        c1, c2 = st.columns(2)
        conf = c1.slider("Confidence", 0.05, 0.9, 0.25, 0.05)
        tgsd = c2.number_input("Model GSD (m)", 0.05, 2.0, 0.25, 0.05,
                               help="Tiles are resampled to this. DOTA ≈ 0.15–0.5 m.")
        all_cls = st.checkbox("All DOTA classes (planes, ships, tanks, pools…)", False)
        m1, m2 = st.columns(2)
        family = m1.selectbox("Model", ["YOLO26", "YOLO11", "Custom .pt"],
                              help="YOLO26: newest, NMS-free, most accurate on DOTA. YOLO11: previous generation.")
        size = m2.selectbox("Size", ["n (fast, CPU ok)", "s", "m (GPU)", "l (GPU)", "x (GPU, best)"],
                            disabled=family == "Custom .pt")
        if family == "Custom .pt":
            weights = st.text_input("Weights path", "", placeholder=r"e.g. C:\GIS\models\best.pt")
        else:
            weights = f"{family.lower()}{size[0]}-obb.pt"
            st.caption(f"Weights: `{weights}` (downloads automatically on first run)")
        if st.button("Detect objects", type="primary", disabled=not ai["ultralytics"] or not weights):
            from geoai.detect import detect_objects
            with st.spinner("Running detector…"):
                g = detect_objects(img1, rgb, weights, None if all_cls else {"small vehicle", "large vehicle"},
                                   conf, tgsd, progress=progress_bar("Detecting"))
            add_layer("Vehicles" if not all_cls else "Objects", g, "vehicles" if not all_cls else "objects",
                      save_vector(g, out_path("objects.gpkg")))
            ss.stats["objects"] = g["class"].value_counts().to_dict() if len(g) else {}
        if not ai["ultralytics"]:
            st.info("Install AI extras: `pip install -r requirements-ai.txt`")
        if ss.stats.get("objects"):
            st.dataframe(ss.stats["objects"], width="stretch")

        st.divider()
        st.markdown("**Segment roads / buildings / trees with AI**")
        backend = st.selectbox("Backend", ["langsam", "yolo-seg"],
                               format_func=lambda b: {"langsam": "LangSAM zero-shot (text prompt)",
                                                      "yolo-seg": "Custom YOLO-seg weights"}[b])
        target = st.selectbox("Target", ["road", "building", "tree"])
        seg_w = st.text_input("Segmentation weights (.pt)", "", disabled=backend != "yolo-seg")
        seg_gsd = st.number_input("Segmentation GSD (m)", 0.05, 3.0, 0.3, 0.05)
        can = ai["samgeo"] if backend == "langsam" else ai["ultralytics"]
        if st.button(f"Segment {target}s", disabled=not can):
            from geoai.detect import make_segmenter, segment_raster
            with st.spinner("Segmenting (first run downloads model weights)…"):
                seg = make_segmenter(backend, prompt=target, weights=seg_w or None)
                mp = out_path(f"{target}_mask.tif")
                segment_raster(img1, mp, seg, rgb, seg_gsd, progress=progress_bar("Segmenting"))
            ss[f"{target}_mask"] = mp
            if target == "building":
                g = footprints.footprints_from_mask(mp, out_path("buildings_ai.gpkg"), height_path=h1)
                add_layer("Buildings (AI)", g, "buildings", out_path("buildings_ai.gpkg"))
            else:
                g = polygonize_mask(mp, min_area_m2=20)
                kind = "roads" if target == "road" else "trees"
                add_layer(f"{target.title()}s (AI)", g, kind, save_vector(g, out_path(f"{target}s_ai.gpkg")))
            st.success(f"{len(g)} {target} polygons")

    # ---------------- building footprints
    with t_bld:
        st.markdown("Height route: `nDSM ≥ min height` and not vegetation → regularised footprints. "
                    "For satellite without height, use **Objects → Segment buildings**.")
        c1, c2, c3 = st.columns(3)
        min_h = c1.number_input("Min height (m)", 1.0, 10.0, 2.5, 0.5)
        min_a = c2.number_input("Min area (m²)", 1.0, 500.0, 12.0)
        simp = c3.number_input("Simplify (m)", 0.0, 5.0, 0.5, 0.1)
        if st.button("Extract footprints", type="primary", disabled=not h1):
            with st.spinner("Extracting footprints…"):
                mp = out_path("building_mask.tif")
                footprints.building_mask_from_height(img1, band_map, h1, mp, min_h,
                                                     progress=progress_bar("Building mask"))
                g = footprints.footprints_from_mask(mp, out_path("buildings.gpkg"), min_a, simplify_m=simp,
                                                    height_path=h1)
            add_layer("Buildings", g, "buildings", out_path("buildings.gpkg"))
        if not h1:
            st.info("Add an nDSM (sidebar → Height models) to use the height route.")
        bl = [n for n, l in ss.layers.items() if l["kind"] == "buildings" and "change" not in n.lower()]
        if bl:
            g = ss.layers[bl[-1]]["gdf"]
            c = st.columns(3)
            c[0].metric("Buildings", len(g))
            c[1].metric("Mean area", f"{g.area_m2.mean():.0f} m²" if len(g) else "–")
            if "height_m" in g:
                c[2].metric("Mean height", f"{g.height_m.dropna().mean():.1f} m" if len(g) else "–")

    # ---------------- change detection
    with t_chg:
        if not img2:
            st.info("Add a date-2 image in the sidebar.")
        else:
            method = st.radio("Method", ["index", "cva", "height", "footprints"], horizontal=True,
                              format_func=lambda m: {"index": f"Index diff ({indices.vegetation_index_name(band_map)})",
                                                     "cva": "All-band CVA", "height": "Height (nDSM)",
                                                     "footprints": "Building footprints"}[m])
            auto = st.checkbox("Automatic threshold", True, key="chg_auto")
            thr = None if auto else st.number_input("Threshold", 0.0, 50.0, 0.2, 0.01)
            min_ca = st.number_input("Min change area (m²)", 0.0, 1e6, 5.0 if not coarse else 1800.0)
            if st.button("Detect change", type="primary"):
                with st.spinner("Comparing dates…"):
                    if method == "footprints":
                        if not (h1 and h2):
                            st.error("Footprint change needs nDSM for both dates (or run AI segmentation per date via CLI).")
                            st.stop()
                        g1 = footprints.footprints_from_mask(
                            footprints.building_mask_from_height(img1, band_map, h1, out_path("bmask_t1.tif")))
                        g2 = footprints.footprints_from_mask(
                            footprints.building_mask_from_height(img2, band_map, h2, out_path("bmask_t2.tif")))
                        g = change.footprint_change(g1, g2)
                        ss.stats["change"] = g.status.value_counts().to_dict()
                    else:
                        a, b = (h1, h2) if method == "height" else (img1, img2)
                        if method == "height" and not (h1 and h2):
                            st.error("Height change needs nDSM for both dates."); st.stop()
                        mp = out_path(f"change_{method}.tif")
                        ss.stats["change"] = change.raster_change(a, b, band_map, mp, method, threshold=thr,
                                                                  progress=progress_bar("Change"))
                        g = polygonize_mask(mp, "code", min_area_m2=min_ca)
                        lab = {1: "changed"} if method == "cva" else {1: "loss", 2: "gain"}
                        g["status"] = g["code"].map(lab)
                add_layer(f"Change ({method})", g, "change", save_vector(g, out_path(f"change_{method}.gpkg")))
            if "change" in ss.stats:
                st.json(ss.stats["change"], expanded=False)

    # ---------------- blocks
    with t_blk:
        src = st.radio("Road source", ["OpenStreetMap", "Road file", "AI road mask"], horizontal=True)
        rfile = None
        if src == "Road file":
            rfile = st.text_input("Road centrelines (GPKG/SHP/GeoJSON)", "").strip().strip('"')
            up = st.file_uploader("…or upload GeoJSON / zipped shapefile", type=["geojson", "json", "zip", "gpkg"])
            if up:
                rfile = save_upload(up, "roads")
        dw = st.number_input("Default road width (m)", 2.0, 40.0, 6.0,
                             help="Used when no width_m field / OSM class width")
        min_blk = st.number_input("Min block area (m²)", 0.0, 1e6, 200.0)
        if st.button("Create blocks", type="primary"):
            with st.spinner("Building blocks…"):
                aoi = raster_footprint(img1)
                if src == "OpenStreetMap":
                    roads = blocks.buffer_roads(blocks.roads_from_osm(aoi), dw)
                elif src == "Road file":
                    roads = blocks.buffer_roads(blocks.roads_from_file(rfile), dw)
                else:
                    if not ss.get("road_mask"):
                        st.error("Run Objects → Segment roads first."); st.stop()
                    roads = blocks.road_polys_from_mask(ss.road_mask)
                bld = next((l["gdf"] for n, l in reversed(ss.layers.items()) if l["kind"] == "buildings"), None)
                veg = next((l["gdf"] for n, l in reversed(ss.layers.items()) if l["kind"] in ("trees", "vegetation")), None)
                g = blocks.make_blocks(aoi, roads, min_blk, bld, veg)
            add_layer("Roads", roads.to_crs(g.crs), "roads")
            add_layer("Blocks", g, "blocks", save_vector(g, out_path("blocks.gpkg")))
        if "Blocks" in ss.layers:
            st.dataframe(ss.layers["Blocks"]["gdf"].drop(columns="geometry"), width="stretch", height=260)

    # ---------------- export
    with t_exp:
        if not ss.layers:
            st.info("Nothing yet – run a tool first.")
        st.caption(f"All results are also saved as GeoPackage in `{workspace().resolve()}` "
                   "(opens directly in ArcGIS Pro and QGIS).")
        for name, l in ss.layers.items():
            g = l["gdf"]
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.write(f"**{name}** — {len(g)} features")
            gj = g.to_crs(4326).to_json()
            c2.download_button("GeoJSON", gj, f"{name}.geojson", "application/geo+json", key=f"gj_{name}")
            csv = g.drop(columns="geometry").to_csv(index=False)
            c3.download_button("CSV", csv, f"{name}.csv", "text/csv", key=f"csv_{name}")

# ------------------------------------------------------------------ map
with right:
    b = gio.raster_bounds_4326(img1)
    m = folium.Map(location=[(b[1] + b[3]) / 2, (b[0] + b[2]) / 2], zoom_start=17, max_zoom=23,
                   tiles=None, control_scale=True)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Imagery", name="Esri imagery", max_zoom=23, max_native_zoom=19).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", max_zoom=23, max_native_zoom=19).add_to(m)
    import base64
    png, bounds = cached_preview(img1, rgb, os.path.getmtime(img1))
    folium.raster_layers.ImageOverlay("data:image/png;base64," + base64.b64encode(png).decode(),
                                      bounds, name="Image date 1", opacity=1).add_to(m)
    if img2:
        png2, bounds2 = cached_preview(img2, rgb, os.path.getmtime(img2))
        folium.raster_layers.ImageOverlay("data:image/png;base64," + base64.b64encode(png2).decode(),
                                          bounds2, name="Image date 2", opacity=1, show=False).add_to(m)
    for name, l in ss.layers.items():
        g = l["gdf"]
        if g is None or g.empty:
            continue
        g4 = g.to_crs(4326)
        g4 = g4[[c for c in g4.columns if c == "geometry" or g4[c].dtype.kind in "iufOb"]]
        fields = [c for c in g4.columns if c != "geometry"][:6]
        if l["kind"] == "change":
            style = lambda f: {"color": CHANGE_COLORS.get(f["properties"].get("status"), "#e67e22"),
                               "fillColor": CHANGE_COLORS.get(f["properties"].get("status"), "#e67e22"),
                               "fillOpacity": 0.5, "weight": 1.5}
        else:
            st_ = STYLES.get(l["kind"], STYLES["objects"])
            style = lambda f, s=st_: s
        folium.GeoJson(json.loads(g4.to_json()), name=f"{name} ({len(g)})", style_function=style,
                       tooltip=folium.GeoJsonTooltip(fields) if fields else None).add_to(m)
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, height=720, use_container_width=True, returned_objects=[])
