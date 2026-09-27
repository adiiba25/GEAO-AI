# GeoAI Studio

A Python GIS app for drone orthomosaics and satellite imagery (high‑res, Sentinel‑2, Landsat):

| Tool | What it produces | Works with |
|---|---|---|
| 🌳 **Vegetation & trees** | Vegetation polygons; trees split from grass when an nDSM is given | RGB drone (ExG), multispectral / Sentinel / Landsat (NDVI) |
| 🚗 **Object detection** | Vehicles (oriented boxes), optionally planes, ships, tanks, pools… | Drone, high‑res satellite (≤ ~1 m) |
| 🤖 **AI segmentation** | Roads, buildings or trees from a text prompt (LangSAM) or your own trained model | Drone, high‑res satellite |
| 🏠 **Building footprints** | Regularised footprints with area, perimeter, orientation, height | Drone + nDSM (no AI needed) or AI building mask |
| 🔁 **Change detection** | Loss/gain polygons (index, all‑band CVA or height) and building change: *new / demolished / modified / unchanged* | Any two dates |
| 🧱 **Urban blocks** | Blocks enclosed by roads with building count, density, building & vegetation cover | Roads from OSM, your own file, or AI road mask |

Everything is processed tile‑by‑tile, so multi‑GB orthomosaics are fine. Results are written as
**GeoPackage** (`.gpkg`) — open them directly in ArcGIS Pro or QGIS — plus GeoJSON/CSV downloads.

---

## 1. Install (Windows)

Use a **separate** Python environment — don't install into ArcGIS Pro's `arcgispro-py3`.

**Easiest:** double‑click `run_app.bat`. First run creates `.venv`, installs the core packages and opens the app in your browser.

**Manual:**
```powershell
cd C:\GIS\geoai_app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**AI extras** (vehicle detection, AI segmentation) — install PyTorch for your hardware first:
```powershell
# NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# or CPU only (works, but slow on big orthos)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-ai.txt
```
Model weights download automatically on first use.

### On a Vast.ai GPU box
```bash
pip install -r requirements.txt -r requirements-ai.txt   # PyTorch images already include torch
streamlit run app.py --server.port 8501 --server.address 0.0.0.0   # open the mapped port
# or headless batch:
python cli.py vehicles /data/ortho.tif --weights yolo26m-obb.pt --device 0 -o /data/out
```

## 2. Try it with sample data

```powershell
python examples\make_sample_data.py
```
Creates a synthetic 300 × 300 m drone survey near Kampala at 10 cm for two dates (RGB ortho + nDSM +
road lines) with 4 new buildings, 2 demolished and a cleared tree patch. In the app, paste the paths from
`sample_data\` into the sidebar and run each tab.

## 3. Recommended workflows

**Drone (DJI Terra):** export the orthomosaic and DSM + DTM. Make the nDSM once
(`nDSM = DSM − DTM`, e.g. ArcGIS Pro *Raster Calculator* or `gdal_calc.py -A dsm.tif -B dtm.tif --calc="A-B" --outfile=ndsm.tif`).
Then: Vegetation (trees vs grass) → Footprints (height route) → Blocks → Change (height or footprints) against the next flight.

**High‑res satellite (0.3–1 m):** set the band preset (e.g. *4‑band B,G,R,NIR*). Vegetation uses NDVI.
Buildings/roads via **Objects → Segment** (LangSAM zero‑shot, or your trained model). Vehicles need ≤ 0.5 m.

**Sentinel‑2 / Landsat (10–30 m):** vegetation and land‑change mapping (NDVI difference or CVA),
and blocks from OSM. Too coarse for individual buildings or vehicles — the app warns you.
Export a band stack in the preset order (e.g. from GEE or SNAP), or set custom band numbers.

## 4. Accuracy notes — read before using results in reports

* **Height‑based footprints** are the most reliable route for drone data (the test data recovers 30/30
  buildings). Watch for trees overhanging roofs, parked trucks and tall walls; tune *min height* and
  *min area*.
* **Vehicles:** the default is **YOLO26‑OBB** (Ultralytics' newest generation, NMS‑free, up to +3.4 mAP over
  YOLO11 on DOTA). YOLO11‑OBB remains selectable. Both were trained on DOTA aerial imagery and have never seen
  Kampala; expect misses in dense boda‑boda / matatu traffic. Use `yolo26m/l/x-obb.pt` on a GPU for better recall.
* **LangSAM** (text prompt) is zero‑shot: great for quick results, inconsistent on informal settlements
  with dense iron‑sheet roofs.
* **For production building/road mapping, train your own model** (`yolo-seg` backend): digitise a few
  hundred buildings from your own imagery in ArcGIS Pro, or start from open African datasets such as
  the *Open Cities AI Challenge* (includes Kampala tiles) or Google Open Buildings for pre‑labels, then
  `yolo segment train data=your.yaml model=yolo26s-seg.pt` on Vast.ai. Point the app at the `best.pt`.
* **Change detection** between different sensors/sun angles produces false change. The app normalises
  radiometry, but use same‑season, same‑sensor pairs where possible and check the threshold.
* Always sample‑check outputs against the imagery before reporting counts.

## 5. Command line

```
python cli.py presets
python cli.py info ortho.tif
python cli.py vegetation ortho.tif --ndsm ndsm.tif -o out
python cli.py footprints ortho.tif --ndsm ndsm.tif -o out
python cli.py segment ortho.tif --target building --backend langsam -o out
python cli.py segment ortho.tif --target building --backend yolo-seg --weights best.pt -o out
python cli.py vehicles ortho.tif --weights yolo26s-obb.pt -o out
python cli.py change t1.tif t2.tif --method index --preset 4 -o out
python cli.py change ndsm_t1.tif ndsm_t2.tif --method height -o out
python cli.py footprint-change out1/buildings.gpkg out2/buildings.gpkg -o out
python cli.py blocks ortho.tif --roads osm --buildings out/buildings.gpkg -o out
```

## 6. Project layout
```
app.py              Streamlit web map UI
cli.py              batch command line
geoai/io.py         raster info, tiling, alignment, previews
geoai/indices.py    band presets, NDVI / ExG / NDWI / NDBI, Otsu
geoai/vegetation.py vegetation & tree mapping
geoai/detect.py     YOLO-OBB detection, LangSAM / YOLO-seg segmentation (tiled, GSD-aware)
geoai/footprints.py footprint extraction & regularisation, zonal height
geoai/change.py     raster change (index / CVA / height), footprint change
geoai/blocks.py     OSM / file / mask roads -> blocks with stats
geoai/vector.py     tiled polygonisation, metrics, export
examples/           synthetic sample data generator
```
