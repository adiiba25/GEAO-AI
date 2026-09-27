"""GeoAI Studio - drone & satellite imagery processing toolkit.

Modules
-------
io          raster reading, tiling, alignment, previews
indices     band presets and spectral indices (NDVI, ExG, NDWI, NDBI)
vegetation  vegetation / tree-canopy mapping
detect      object detection (vehicles) and segmentation (buildings, roads)
footprints  mask -> regularised building footprints
change      raster change detection and footprint (object) change
blocks      urban blocks from road networks (OSM, file or detected)
vector      polygonising masks tile-by-tile, area helpers
"""
__version__ = "0.1.0"
