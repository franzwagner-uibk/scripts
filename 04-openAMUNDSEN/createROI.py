# -*- coding: utf-8 -*-
"""
Script: create_ascii_bbox_raster.py
Author: Franz Wagner
Date: 2025-10-13
Description:
    Creates a 1/0 ROI ASCII raster aligned to a DEM grid.
    ROI = bounding box of either:
      A) a dataset given by LAYER_DATASET_PATH (feature class / shapefile), or
      B) a layer (LAYER_NAME) inside MAP_NAME of an APRX.
    After conversion, the ROI ASCII header is rewritten to match the DEM's
    header exactly (ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value).

Notes:
    * All RELATIVE paths are resolved against BASE_DIR (default = os.getcwd()).
    * No auto-search; no forcing paths into the script directory.
"""

from __future__ import annotations
import arcpy
import os
import sys
import logging
from datetime import datetime
from typing import Optional, Tuple

# ----------------- CONFIG (EDIT HERE) -----------------
# Base directory for resolving relative paths. Default uses current working directory at runtime.
BASE_DIR              = os.getcwd()  # e.g., r"C:\Daten\PhD" to pin it explicitly

# Option A (recommended): direct dataset whose EXTENT defines ROI
LAYER_DATASET_PATH    = None  # e.g., r"02-Daten\gmba\gmba_l8_15422.shp" or r"my.gdb\gmba_l8_15422"

# Option B: APRX + Map + Layer name (used only if LAYER_DATASET_PATH is None)
PROJECT_PATH          = r"04-APRX\Euregio\Euregio.aprx"
MAP_NAME              = "Tux"
LAYER_NAME            = "GMBA_Inventory_L8_15422"

# DEM ASCII to align to, and output ASCII path
REFERENCE_DEM_ASCII   = r"02-Daten\15422\grids\dem_15422_100.asc"
OUTPUT_ASCII          = r"02-Daten\15422\grids\roi_15422_100.asc"

# ROI values and header handling
VAL_INSIDE, VAL_OUTSIDE = 1, 0
FORCE_NODATA          = -9999  # or None to keep DEM header's NODATA

# Logging
LOG_FOLDER            = r"C:\temp\logs"
# ------------------------------------------------------

# ---------------- LOGGING (after CONFIG) ----------------
def _init_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, f"roi_{datetime.now():%Y%m%d_%H%M%S}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(logfile, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    )
    logging.info("Logging initialized.")

# ---------------- PATH HELPERS ----------------
def resolve(path: str) -> str:
    """Resolve absolute path. If relative, resolve against BASE_DIR (not script dir)."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(BASE_DIR, path))

def must_exist(path: str, label: str) -> None:
    """Raise with clear message if path does not exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")

# ---------------- ASCII HEADER FIX ----------------
def rewrite_header_like_dem(roi_ascii: str, dem_ascii: str, nodata: Optional[int]) -> None:
    """Rewrite ROI ASCII header with DEM ASCII header values; optionally force NODATA_value."""
    with open(dem_ascii, "r", encoding="utf-8") as f:
        dem_lines = f.readlines()
    with open(roi_ascii, "r", encoding="utf-8") as f:
        roi_lines = f.readlines()

    roi_lines[:6] = dem_lines[:6]  # copy header lines from DEM

    if nodata is not None:
        for i, l in enumerate(roi_lines[:12]):  # robust rewrite if line order differs
            if l.lower().startswith("nodata_value"):
                roi_lines[i] = f"NODATA_value {nodata}\n"

    with open(roi_ascii, "w", encoding="utf-8") as f:
        f.writelines(roi_lines)
    logging.info("ROI header normalized to DEM.")

# ---------------- EXTENT HELPERS ----------------
def get_extent_from_dataset(dataset_path: str, target_sr) -> Tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) of dataset, projected to target_sr if needed."""
    desc = arcpy.Describe(dataset_path)
    ext = desc.extent
    if desc.spatialReference and target_sr and desc.spatialReference.name != target_sr.name:
        ext = ext.projectAs(target_sr)
    return ext.XMin, ext.YMin, ext.XMax, ext.YMax

def get_extent_from_aprx_layer(aprx_path: str, map_name: str, layer_name: str, target_sr) -> Tuple[float, float, float, float]:
    """Open APRX and read extent of named layer (project to target_sr if needed)."""
    aprx = arcpy.mp.ArcGISProject(aprx_path)
    maps = aprx.listMaps(map_name)
    if not maps:
        raise ValueError(f"Map '{map_name}' not found in APRX: {aprx_path}")
    lyr_candidates = [l for l in maps[0].listLayers() if l.name == layer_name]
    if not lyr_candidates:
        raise ValueError(f"Layer '{layer_name}' not found in map '{map_name}'.")
    d = arcpy.Describe(lyr_candidates[0])
    ext = d.extent
    if d.spatialReference and target_sr and d.spatialReference.name != target_sr.name:
        ext = ext.projectAs(target_sr)
    return ext.XMin, ext.YMin, ext.XMax, ext.YMax

def build_bbox_fc(extent_xyxy: Tuple[float, float, float, float], spatial_ref) -> str:
    """Create an in_memory polygon from (xmin, ymin, xmax, ymax)."""
    xmin, ymin, xmax, ymax = extent_xyxy
    arr = arcpy.Array([
        arcpy.Point(xmin, ymin),
        arcpy.Point(xmin, ymax),
        arcpy.Point(xmax, ymax),
        arcpy.Point(xmax, ymin),
        arcpy.Point(xmin, ymin),
    ])
    poly = arcpy.Polygon(arr, spatial_ref)
    fc = r"in_memory\bbox"
    if arcpy.Exists(fc):
        arcpy.Delete_management(fc)
    arcpy.CreateFeatureclass_management("in_memory", "bbox", "POLYGON", spatial_reference=spatial_ref)
    with arcpy.da.InsertCursor(fc, ["SHAPE@"]) as cur:
        cur.insertRow([poly])
    logging.info("In-memory bbox feature class created.")
    return fc

# ---------------- CORE ----------------
def create_roi_raster_as_ascii(
    project_path: Optional[str],
    map_name: str,
    layer_name: str,
    dataset_path: Optional[str],
    dem_ascii_path: str,
    output_ascii_path: str,
    val_inside: int,
    val_outside: int,
    force_nodata: Optional[int]
) -> None:
    """Create the 1/0 ROI ASCII aligned to DEM ASCII."""
    # Check inputs
    must_exist(dem_ascii_path, "Reference DEM ASCII")
    os.makedirs(os.path.dirname(output_ascii_path) or ".", exist_ok=True)

    # DEM raster defines environment + target SR
    dem_ras = arcpy.Raster(dem_ascii_path)
    arcpy.env.snapRaster = dem_ras
    arcpy.env.extent = dem_ras.extent
    arcpy.env.cellSize = dem_ras.meanCellWidth
    arcpy.env.outputCoordinateSystem = dem_ras.spatialReference
    arcpy.env.overwriteOutput = True
    logging.info("Environment set (snapRaster, extent, cellSize, outputCoordinateSystem).")

    # Determine extent
    if dataset_path:
        must_exist(dataset_path, "Layer dataset")
        xmin, ymin, xmax, ymax = get_extent_from_dataset(dataset_path, dem_ras.spatialReference)
        logging.info(f"Extent from dataset: {dataset_path}")
    else:
        must_exist(project_path, "APRX project")
        xmin, ymin, xmax, ymax = get_extent_from_aprx_layer(project_path, map_name, layer_name, dem_ras.spatialReference)
        logging.info(f"Extent from APRX layer: {layer_name} in map '{map_name}'")

    # Build bbox polygon
    bbox_fc = build_bbox_fc((xmin, ymin, xmax, ymax), dem_ras.spatialReference)

    # Build ROI raster: inside=val_inside, outside=val_outside (in memory)
    ones = arcpy.sa.CreateConstantRaster(val_inside, "INTEGER", dem_ras.meanCellWidth, dem_ras.extent)
    masked = arcpy.sa.Con(arcpy.sa.IsNull(arcpy.sa.ExtractByMask(ones, bbox_fc)), val_outside, val_inside)
    logging.info("ROI raster created in memory.")

    # Convert directly to ASCII (no temp save needed)
    arcpy.conversion.RasterToASCII(masked, output_ascii_path)
    logging.info(f"RasterToASCII written: {output_ascii_path}")

    # Normalize header to DEM
    rewrite_header_like_dem(output_ascii_path, dem_ascii_path, force_nodata)

# ---------------- MAIN ----------------
def main() -> None:
    _init_logging(LOG_FOLDER)

    # Resolve all paths against BASE_DIR (only if they are relative)
    abs_project = resolve(PROJECT_PATH) if PROJECT_PATH else None
    abs_dataset = resolve(LAYER_DATASET_PATH) if LAYER_DATASET_PATH else None
    abs_dem     = resolve(REFERENCE_DEM_ASCII)
    abs_out     = resolve(OUTPUT_ASCII)

    logging.info("----- CONFIG -----")
    logging.info(f"BASE_DIR            = {BASE_DIR}")
    logging.info(f"CWD                 = {os.getcwd()}")
    logging.info(f"LAYER_DATASET_PATH  = {abs_dataset}")
    logging.info(f"PROJECT_PATH        = {abs_project}")
    logging.info(f"MAP_NAME            = {MAP_NAME}")
    logging.info(f"LAYER_NAME          = {LAYER_NAME}")
    logging.info(f"REFERENCE_DEM_ASCII = {abs_dem}")
    logging.info(f"OUTPUT_ASCII        = {abs_out}")
    logging.info(f"VAL_INSIDE/OUTSIDE  = {VAL_INSIDE}/{VAL_OUTSIDE}")
    logging.info(f"FORCE_NODATA        = {FORCE_NODATA}")
    logging.info("-------------------")

    arcpy.CheckOutExtension("Spatial")
    try:
        create_roi_raster_as_ascii(
            project_path=abs_project,
            map_name=MAP_NAME,
            layer_name=LAYER_NAME,
            dataset_path=abs_dataset,
            dem_ascii_path=abs_dem,
            output_ascii_path=abs_out,
            val_inside=VAL_INSIDE,
            val_outside=VAL_OUTSIDE,
            force_nodata=FORCE_NODATA
        )
        logging.info("Done.")
    finally:
        arcpy.CheckInExtension("Spatial")
        logging.info("Spatial Analyst license returned.")

if __name__ == "__main__":
    main()
