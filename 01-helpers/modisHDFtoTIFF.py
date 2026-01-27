#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: modis_hdf_to_geotiff.py
Description:
    Batch-convert MODIS HDF/HDF-EOS downloads (e.g., MOD10A1/MOD10A2, etc.)
    into per-layer GeoTIFFs (or ASCII grids), reprojected to a target CRS (e.g., UTM),
    with optional clipping to an AOI vector.

    The script:
      1) Recursively scans an input folder for .hdf files (XMLs are ignored).
      2) Lists ALL subdatasets in each HDF and processes each one.
      3) Reprojects to TARGET_EPSG (e.g., EPSG:32632 for Tyrol West or 32633 East).
      4) Optional AOI crop:
         - Fast rectangle crop using AOI envelope in target CRS (outputBounds) [default], or
         - Cutline crop to the exact AOI polygon (slower).
      5) Writes outputs into a clean folder structure: <OUTPUT_DIR>/<PRODUCT>/<YYYY-MM-DD>/<TILE>/<SDS>.tif
         (Handles products with or without MODIS h/v tile codes.)
      6) Safe defaults: nearest-neighbor resampling for classes, compression, and overviews.

Author: Franz Wagner
Date: 2025-10-15
"""

from __future__ import annotations

import os
import re
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from osgeo import gdal, osr, ogr  # <- added ogr

# =============================================================================
# ============================ USER CONFIG (GLOBAL) ============================
# =============================================================================

# --- REQUIRED PATHS ---
INPUT_DIR: str  = r"02-Daten\MOD10A1_61-20251015_114228"    # <- set me
OUTPUT_DIR: str = r"02-Daten\MODIS-Test"                    # <- set me
LOG_DIR: str    = os.path.join(OUTPUT_DIR, "log")           # logs go here

# --- TARGET COORDINATE SYSTEM (UTM for Tyrol) ---
# Western Tyrol ~ EPSG:32632, Eastern Tyrol ~ EPSG:32633
TARGET_EPSG: int = 32632

# When reprojecting to meters-based UTM, choose output pixel size (meters).
# For MODIS 500 m products, set to 500.0 to keep close to native scale.
# Set to None to let GDAL choose automatically.
TARGET_RES_METERS: Optional[float] = 500.0

# Snap output grid to the chosen resolution to avoid half-pixel shifts
TARGET_ALIGNED_PIXELS: bool = True

# --- CLIP TO AOI ---
# Provide any vector format GDAL can read (SHP, GPKG, GeoJSON, etc.)
CLIP_TO_AOI: bool = True
AOI_PATH: Optional[str] = r"02-Daten\AOI\AOI-Euregio-BoundingBox.shp"    # <- set me or keep None

# Choose *one* AOI mode:
# - If True  -> fast rectangle crop using AOI envelope in target CRS via outputBounds.
# - If False -> exact polygon cutline (slower; uses CUTLINE_* settings below).
USE_AOI_ENVELOPE_BOUNDS: bool = True

# Cutline settings (used only when USE_AOI_ENVELOPE_BOUNDS=False)
CUTLINE_CROP: bool = True               # crop to AOI extent (cutline)
CUTLINE_BLEND: int = 0                  # 0 = hard edge; >0 blends edge in pixels

# --- OUTPUT FORMAT ---
# "GTiff" (GeoTIFF) or "AAIGrid" (ESRI ASCII grid)
OUTPUT_FORMAT: str = "GTiff"

# --- BEHAVIOR ---
OVERWRITE: bool = True
RECURSIVE: bool = True
NUM_THREADS: str = "ALL_CPUS"  # informational; GDAL Warp uses multithread=True internally

# --- OPTIONAL FILTER: process only certain SDS names (None = all) ---
# e.g., SDS_FILTER = ["Snow_Cover_Daily_Tile", "Fractional_Snow_Cover"]
SDS_FILTER: Optional[List[str]] = None

# --- RESAMPLING SETTINGS ---
# Default resampling for reprojection; categorical (e.g., class maps): "near"
RESAMPLE_DEFAULT: str = "near"
# Per-layer overrides (e.g. continuous variables). Example:
# RESAMPLE_OVERRIDES = {"NDSI": "bilinear", "Snow_Albedo_Daily_Tile": "bilinear"}
RESAMPLE_OVERRIDES: Dict[str, str] = {}

# --- GDAL CREATION OPTIONS ---
# Reasonable GeoTIFF defaults: compression, tiling, bigtiff auto
GTIFF_CREATION_OPTS: List[str] = ["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2", "BIGTIFF=IF_SAFER"]
# ESRI ASCII grid has no creation options typically
AAIGRID_CREATION_OPTS: List[str] = []

# --- OVERVIEWS ---
BUILD_OVERVIEWS: bool = True
OVERVIEW_LEVELS: List[int] = [2, 4, 8, 16]

# =============================================================================
# ================================ LOGGING ====================================
# =============================================================================

def setup_logging() -> None:
    """Configure logging to file and console."""
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"modis_hdf_to_geotiff_{ts}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Logging initialized.")
    logging.info("GDAL version: %s", gdal.VersionInfo())

def verify_gdal_drivers() -> None:
    """Log presence of critical drivers (HDF4/HDF5 variations)."""
    drv_names = ["HDF4", "HDF4Image", "HDF5", "HDF5Image"]
    present = []
    for d in drv_names:
        if gdal.GetDriverByName(d):
            present.append(d)
    if present:
        logging.info("Required GDAL drivers present: %s", ", ".join(present))
    else:
        logging.warning("No HDF drivers detected! Ensure GDAL with HDF support is installed.")

# =============================================================================
# ============================== HELPER UTILS =================================
# =============================================================================

MODIS_NAME_RE = re.compile(
    r"^(?P<product>[A-Z0-9]{7,})\.A(?P<yeardoy>\d{7})(?:\.(?P<hv>h\d{2}v\d{2}))?",
    re.IGNORECASE
)

def list_hdf_files(root: str, recursive: bool = True) -> List[str]:
    """List .hdf (or .HDF) files under root."""
    files: List[str] = []
    if recursive:
        for dpath, _, fnames in os.walk(root):
            for fn in fnames:
                if fn.lower().endswith(".hdf"):
                    files.append(os.path.join(dpath, fn))
    else:
        for fn in os.listdir(root):
            if fn.lower().endswith(".hdf"):
                files.append(os.path.join(root, fn))
    return files

def parse_modis_filename(path: str) -> Dict[str, Optional[str]]:
    """Parse product, YYYYDOY, and tile (hXXvYY) from a MODIS HDF filename."""
    base = os.path.basename(path)
    m = MODIS_NAME_RE.match(base)
    if not m:
        return {"product": None, "yeardoy": None, "tile": None}
    return m.groupdict()

def yeardoy_to_datestr(yeardoy: str) -> str:
    """Convert YYYYDOY to 'YYYY-MM-DD'."""
    year = int(yeardoy[:4])
    doy  = int(yeardoy[4:])
    dt = datetime.strptime(f"{year}-{doy:03d}", "%Y-%j")
    return dt.strftime("%Y-%m-%d")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def get_creation_options(fmt: str) -> List[str]:
    if fmt.lower() == "gtiff":
        return GTIFF_CREATION_OPTS
    if fmt.lower() == "aaigrid":
        return AAIGRID_CREATION_OPTS
    return []

def detect_resample(sds_name: str) -> str:
    """Choose resampling method per SDS name (override else default)."""
    for key, method in RESAMPLE_OVERRIDES.items():
        if key.lower() == sds_name.lower():
            return method
    return RESAMPLE_DEFAULT

def build_output_dir(product: Optional[str], yeardoy: Optional[str], tile: Optional[str]) -> str:
    """Build hierarchical output dir."""
    parts = [OUTPUT_DIR, product or "UNKNOWN_PRODUCT"]
    if yeardoy:
        parts.append(yeardoy_to_datestr(yeardoy))  # e.g., 2018-02-01
    if tile:
        parts.append(tile)
    out_dir = os.path.join(*parts)
    ensure_dir(out_dir)
    return out_dir

def sanitize_name(name: str) -> str:
    """Make a safe filename from an SDS name/description."""
    name = re.sub(r'[^A-Za-z0-9_\-]+', '_', name.strip())
    return name.strip("_")

def is_geographic_wgs84(dataset: gdal.Dataset) -> bool:
    """Check if dataset is geographic WGS84 (EPSG:4326)."""
    wkt = dataset.GetProjection()
    if not wkt:
        return False
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    return srs.IsGeographic() == 1

# ---------------- AOI envelope helper (fast outputBounds crop) ----------------

_AOI_ENV_CACHE: Optional[Tuple[float, float, float, float]] = None
_AOI_ENV_CACHE_EPSG: Optional[int] = None
_AOI_ENV_CACHE_PATH: Optional[str] = None

def aoi_envelope_in_epsg(aoi_path: str, target_epsg: int) -> Tuple[float, float, float, float]:
    """
    Return (minx, miny, maxx, maxy) of AOI reprojected to EPSG:target_epsg.
    Caches result per (path, epsg) to avoid repeated IO.
    """
    global _AOI_ENV_CACHE, _AOI_ENV_CACHE_EPSG, _AOI_ENV_CACHE_PATH
    if (_AOI_ENV_CACHE is not None
        and _AOI_ENV_CACHE_EPSG == target_epsg
        and _AOI_ENV_CACHE_PATH == os.path.abspath(aoi_path)):
        return _AOI_ENV_CACHE

    ds = ogr.Open(aoi_path, 0)
    if ds is None:
        raise RuntimeError(f"Cannot open AOI: {aoi_path}")
    lyr = ds.GetLayer(0)
    src_srs = lyr.GetSpatialRef()
    tgt_srs = osr.SpatialReference(); tgt_srs.ImportFromEPSG(int(target_epsg))
    tx = osr.CoordinateTransformation(src_srs, tgt_srs) if (src_srs and not src_srs.IsSame(tgt_srs)) else None

    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for f in lyr:
        g = f.GetGeometryRef()
        if g is None:
            continue
        g = g.Clone()
        g.FlattenTo2D()
        if tx:
            g.Transform(tx)
        e = g.GetEnvelope()  # (minx, maxx, miny, maxy)
        minx = min(minx, e[0]); maxx = max(maxx, e[1])
        miny = min(miny, e[2]); maxy = max(maxy, e[3])
    ds = None

    if minx == float("inf"):
        raise RuntimeError("AOI has no valid geometry.")

    env = (minx, miny, maxx, maxy)
    _AOI_ENV_CACHE = env
    _AOI_ENV_CACHE_EPSG = target_epsg
    _AOI_ENV_CACHE_PATH = os.path.abspath(aoi_path)
    return env

# =============================================================================
# ============================ CORE PROCESSING ================================
# =============================================================================

def list_subdatasets(hdf_path: str) -> List[Tuple[str, str]]:
    """
    Return list of (subdataset_name, description) from an HDF file.
    """
    ds = gdal.Open(hdf_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open HDF: {hdf_path}")
    sds_list = ds.GetSubDatasets() or []
    if not sds_list:
        logging.warning("No subdatasets found in %s; attempting to process the root as a single dataset.", hdf_path)
        return [(hdf_path, os.path.basename(hdf_path))]
    return [(sds_name, desc) for sds_name, desc in sds_list]

def export_layer_to_target(
    subdataset: str,
    out_path: str,
    target_epsg: int,
    out_format: str,
    resample: str,
) -> None:
    """
    Reproject/write a subdataset directly to target CRS and desired format, with optional AOI clipping.
    """
    creation_opts = get_creation_options(out_format)

    warp_kwargs = dict(
        dstSRS=f"EPSG:{target_epsg}",
        format=out_format,
        resampleAlg=resample,
        multithread=True,                     # enable internal warper threading
        creationOptions=creation_opts,
        dstNodata=None,                       # keep native nodata if present
        targetAlignedPixels=TARGET_ALIGNED_PIXELS,
    )

    # AOI: fast rectangle crop via outputBounds (default), or exact cutline
    aoi_label = "none"
    if CLIP_TO_AOI and AOI_PATH:
        aoi_label = os.path.basename(AOI_PATH)
        if USE_AOI_ENVELOPE_BOUNDS:
            bounds = aoi_envelope_in_epsg(AOI_PATH, target_epsg)
            warp_kwargs.update({
                "outputBounds": bounds,
                "outputBoundsSRS": f"EPSG:{target_epsg}",
            })
        else:
            warp_kwargs.update({
                "cutlineDSName": AOI_PATH,
                "cropToCutline": CUTLINE_CROP,
                "cutlineBlend": CUTLINE_BLEND,
            })

    # Pixel size (meters) for UTM
    if TARGET_RES_METERS is not None:
        warp_kwargs.update({
            "xRes": float(TARGET_RES_METERS),
            "yRes": float(TARGET_RES_METERS),
        })

    warp_opts = gdal.WarpOptions(**warp_kwargs)

    # Log what we’re doing
    log_aoi_mode = (
        f"outputBounds={bounds}" if (CLIP_TO_AOI and AOI_PATH and USE_AOI_ENVELOPE_BOUNDS)
        else ("cutline" if (CLIP_TO_AOI and AOI_PATH) else "none")
    )
    logging.info(
        "  -> gdal.Warp to %s | EPSG=%s | resample=%s | fmt=%s | xRes/yRes=%s | AOI=%s | %s",
        out_path, target_epsg, resample, out_format,
        TARGET_RES_METERS if TARGET_RES_METERS is not None else "auto",
        aoi_label, log_aoi_mode,
    )

    if os.path.exists(out_path) and OVERWRITE:
        try:
            os.remove(out_path)
        except Exception:
            pass

    res = gdal.Warp(out_path, subdataset, options=warp_opts)
    if res is None:
        raise RuntimeError(f"gdal.Warp failed for: {subdataset}")
    res = None  # flush

    if BUILD_OVERVIEWS and out_format.lower() == "gtiff":
        ds = gdal.Open(out_path, gdal.GA_Update)
        if ds:
            logging.info("  -> building overviews %s", OVERVIEW_LEVELS)
            ds.BuildOverviews("NEAREST", OVERVIEW_LEVELS)
            ds = None

def process_hdf_file(hdf_path: str) -> None:
    """Process a single HDF: extract every SDS (optionally filtered) and export."""
    meta = parse_modis_filename(hdf_path)
    product = meta.get("product")
    yeardoy = meta.get("yeardoy")
    tile    = meta.get("hv")

    logging.info("Processing: %s | product=%s | date(yeardoy)=%s | tile=%s",
                 os.path.basename(hdf_path), product, yeardoy, tile)

    try:
        subdatasets = list_subdatasets(hdf_path)
    except Exception as e:
        logging.exception("Failed reading subdatasets: %s", e)
        return

    # Filter if requested
    if SDS_FILTER:
        subdatasets = [s for s in subdatasets if any(x.lower() in s[0].lower() for x in SDS_FILTER)]
        if not subdatasets:
            logging.warning("No subdatasets matched SDS_FILTER in %s", hdf_path)
            return

    out_dir = build_output_dir(product, yeardoy, tile)

    for sds_name, sds_desc in subdatasets:
        # Label from subdataset or description
        m = re.search(r":([^:]+)$", sds_name)
        sds_label = m.group(1) if m else (sds_desc.split(":")[-1] if ":" in sds_desc else os.path.basename(sds_name))
        sds_label = sanitize_name(sds_label)

        ext = ".tif" if OUTPUT_FORMAT.lower() == "gtiff" else ".asc"
        out_path = os.path.join(out_dir, f"{sds_label}{ext}")

        # Decide resample per SDS
        resample = detect_resample(sds_label)

        try:
            export_layer_to_target(
                subdataset=sds_name,
                out_path=out_path,
                target_epsg=TARGET_EPSG,
                out_format=OUTPUT_FORMAT,
                resample=resample,
            )
            logging.info("✓ Wrote: %s", out_path)
        except Exception:
            logging.exception("✗ Failed exporting %s (%s)", sds_label, sds_name)

# =============================================================================
# ================================== MAIN =====================================
# =============================================================================

def main() -> None:
    """
    Entry point: validates config, scans input, and processes all HDF files.
    """
    setup_logging()
    verify_gdal_drivers()

    # Basic checks
    if not os.path.isdir(INPUT_DIR):
        logging.error("INPUT_DIR does not exist: %s", INPUT_DIR)
        sys.exit(1)
    ensure_dir(OUTPUT_DIR)

    # Informative logs
    logging.info("INPUT_DIR  = %s", INPUT_DIR)
    logging.info("OUTPUT_DIR = %s", OUTPUT_DIR)
    logging.info("LOG_DIR    = %s", LOG_DIR)
    logging.info("TARGET_EPSG= %s", TARGET_EPSG)
    logging.info("OUTPUT_FORMAT = %s", OUTPUT_FORMAT)
    logging.info("NUM_THREADS   = %s", NUM_THREADS)
    logging.info("RESAMPLE_DEFAULT  = %s", RESAMPLE_DEFAULT)
    logging.info("RESAMPLE_OVERRIDES= %s", RESAMPLE_OVERRIDES)
    logging.info("CLIP_TO_AOI = %s | AOI_PATH = %s | USE_AOI_ENVELOPE_BOUNDS = %s | CUTLINE_CROP = %s | CUTLINE_BLEND = %s",
                 CLIP_TO_AOI, AOI_PATH, USE_AOI_ENVELOPE_BOUNDS, CUTLINE_CROP, CUTLINE_BLEND)
    logging.info("TARGET_RES_METERS = %s | TARGET_ALIGNED_PIXELS = %s",
                 TARGET_RES_METERS if TARGET_RES_METERS is not None else "auto",
                 TARGET_ALIGNED_PIXELS)

    # If we'll need the AOI envelope, try computing once up front (fail-fast).
    if CLIP_TO_AOI and AOI_PATH and USE_AOI_ENVELOPE_BOUNDS:
        try:
            env = aoi_envelope_in_epsg(AOI_PATH, TARGET_EPSG)
            logging.info("AOI envelope in EPSG:%s -> %s", TARGET_EPSG, env)
        except Exception as e:
            logging.exception("Failed computing AOI envelope: %s", e)
            sys.exit(1)

    # Scan HDF files
    hdf_files = list_hdf_files(INPUT_DIR, recursive=RECURSIVE)
    if not hdf_files:
        logging.warning("No .hdf files found in %s (recursive=%s)", INPUT_DIR, RECURSIVE)
        return

    logging.info("Found %d HDF file(s).", len(hdf_files))

    # Process each HDF
    for idx, hdf in enumerate(sorted(hdf_files), start=1):
        logging.info("[%d/%d] %s", idx, len(hdf_files), hdf)
        process_hdf_file(hdf)

    logging.info("All done.")

if __name__ == "__main__":
    main()
