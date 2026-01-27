#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
geoConverter.py

Simple batch and single-file converter for:
  - Shapefile (.shp) <-> GeoPackage (.gpkg)
  - Shapefile (.shp) / GeoPackage (.gpkg) -> GeoJSON (.geojson)

The script is controlled via the CONFIG section below. It does not
provide a CLI – just edit the variables and run:

    python geoConverter.py

Author: Franz Wagner
Date: 2025-12-02
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import geopandas as gpd


# =========================
# ===== CONFIG (GLOBAL) ===
# =========================

# Mode of operation:
#   "shp2gpkg"  - convert Shapefile  -> GeoPackage
#   "gpkg2shp"  - convert GeoPackage -> Shapefile
#   "shp2json"  - convert Shapefile  -> GeoJSON
#   "gpkg2json" - convert GeoPackage -> GeoJSON
MODE: str = "shp2gpkg"

# Input can be a single file or a directory.
INPUT_PATH: str = r"F:\fram3s\01-data\01-aoi\TESTSITES\testsites_rofental_extended.shp"

# Output directory. If None, outputs are written next to the inputs.
OUTPUT_DIR: Optional[str] = None

# If INPUT_PATH is a directory, decide whether to recurse into subfolders.
RECURSIVE: bool = False

# Overwrite existing output files?
OVERWRITE: bool = False

# Logging level: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL: str = "INFO"


# =================
# LOGGING SETUP
# =================
def configure_logging(level: str = LOG_LEVEL) -> None:
    """Configure simple console logging."""
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
    )
    logging.info("Logging initialized (level=%s).", level.upper())


# =================
# HELPER FUNCTIONS
# =================
def _expected_suffix(mode: str) -> str:
    if mode in ("shp2gpkg", "shp2json"):
        return ".shp"
    if mode in ("gpkg2shp", "gpkg2json"):
        return ".gpkg"
    raise ValueError(f"Unsupported MODE: {mode}")


def discover_input_files(mode: str, input_path: Path, recursive: bool) -> List[Path]:
    """Return list of input files based on mode and INPUT_PATH."""
    expected = _expected_suffix(mode)

    if input_path.is_file():
        if input_path.suffix.lower() != expected:
            logging.error("INPUT_PATH has wrong extension for MODE %s: %s", mode, input_path)
            return []
        return [input_path]

    if not input_path.is_dir():
        logging.error("INPUT_PATH does not exist or is not a directory: %s", input_path)
        return []

    pattern = f"*{expected}"
    if recursive:
        files = sorted(p for p in input_path.rglob(pattern) if p.is_file())
    else:
        files = sorted(p for p in input_path.glob(pattern) if p.is_file())

    logging.info("Found %d input file(s) in %s (recursive=%s).", len(files), input_path, recursive)
    return files


def ensure_output_dir(path: Path) -> None:
    """Create output directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def output_directory_for(src: Path, configured_output: Optional[Path]) -> Path:
    """Decide where to place the converted file."""
    if configured_output is not None:
        return configured_output
    # Default: same folder as source
    return src.parent


def convert_shp_to_gpkg(src: Path, dst_dir: Path, overwrite: bool) -> bool:
    """Convert a single shapefile to GeoPackage."""
    try:
        ensure_output_dir(dst_dir)
        dst = dst_dir / f"{src.stem}.gpkg"

        if dst.exists():
            if not overwrite:
                logging.info("Skipping existing GPKG (overwrite=False): %s", dst)
                return True
            logging.warning("Overwriting existing GPKG: %s", dst)
            dst.unlink()

        logging.debug("Reading SHP: %s", src)
        gdf = gpd.read_file(src)

        logging.debug("Writing GPKG: %s", dst)
        gdf.to_file(dst, layer=src.stem, driver="GPKG")

        logging.info("Converted SHP -> GPKG: %s -> %s", src, dst)
        return True
    except Exception as exc:
        logging.error("Failed SHP -> GPKG: %s | %s", src, exc)
        return False


def convert_gpkg_to_shp(src: Path, dst_dir: Path, overwrite: bool) -> bool:
    """Convert a single GeoPackage (first layer) to Shapefile."""
    try:
        ensure_output_dir(dst_dir)
        dst = dst_dir / f"{src.stem}.shp"

        if dst.exists():
            if not overwrite:
                logging.info("Skipping existing SHP (overwrite=False): %s", dst)
                return True
            logging.warning("Overwriting existing SHP: %s", dst)
            # Remove all shapefile sidecar files
            for side in dst_dir.glob(f"{src.stem}.*"):
                side.unlink()

        logging.debug("Reading GPKG: %s", src)
        # By default, GeoPandas reads the first layer.
        gdf = gpd.read_file(src)

        logging.debug("Writing SHP: %s", dst)
        gdf.to_file(dst, driver="ESRI Shapefile")

        logging.info("Converted GPKG -> SHP: %s -> %s", src, dst)
        return True
    except Exception as exc:
        logging.error("Failed GPKG -> SHP: %s | %s", src, exc)
        return False


def convert_shp_to_geojson(src: Path, dst_dir: Path, overwrite: bool) -> bool:
    """Convert a single shapefile to GeoJSON."""
    try:
        ensure_output_dir(dst_dir)
        dst = dst_dir / f"{src.stem}.geojson"

        if dst.exists():
            if not overwrite:
                logging.info("Skipping existing GeoJSON (overwrite=False): %s", dst)
                return True
            logging.warning("Overwriting existing GeoJSON: %s", dst)
            dst.unlink()

        logging.debug("Reading SHP: %s", src)
        gdf = gpd.read_file(src)

        logging.debug("Writing GeoJSON: %s", dst)
        gdf.to_file(dst, driver="GeoJSON")

        logging.info("Converted SHP -> GeoJSON: %s -> %s", src, dst)
        return True
    except Exception as exc:
        logging.error("Failed SHP -> GeoJSON: %s | %s", src, exc)
        return False


def convert_gpkg_to_geojson(src: Path, dst_dir: Path, overwrite: bool) -> bool:
    """Convert a single GeoPackage (first layer) to GeoJSON."""
    try:
        ensure_output_dir(dst_dir)
        dst = dst_dir / f"{src.stem}.geojson"

        if dst.exists():
            if not overwrite:
                logging.info("Skipping existing GeoJSON (overwrite=False): %s", dst)
                return True
            logging.warning("Overwriting existing GeoJSON: %s", dst)
            dst.unlink()

        logging.debug("Reading GPKG: %s", src)
        # By default, GeoPandas reads the first layer.
        gdf = gpd.read_file(src)

        logging.debug("Writing GeoJSON: %s", dst)
        gdf.to_file(dst, driver="GeoJSON")

        logging.info("Converted GPKG -> GeoJSON: %s -> %s", src, dst)
        return True
    except Exception as exc:
        logging.error("Failed GPKG -> GeoJSON: %s | %s", src, exc)
        return False


# =================
# MAIN ORCHESTRATION
# =================
def main() -> None:
    configure_logging(LOG_LEVEL)

    mode = MODE.lower()
    if mode not in {"shp2gpkg", "gpkg2shp", "shp2json", "gpkg2json"}:
        logging.error("Unsupported MODE: %s", MODE)
        return

    input_path = Path(INPUT_PATH)
    output_root = Path(OUTPUT_DIR) if OUTPUT_DIR is not None else None

    logging.info("Mode        : %s", mode)
    logging.info("INPUT_PATH  : %s", input_path)
    logging.info("OUTPUT_DIR  : %s", output_root if output_root is not None else "<same as input>")
    logging.info("RECURSIVE   : %s", RECURSIVE)
    logging.info("OVERWRITE   : %s", OVERWRITE)

    files = discover_input_files(mode, input_path, RECURSIVE)
    if not files:
        logging.warning("No input files found. Nothing to do.")
        return

    success = 0
    total = len(files)

    for idx, src in enumerate(files, start=1):
        logging.info("[%d/%d] Processing: %s", idx, total, src)
        out_dir = output_directory_for(src, output_root)

        if mode == "shp2gpkg":
            ok = convert_shp_to_gpkg(src, out_dir, OVERWRITE)
        elif mode == "gpkg2shp":
            ok = convert_gpkg_to_shp(src, out_dir, OVERWRITE)
        elif mode == "shp2json":
            ok = convert_shp_to_geojson(src, out_dir, OVERWRITE)
        elif mode == "gpkg2json":
            ok = convert_gpkg_to_geojson(src, out_dir, OVERWRITE)
        else:
            ok = False  # Should not reach here due to earlier check.

        if ok:
            success += 1

    logging.info("Finished. %d/%d file(s) converted successfully.", success, total)


if __name__ == "__main__":
    main()

