#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nc_to_tif_batch.py

Batch-convert SnowFLAKES NetCDF rasters (e.g., FSC + uncertainty) to GeoTIFF.
Configuration is done via the CONFIG section below – no CLI flags.

Default assumptions
- NetCDF has 2-D raster variables with dimensions (y, x) in a projected CRS.
- FSC values are 0..100 (%) and NoData uses the variable _FillValue (e.g., 255).
- Dates are parsed from the filename using a regex (default: YYYYMMDD or YYYY_MM_DD).

Example usage
    python nc_to_tif_batch.py

Author: Franz Wagner
Date: 2025-12-18
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Union

import netCDF4
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

# =========================
# ===== CONFIG (GLOBAL) ===
# =========================

# Input can be a directory (recommended) or a single NetCDF file.
INPUT_PATH: str = r"F:\\fram3s\\eurac\\SCF_v0"

# Output directory. If None, writes next to each input file.
OUTPUT_DIR: Optional[str] = r"F:\\fram3s\\eurac\\SCF_v0_tif"

# Recurse into subdirectories when INPUT_PATH is a directory.
RECURSIVE: bool = False

# Overwrite existing GeoTIFFs if present.
OVERWRITE: bool = False

# Variables to export from the NetCDF. Only variables listed here are processed.
# Typical SnowFLAKES variables: ["fsc", "uncertainty"].
VARIABLES_TO_WRITE: Sequence[str] = ("fsc", "uncertainty")

# ---- Date handling ----
# How to derive the acquisition date for filename tagging.
#   "filename" : parse from filename with DATE_REGEX (default)
#   "timevar"  : read the first entry from the NetCDF variable TIME_VAR_NAME
#   "static"   : always use DATE_STATIC
DATE_MODE: str = "filename"

# Regex applied to the filename (without path). The first match is used.
# Default matches YYYYMMDD or YYYY_MM_DD anywhere in the name.
DATE_REGEX: str = r"(\d{4})[_-]?(\d{2})[_-]?(\d{2})"

# If DATE_MODE == "timevar", this variable is read and converted from its units.
TIME_VAR_NAME: str = "time"

# If DATE_MODE == "static", this date is used.
DATE_STATIC: str = "2016-10-06"

# Output filename template. Placeholders: {date}, {var}, {stem}
# - {date} is formatted as YYYY_MM_DD
# - {stem} is the input filename without extension
OUTPUT_TEMPLATE: str = "{stem}_{date}_{var}.tif"

# Force output NoData value (set to None to keep the source _FillValue)
NODATA_OVERRIDE: Optional[float] = None

# Logging level: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL: str = "INFO"


# =================
# LOGGING SETUP
# =================
def configure_logging(level: str = LOG_LEVEL) -> None:
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)
    logging.info("Logging initialized (level=%s)", level.upper())


# =================
# HELPER FUNCTIONS
# =================
def discover_input_files(root: Path, recursive: bool) -> List[Path]:
    """Collect NetCDF files from a file or directory."""
    if root.is_file() and root.suffix.lower() == ".nc":
        return [root]
    if not root.is_dir():
        logging.error("INPUT_PATH is neither a .nc file nor a directory: %s", root)
        return []
    files = sorted(root.rglob("*.nc")) if recursive else sorted(root.glob("*.nc"))
    logging.info("Found %d NetCDF file(s) in %s (recursive=%s).", len(files), root, recursive)
    return files


def parse_date_from_filename(name: str, regex: str) -> Optional[datetime]:
    """Parse YYYY/MM/DD from filename using regex groups 1-3."""
    m = re.search(regex, name)
    if not m:
        return None
    try:
        y, mo, d = (int(m.group(i)) for i in range(1, 4))
        return datetime(year=y, month=mo, day=d)
    except Exception as exc:
        logging.warning("Failed parsing date from filename %s: %s", name, exc)
        return None


def parse_date_from_timevar(nc_path: Path, var_name: str) -> Optional[datetime]:
    """Read first time value from the NetCDF variable var_name."""
    try:
        with netCDF4.Dataset(nc_path, "r") as ds:
            if var_name not in ds.variables:
                logging.error("time var '%s' not found in %s", var_name, nc_path)
                return None
            tvar = ds.variables[var_name]
            units = getattr(tvar, "units", None)
            calendar = getattr(tvar, "calendar", "standard")
            vals = tvar[:]
            if vals.size == 0:
                logging.error("time var '%s' is empty in %s", var_name, nc_path)
                return None
            dt = netCDF4.num2date(vals.flat[0], units=units, calendar=calendar)
            return dt if isinstance(dt, datetime) else dt.to_datetime()
    except Exception as exc:
        logging.error("Failed reading time var '%s' in %s: %s", var_name, nc_path, exc)
        return None


def resolve_date_tag(nc_path: Path) -> str:
    """Return a YYYY_MM_DD string based on DATE_MODE."""
    if DATE_MODE.lower() == "filename":
        dt = parse_date_from_filename(nc_path.name, DATE_REGEX)
    elif DATE_MODE.lower() == "timevar":
        dt = parse_date_from_timevar(nc_path, TIME_VAR_NAME)
    elif DATE_MODE.lower() == "static":
        dt = datetime.fromisoformat(DATE_STATIC)
    else:
        logging.error("Unsupported DATE_MODE: %s", DATE_MODE)
        dt = None

    if not dt:
        logging.warning("Using fallback date '9999_99_99' for %s", nc_path.name)
        return "9999_99_99"
    return dt.strftime("%Y_%m_%d")


def ensure_output_dir(path: Path) -> None:
    """Ensure output directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def output_path_for(nc_path: Path, out_dir: Optional[Path], var: str, date_tag: str) -> Path:
    """Build destination path for a variable."""
    base_dir = out_dir if out_dir is not None else nc_path.parent
    ensure_output_dir(base_dir)
    stem = nc_path.stem
    fname = OUTPUT_TEMPLATE.format(date=date_tag, var=var, stem=stem)
    return base_dir / fname


def _parse_geotransform(values):
    """Return Affine from 6- or 9-value GeoTransform-like sequences."""
    if len(values) == 6:
        return Affine.from_gdal(*values)
    if len(values) == 9:
        a, b, c, d, e, f, *_ = values
        return Affine(a, b, c, d, e, f)
    return None


def _extract_transform(ds: netCDF4.Dataset, var_name: str) -> Optional[Affine]:
    """Try variable-level then global geotransform attributes."""
    def _split(val: Union[str, Sequence, np.ndarray]):
        if isinstance(val, np.ndarray):
            val = val.flatten().tolist()
        if isinstance(val, (list, tuple)):
            try:
                return [float(v) for v in val]
            except Exception:
                return None
        if isinstance(val, bytes):
            val = val.decode("utf-8", errors="ignore")
        if isinstance(val, str):
            parts = [p for p in re.split(r"[ ,]+", val.strip()) if p]
            try:
                return [float(p) for p in parts]
            except Exception:
                return None
        return None

    var = ds.variables.get(var_name)
    candidates = []
    if var is not None:
        for key in ("GeoTransform", "geotransform", "transform"):
            if key in var.ncattrs():
                parsed = _split(getattr(var, key))
                if parsed:
                    candidates.append(parsed)
    for key in ("GeoTransform", "geotransform", "transform"):
        if key in ds.ncattrs():
            parsed = _split(getattr(ds, key))
            if parsed:
                candidates.append(parsed)

    for vals in candidates:
        aff = _parse_geotransform(vals)
        if aff:
            return aff
    return None


def _extract_crs(ds: netCDF4.Dataset, var_name: str) -> Optional[CRS]:
    """Try variable-level CRS WKT or CF grid_mapping attr."""
    def _as_str(val):
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="ignore")
        if isinstance(val, np.ndarray):
            val = val.flatten()
            if val.size == 1:
                return _as_str(val[0])
            return None
        return str(val)

    var = ds.variables.get(var_name)
    if var is not None:
        for key in ("crs", "spatial_ref", "crs_wkt", "grid_mapping"):
            if key in var.ncattrs():
                try:
                    return CRS.from_string(_as_str(getattr(var, key)))
                except Exception:
                    pass
    for key in ("crs", "spatial_ref", "crs_wkt"):
        if key in ds.ncattrs():
            try:
                return CRS.from_string(_as_str(getattr(ds, key)))
            except Exception:
                pass
    if var is not None and "grid_mapping" in var.ncattrs():
        gm_name = getattr(var, "grid_mapping")
        gm = ds.variables.get(gm_name)
        if gm is not None:
            for key in ("crs_wkt", "spatial_ref", "crs"):
                if key in gm.ncattrs():
                    try:
                        return CRS.from_string(_as_str(getattr(gm, key)))
                    except Exception:
                        pass
    return None


def write_variable_as_tif(nc_path: Path, var: str, dst: Path, overwrite: bool) -> bool:
    """Write one NetCDF variable as GeoTIFF using netCDF4 + rasterio writer."""
    try:
        with netCDF4.Dataset(nc_path, "r") as ds:
            if var not in ds.variables:
                logging.error("Variable '%s' missing in %s", var, nc_path.name)
                return False
            v = ds.variables[var]
            if v.ndim != 2:
                logging.error("Variable '%s' is not 2-D (dims=%s) in %s", var, v.dimensions, nc_path.name)
                return False
            data = np.array(v[:], copy=True)
            nodata_source = getattr(v, "_FillValue", None)
            nodata = NODATA_OVERRIDE if NODATA_OVERRIDE is not None else nodata_source

            # Build mask from NaNs/Infs and from source nodata, then assign chosen nodata
            mask = ~np.isfinite(data)
            if nodata_source is not None:
                mask |= data == nodata_source
            if nodata is not None:
                data = data.astype(float if isinstance(nodata, float) else data.dtype, copy=True)
                data[mask] = nodata
            transform = _extract_transform(ds, var)
            crs = _extract_crs(ds, var)
            height, width = data.shape
            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": data.dtype,
                "nodata": nodata,
                "compress": "lzw",
            }
            if transform:
                profile["transform"] = transform
            if crs:
                profile["crs"] = crs

        if dst.exists():
            if not overwrite:
                logging.info("Skip existing (overwrite=False): %s", dst)
                return True
            logging.warning("Overwriting: %s", dst)
            dst.unlink()

        with rasterio.open(dst, "w", **profile) as dst_ds:
            dst_ds.write(data, 1)

        logging.info("Wrote %s -> %s", var, dst)
        return True
    except Exception as exc:
        logging.error("Failed writing %s from %s: %s", var, nc_path.name, exc)
        return False


def process_file(nc_path: Path, out_dir: Optional[Path]) -> None:
    """Process one NetCDF file for configured variables."""
    date_tag = resolve_date_tag(nc_path)
    for var in VARIABLES_TO_WRITE:
        dst = output_path_for(nc_path, out_dir, var, date_tag)
        write_variable_as_tif(nc_path, var, dst, OVERWRITE)


def main() -> None:
    configure_logging()
    root = Path(INPUT_PATH)
    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else None
    files = discover_input_files(root, RECURSIVE)
    if not files:
        logging.error("No NetCDF files to process. Exiting.")
        return
    for nc in files:
        logging.info("Processing %s", nc)
        process_file(nc, out_dir)


if __name__ == "__main__":
    main()
