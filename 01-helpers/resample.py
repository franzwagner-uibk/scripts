#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch resample the testsite grids to user-defined resolutions aligned to an
existing reference raster. Edit the CONFIG section and run:

    python resampleTo500m.py

Supports multiple target resolutions per run (e.g. 10/50/100/250/500/1000) and
either a fixed resampling method (bilinear/nearest) or automatic selection
based on filename rules.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

# =========================
# ======== CONFIG =========
# =========================
# Folder with the source .asc rasters
INPUT_DIR = Path(
    r"F:\fram3s\01-data\06-dem\rofental"
)
# Where to write the resampled rasters (can be the same as INPUT_DIR)
OUTPUT_DIR = Path(
    r"F:\fram3s\01-data\06-dem\rofental"
)
# Reference raster that defines the extent/origin for alignment
# Use the clipped rofental subset as the alignment grid.
REFERENCE_RASTER = Path(
    r"F:\fram3s\01-data\06-dem\euregio\dem_euregio_50.asc"
)
# If the reference raster lacks CRS, set a fallback (e.g., "EPSG:32632"); otherwise leave None
# Set to EPSG:32632 (UTM 32N) for testsite grids.
REF_CRS_OVERRIDE: str | None = "EPSG:32632"

# Target cell sizes in meters (can be multiple)
TARGET_RESOLUTIONS = [100, 250, 500]

# Resampling mode:
#   "fixed" -> use RESAMPLING_METHOD for all rasters
#   "auto"  -> pick from RESAMPLING_RULES by filename substring
RESAMPLING_MODE: str = "auto"

# If RESAMPLING_MODE == "fixed", choose one of: "bilinear" or "nearest"
RESAMPLING_METHOD: str = "bilinear"

# If RESAMPLING_MODE == "auto", substring -> resampling rule (first match wins)
RESAMPLING_RULES: Tuple[Tuple[str, Resampling], ...] = (
    ("lc", Resampling.nearest),
    ("landcover", Resampling.nearest),
    ("dem", Resampling.bilinear),
    ("srf", Resampling.bilinear),
    ("svf", Resampling.bilinear),
)

# Overwrite existing outputs?
OVERWRITE: bool = False

# Logging
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


# =================
# HELPER FUNCTIONS
# =================
def discover_inputs(directory: Path) -> Iterable[Path]:
    """Return all .asc rasters in the directory (non-recursive)."""
    return sorted(p for p in directory.glob("*.asc") if p.is_file())


def derive_target_grid(ref_path: Path, target_res: float) -> Tuple[Affine, int, int]:
    """
    Build the target transform/shape from a reference raster, snapping to its
    upper-left corner.
    """
    with rasterio.open(ref_path) as ref:
        left, bottom, right, top = ref.bounds

    width = math.ceil((right - left) / target_res)
    height = math.ceil((top - bottom) / target_res)
    transform = Affine(target_res, 0, left, 0, -target_res, top)
    return transform, width, height


def choose_resampling(path: Path) -> Resampling | None:
    """Pick resampling method from RESAMPLING_RULES based on filename."""
    name = path.name.lower()
    for key, method in RESAMPLING_RULES:
        if key in name:
            return method
    return None


def parse_fixed_resampling(method: str) -> Resampling:
    """Convert string to rasterio.Resampling."""
    m = method.strip().lower()
    if m in ("bilinear", "linear"):
        return Resampling.bilinear
    if m in ("nearest", "nearest_neighbor", "nn"):
        return Resampling.nearest
    raise ValueError(f"Unsupported RESAMPLING_METHOD: {method}")


def output_path_for(src: Path, out_dir: Path, target_res: float) -> Path:
    """
    Build output filename by swapping the resolution tag (e.g. 100 -> 500).
    If no numeric tag is found, append _{target_res}.
    """
    target_tag = f"_{int(target_res)}"
    stem = src.stem

    # Replace the first occurrence of a _<number>[m] token (e.g., _100 or _100m)
    def _repl(match: re.Match[str]) -> str:
        prefix = match.group(0)
        suffix = "_" if prefix.endswith("_") else ""
        return f"{target_tag}{suffix}"

    new_stem, count = re.subn(r"_(\d+)(?:m)?(?=_|$)", _repl, stem, count=1)
    if count == 0:
        new_stem = f"{stem}{target_tag}"

    return out_dir / f"{new_stem}.asc"


def clean_profile(profile: dict, dst_transform: rasterio.Affine, width: int, height: int, dtype: str) -> dict:
    """Prepare a rasterio profile for writing AAIGrid."""
    prof = profile.copy()
    for key in ("driver", "tiled", "compress", "interleave", "blockxsize", "blockysize"):
        prof.pop(key, None)
    prof.update(
        driver="AAIGrid",
        transform=dst_transform,
        width=width,
        height=height,
        dtype=dtype,
        count=1,
    )
    return prof


def load_reference_crs(ref_path: Path) -> CRS | None:
    """Return CRS from the reference raster or REF_CRS_OVERRIDE."""
    ref_crs: CRS | None
    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
    if ref_crs is None and REF_CRS_OVERRIDE:
        ref_crs = CRS.from_user_input(REF_CRS_OVERRIDE)
    return ref_crs


def resample_raster(
    src_path: Path,
    dst_path: Path,
    dst_transform: rasterio.Affine,
    dst_width: int,
    dst_height: int,
    resampling: Resampling,
    ref_crs: CRS | None,
) -> None:
    """Resample one raster and write to AAIGrid."""
    with rasterio.open(src_path) as src:
        src_array = src.read(1)
        src_nodata = src.nodata
        dst_nodata = src_nodata
        dst_dtype = src.dtypes[0] if resampling == Resampling.nearest else "float32"

        src_crs = src.crs or ref_crs
        if src_crs is None:
            raise ValueError(f"No CRS found for {src_path} and no REF_CRS_OVERRIDE provided.")

        if src_nodata is not None:
            fill_value = src_nodata
        elif np.issubdtype(np.dtype(dst_dtype), np.integer):
            fill_value = 0
        else:
            fill_value = np.nan

        dst_array = np.full((dst_height, dst_width), fill_value=fill_value, dtype=dst_dtype)

        reproject(
            source=src_array,
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src_crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=src_crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )

        # Round categorical rasters after nearest-neighbor to keep integer classes
        if resampling == Resampling.nearest and np.issubdtype(np.dtype(dst_dtype), np.integer):
            dst_array = np.rint(dst_array).astype(dst_dtype)

        profile = clean_profile(src.profile, dst_transform, dst_width, dst_height, dst_dtype)
        profile["nodata"] = dst_nodata

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(dst_array, 1)


# =================
# ====== MAIN =====
# =================
def main() -> None:
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    logger = logging.getLogger("resampleTo500m")

    if RESAMPLING_MODE not in {"auto", "fixed"}:
        raise ValueError("RESAMPLING_MODE must be 'auto' or 'fixed'.")
    fixed_method = parse_fixed_resampling(RESAMPLING_METHOD) if RESAMPLING_MODE == "fixed" else None

    if not REFERENCE_RASTER.is_file():
        raise FileNotFoundError(f"Reference raster not found: {REFERENCE_RASTER}")
    ref_crs = load_reference_crs(REFERENCE_RASTER)
    if ref_crs is None:
        raise ValueError("Reference raster has no CRS and REF_CRS_OVERRIDE is not set.")

    inputs = list(discover_inputs(INPUT_DIR))
    if not inputs:
        logger.warning("No .asc rasters found in %s", INPUT_DIR)
        return

    for target_res in TARGET_RESOLUTIONS:
        dst_transform, dst_w, dst_h = derive_target_grid(REFERENCE_RASTER, target_res)
        logger.info("Target grid: res=%.2f m, width=%d, height=%d", target_res, dst_w, dst_h)

        for src in inputs:
            method = fixed_method
            if method is None:
                method = choose_resampling(src)
                if method is None:
                    logger.info("Skipping (no rule): %s", src.name)
                    continue

            dst = output_path_for(src, OUTPUT_DIR, target_res)
            if dst.exists() and not OVERWRITE:
                logger.info("Skipping existing (overwrite=False): %s", dst.name)
                continue

            logger.info("Resampling %s -> %s (%s)", src.name, dst.name, method.name)
            resample_raster(src, dst, dst_transform, dst_w, dst_h, method, ref_crs)

    logger.info("Done.")


if __name__ == "__main__":
    main()
