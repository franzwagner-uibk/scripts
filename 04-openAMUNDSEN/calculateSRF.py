#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute openness-based Snow Redistribution Factor (SRF) for openAMUNDSEN.

This version takes a single DEM raster file (e.g. TIF/ASC) as input
instead of searching in a directory. The SRF is written as an ASCII
grid (AAIGrid) in the same folder by default.

Run from PowerShell with live logs (conda env "openamundsen"):
    & "$env:USERPROFILE\\miniforge3\\Scripts\\conda.exe" run --live-stream -n openamundsen python -u "C:/Users/franz/Nextcloud/PhD/05-Script/04-openAMUNDSEN/calculateSRF.py"

Implements colleague's method (Hanzer et al., 2016) with elevation dependency:

    no5000 = openness(dem, resolution, 5000 m, negative=True)
    no100  = openness(dem, resolution, 100  m, negative=True)

    srf    = ( clamp(3*(no5000 - 1),   0.1, 1.6)
             + clamp(3*(no100  - 1.2), 0.1, 1.6) ) / 2

    elev_factor = dem / max(dem)
    srf[>=1] = 1 + (srf-1) * elev_factor
    srf[< 1] = 1 - (1 - srf) * elev_factor

Optionally (disabled by default) normalize SRF so mean = 1.0.

Author: Franz Wagner
Date: 2025-10-02
"""

# =========================
# USER INPUT VARIABLES
# =========================
# Set the path to your input DEM (ASCII/TIF) here.
DEM_PATH: str = r"F:\fram3s\01-data\06-dem\dem_euregio_1000.asc"

# =========================
# IMPORTS
# =========================
import logging
import time
from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.transform import Affine

# openAMUNDSEN
import openamundsen as oa  # type: ignore

# =========================
# USER / GLOBAL VARIABLES
# =========================
# Openness radii (meters) per colleague´s code: 5000 m (broad scale) and ~100 m (local scale)
OPENNESS_RADIUS_LARGE_M: int = 5000
OPENNESS_RADIUS_SMALL_M: int = 100   # was "no50" in snippet but used 100 m radius there

# Clamp bounds for SRF terms
SRF_MIN: float = 0.1
SRF_MAX: float = 1.6

# Optional: preserve total snowfall by normalizing SRF mean to 1.0
NORMALIZE_MEAN_TO_ONE: bool = True

# Optional: show a step-based progress bar (requires tqdm; falls back to logging if not installed)
USE_PROGRESS_BAR: bool = True

# Logging
LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"
# Optional: also write logs to a file next to this script
LOG_TO_FILE: bool = True
LOG_FILE_PATH: Path = Path(__file__).with_suffix(".log")


# =================
# LOGGING SETUP
# =================
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(LOG_LEVEL)
handler.setFormatter(logging.Formatter(LOG_FORMAT))
handlers = [handler]
if LOG_TO_FILE:
    try:
        LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handlers.append(file_handler)
    except Exception as exc:  # pragma: no cover
        # Fall back to console-only logging if file handler fails
        print(f"WARNING: Could not set up file logging ({exc}); continuing with console logging only.")

logging.basicConfig(
    handlers=handlers,
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    force=True,  # ensure our config is applied even if another lib configured logging earlier
)
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


# =================
# FUNCTIONS
# =================
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


class StepProgress:
    def __init__(self, total_steps: int, enabled: bool) -> None:
        if enabled and tqdm is None:
            logger.info("Progress bar requested, but 'tqdm' is not installed; continuing without progress bar.")
        self._enabled = bool(enabled) and tqdm is not None
        self._bar = tqdm(total=total_steps, unit="step") if self._enabled else None

    def set(self, message: str) -> None:
        if self._bar is not None:
            self._bar.set_description(str(message))

    def step(self, message: str) -> None:
        self.set(message)
        if self._bar is not None:
            self._bar.update(1)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def read_dem(path: str) -> tuple[np.ndarray, dict]:
    """Read DEM as masked float32 array and return (arr, profile)."""
    with rasterio.open(path) as ds:
        arr = ds.read(1, masked=True).astype(np.float32)
        profile = ds.profile.copy()
    arr = np.ma.masked_invalid(arr)
    logger.info(
        "DEM info: shape=%s, crs=%s, nodata=%s",
        arr.shape,
        profile.get("crs"),
        profile.get("nodata"),
    )
    return arr, profile


def log_array_stats(name: str, arr: np.ndarray, with_percentiles: bool = False) -> None:
    """Log basic stats for a masked/regular array."""
    data = np.ma.masked_invalid(arr)
    if data.count() == 0:
        logger.warning("%s stats: no valid cells", name)
        return
    vals = data.compressed()
    if with_percentiles:
        logger.info(
            "%s stats: min=%.3f, p5=%.3f, mean=%.3f, p95=%.3f, max=%.3f",
            name,
            float(np.min(vals)),
            float(np.percentile(vals, 5)),
            float(np.mean(vals)),
            float(np.percentile(vals, 95)),
            float(np.max(vals)),
        )
    else:
        logger.info(
            "%s stats: min=%.3f, mean=%.3f, max=%.3f",
            name,
            float(np.min(vals)),
            float(np.mean(vals)),
            float(np.max(vals)),
        )


def infer_resolution(transform: Affine) -> float:
    """Infer resolution from affine transform; warn on non-square pixels."""
    res_x = float(abs(transform.a))
    res_y = float(abs(transform.e))
    if not np.isclose(res_x, res_y):
        logger.warning("Non-square pixels: res_x=%.3f, res_y=%.3f; using res_x", res_x, res_y)
    return res_x


def compute_openness_negative(dem: np.ndarray, resolution_m: int, radius_m: int) -> np.ndarray:
    """Compute NEGATIVE openness via openAMUNDSEN terrain.openness."""
    logger.info("Computing negative openness: radius=%d m ...", radius_m)
    t0 = time.perf_counter()
    # oa.terrain.openness expects raw ndarray (not masked). Use NaN for invalid cells.
    dem_ma = np.ma.masked_invalid(dem)
    dem_filled = dem_ma.filled(np.nan).astype(float)
    # oa.terrain.openness handles negative=True (negative openness)
    no = oa.terrain.openness(dem_filled, resolution_m, radius_m, negative=True)
    # Re-apply mask where DEM invalid
    no = np.ma.masked_array(no.astype(np.float32), mask=np.ma.getmaskarray(dem_ma))
    logger.info("Openness done: radius=%d m (%.2f s)", radius_m, time.perf_counter() - t0)
    return no


def clamp(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clamp array to [lo, hi] preserving mask."""
    out = np.ma.clip(arr, lo, hi).astype(np.float32)
    return np.ma.masked_array(out, mask=np.ma.getmaskarray(arr))


def apply_elevation_dependency(srf: np.ndarray, dem: np.ndarray) -> np.ndarray:
    """Scale deviations from 1 by normalized elevation factor (0..1)."""
    srf_ma = np.ma.masked_invalid(srf)
    dem_ma = np.ma.masked_invalid(dem)
    if dem_ma.count() == 0:
        logger.warning("DEM has no valid elevation values; skipping elevation dependency.")
        return srf_ma
    dem_max = float(dem_ma.max())
    if not np.isfinite(dem_max) or dem_max <= 0:
        logger.warning("DEM max elevation invalid; skipping elevation dependency.")
        return srf_ma
    elev_factor = (dem_ma / dem_max).astype(np.float32)

    srf_adj = np.ma.where(
        srf_ma >= 1.0,
        1.0 + (srf_ma - 1.0) * elev_factor,
        1.0 - (1.0 - srf_ma) * elev_factor,
    )

    combined_mask = np.ma.getmaskarray(srf_ma) | np.ma.getmaskarray(dem_ma)
    return np.ma.masked_array(srf_adj, mask=combined_mask)


def normalize_mean_to_one(arr: np.ndarray) -> np.ndarray:
    """Scale array so mean over valid cells = 1.0."""
    data = np.ma.masked_invalid(arr)
    if data.count() == 0:
        logger.warning("No valid cells for normalization; skipping.")
        return data
    mean_val = float(data.mean())
    if not np.isfinite(mean_val) or mean_val == 0.0:
        logger.warning("Invalid mean for normalization (%.4f); skipping.", mean_val)
        return data
    logger.info("Normalizing SRF mean to 1.0 (mean before=%.4f)", mean_val)
    return (data / mean_val).astype(np.float32)


def write_aaigrid_ascii(path: Path, data: np.ndarray, transform: Affine, nodata: float = -9999.0) -> None:
    """Write Arc/Info ASCII grid aligned to DEM transform."""
    # Build minimal profile for AAIGrid
    prof = {
        "driver": "AAIGrid",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": None,            # AAIGrid won't store CRS; OA uses alignment
        "transform": transform,
        "nodata": np.float32(nodata),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(path), "w", **prof) as dst:
        out = np.ma.masked_invalid(data)
        out_filled = out.filled(nodata).astype(np.float32)
        dst.write(out_filled, 1)
    logger.info("Wrote SRF ASCII: %s", path)


def compute_srf(dem: np.ndarray, resolution_m: float, progress: "StepProgress | None" = None) -> np.ndarray:
    """Compute SRF from DEM at given resolution (meters)."""
    res_int = int(round(resolution_m))

    if progress is not None:
        progress.set(f"Openness {OPENNESS_RADIUS_LARGE_M} m")
    no_large = compute_openness_negative(dem, res_int, OPENNESS_RADIUS_LARGE_M)
    if progress is not None:
        progress.step(f"Openness {OPENNESS_RADIUS_LARGE_M} m")

    if progress is not None:
        progress.set(f"Openness {OPENNESS_RADIUS_SMALL_M} m")
    no_small = compute_openness_negative(dem, res_int, OPENNESS_RADIUS_SMALL_M)
    if progress is not None:
        progress.step(f"Openness {OPENNESS_RADIUS_SMALL_M} m")

    if progress is not None:
        progress.set("Compute SRF")

    term_large = clamp(3.0 * (no_large - 1.0), SRF_MIN, SRF_MAX)
    term_small = clamp(3.0 * (no_small - 1.2), SRF_MIN, SRF_MAX)
    srf = ((term_large + term_small) / 2.0).astype(np.float32)
    srf = np.ma.masked_array(srf, mask=np.ma.getmaskarray(dem))

    srf = apply_elevation_dependency(srf, dem)
    if NORMALIZE_MEAN_TO_ONE:
        srf = normalize_mean_to_one(srf)

    if progress is not None:
        progress.step("Compute SRF")

    return srf


def main() -> None:
    logger.info("=== SRF computation (openness + elevation dependency) started ===")
    if LOG_TO_FILE:
        logger.info("Log file: %s", LOG_FILE_PATH.resolve())
    progress = StepProgress(total_steps=6, enabled=USE_PROGRESS_BAR)
    try:
        # 1) Load DEM from user-defined path at top of script
        dem_path = DEM_PATH
        if not dem_path:
            raise ValueError("DEM_PATH is empty. Please set DEM_PATH at the top of the script.")

        dem_path_obj = Path(dem_path)
        if not dem_path_obj.is_file():
            raise FileNotFoundError(f"DEM not found: {dem_path_obj}")

        logger.info("Input DEM: %s", dem_path_obj)
        logger.info(
            "Parameters: large_radius=%d m, small_radius=%d m, clamp=[%.2f, %.2f], normalize=%s",
            OPENNESS_RADIUS_LARGE_M,
            OPENNESS_RADIUS_SMALL_M,
            SRF_MIN,
            SRF_MAX,
            NORMALIZE_MEAN_TO_ONE,
        )

        progress.set("Read DEM")
        dem, dem_prof = read_dem(str(dem_path_obj))
        log_array_stats("DEM", dem)
        progress.step("Read DEM")

        # Determine grid resolution from DEM transform
        transform: Affine = dem_prof["transform"]
        # Assume square pixels; use absolute x-scale as resolution in meters.
        resolution = infer_resolution(transform)
        logger.info("Using resolution inferred from DEM: %.3f m", resolution)

        # Determine output path (same directory, prefixed with 'srf_')
        stem = dem_path_obj.stem
        # Prefer replacing leading "dem_" with "srf_" to keep concise names (e.g., dem_euregio_50 -> srf_euregio_50)
        if stem.startswith("dem_"):
            out_stem = f"srf_{stem.removeprefix('dem_')}"
        else:
            out_stem = f"srf_{stem}"
        output_srf_asc = dem_path_obj.with_name(f"{out_stem}.asc")

        # 2) SRF computation
        srf = compute_srf(dem, resolution, progress=progress)
        log_array_stats("SRF", srf, with_percentiles=True)

        # 3) Write output
        progress.set("Write output")
        write_aaigrid_ascii(output_srf_asc, srf, dem_prof["transform"], nodata=-9999.0)
        progress.step("Write output")

        logger.info("Output ready: %s", output_srf_asc)
        logger.info("=== SRF computation finished ===")
        progress.step("Done")
    finally:
        progress.close()


if __name__ == "__main__":
    main()
