#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script:    02-createSubregionOAInputData.py
Author:    Franz Wagner (with ChatGPT)
Date:      2025-10-02

Description:
    Prepare subregional meteorological inputs for openAMUNDSEN.

    Workflow:
      1) Read an ESRI ASCII grid (*.asc) representing the Region of Interest (ROI).
         Use header (ncols, nrows, xllcorner/xllcenter, yllcorner/yllcenter, cellsize)
         to compute the ROI bounding box.
      2) Load the global stations overview CSV (must include station id + coordinates),
         filter stations whose (x, y) lie inside the ROI bounding box.
      3) Recursively index the meteo directory for per-station CSVs. Station ID is taken
         from the filename stem (e.g., '00051014-1.csv' -> '00051014-1'). The file
         'stations.csv' is ignored.
      4) Copy all matched station CSVs into the output directory.
      5) Write a filtered 'stations.csv' into the output directory with the selected rows.

Assumptions:
    * ROI ASCII and station coordinates share the same projected CRS.
    * Station CSV filenames are '<station_id>.csv'. IDs may contain digits, letters,
      dashes, and leading zeros (e.g., 'E00106', '00051014-1').
    * The overview file in the meteo folder is named 'stations.csv' and must be ignored.
"""

# =========================
# ====== CONFIGURE ME =====
# =========================

# --- Paths (SET THESE!) ---
ROI_ASC_PATH: str = r"09-openAMUNDSEN\testsite\grids\testsite_100.asc"
STATIONS_CSV_PATH: str = r"02-Daten\euregio\meteo\stations\stations.csv"
METEO_INPUT_DIR: str = r"02-Daten\euregio\meteo\stations"
OUTPUT_DIR: str = r"09-openAMUNDSEN\testsite\meteo"

# --- Behavior switches ---
RECURSIVE_SEARCH: bool = True          # search meteo dir recursively for station CSVs
OVERWRITE_OUTPUT: bool = True          # overwrite destination files if they exist
WRITE_FULL_LOGFILE: bool = False       # if True, also write a timestamped log file

# --- Output filenames ---
FILTERED_STATIONS_FILENAME: str = "stations.csv"   # written into OUTPUT_DIR

# --- Stations CSV required columns ---
STATION_ID_COL: str = "id"
STATION_X_COL: str = "x"
STATION_Y_COL: str = "y"

# =========================
# ====== IMPORTS ETC. =====
# =========================

import os
import re
import sys
import shutil
import logging
from typing import Dict, List, Tuple

from datetime import datetime
import pandas as pd


# =========================
# ====== LOGGING SETUP ====
# =========================

def setup_logging() -> None:
    """
    Configure logging (console + optional file) as per user preference.
    """
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)

    if WRITE_FULL_LOGFILE:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        logfile = f"extract_subregion_meteo_{ts}.log"
        fh = logging.FileHandler(logfile, mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(log_fmt))
        logging.getLogger().addHandler(fh)
        logging.info("Log file: %s", os.path.abspath(logfile))


# ==========================================
# ====== CORE FUNCTIONALITY (MODULAR) ======
# ==========================================

def _parse_decimal(text: str) -> float:
    """
    Parse a number with optional comma decimal (e.g., '680794,892' -> 680794.892).
    """
    return float(text.strip().replace(",", "."))


def read_esri_ascii_bbox(asc_path: str) -> Tuple[float, float, float, float]:
    """
    Read ESRI ASCII header and compute bounding box: (xmin, ymin, xmax, ymax).
    Supports xllcorner/yllcorner and xllcenter/yllcenter.
    """
    logging.info("Reading ROI ASCII header: %s", asc_path)
    if not os.path.isfile(asc_path):
        raise FileNotFoundError(f"ASCII grid not found: {asc_path}")

    header: Dict[str, str] = {}

    with open(asc_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(20):  # read a handful of header lines
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            m = re.match(r"^\s*([A-Za-z_]+)\s+([^\s]+)", line)
            if m:
                key = m.group(1).lower()
                val = m.group(2)
                header[key] = val
            else:
                f.seek(pos)  # grid rows begin
                break

    # Validate essentials
    req = {"ncols", "nrows", "cellsize"}
    missing = req - set(header)
    if missing:
        raise ValueError(f"Missing ASCII header keys: {sorted(missing)}")

    ncols = int(float(header["ncols"].replace(",", ".")))
    nrows = int(float(header["nrows"].replace(",", ".")))
    cellsize = _parse_decimal(header["cellsize"])

    have_corner = "xllcorner" in header and "yllcorner" in header
    have_center = "xllcenter" in header and "yllcenter" in header
    if not (have_corner or have_center):
        raise ValueError("Header must contain either (xllcorner,yllcorner) or (xllcenter,yllcenter).")

    if have_corner:
        xll = _parse_decimal(header["xllcorner"])
        yll = _parse_decimal(header["yllcorner"])
        xmin = xll
        ymin = yll
        xmax = xll + ncols * cellsize
        ymax = yll + nrows * cellsize
    else:
        xlc = _parse_decimal(header["xllcenter"])
        ylc = _parse_decimal(header["yllcenter"])
        xmin = xlc - 0.5 * cellsize
        ymin = ylc - 0.5 * cellsize
        xmax = xmin + ncols * cellsize
        ymax = ymin + nrows * cellsize

    logging.info("ROI bbox computed: xmin=%.3f, ymin=%.3f, xmax=%.3f, ymax=%.3f",
                 xmin, ymin, xmax, ymax)
    return xmin, ymin, xmax, ymax


def load_and_filter_stations(stations_csv: str,
                             bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Load stations CSV and filter to those within bbox.
    Expects at least columns: id, x, y (names configurable at top).
    """
    xmin, ymin, xmax, ymax = bbox
    logging.info("Loading stations from: %s", stations_csv)
    if not os.path.isfile(stations_csv):
        raise FileNotFoundError(f"Stations CSV not found: {stations_csv}")

    df = pd.read_csv(stations_csv)

    # Check required columns
    required = {STATION_ID_COL, STATION_X_COL, STATION_Y_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Stations CSV missing columns: {sorted(missing)}")

    # Normalize types
    df[STATION_ID_COL] = df[STATION_ID_COL].astype(str).str.strip()
    df[STATION_X_COL] = pd.to_numeric(df[STATION_X_COL], errors="coerce")
    df[STATION_Y_COL] = pd.to_numeric(df[STATION_Y_COL], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[STATION_X_COL, STATION_Y_COL])
    dropped = before - len(df)
    if dropped > 0:
        logging.warning("Dropped %d stations with invalid coordinates.", dropped)

    mask = (
        (df[STATION_X_COL] >= xmin) &
        (df[STATION_X_COL] <= xmax) &
        (df[STATION_Y_COL] >= ymin) &
        (df[STATION_Y_COL] <= ymax)
    )
    filtered = df.loc[mask].copy()
    logging.info("Stations in ROI: %d (of %d total)", len(filtered), before)
    return filtered


def build_station_file_index(meteo_dir: str,
                             recursive: bool = True) -> Dict[str, str]:
    """
    Create index mapping station_id -> absolute CSV filepath by scanning the meteo dir.

    IMPORTANT:
        * Any '<id>.csv' is indexed, where <id> is the filename stem (alphanumeric,
          dashes, leading zeros allowed).
        * The file named exactly 'stations.csv' is ignored.

    Returns:
        dict like {"00051014-1": "C:/.../00051014-1.csv", "E00106": "C:/.../E00106.csv", ...}
    """
    logging.info("Indexing station files in: %s (recursive=%s)", meteo_dir, recursive)
    if not os.path.isdir(meteo_dir):
        raise NotADirectoryError(f"Meteo directory not found: {meteo_dir}")

    index: Dict[str, str] = {}

    def handle_file(path: str) -> None:
        name = os.path.basename(path)
        stem, ext = os.path.splitext(name)
        if ext.lower() != ".csv":
            return
        if stem.lower() == "stations":  # skip the overview file
            return
        sid = stem  # accept any stem as station id
        full = os.path.abspath(path)
        if sid in index:
            logging.warning("Duplicate station CSV for id %s:\n  - %s\n  - %s",
                            sid, index[sid], full)
            return
        index[sid] = full

    if recursive:
        for root, _, files in os.walk(meteo_dir):
            for fn in files:
                handle_file(os.path.join(root, fn))
    else:
        for fn in os.listdir(meteo_dir):
            handle_file(os.path.join(meteo_dir, fn))

    logging.info("Indexed %d station CSV files.", len(index))
    return index


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def copy_selected_station_files(filtered_stations: pd.DataFrame,
                                file_index: Dict[str, str],
                                output_dir: str,
                                overwrite: bool = True) -> Tuple[int, List[str], List[str]]:
    """
    Copy per-station CSVs for filtered stations to output_dir.

    Returns:
        (copied_count, missing_ids, copied_files)
    """
    ensure_dir(output_dir)
    missing: List[str] = []
    copied_files: List[str] = []

    for _, row in filtered_stations.iterrows():
        sid = str(row[STATION_ID_COL]).strip()
        src = file_index.get(sid)
        if not src or not os.path.isfile(src):
            missing.append(sid)
            logging.warning("No CSV found for station id %s", sid)
            continue

        dst = os.path.join(output_dir, f"{sid}.csv")
        if os.path.exists(dst) and not overwrite:
            logging.info("Skip existing file (overwrite=False): %s", dst)
            continue

        shutil.copy2(src, dst)
        copied_files.append(dst)
        logging.info("Copied: %s -> %s", src, dst)

    return len(copied_files), missing, copied_files


def write_filtered_stations_csv(filtered_stations: pd.DataFrame,
                                output_dir: str,
                                filename: str = "stations.csv") -> str:
    """
    Write the filtered stations to output_dir/filename and return its absolute path.
    """
    ensure_dir(output_dir)
    out_path = os.path.abspath(os.path.join(output_dir, filename))
    filtered_stations.to_csv(out_path, index=False)
    logging.info("Wrote filtered stations CSV: %s (rows=%d)",
                 out_path, len(filtered_stations))
    return out_path


# =========================
# ========= MAIN ==========
# =========================

def main() -> None:
    """
    Orchestrate the subregion meteo extraction workflow.
    """
    setup_logging()
    logging.info("=== Subregion meteo extraction started ===")

    try:
        bbox = read_esri_ascii_bbox(ROI_ASC_PATH)
        stations_roi = load_and_filter_stations(STATIONS_CSV_PATH, bbox)
        file_index = build_station_file_index(METEO_INPUT_DIR, recursive=RECURSIVE_SEARCH)

        copied_count, missing_ids, _ = copy_selected_station_files(
            stations_roi, file_index, OUTPUT_DIR, overwrite=OVERWRITE_OUTPUT
        )

        stations_out = write_filtered_stations_csv(
            stations_roi, OUTPUT_DIR, filename=FILTERED_STATIONS_FILENAME
        )

        logging.info("--- Summary ---")
        logging.info("Stations in ROI: %d", len(stations_roi))
        logging.info("Files copied:    %d", copied_count)
        logging.info("Missing files:   %d", len(missing_ids))
        if missing_ids:
            logging.info("Missing station ids: %s", ", ".join(sorted(set(missing_ids))))
        logging.info("Filtered stations CSV: %s", stations_out)

    except Exception as exc:
        logging.exception("Processing failed: %s", exc)
        sys.exit(1)

    logging.info("=== Done. ===")


if __name__ == "__main__":
    main()
