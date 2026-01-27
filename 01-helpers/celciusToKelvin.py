#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: check_and_convert_temperature_to_kelvin.py
Author: Franz Wagner
Date: 2025-10-09 (Europe/Vienna)

Description
-----------
Iterates through a directory containing meteorological forcing CSVs for openAMUNDSEN,
checks whether the temperature values (column 'temp') are in Kelvin or Celsius,
and converts them to Kelvin if necessary.

The script assumes:
- Input CSVs have at least one 'temp' column (exact name, case-sensitive).
- Temperature in °C is detected if values fall within a realistic range for
  European mountain climates (approx. -50°C to +50°C).
- Temperature in Kelvin typically ranges from ~223K to ~323K.

If conversion is applied, the script overwrites the original CSV (or optionally
writes to a new file). All results are logged.

Example
-------
# Example input file (CSV)
date,temp,precip
1990-10-01 00:00:00,-3.1,0.0
1990-10-01 01:00:00,-2.9,0.0

# After conversion
date,temp,precip
1990-10-01 00:00:00,270.05,0.0
1990-10-01 01:00:00,270.25,0.0

Configuration
-------------
Adjust the variables in the CONFIG AREA section below for your file paths.
"""

import os
import pandas as pd
import logging
from typing import Optional

# ==================================
# ======== CONFIG AREA =============
# ==================================

# Path to the folder containing all meteorological CSVs
INPUT_DIR: str = r"02-Daten\euregio\meteo\stations"

# Whether to overwrite the original files (True) or save copies (False)
OVERWRITE: bool = True

# If not overwriting, suffix for converted files
OUTPUT_SUFFIX: str = "_K"

# Acceptable range thresholds
CELSIUS_RANGE = (-50, 50)
KELVIN_RANGE = (200, 330)

# Log file name
LOGFILE_NAME: str = "check_temperature_units.log"


# ==================================
# ======== LOGGING SETUP ===========
# ==================================

def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, LOGFILE_NAME)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logfile, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized. Log file: %s", logfile)


# ==================================
# ======= CORE FUNCTIONS ===========
# ==================================

def detect_temperature_unit(series: pd.Series) -> Optional[str]:
    """Detect whether the temperature series is in Celsius or Kelvin."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None

    min_val, max_val = s.min(), s.max()
    logging.debug(f"Detected temp range: {min_val:.2f} to {max_val:.2f}")

    if CELSIUS_RANGE[0] <= min_val <= CELSIUS_RANGE[1] and CELSIUS_RANGE[0] <= max_val <= CELSIUS_RANGE[1]:
        return "C"
    elif KELVIN_RANGE[0] <= min_val <= KELVIN_RANGE[1] and KELVIN_RANGE[0] <= max_val <= KELVIN_RANGE[1]:
        return "K"
    else:
        # Ambiguous / mixed / corrupted values
        return None


def convert_to_kelvin(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the 'temp' column from Celsius to Kelvin."""
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce") + 273.15
    return df


def process_csv_file(csv_path: str, overwrite: bool, suffix: str) -> None:
    """Process a single CSV: detect and convert temperature if needed."""
    fname = os.path.basename(csv_path)
    logging.info(f"Checking {fname} ...")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Failed to read {fname}: {e}")
        return

    if "temp" not in df.columns:
        logging.error(f"No 'temp' column found in {fname}! Skipping.")
        return

    unit = detect_temperature_unit(df["temp"])

    if unit == "C":
        logging.info(f"Detected °C in {fname} → converting to K.")
        df = convert_to_kelvin(df)
        if overwrite:
            df.to_csv(csv_path, index=False)
        else:
            new_path = os.path.splitext(csv_path)[0] + suffix + ".csv"
            df.to_csv(new_path, index=False)
        logging.info(f"Conversion complete for {fname}.")
    elif unit == "K":
        logging.info(f"{fname}: already in Kelvin — no change.")
    else:
        logging.warning(f"Could not determine unit for {fname} (range out of bounds). Skipped.")


def main() -> None:
    log_dir = os.path.join(os.path.dirname(INPUT_DIR), "logs")
    setup_logging(log_dir)
    logging.info("Starting temperature unit verification for directory: %s", INPUT_DIR)

    if not os.path.isdir(INPUT_DIR):
        logging.error("Input directory does not exist: %s", INPUT_DIR)
        return

    csv_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".csv")]
    if not csv_files:
        logging.warning("No CSV files found in input directory.")
        return

    for f in csv_files:
        csv_path = os.path.join(INPUT_DIR, f)
        process_csv_file(csv_path, OVERWRITE, OUTPUT_SUFFIX)

    logging.info("Finished checking all CSVs in %s", INPUT_DIR)


if __name__ == "__main__":
    main()
