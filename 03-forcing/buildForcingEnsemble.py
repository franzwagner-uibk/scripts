#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: build_forcing_ensemble.py
Author: Franz Wagner
Date: 2025-10-08 (Europe/Vienna)

Description
-----------
Framework to generate an ensemble of perturbed meteorological forcings for openAMUNDSEN.

Core features
-------------
- Configurable ensemble size
- Temperature: ΔT_i ~ N(0, SIGMA_T^2) (additive, stationary per member)
- Precipitation: f_p,i ~ logN(MU_P, SIGMA_P^2) (multiplicative, stationary per member)
- Exact CSV schema preserved (names/order); only temp/precip values are changed
- Optional inclusive date filter [START_DATE .. END_DATE]
- Output per member:
    member_xxx/
      ├─ meteo/     (perturbed CSVs + stations.csv unchanged)
      ├─ results/   (empty placeholder)
      └─ INFO.txt   (ΔT, f_p, time window, columns used, stats, duration)
- open_loop/ contains (optionally time-filtered) unperturbed inputs
- Output root auto-suffixed to avoid overwrite:
    <OUTPUT_ROOT>_N{ENSEMBLE_SIZE}_sigT{SIGMA_T}_sigP{SIGMA_P}[_run2|_run3|...]

Assumptions
-----------
- INPUT_METEO_DIR contains stations.csv + one <station_id>.csv per station
- If column names are unknown, robust autodetection is applied (global scan + per-file fallback)

Logging
-------
- INFO-level console + file logging (timestamped)

Coding guidelines
-----------------
- All variables & paths are configured at the top
- Detailed comments for variables to be set
- Functionality is modularized; script uses global-style config vars to remain generic
"""

from __future__ import annotations
import os
import re
import sys
import shutil
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd

# =========================
# ====== CONFIG AREA ======
# =========================

# ---- INPUTS ----
# Base directory that contains the meteo inputs (stations.csv + one CSV per station).
INPUT_METEO_DIR: str = r"02-Daten\15422\meteo"
STATIONS_CSV_NAME: str = "stations.csv"

# Explicit column names (set to None to rely on autodetection)
TIME_COL:   Optional[str] = "date"
TEMP_COL:   Optional[str] = "temp"
PRECIP_COL: Optional[str] = "precip"

# Regex patterns for autodetection (case-insensitive)
TIME_PATTERNS:   List[str] = [r"time", r"date", r"datetime", r"timestamp"]
TEMP_PATTERNS:   List[str] = [r"^temp(erature)?$", r"^ta$", r"^t$", r"^t2m$", r"air.?temp", r"^tt$", r"^tg$"]
PRECIP_PATTERNS: List[str] = [r"^precip(it(a|)tion)?$", r"^psum$", r"^rr$", r"^rrr$", r"^pr(cp)?$", r"^p$", r"^rf$",
                              r"niederschlag", r"^ppt$", r"^rain$"]

# ---- DATE FILTER (inclusive) ----
ENABLE_DATE_FILTER: bool = True
START_DATE: str = "2010-10-01 00:00:00"
END_DATE:   str = "2024-09-30 23:59:59"

# ---- OUTPUT ROOT (BASE) ----
# IMPORTANT: Use EXACT path (absolute or relative). The script will append a suffix to avoid overwrite.
# Example: r"C:\Daten\PhD\forcing_ensemble"
OUTPUT_ROOT: str = r"02-Daten\15422\forcing_ensemble"   # base path provided by the user

# ---- ENSEMBLE SETTINGS ----
ENSEMBLE_SIZE: int = 15
RANDOM_SEED: int = 42

# Temperature prior ΔT (same units as your temperature column)
SIGMA_T: float = 0.5

# Precip factor f_p ~ LogNormal(MU_P, SIGMA_P^2)
# Tip: set MU_P = -0.5 * SIGMA_P**2 to keep E[f_p] ≈ 1
MU_P: float = 0.0
SIGMA_P: float = 0.5

# ---- TEST MODE ----
TEST_MODE: bool = False
TEST_MAX_FILES: int = 10

# ---- RUNTIME OUTPUT ROOT (computed in main; do not edit) ----
OUT_ROOT: str = OUTPUT_ROOT  # will be replaced with suffixed path in main


# ==============================
# ====== LOGGING SETUP =========
# ==============================

def _setup_logging() -> None:
    """Configure INFO-level logging to console and to a timestamped logfile under OUT_ROOT/logs."""
    log_dir = os.path.join(OUT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    logfile = os.path.join(log_dir, f"build_forcing_ensemble_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logfile, encoding="utf-8")]
    )
    logging.info("Logging initialized.")
    logging.info("Log file: %s", logfile)


# =====================================
# ====== UTILS & CORE FUNCTIONS =======
# =====================================

def _first_regex_match(cols: List[str], patterns: List[str]) -> Optional[str]:
    for c in cols:
        if any(re.search(p, c.lower(), re.IGNORECASE) for p in patterns):
            return c
    return None


def _fmt_param(x: float) -> str:
    """Format numeric parameter for folder name (e.g., 0.5 -> '0p5', 1.0 -> '1')."""
    s = f"{x:.6g}"
    return s.replace(".", "p").replace("-", "m")


def _compute_out_root(base: str) -> str:
    """
    Build a suffixed output root:
      <base>_N{ENSEMBLE_SIZE}_sigT{SIGMA_T}_sigP{SIGMA_P}
    If it exists, append _run2, _run3, ... until unique.
    """
    suffix = f"_N{ENSEMBLE_SIZE}_sigT{_fmt_param(SIGMA_T)}_sigP{_fmt_param(SIGMA_P)}"
    candidate = f"{base}{suffix}"
    if not os.path.exists(candidate):
        return candidate
    k = 2
    while True:
        c = f"{candidate}_run{k}"
        if not os.path.exists(c):
            return c
        k += 1


def list_station_files(meteo_dir: str, stations_csv_name: str) -> Tuple[str, List[str]]:
    stations_csv_path = os.path.join(meteo_dir, stations_csv_name)
    if not os.path.isfile(stations_csv_path):
        raise FileNotFoundError(f"Missing stations CSV: {stations_csv_path}")
    all_csvs = [f for f in sorted(os.listdir(meteo_dir)) if f.lower().endswith(".csv")]
    station_csvs = [os.path.join(meteo_dir, f) for f in all_csvs if f != stations_csv_name]
    if TEST_MODE:
        station_csvs = station_csvs[:TEST_MAX_FILES]
        logging.info("TEST_MODE active: limiting to first %d station CSVs", len(station_csvs))
    if not station_csvs:
        logging.warning("No station CSVs found under %s (besides %s).", meteo_dir, stations_csv_name)
    else:
        logging.info("Found %d station CSV files.", len(station_csvs))
    return stations_csv_path, station_csvs


def detect_columns_across_files(
    station_files: List[str],
    temp_hint: Optional[str],
    precip_hint: Optional[str],
    temp_patterns: List[str],
    precip_patterns: List[str]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find global default temp/precip columns:
    - Prefer hints if present in any scanned file
    - Else try regex patterns across up to 20 files
    """
    scan = station_files[:min(20, len(station_files))]
    tcol, pcol = None, None

    # Try hints
    if temp_hint or precip_hint:
        for path in scan:
            try:
                df = pd.read_csv(path, nrows=5)
            except Exception:
                continue
            if temp_hint and temp_hint in df.columns:
                tcol = temp_hint
            if precip_hint and precip_hint in df.columns:
                pcol = precip_hint
            if tcol and pcol:
                break

    # Try patterns
    if not (tcol and pcol):
        for path in scan:
            try:
                df = pd.read_csv(path, nrows=5)
            except Exception:
                continue
            t = tcol or _first_regex_match(list(df.columns), temp_patterns)
            p = pcol or _first_regex_match(list(df.columns), precip_patterns)
            if t and p:
                tcol, pcol = t, p
                break

    if tcol and pcol:
        logging.info("Global columns -> TEMP='%s', PRECIP='%s'", tcol, pcol)
    else:
        logging.warning("Global detection incomplete. TEMP=%s, PRECIP=%s. "
                        "Per-file fallback will be used.", tcol, pcol)
    return tcol, pcol


def autodetect_time_column(df: pd.DataFrame, time_hint: Optional[str]) -> Optional[str]:
    if time_hint and time_hint in df.columns:
        return time_hint
    return _first_regex_match(list(df.columns), TIME_PATTERNS)


def parse_time_index(df: pd.DataFrame, time_col: Optional[str]) -> Optional[pd.Series]:
    if not time_col or time_col not in df.columns:
        return None
    t = pd.to_datetime(df[time_col], errors="coerce", utc=False)
    return None if t.isna().all() else t


def filter_df_by_dates(df: pd.DataFrame,
                       time_col: Optional[str],
                       start_date: str,
                       end_date: str) -> pd.DataFrame:
    if time_col is None or time_col not in df.columns:
        return df
    t = parse_time_index(df, time_col)
    if t is None:
        logging.warning("Time parsing failed for '%s'. Skipping date filtering.", time_col)
        return df
    start = pd.to_datetime(start_date, errors="coerce")
    end   = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        logging.warning("Invalid date bounds; skipping filter. START=%s END=%s", start_date, end_date)
        return df
    mask = (t >= start) & (t <= end)
    before, after = len(df), int(mask.sum())
    logging.info("Date filter [%s..%s]: %d -> %d rows", start_date, end_date, before, after)
    return df.loc[mask].copy()


def resolve_cols_for_df(df: pd.DataFrame,
                        global_temp: Optional[str],
                        global_precip: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Per-file robust resolution: use global names if present, else regex autodetect on this df."""
    cols = list(df.columns)
    tcol = global_temp if (global_temp and global_temp in cols) else _first_regex_match(cols, TEMP_PATTERNS)
    pcol = global_precip if (global_precip and global_precip in cols) else _first_regex_match(cols, PRECIP_PATTERNS)
    return tcol, pcol


def perturb_values(df: pd.DataFrame,
                   tcol: Optional[str],
                   pcol: Optional[str],
                   delta_t: float,
                   f_p: float) -> pd.DataFrame:
    """Column-wise perturbation (apply what exists, leave the rest untouched)."""
    out = df.copy()
    if tcol and tcol in out.columns:
        out[tcol] = pd.to_numeric(out[tcol], errors="coerce") + delta_t
    if pcol and pcol in out.columns:
        out[pcol] = pd.to_numeric(out[pcol], errors="coerce").clip(lower=0) * f_p
    return out


def write_csv(df: pd.DataFrame, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    df.to_csv(dst_path, index=False)


def copy_stations_csv(src_stations_csv: str, dest_meteo_dir: str) -> None:
    os.makedirs(dest_meteo_dir, exist_ok=True)
    shutil.copy2(src_stations_csv, os.path.join(dest_meteo_dir, STATIONS_CSV_NAME))


def make_member_dirs(member_name: str) -> Tuple[str, str, str]:
    base = os.path.join(OUT_ROOT, member_name)
    meteo_dir = os.path.join(base, "meteo")
    results_dir = os.path.join(base, "results")
    os.makedirs(meteo_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    logging.info("Prepared %s/{meteo,results}", base)
    return base, meteo_dir, results_dir


def format_timedelta(dt: timedelta) -> str:
    s = int(dt.total_seconds())
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d} (hh:mm:ss)"


@dataclass
class MemberStats:
    member_name: str
    delta_t: float
    precip_factor: float
    random_seed: int
    temp_col_global: Optional[str]
    precip_col_global: Optional[str]
    time_col: Optional[str]
    input_dir: str
    output_dir: str
    date_filter_enabled: bool
    start_date: Optional[str]
    end_date: Optional[str]
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    n_files: int = 0
    n_changed_t: int = 0
    n_changed_p: int = 0
    total_rows_before: int = 0
    total_rows_after: int = 0

    def finish(self) -> None:
        self.end_time = datetime.now()

    def duration(self) -> timedelta:
        return (self.end_time or datetime.now()) - self.start_time


def process_one_file(src_path: str,
                     dst_dir: str,
                     time_col: Optional[str],
                     do_filter: bool,
                     t_global: Optional[str],
                     p_global: Optional[str],
                     delta_t: float,
                     f_p: float,
                     apply_perturbations: bool,
                     stats: Optional[MemberStats]) -> None:
    """
    Unified station handler:
      - read
      - optional date filter
      - per-file column resolution
      - optional per-variable perturbation
      - write
      - update stats
    """
    df = pd.read_csv(src_path)
    before = len(df)

    if do_filter:
        df = filter_df_by_dates(df, time_col, START_DATE, END_DATE)

    tcol, pcol = resolve_cols_for_df(df, t_global, p_global)
    changed_t = changed_p = False

    if apply_perturbations:
        df_out = perturb_values(df, tcol, pcol, delta_t, f_p)
        changed_t = tcol is not None and tcol in df.columns
        changed_p = pcol is not None and pcol in df.columns
    else:
        df_out = df

    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    write_csv(df_out, dst_path)

    if stats is not None:
        stats.n_files += 1
        stats.total_rows_before += before
        stats.total_rows_after += len(df)
        stats.n_changed_t += int(changed_t and apply_perturbations)
        stats.n_changed_p += int(changed_p and apply_perturbations)

        if apply_perturbations:
            msg_bits = []
            if changed_t: msg_bits.append(f"T:+{delta_t:.3f}")
            if changed_p: msg_bits.append(f"P×{f_p:.3f}")
            if msg_bits:
                logging.info("[%s] %s -> %s", stats.member_name, os.path.basename(src_path), ", ".join(msg_bits))
            else:
                logging.info("[%s] %s -> copied unchanged (no T/P cols)", stats.member_name, os.path.basename(src_path))


def write_member_info(stats: MemberStats) -> None:
    info_path = os.path.join(stats.output_dir, "INFO.txt")
    lines = [
        f"Member: {stats.member_name}",
        f"Timestamp (local): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Random seed: {stats.random_seed}",
        "",
        "Applied perturbations (stationary per member):",
        f"  ΔT (additive): {stats.delta_t:+.3f}",
        f"  f_p (precip factor): {stats.precip_factor:.3f}",
        "",
        "Date filter:",
        f"  Enabled: {stats.date_filter_enabled}",
        f"  START_DATE: {stats.start_date or 'N/A'}",
        f"  END_DATE:   {stats.end_date or 'N/A'}",
        "",
        "Columns (global defaults):",
        f"  TIME_COL:   {stats.time_col or 'auto'}",
        f"  TEMP_COL:   {stats.temp_col_global or 'auto'}",
        f"  PRECIP_COL: {stats.precip_col_global or 'auto'}",
        "",
        "Processing statistics:",
        f"  Station files processed: {stats.n_files}",
        f"  Files with temperature changed: {stats.n_changed_t}",
        f"  Files with precipitation changed: {stats.n_changed_p}",
        f"  Total rows before filter: {stats.total_rows_before}",
        f"  Total rows after  filter: {stats.total_rows_after}",
        f"  Duration: {format_timedelta(stats.duration())}",
        "",
        "Paths:",
        f"  Input dir:  {stats.input_dir}",
        f"  Output dir: {stats.output_dir}",
    ]
    os.makedirs(stats.output_dir, exist_ok=True)
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def process_open_loop(stations_csv_path: str,
                      station_files: List[str],
                      time_col: Optional[str]) -> None:
    base, meteo_dir, _ = make_member_dirs("open_loop")
    for src in station_files:
        try:
            process_one_file(src, meteo_dir, time_col, ENABLE_DATE_FILTER,
                             t_global=None, p_global=None,  # no perturbation
                             delta_t=0.0, f_p=1.0,
                             apply_perturbations=False, stats=None)
        except Exception as ex:
            logging.exception("[open_loop] Failed %s: %s", os.path.basename(src), ex)
            shutil.copy2(src, os.path.join(meteo_dir, os.path.basename(src)))
    copy_stations_csv(stations_csv_path, meteo_dir)
    logging.info("Open-loop inputs ready at: %s", base)


def process_member(member_index: int,
                   stations_csv_path: str,
                   station_files: List[str],
                   rng: np.random.Generator,
                   t_global: Optional[str],
                   p_global: Optional[str],
                   time_col: Optional[str]) -> None:
    member = f"member_{member_index:03d}"
    base, meteo_dir, _ = make_member_dirs(member)
    delta_t = rng.normal(0.0, SIGMA_T)
    f_p = float(rng.lognormal(mean=MU_P, sigma=SIGMA_P))
    logging.info("[%s] ΔT=%+0.3f ; f_p=%0.3f", member, delta_t, f_p)

    stats = MemberStats(
        member_name=member,
        delta_t=delta_t,
        precip_factor=f_p,
        random_seed=RANDOM_SEED,
        temp_col_global=t_global,
        precip_col_global=p_global,
        time_col=time_col,
        input_dir=os.path.abspath(INPUT_METEO_DIR),
        output_dir=base,
        date_filter_enabled=ENABLE_DATE_FILTER,
        start_date=START_DATE if ENABLE_DATE_FILTER else None,
        end_date=END_DATE if ENABLE_DATE_FILTER else None
    )

    for src in station_files:
        try:
            process_one_file(src, meteo_dir, time_col, ENABLE_DATE_FILTER,
                             t_global=t_global, p_global=p_global,
                             delta_t=delta_t, f_p=f_p,
                             apply_perturbations=True, stats=stats)
        except Exception as ex:
            logging.exception("[%s] Failed %s: %s", member, os.path.basename(src), ex)
            shutil.copy2(src, os.path.join(meteo_dir, os.path.basename(src)))

    copy_stations_csv(stations_csv_path, meteo_dir)
    stats.finish()
    write_member_info(stats)


def summarize_outputs() -> None:
    if not os.path.isdir(OUT_ROOT):
        return
    members = [d for d in sorted(os.listdir(OUT_ROOT))
               if os.path.isdir(os.path.join(OUT_ROOT, d))]
    logging.info("Created the following folders under %s:", OUT_ROOT)
    for m in members:
        mpath = os.path.join(OUT_ROOT, m, "meteo")
        n_csv = len([f for f in os.listdir(mpath)]) if os.path.isdir(mpath) else 0
        logging.info(" - %s: %d CSV(s) in meteo/", m, n_csv)


# ============================
# ====== MAIN EXECUTION ======
# ============================

def main() -> None:
    global OUT_ROOT

    # Build suffixed, unique OUT_ROOT based on ensemble settings
    OUT_ROOT = _compute_out_root(OUTPUT_ROOT)
    os.makedirs(OUT_ROOT, exist_ok=True)

    _setup_logging()
    logging.info("Output root (final): %s", os.path.abspath(OUT_ROOT))
    logging.info("Date filter: %s | [%s .. %s] | TIME_COL=%s",
                 ENABLE_DATE_FILTER, START_DATE, END_DATE, TIME_COL or "auto")
    logging.info("Ensemble settings: N=%d, SIGMA_T=%s, MU_P=%s, SIGMA_P=%s",
                 ENSEMBLE_SIZE, SIGMA_T, MU_P, SIGMA_P)

    # Deterministic sampling
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Inputs
    stations_csv_path, station_files = list_station_files(INPUT_METEO_DIR, STATIONS_CSV_NAME)
    logging.info("Stations CSV: %s", stations_csv_path)

    # Global default columns (with per-file fallback in processing)
    t_global, p_global = detect_columns_across_files(
        station_files, TEMP_COL, PRECIP_COL, TEMP_PATTERNS, PRECIP_PATTERNS
    )

    # Time column detection (small sample)
    time_col = None
    if station_files:
        try:
            sample_df = pd.read_csv(station_files[0], nrows=5)
            time_col = autodetect_time_column(sample_df, TIME_COL)
        except Exception:
            time_col = TIME_COL

    # open_loop (unperturbed)
    process_open_loop(stations_csv_path, station_files, time_col)

    # ensemble members
    for i in range(1, ENSEMBLE_SIZE + 1):
        process_member(i, stations_csv_path, station_files, rng, t_global, p_global, time_col)

    summarize_outputs()
    logging.info("DONE. Point openAMUNDSEN to any member_xxx/meteo (or open_loop/meteo).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # If logging not yet configured (e.g., failure before _setup_logging), print to stderr too
        print(f"Fatal error: {e}", file=sys.stderr)
        logging.exception("Fatal error: %s", e)
        sys.exit(1)
