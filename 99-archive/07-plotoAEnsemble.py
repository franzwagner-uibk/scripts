#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 07-plotoAEnsemble.py
Author: Franz Wagner (+ ChatGPT assist)
Date: 2025-10-13 (Europe/Vienna)

Description
-----------
Plot openAMUNDSEN point results as an ensemble:
- Reads result CSVs from <OUTPUT_ROOT>/member_XXX/results/point_*.csv
- Optionally includes open_loop/results if available (rendered in black)
- Variable to plot is user-defined (e.g., 'swe' (mm), 'snow_depth' (m))
- Time window (inclusive) + optional resampling/smoothing
- Ensemble mean & 5–95% band
- Robust to duplicate timestamps (e.g., 3-hourly) by collapsing duplicates
- Legend below the plot (single block, 4 columns)
- Auto-creates unique output folder: <OUTPUT_ROOT>/plots_results_<mm-dd-yy>_<mm-dd-yy>[/_vN]
- Adds station altitude (from stations.csv) to the plot subtitle

Dependencies
------------
- pandas, numpy, matplotlib
"""

from __future__ import annotations
import os
import re
import sys
import logging
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cycler

# =========================
# ====== CONFIG AREA ======
# =========================

# Root produced by build_forcing_ensemble.py
OUTPUT_ROOT: str = r"02-Daten\15422\forcing_ensemble_N15_sigT0p5_sigP0p5"

# Path to stations metadata CSV (must contain either ['id','name','alt'] or ['name','alt'])
STATIONS_CSV_PATH: Optional[str] = r"02-Daten\euregio\meteo\stations\stations.csv"

# Variable to plot (column in OA point CSVs)
VAR_NAME: str = "swe"  # e.g., "swe" (mm) or "snow_depth" (m)
# Units & pretty names for legend title and axis labeling
VAR_UNITS_MAP: Dict[str, str] = {
    "swe": "mm",
    "snow_depth": "m",
}
VAR_PRETTY_MAP: Dict[str, str] = {
    "swe": "Snow Water Equivalent",
    "snow_depth": "Snow Depth",
}

# OpenAMUNDSEN results time column name
OA_TIME_COL: str = "time"

# Plot window (inclusive). Leave None to plot full span.
PLOT_START: Optional[str] = "2017-10-01 00:00:00"
PLOT_END:   Optional[str] = "2018-06-30 23:59:59"

# Optional resampling / smoothing
RESAMPLE_RULE: Optional[str] = None  # e.g. 'D', 'W', or None
RESAMPLE_AGG: str = "mean"          # typical for state variables
ROLLING_WINDOW: Optional[int] = None # e.g. 8 samples for ~1 day at 3-hourly

# Styling
FIGSIZE = (14, 7)
ALPHA_MEMBERS = 0.28
LW_OPEN = 1.9
LW_MEMBER = 1.0
LW_MEAN = 1.6
SHOW_ENS_BAND = True
BAND_ALPHA = 0.3
INCLUDE_PERTURB_IN_LEGEND = True     # show (ΔT, f_p) next to each member label

# Distinct member colors (tweak as you like)
COLOR_CYCLE = [
    "#00429d", "#73a2c6", "#f4777f", "#93003a", "#fdb863",
    "#e66101", "#b2abd2", "#5e3c99", "#1b7837", "#a6dba0",
    "#ca0020", "#0571b0", "#92c5de", "#f4a582", "#7b3294",
    "#008837", "#4dac26", "#d01c8b", "#f6e8c3", "#01665e"
]

# Max stations to plot (None -> all)
MAX_STATIONS: Optional[int] = None

# Logging
LOG_NAME = "plot_oa_results_ensemble.log"


# ==============================
# ====== LOGGING SETUP =========
# ==============================

def _setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, LOG_NAME)
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

def _parse_ts(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        logging.warning("Invalid timestamp for plot window: %s", s)
        return None
    return ts

def _window_suffix(start_ts: Optional[pd.Timestamp], end_ts: Optional[pd.Timestamp]) -> str:
    if start_ts is None and end_ts is None:
        return "full-span"
    def fmt(ts: Optional[pd.Timestamp]) -> str:
        return ts.strftime("%m-%d-%y") if ts is not None else "..."
    return f"{fmt(start_ts)}_{fmt(end_ts)}"

def _ensure_unique_dir(base_dir: str) -> str:
    if not os.path.exists(base_dir):
        return base_dir
    k = 2
    while True:
        cand = f"{base_dir}_v{k}"
        if not os.path.exists(cand):
            return cand
        k += 1

def _color_cycler() -> cycler.Cycler:
    return cycler(color=COLOR_CYCLE)

def _member_id(folder: str) -> str:
    m = re.match(r"member_(\d{3})$", folder)
    return f"member {m.group(1)}" if m else folder  # clearer than "m001"

def _collapse_duplicate_times(df: pd.DataFrame, prefer: str = "mean") -> pd.DataFrame:
    """
    Collapse duplicate timestamps on a *single-variable* dataframe.
    The dataframe must already have a DatetimeIndex and only your VAR column.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or df.shape[1] != 1:
        return df
    df = df.sort_index()
    if df.index.is_unique:
        return df
    if prefer == "sum":
        return df.groupby(level=0).sum()
    elif prefer == "first":
        return df.groupby(level=0).first()
    elif prefer == "last":
        return df.groupby(level=0).last()
    else:
        return df.groupby(level=0).mean()

def _maybe_resample_and_smooth(df: Optional[pd.DataFrame], var: str) -> Optional[pd.DataFrame]:
    if df is None or not isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df.copy()
    if RESAMPLE_RULE:
        if RESAMPLE_AGG == "sum":
            out = out.resample(RESAMPLE_RULE).sum()
        elif RESAMPLE_AGG == "first":
            out = out.resample(RESAMPLE_RULE).first()
        elif RESAMPLE_AGG == "last":
            out = out.resample(RESAMPLE_RULE).last()
        else:
            out = out.resample(RESAMPLE_RULE).mean()
    if ROLLING_WINDOW and ROLLING_WINDOW > 1 and var in out.columns:
        out[var] = out[var].rolling(ROLLING_WINDOW, min_periods=1).mean()
    return out

def _apply_window(df: Optional[pd.DataFrame], start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Optional[pd.DataFrame]:
    if df is None or not isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df
    if start is not None:
        out = out[out.index >= start]
    if end is not None:
        out = out[out.index <= end]
    return out

def _series_has_data(s: pd.Series) -> bool:
    return pd.to_numeric(s, errors="coerce").notna().any() if s is not None else False

def _any_data_in_window(ol_df: Optional[pd.DataFrame], mem_dfs: List[pd.DataFrame], col: str) -> bool:
    series = []
    if ol_df is not None and col in ol_df.columns:
        series.append(ol_df[col])
    for d in mem_dfs:
        if col in d.columns:
            series.append(d[col])
    return any(_series_has_data(s) for s in series)

def _envelope(dfs: List[pd.DataFrame], col: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if not dfs:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    clean = []
    for d in dfs:
        if col in d.columns:
            clean.append(_collapse_duplicate_times(d[[col]], prefer="mean"))
    if not clean:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    aligned = pd.concat(clean, axis=1, join="inner")
    if aligned.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    mean = aligned.mean(axis=1)
    p05  = aligned.quantile(0.05, axis=1)
    p95  = aligned.quantile(0.95, axis=1)
    return mean, p05, p95

def _legend_below(ax: plt.Axes, title: str, ncol: int = 4) -> None:
    """
    Place one legend below the plot, centered, with N columns.
    """
    plt.subplots_adjust(bottom=0.20)  # room for legend
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=ncol,
        frameon=True,
        title=title,
        columnspacing=1.3,
        handletextpad=0.6,
    )
    if leg:
        leg._legend_box.align = "left"


# ============== DISCOVERY & IO ==============

def _list_member_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"OUTPUT_ROOT not found: {root}")
    members = [d for d in os.listdir(root)
               if re.match(r"member_\d{3}$", d) and os.path.isdir(os.path.join(root, d, "results"))]
    members.sort()
    return members

def _list_point_files(root: str) -> List[str]:
    """
    List candidate point_*.csv files to plot.
    Preference: open_loop/results if exists, else first member/results found.
    """
    ol_res = os.path.join(root, "open_loop", "results")
    candidates: List[str] = []
    if os.path.isdir(ol_res):
        candidates = [f for f in os.listdir(ol_res) if f.startswith("point_") and f.lower().endswith(".csv")]
    if not candidates:
        for m in _list_member_dirs(root):
            p = os.path.join(root, m, "results")
            if os.path.isdir(p):
                candidates = [f for f in os.listdir(p) if f.startswith("point_") and f.lower().endswith(".csv")]
                if candidates:
                    break
    candidates.sort()
    return candidates

def _read_point_csv(csv_path: str, var: str) -> Optional[pd.DataFrame]:
    """
    Read a results CSV and return a single-column DataFrame [var] indexed by time,
    with duplicate timestamps collapsed (mean).
    """
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)

    if OA_TIME_COL not in df.columns:
        logging.warning("Missing '%s' column in %s", OA_TIME_COL, csv_path)
        return None
    if var not in df.columns:
        logging.warning("Variable '%s' not in %s", var, os.path.basename(csv_path))
        return None

    time_raw = df[OA_TIME_COL].astype(str)

    # --- robust datetime parsing ---
    try:
        t = pd.to_datetime(time_raw, format="%Y-%m-%d %H:%M:%S", errors="coerce")
        if t.isna().all():
            t = pd.to_datetime(time_raw, format="%Y-%m-%d", errors="coerce")
    except Exception:
        t = pd.to_datetime(time_raw, errors="coerce")

    df_var = pd.DataFrame({var: pd.to_numeric(df[var], errors="coerce")})
    df_var.index = t
    df_var = df_var[~df_var.index.isna()]

    if df_var.index.nunique() < 10:
        logging.warning("Time parsing issue in %s — only %d unique timestamps found.",
                        os.path.basename(csv_path), df_var.index.nunique())

    df_var = _collapse_duplicate_times(df_var, prefer="mean")
    return df_var

def _read_member_perturb_info_once(root: str, members: List[str]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Return {member_dir: (delta_t, f_p)} from INFO.txt; missing -> (None, None)."""
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for md in members:
        info_path = os.path.join(root, md, "INFO.txt")
        dt_val: Optional[float] = None
        fp_val: Optional[float] = None
        if os.path.isfile(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if "ΔT" in line or "Delta" in line or "dT" in line:
                            m = re.search(r"([-+]?[\d\.]+)", line)
                            if m: dt_val = float(m.group(1))
                        elif "f_p" in line or "precip factor" in line:
                            m = re.search(r"([-+]?[\d\.]+)", line)
                            if m: fp_val = float(m.group(1))
            except Exception as ex:
                logging.warning("Failed reading INFO.txt for %s: %s", md, ex)
        out[md] = (dt_val, fp_val)
    return out


# ============== STATIONS / ALTITUDE LOOKUP ==============

def _load_stations(csv_path: Optional[str]) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Read stations CSV and return a mapping by both ID and NAME for robust lookup.

    Expected columns:
      - Either ['id','name','alt']  OR  ['name','alt'].
    Returns dict with keys for both id and name (lowercased), values {'name':<str>, 'alt':<float>}.
    """
    mapping: Dict[str, Dict[str, Optional[float]]] = {}
    if not csv_path:
        logging.info("No STATIONS_CSV_PATH configured; altitude will not be shown.")
        return mapping
    if not os.path.isfile(csv_path):
        logging.warning("Stations CSV not found: %s", csv_path)
        return mapping

    try:
        df = pd.read_csv(csv_path)
    except Exception as ex:
        logging.warning("Failed to read stations CSV: %s", ex)
        return mapping

    cols = {c.lower().strip(): c for c in df.columns}
    has_id = "id" in cols
    has_name = "name" in cols
    has_alt = "alt" in cols

    if not (has_name and has_alt) and not (has_id and has_alt):
        logging.warning("Stations CSV must include 'name' and 'alt' (optionally 'id'). Columns found: %s", list(df.columns))
        return mapping

    for _, row in df.iterrows():
        name_val = str(row[cols["name"]]).strip() if has_name else None
        alt_val = None
        try:
            alt_val = float(row[cols["alt"]])
        except Exception:
            alt_val = None

        rec = {"name": name_val, "alt": alt_val}
        if has_name and name_val:
            mapping[name_val.lower()] = rec
        if has_id and not pd.isna(row[cols["id"]]):
            key_id = str(row[cols["id"]]).strip().lower()
            mapping[key_id] = rec
    logging.info("Loaded %d station metadata entries from %s", len(mapping), csv_path)
    return mapping

def _lookup_station_info(stations_map: Dict[str, Dict[str, Optional[float]]],
                         station_token: str) -> Dict[str, Optional[str | float]]:
    """
    Try by id first, then by name (case-insensitive).
    station_token is derived from filename 'point_<token>.csv'.
    """
    key = station_token.strip().lower()
    rec = stations_map.get(key)
    if rec:
        return rec
    # fallback: attempt raw token as a name
    return {"name": station_token, "alt": None}


# ============== PLOTTING ==============

def _apply_xlim(ax: plt.Axes, idx: Optional[pd.DatetimeIndex],
                start_ts: Optional[pd.Timestamp], end_ts: Optional[pd.Timestamp]) -> None:
    if start_ts is None and end_ts is None:
        return
    xmin = start_ts if start_ts is not None else (idx.min() if idx is not None and len(idx) else None)
    xmax = end_ts   if end_ts   is not None else (idx.max() if idx is not None and len(idx) else None)
    try:
        ax.set_xlim(xmin, xmax)
    except Exception as ex:
        logging.warning("Failed to set x-limits: %s", ex)

def _var_units_label(var: str) -> str:
    return VAR_UNITS_MAP.get(var, "")

def _var_pretty(var: str) -> str:
    return VAR_PRETTY_MAP.get(var, var)

def plot_point_file(point_file: str,
                    ol_df: Optional[pd.DataFrame],
                    mem_dfs: List[Tuple[str, Optional[pd.DataFrame]]],
                    member_info: Dict[str, Tuple[Optional[float], Optional[float]]],
                    stations_map: Dict[str, Dict[str, Optional[float]]],
                    var: str,
                    outdir: str,
                    start_ts: Optional[pd.Timestamp],
                    end_ts: Optional[pd.Timestamp]) -> None:

    station_token = os.path.splitext(os.path.basename(point_file))[0].replace("point_", "")
    os.makedirs(outdir, exist_ok=True)

    # Station meta (name + altitude)
    st = _lookup_station_info(stations_map, station_token)
    station_name = st.get("name") or station_token
    station_alt = st.get("alt")
    alt_str = f" ({int(round(station_alt))} m)" if isinstance(station_alt, (int, float)) else ""

    # Resample / smooth
    ol_proc = _maybe_resample_and_smooth(ol_df, var) if ol_df is not None else None
    mem_proc = [(mdir, _maybe_resample_and_smooth(d, var) if d is not None else None) for mdir, d in mem_dfs]

    # Time window
    ol_proc = _apply_window(ol_proc, start_ts, end_ts) if ol_proc is not None else None
    mem_proc = [(mdir, _apply_window(d, start_ts, end_ts) if d is not None else None) for mdir, d in mem_proc]

    # Skip if no data in window
    mem_only = [d for _, d in mem_proc if d is not None]
    if not _any_data_in_window(ol_proc, mem_only, var):
        logging.info("Skip plot for %s: all values are NaN in window.", station_token)
        return

    # --- Plot ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_prop_cycle(_color_cycler())

    # Members
    for mdir, d in mem_proc:
        if d is None or var not in d.columns:
            continue
        dt, fp = member_info.get(mdir, (None, None))
        label = _member_id(mdir)
        if INCLUDE_PERTURB_IN_LEGEND and (dt is not None and fp is not None):
            label = f"{label} (ΔT={dt:+.3f}, f_p={fp:.3f})"
        ax.plot(d.index, d[var], linewidth=LW_MEMBER, alpha=ALPHA_MEMBERS, label=label)

    # Ensemble mean + band
    mean, p05, p95 = _envelope([d for _, d in mem_proc if d is not None], var)
    if not mean.empty:
        if SHOW_ENS_BAND and not p05.empty and not p95.empty:
            ax.fill_between(p05.index, p05.values, p95.values, alpha=BAND_ALPHA,
                            color="#5A96E8", zorder=1, label="ensemble 5–95%")
        ax.plot(mean.index, mean.values, linewidth=LW_MEAN, color="#F54927",
                label="ensemble mean", zorder=3)

    # Open loop (if present): black and on top
    if ol_proc is not None and var in ol_proc.columns:
        ax.plot(ol_proc.index, ol_proc[var], linewidth=LW_OPEN, color="black",
                label="open loop", zorder=4)

    # ---- Titles / labels ----
    units = _var_units_label(var)
    pretty = _var_pretty(var)
    var_title = f"{pretty} ({units})" if units else pretty  
    
    # --- Main title (centered) + subtitle (minimal gap) ---
    # Use a single title approach with combined text
    combined_title = f"openAMUNDSEN Ensemble results with perturbed forcing\n{station_name}{alt_str} — {var_title}"
    ax.set_title(combined_title, fontsize=14, fontweight='bold', pad=12, loc='center')
    
    ax.set_xlabel("Time")
    ax.set_ylabel(var_title)
    ax.grid(True)   
    
    # --- Layout & legend ---
    # Adjust the top margin to bring title closer
    fig.tight_layout(rect=[0.06, 0.18, 0.98, 0.94])  # Increased top from 0.905 to 0.94   

    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=True,
        title=var_title,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    if leg:
        leg._legend_box.align = "left"  

    out_path = os.path.join(outdir, f"{station_token}_{var}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    logging.info("Wrote %s", out_path)






# ============================
# ====== MAIN EXECUTION ======
# ============================

def main() -> None:
    # Resolve window + output folder
    start_ts = _parse_ts(PLOT_START)
    end_ts   = _parse_ts(PLOT_END)
    win_suffix = _window_suffix(start_ts, end_ts)
    base_plots_dir = os.path.join(OUTPUT_ROOT, f"plots_results_{win_suffix}")
    plots_dir = _ensure_unique_dir(base_plots_dir)
    os.makedirs(plots_dir, exist_ok=True)

    _setup_logging(plots_dir)
    logging.info("OUTPUT_ROOT: %s", os.path.abspath(OUTPUT_ROOT))
    logging.info("Plots will be written to: %s", os.path.abspath(plots_dir))
    logging.info("Plot window: [%s .. %s]", PLOT_START or "-", PLOT_END or "-")

    units = _var_units_label(VAR_NAME)
    pretty = _var_pretty(VAR_NAME)
    logging.info("Variable: %s%s", VAR_NAME, f" ({pretty}, units: {units})" if units else "")

    logging.info("Expecting time column named exactly '%s' in OA CSVs.", OA_TIME_COL)

    # Stations metadata (for altitude in titles)
    stations_map = _load_stations(STATIONS_CSV_PATH)

    # Discover members + perturbation info
    members_dirs = _list_member_dirs(OUTPUT_ROOT)
    logging.info("Found %d member result folders.", len(members_dirs))
    member_info = _read_member_perturb_info_once(OUTPUT_ROOT, members_dirs)

    # Discover point files
    point_files = _list_point_files(OUTPUT_ROOT)
    if MAX_STATIONS is not None:
        point_files = point_files[:MAX_STATIONS]
    logging.info("Found %d point CSVs to plot.", len(point_files))

    # Iterate stations
    for fname in point_files:
        # Open loop (if exists)
        ol_path = os.path.join(OUTPUT_ROOT, "open_loop", "results", fname)
        ol_df = _read_point_csv(ol_path, VAR_NAME) if os.path.isfile(ol_path) else None

        # Member frames
        mem_frames: List[Tuple[str, Optional[pd.DataFrame]]] = []
        for md in members_dirs:
            mp = os.path.join(OUTPUT_ROOT, md, "results", fname)
            if not os.path.isfile(mp):
                logging.warning("Missing in %s/results: %s", md, fname)
                mem_frames.append((md, None))
                continue
            mem_frames.append((md, _read_point_csv(mp, VAR_NAME)))

        plot_point_file(
            point_file=fname,
            ol_df=ol_df,
            mem_dfs=mem_frames,
            member_info=member_info,
            stations_map=stations_map,
            var=VAR_NAME,
            outdir=plots_dir,
            start_ts=start_ts,
            end_ts=end_ts
        )

    logging.info("DONE. Compare PNGs in %s", os.path.abspath(plots_dir))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Fatal error: %s", e)
        sys.exit(1)
