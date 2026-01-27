#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: plot_ensemble_comprehensive.py
Author: Franz Wagner (+ ChatGPT assist)
Date: 2025-10-14 (Europe/Vienna)

Description
-----------
Combined script for plotting both forcing ensembles and openAMUNDSEN results.
User can choose to plot either/both types of data.

Features:
- Forcing: Temperature + cumulative precipitation comparisons
- OA Results: Snow variables (SWE, snow depth, etc.) as ensemble
- Optional time window, resampling, smoothing
- Ensemble mean & 5–95% bands
- Legend below plot in 4 columns
- Station altitude from metadata
- Auto-creates unique output folders

Dependencies: pandas, numpy, matplotlib

Notes on Layout
---------------
Whitespace between the figure title (suptitle) and the plot area is controlled
globally via constants below and the function `apply_titles_and_layout()`.
Adjust `TITLE_Y` and `TIGHT_RECT` to fine-tune spacing across all plots.
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

# Root directory containing ensemble data
OUTPUT_ROOT: str = r"C:\Daten\PhD\openamundsen_da\examples\test-project\propagation\season_2017-2018\step_00_init\ensembles\prior"

# What to plot
PLOT_FORCING: bool = True      # Plot forcing ensembles (temperature + precipitation)
PLOT_OA_RESULTS: bool = True   # Plot openAMUNDSEN results (snow variables)

# Path to stations metadata CSV
STATIONS_CSV_PATH: Optional[str] = r"C:\Daten\PhD\openamundsen_da\examples\test-project\propagation\season_2017-2018\step_00_init\ensembles\prior\member_003\meteo\stations.csv"

# ====== FORCING-SPECIFIC CONFIG ======
# Column names for forcing data (set to None to autodetect)
TIME_COL:   Optional[str] = "date"
TEMP_COL:   Optional[str] = "temp"
PRECIP_COL: Optional[str] = "precip"

# Autodetection patterns (case-insensitive)
TIME_PATTERNS  = [r"time", r"date", r"datetime", r"timestamp"]
TEMP_PATTERNS  = [r"^temp(erature)?$", r"^ta$", r"^t$", r"^t2m$", r"air.?temp", r"^tt$", r"^tg$"]
PREC_PATTERNS  = [r"^precip(it(a|)tion)?$", r"^psum$", r"^rr$", r"^rrr$", r"^pr(cp)?$", r"^p$", r"^rf$", r"niederschlag", r"^ppt$", r"^rain$"]

# Hydrological year for cumulative precipitation
HYDRO_START_MONTH: int = 10
HYDRO_START_DAY:   int = 1
PRECIP_CUM_SUFFIX: str = "_cum"  # internal only

# ====== OA RESULTS-SPECIFIC CONFIG ======
# Variable to plot for OA results
OA_VAR_NAME: str = "swe"  # e.g., "swe" (mm) or "snow_depth" (m)

# Units & pretty names for OA variables
OA_VAR_UNITS_MAP: Dict[str, str] = {
    "swe": "mm",
    "snow_depth": "m",
}
OA_VAR_PRETTY_MAP: Dict[str, str] = {
    "swe": "Snow Water Equivalent",
    "snow_depth": "Snow Depth",
}

# OpenAMUNDSEN results time column name
OA_TIME_COL: str = "time"

# ====== COMMON PLOTTING CONFIG ======
# Plot window (inclusive); None -> full span
PLOT_START: Optional[str] = "2017-10-01 00:00:00"
PLOT_END:   Optional[str] = "2018-01-10 23:59:59"

# Optional resampling / smoothing
RESAMPLE_RULE: Optional[str] = None  # e.g. 'D', 'W', None
FORCING_AGG_TEMP: str = "mean"
FORCING_AGG_PREC: str = "sum"
OA_RESAMPLE_AGG: str = "mean"
ROLLING_WINDOW: Optional[int] = None  # e.g. 8 samples for ~1 day at 3-hourly

# Plot style
FIGSIZE = (14, 7)
ALPHA_MEMBERS = 1.0
LW_OPEN = 2.5
LW_MEMBER = 1.8
LW_MEAN = 2.2
SHOW_ENS_BAND = True
BAND_ALPHA = 0.4
ANNOTATE_PERTURBATIONS_IN_LEGEND = True  # include ΔT/f_p next to each member in legend

# Distinct member colors (calmer qualitative palette)
COLOR_CYCLE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#393b79",  # dark blue
    "#637939",  # olive green
    "#8c6d31",  # dark brown
    "#843c39",  # dark red
    "#7b4173",  # violet
    "#cedb9c",  # light green
    "#9c9ede",  # light blue
    "#e7cb94",  # beige
    "#ad494a",  # muted red
    "#a55194",  # muted magenta
]

# ====== GLOBAL ENSEMBLE STYLING (centralized) ======
OPEN_LOOP_COLOR: str = "black"           # keep open-loop in black
ENSEMBLE_MEAN_COLOR: str = "#F54927"     # strong red-orange (global)
ENSEMBLE_MEAN_LW: float = LW_MEAN        # use existing thickness
ENSEMBLE_MEAN_LABEL: str = "ensemble mean"

ENSEMBLE_BAND_COLOR: str = "#5A96E8"     # global band color (5–95%)
ENSEMBLE_BAND_LABEL: str = "ensemble 5–95%"

# Limits / selection
MAX_STATIONS: Optional[int] = None

# Logging
LOG_NAME = "plot_ensemble_comprehensive.log"

# ====== GLOBAL LAYOUT TUNING (whitespace control) ======
# Lower TITLE_Y (e.g., 0.92–0.95) to bring suptitle closer to the axes area
TITLE_Y: float = 0.94
# Subtitle (axes-title) padding; keep small to reduce gap above the plot
SUBTITLE_PAD: float = 1.0
# Area reserved for axes; increase top (index 3) to reduce top whitespace
# Format: (left, bottom, right, top)
TIGHT_RECT: Tuple[float, float, float, float] = (0.06, 0.18, 0.98, 0.94)
# Optional post-tight_layout top adjustment; set to None to skip
SUBPLOTS_ADJUST_TOP: Optional[float] = None  # e.g., 0.93


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
    return f"member {m.group(1)}" if m else folder

def _collapse_duplicate_times(df: pd.DataFrame, prefer: str = "mean") -> pd.DataFrame:
    """Collapse duplicate timestamps on a single-variable dataframe."""
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
    """Read stations CSV and return mapping by both ID and NAME for robust lookup."""
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
    """Try by id first, then by name (case-insensitive)."""
    key = station_token.strip().lower()
    rec = stations_map.get(key)
    if rec:
        return rec
    return {"name": station_token, "alt": None}


# ============== FORCING-SPECIFIC FUNCTIONS ==============

def _first_match(cols: List[str], pats: List[str]) -> Optional[str]:
    for c in cols:
        if any(re.search(p, c.lower(), re.IGNORECASE) for p in pats):
            return c
    return None

def autodetect_columns(df: pd.DataFrame,
                       time_hint: Optional[str],
                       temp_hint: Optional[str],
                       prec_hint: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    cols = list(df.columns)
    tcol = time_hint if (time_hint and time_hint in cols) else _first_match(cols, TIME_PATTERNS)
    xcol = temp_hint if (temp_hint and temp_hint in cols) else _first_match(cols, TEMP_PATTERNS)
    pcol = prec_hint if (prec_hint and prec_hint in cols) else _first_match(cols, PREC_PATTERNS)
    return tcol, xcol, pcol

def list_member_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"OUTPUT_ROOT not found: {root}")
    members = [d for d in os.listdir(root) if re.match(r"member_\d{3}$", d) and os.path.isdir(os.path.join(root, d, "meteo"))]
    members.sort()
    return members

def list_open_loop_stations(root: str, stations_csv_name: str = "stations.csv") -> List[str]:
    ol_meteo = os.path.join(root, "open_loop", "meteo")
    if not os.path.isdir(ol_meteo):
        raise FileNotFoundError(f"open_loop/meteo not found under {root}")
    files = [f for f in os.listdir(ol_meteo) if f.lower().endswith(".csv") and f != stations_csv_name]
    files.sort()
    return files

def read_station_series(csv_path: str,
                        time_col_hint: Optional[str],
                        temp_col_hint: Optional[str],
                        prec_col_hint: Optional[str]) -> Tuple[pd.DataFrame, Optional[str], Optional[str], Optional[str]]:
    df = pd.read_csv(csv_path)
    tcol, xcol, pcol = autodetect_columns(df, time_col_hint, temp_col_hint, prec_col_hint)
    if tcol is None:
        logging.warning("No time column detected for %s; returning df as-is.", os.path.basename(csv_path))
        return df, None, xcol, pcol
    t = pd.to_datetime(df[tcol], errors="coerce", utc=False)
    if t.isna().all():
        logging.warning("Time parsing failed for %s; returning df as-is.", os.path.basename(csv_path))
        return df, None, xcol, pcol
    df = df.set_index(t)
    return df, tcol, xcol, pcol

def maybe_resample_and_smooth_forcing(df: pd.DataFrame,
                                      tcol: Optional[str],
                                      pcol: Optional[str]) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    if RESAMPLE_RULE:
        agg: Dict[str, str] = {}
        if tcol and tcol in df.columns: agg[tcol] = FORCING_AGG_TEMP
        if pcol and pcol in df.columns: agg[pcol] = FORCING_AGG_PREC
        if agg:
            df = df.resample(RESAMPLE_RULE).agg(agg)
    if ROLLING_WINDOW and ROLLING_WINDOW > 1:
        for c in [tcol, pcol]:
            if c and c in df.columns:
                df[c] = df[c].rolling(ROLLING_WINDOW, min_periods=1).mean()
    return df

def apply_window(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if df is None or not isinstance(df.index, pd.DatetimeIndex):
        return df
    if start is not None: df = df[df.index >= start]
    if end   is not None: df = df[df.index <= end]
    return df

def _hydro_year_index(idx: pd.DatetimeIndex, m0: int, d0: int) -> np.ndarray:
    before = (idx.month < m0) | ((idx.month == m0) & (idx.day < d0))
    return (idx.year - before.astype(int)).astype(int, copy=False)

def to_hydro_cumulative(df: pd.DataFrame, pcol: str, m0: int, d0: int) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex) or pcol not in df.columns:
        return df
    out = df.copy()
    pr = pd.to_numeric(out[pcol], errors="coerce").fillna(0.0).clip(lower=0.0)
    hy = _hydro_year_index(out.index, m0, d0)
    out[pcol + PRECIP_CUM_SUFFIX] = pr.groupby(hy).cumsum()
    return out

def collect_member_series(station_file: str, members: List[str]) -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    for m in members:
        path = os.path.join(OUTPUT_ROOT, m, "meteo", station_file)
        if not os.path.isfile(path):
            logging.warning("Missing in %s: %s", m, station_file)
            continue
        df, _, _, _ = read_station_series(path, TIME_COL, TEMP_COL, PRECIP_COL)
        if isinstance(df.index, pd.DatetimeIndex):
            dfs.append(df)
        else:
            logging.warning("Skipping %s for %s: time index not parsed.", station_file, m)
    return dfs


# ============== OA RESULTS-SPECIFIC FUNCTIONS ==============

def _list_oa_point_files(root: str) -> List[str]:
    """List candidate point_*.csv files to plot for OA results."""
    ol_res = os.path.join(root, "open_loop", "results")
    candidates: List[str] = []
    if os.path.isdir(ol_res):
        candidates = [f for f in os.listdir(ol_res) if f.startswith("point_") and f.lower().endswith(".csv")]
    if not candidates:
        for m in list_member_dirs(root):
            p = os.path.join(root, m, "results")
            if os.path.isdir(p):
                candidates = [f for f in os.listdir(p) if f.startswith("point_") and f.lower().endswith(".csv")]
                if candidates:
                    break
    candidates.sort()
    return candidates

def _read_oa_point_csv(csv_path: str, var: str) -> Optional[pd.DataFrame]:
    """Read OA results CSV and return single-column DataFrame indexed by time."""
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

def _maybe_resample_and_smooth_oa(df: Optional[pd.DataFrame], var: str) -> Optional[pd.DataFrame]:
    if df is None or not isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df.copy()
    if RESAMPLE_RULE:
        if OA_RESAMPLE_AGG == "sum":
            out = out.resample(RESAMPLE_RULE).sum()
        elif OA_RESAMPLE_AGG == "first":
            out = out.resample(RESAMPLE_RULE).first()
        elif OA_RESAMPLE_AGG == "last":
            out = out.resample(RESAMPLE_RULE).last()
        else:
            out = out.resample(RESAMPLE_RULE).mean()
    if ROLLING_WINDOW and ROLLING_WINDOW > 1 and var in out.columns:
        out[var] = out[var].rolling(ROLLING_WINDOW, min_periods=1).mean()
    return out


# ============== GLOBAL TITLE/LAYOUT FUNCTION ==============

def apply_titles_and_layout(fig: plt.Figure,
                            ax: plt.Axes,
                            main_title: str,
                            subtitle: str,
                            title_y: Optional[float] = None,
                            subtitle_pad: Optional[float] = None,
                            tight_rect: Optional[Tuple[float, float, float, float]] = None,
                            subplots_adjust_top: Optional[float] = None) -> None:
    """
    Apply a consistent suptitle + subtitle layout with minimal whitespace.

    Parameters
    ----------
    fig : plt.Figure
        Figure handle.
    ax : plt.Axes
        Axes handle.
    main_title : str
        Text for the figure suptitle.
    subtitle : str
        Text for the axes title (subtitle line above the plot).
    title_y : float, optional
        Relative vertical position of suptitle (0..1). Defaults to TITLE_Y.
    subtitle_pad : float, optional
        Padding between subtitle and axes (in points). Defaults to SUBTITLE_PAD.
    tight_rect : tuple, optional
        (left, bottom, right, top) reserved for axes in tight_layout. Defaults to TIGHT_RECT.
    subplots_adjust_top : float, optional
        If provided, call plt.subplots_adjust(top=...). Defaults to SUBPLOTS_ADJUST_TOP.
    """
    ty = TITLE_Y if title_y is None else title_y
    sp = SUBTITLE_PAD if subtitle_pad is None else subtitle_pad
    rect = TIGHT_RECT if tight_rect is None else tight_rect
    top_adj = SUBPLOTS_ADJUST_TOP if subplots_adjust_top is None else subplots_adjust_top

    # Apply titles
    fig.suptitle(main_title, fontsize=14, fontweight="bold", y=ty, x=0.5, ha="center")
    ax.set_title(subtitle, fontsize=11, pad=sp, loc="center")

    # Layout: reserve space and optionally tighten the top even further
    fig.tight_layout(rect=rect)
    if isinstance(top_adj, (int, float)):
        plt.subplots_adjust(top=top_adj)


# ============== PLOTTING FUNCTIONS ==============

def plot_forcing_station(station_file: str,
                         ol_df: pd.DataFrame,
                         mem_dfs: List[pd.DataFrame],
                         members_dirs: List[str],
                         member_info: Dict[str, Tuple[Optional[float], Optional[float]]],
                         temp_col: Optional[str],
                         prec_col: Optional[str],
                         outdir: str,
                         start_ts: Optional[pd.Timestamp],
                         end_ts: Optional[pd.Timestamp],
                         stations_map: Dict[str, Dict[str, Optional[float]]]) -> None:

    station_token = os.path.splitext(os.path.basename(station_file))[0]
    os.makedirs(outdir, exist_ok=True)

    # Station meta
    st = _lookup_station_info(stations_map, station_token)
    station_name = st.get("name") or station_token
    station_alt = st.get("alt")
    alt_str = f" ({int(round(station_alt))} m)" if isinstance(station_alt, (int, float)) else ""

    # resample/smooth
    ol_proc = maybe_resample_and_smooth_forcing(ol_df.copy(), temp_col, prec_col)
    mem_proc = [maybe_resample_and_smooth_forcing(d.copy(), temp_col, prec_col) for d in mem_dfs]

    # time window
    ol_proc = apply_window(ol_proc, start_ts, end_ts)
    mem_proc = [apply_window(d, start_ts, end_ts) for d in mem_proc]

    # -------- Temperature --------
    if temp_col and temp_col in ol_proc.columns:
        if not _any_data_in_window(ol_proc, mem_proc, temp_col):
            logging.info("Skip temp plot for %s: all values are NaN in window.", station_token)
        else:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.set_prop_cycle(_color_cycler())

            # Members
            for d, mdir in zip(mem_proc, members_dirs):
                if temp_col in d.columns:
                    dt, fp = member_info.get(mdir, (None, None))
                    lab = _member_id(mdir)
                    if ANNOTATE_PERTURBATIONS_IN_LEGEND and (dt is not None and fp is not None):
                        lab = f"{lab} (ΔT={dt:+.3f}, f_p={fp:.3f})"
                    ax.plot(d.index, d[temp_col], linewidth=LW_MEMBER, alpha=ALPHA_MEMBERS, label=lab)

            # Ensemble mean & band
            mean, p05, p95 = _envelope(mem_proc, temp_col)
            if not mean.empty:
                if SHOW_ENS_BAND and not p05.empty and not p95.empty:
                    ax.fill_between(p05.index, p05.values, p95.values,
                                    alpha=BAND_ALPHA, color=ENSEMBLE_BAND_COLOR, label=ENSEMBLE_BAND_LABEL)
                ax.plot(mean.index, mean.values,
                        linewidth=ENSEMBLE_MEAN_LW,
                        color=ENSEMBLE_MEAN_COLOR,
                        label=ENSEMBLE_MEAN_LABEL)

            # Open-loop (black)
            ax.plot(ol_proc.index, ol_proc[temp_col], linewidth=LW_OPEN, color=OPEN_LOOP_COLOR, label="open loop", zorder=10)

            # Titles / labels / legend
            main_title = "Comparison of Open-Loop and Perturbed Forcing Ensembles"
            subtitle = f"{station_name}{alt_str} — Air Temperature (K)"

            ax.set_xlabel("Time")
            ax.set_ylabel("Air temperature (K)")
            ax.grid(True)

            # Legend
            leg = ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=4,
                frameon=True,
                title="Air temperature (K)",
                columnspacing=1.2,
                handletextpad=0.5,
            )
            if leg:
                leg._legend_box.align = "left"

            _apply_xlim(ax, ol_proc.index if isinstance(ol_proc.index, pd.DatetimeIndex) else None, start_ts, end_ts)

            # >>> Unified layout call <<<
            apply_titles_and_layout(fig, ax, main_title, subtitle)

            out_path = os.path.join(outdir, f"{station_token}_temp.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
            plt.close(fig)
            logging.info("Wrote %s", out_path)

    # -------- Precipitation (cumulative hydro-year) --------
    if prec_col and prec_col in ol_proc.columns:
        ol_cum = to_hydro_cumulative(ol_proc[[prec_col]].copy(), prec_col, HYDRO_START_MONTH, HYDRO_START_DAY)
        cum_col = prec_col + PRECIP_CUM_SUFFIX

        mem_cum_pairs: List[Tuple[str, pd.DataFrame]] = []
        for d, mdir in zip(mem_proc, members_dirs):
            if prec_col in d.columns:
                mem_cum_pairs.append((mdir, to_hydro_cumulative(d[[prec_col]].copy(), prec_col, HYDRO_START_MONTH, HYDRO_START_DAY)))
        mem_cum_only = [d for _, d in mem_cum_pairs]

        has_any = _any_data_in_window(ol_cum, mem_cum_only, cum_col) if cum_col in ol_cum.columns else any(
            cum_col in d.columns and _series_has_data(d[cum_col]) for d in mem_cum_only
        )

        if not has_any:
            logging.info("Skip precip plot for %s: all values are NaN in window.", station_token)
        else:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.set_prop_cycle(_color_cycler())

            # Members cumulative
            for mdir, d in mem_cum_pairs:
                if cum_col in d.columns:
                    dt, fp = member_info.get(mdir, (None, None))
                    lab = _member_id(mdir)
                    if ANNOTATE_PERTURBATIONS_IN_LEGEND and (dt is not None and fp is not None):
                        lab = f"{lab} (ΔT={dt:+.3f}, f_p={fp:.3f})"
                    ax.plot(d.index, d[cum_col], linewidth=LW_MEMBER, alpha=ALPHA_MEMBERS, label=lab)

            # Ensemble mean & band
            mean, p05, p95 = _envelope(mem_cum_only, cum_col)
            if not mean.empty:
                if SHOW_ENS_BAND and not p05.empty and not p95.empty:
                    ax.fill_between(p05.index, p05.values, p95.values,
                                    alpha=BAND_ALPHA, color=ENSEMBLE_BAND_COLOR, label=ENSEMBLE_BAND_LABEL)
                ax.plot(mean.index, mean.values,
                        linewidth=ENSEMBLE_MEAN_LW,
                        color=ENSEMBLE_MEAN_COLOR,
                        label=ENSEMBLE_MEAN_LABEL)

            # Open-loop cumulative (black)
            if cum_col in ol_cum.columns:
                ax.plot(ol_cum.index, ol_cum[cum_col], linewidth=LW_OPEN, color=OPEN_LOOP_COLOR, label="open loop", zorder=10)

            # Titles / labels / legend
            main_title = "Comparison of Open-Loop and Perturbed Forcing Ensembles"
            subtitle = f"{station_name}{alt_str} — Cumulative precipitation (mm)"

            ax.set_xlabel("Time")
            ax.set_ylabel("Cumulative precipitation (mm)")
            ax.grid(True)

            leg = ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=4,
                frameon=True,
                title="Cumulative precipitation (mm)",
                columnspacing=1.2,
                handletextpad=0.5,
            )
            if leg:
                leg._legend_box.align = "left"

            _apply_xlim(ax, ol_cum.index if isinstance(ol_cum.index, pd.DatetimeIndex) else None, start_ts, end_ts)

            # >>> Unified layout call <<<
            apply_titles_and_layout(fig, ax, main_title, subtitle)

            out_path = os.path.join(outdir, f"{station_token}_precip_cum.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
            plt.close(fig)
            logging.info("Wrote %s", out_path)


def plot_oa_point_file(point_file: str,
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

    # Station meta
    st = _lookup_station_info(stations_map, station_token)
    station_name = st.get("name") or station_token
    station_alt = st.get("alt")
    alt_str = f" ({int(round(station_alt))} m)" if isinstance(station_alt, (int, float)) else ""

    # Resample / smooth
    ol_proc = _maybe_resample_and_smooth_oa(ol_df, var) if ol_df is not None else None
    mem_proc = [(mdir, _maybe_resample_and_smooth_oa(d, var) if d is not None else None) for mdir, d in mem_dfs]

    # Time window
    ol_proc = apply_window(ol_proc, start_ts, end_ts) if ol_proc is not None else None
    mem_proc = [(mdir, apply_window(d, start_ts, end_ts) if d is not None else None) for mdir, d in mem_proc]

    # Skip if no data in window
    mem_only = [d for _, d in mem_proc if d is not None]
    if not _any_data_in_window(ol_proc, mem_only, var):
        logging.info("Skip OA plot for %s: all values are NaN in window.", station_token)
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
        if ANNOTATE_PERTURBATIONS_IN_LEGEND and (dt is not None and fp is not None):
            label = f"{label} (ΔT={dt:+.3f}, f_p={fp:.3f})"
        ax.plot(d.index, d[var], linewidth=LW_MEMBER, alpha=ALPHA_MEMBERS, label=label)

    # Ensemble mean + band
    mean, p05, p95 = _envelope([d for _, d in mem_proc if d is not None], var)
    if not mean.empty:
        if SHOW_ENS_BAND and not p05.empty and not p95.empty:
            ax.fill_between(p05.index, p05.values, p95.values, alpha=BAND_ALPHA,
                            color=ENSEMBLE_BAND_COLOR, zorder=1, label=ENSEMBLE_BAND_LABEL)
        ax.plot(mean.index, mean.values,
                linewidth=ENSEMBLE_MEAN_LW,
                color=ENSEMBLE_MEAN_COLOR,
                label=ENSEMBLE_MEAN_LABEL,
                zorder=3)

    # Open loop (if present): black and on top
    if ol_proc is not None and var in ol_proc.columns:
        ax.plot(ol_proc.index, ol_proc[var], linewidth=LW_OPEN, color=OPEN_LOOP_COLOR,
                label="open loop", zorder=4)

    # ---- Titles / labels ----
    units = OA_VAR_UNITS_MAP.get(var, "")
    pretty = OA_VAR_PRETTY_MAP.get(var, var)
    var_title = f"{pretty} ({units})" if units else pretty

    main_title = "openAMUNDSEN Ensemble results with perturbed forcing"
    subtitle = f"{station_name}{alt_str} — {var_title}"

    ax.set_xlabel("Time")
    ax.set_ylabel(var_title)
    ax.grid(True)

    # Legend
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=True,
        title=var_title,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    if leg:
        leg._legend_box.align = "left"

    _apply_xlim(ax, ol_proc.index if ol_proc is not None and isinstance(ol_proc.index, pd.DatetimeIndex) else None,
                start_ts, end_ts)

    # >>> Unified layout call <<<
    apply_titles_and_layout(fig, ax, main_title, subtitle)

    out_path = os.path.join(outdir, f"{station_token}_{var}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    logging.info("Wrote %s", out_path)


# ============================
# ====== MAIN EXECUTION ======
# ============================

def main() -> None:
    # Resolve window + output folders
    start_ts = _parse_ts(PLOT_START)
    end_ts   = _parse_ts(PLOT_END)
    win_suffix = _window_suffix(start_ts, end_ts)
    
    # Create separate output directories for each plot type
    base_dir = os.path.join(OUTPUT_ROOT, f"plots_comprehensive_{win_suffix}")
    plots_dir = _ensure_unique_dir(base_dir)
    os.makedirs(plots_dir, exist_ok=True)

    forcing_dir = os.path.join(plots_dir, "forcing")
    oa_dir = os.path.join(plots_dir, "oa_results")
    
    if PLOT_FORCING:
        os.makedirs(forcing_dir, exist_ok=True)
    if PLOT_OA_RESULTS:
        os.makedirs(oa_dir, exist_ok=True)

    _setup_logging(plots_dir)
    logging.info("OUTPUT_ROOT: %s", os.path.abspath(OUTPUT_ROOT))
    logging.info("Plots will be written to: %s", os.path.abspath(plots_dir))
    logging.info("Plot window: [%s .. %s]", PLOT_START or "-", PLOT_END or "-")
    logging.info("Plot forcing: %s, Plot OA results: %s", PLOT_FORCING, PLOT_OA_RESULTS)

    # Load stations metadata
    stations_map = _load_stations(STATIONS_CSV_PATH)

    # Discover members + perturbation info
    members_dirs = list_member_dirs(OUTPUT_ROOT)
    logging.info("Found %d member folders.", len(members_dirs))
    member_info = _read_member_perturb_info_once(OUTPUT_ROOT, members_dirs)

    # ====== PLOT FORCING DATA ======
    if PLOT_FORCING:
        logging.info("Processing forcing data...")
        station_files = list_open_loop_stations(OUTPUT_ROOT, stations_csv_name="stations.csv")
        if MAX_STATIONS is not None:
            station_files = station_files[:MAX_STATIONS]
        logging.info("Found %d station CSVs in open_loop/meteo.", len(station_files))

        for fname in station_files:
            ol_path = os.path.join(OUTPUT_ROOT, "open_loop", "meteo", fname)
            ol_df, _, temp_col, prec_col = read_station_series(ol_path, TIME_COL, TEMP_COL, PRECIP_COL)
            if not isinstance(ol_df.index, pd.DatetimeIndex):
                logging.warning("Skipping %s: cannot parse time index.", fname)
                continue

            mem_dfs = collect_member_series(fname, members_dirs)
            if not mem_dfs:
                logging.warning("No member dataframes parsed for %s; plotting open loop only.", fname)

            plot_forcing_station(
                station_file=fname,
                ol_df=ol_df,
                mem_dfs=mem_dfs,
                members_dirs=members_dirs,
                member_info=member_info,
                temp_col=temp_col,
                prec_col=prec_col,
                outdir=forcing_dir,
                start_ts=start_ts,
                end_ts=end_ts,
                stations_map=stations_map
            )

    # ====== PLOT OA RESULTS ======
    if PLOT_OA_RESULTS:
        logging.info("Processing OA results...")
        point_files = _list_oa_point_files(OUTPUT_ROOT)
        if MAX_STATIONS is not None:
            point_files = point_files[:MAX_STATIONS]
        logging.info("Found %d point CSVs to plot.", len(point_files))

        for fname in point_files:
            # Open loop (if exists)
            ol_path = os.path.join(OUTPUT_ROOT, "open_loop", "results", fname)
            ol_df = _read_oa_point_csv(ol_path, OA_VAR_NAME) if os.path.isfile(ol_path) else None

            # Member frames
            mem_frames: List[Tuple[str, Optional[pd.DataFrame]]] = []
            for md in members_dirs:
                mp = os.path.join(OUTPUT_ROOT, md, "results", fname)
                if not os.path.isfile(mp):
                    logging.warning("Missing in %s/results: %s", md, fname)
                    mem_frames.append((md, None))
                    continue
                mem_frames.append((md, _read_oa_point_csv(mp, OA_VAR_NAME)))

            plot_oa_point_file(
                point_file=fname,
                ol_df=ol_df,
                mem_dfs=mem_frames,
                member_info=member_info,
                stations_map=stations_map,
                var=OA_VAR_NAME,
                outdir=oa_dir,
                start_ts=start_ts,
                end_ts=end_ts
            )

    logging.info("DONE. Output in: %s", os.path.abspath(plots_dir))
    if PLOT_FORCING:
        logging.info("Forcing plots: %s", os.path.abspath(forcing_dir))
    if PLOT_OA_RESULTS:
        logging.info("OA results plots: %s", os.path.abspath(oa_dir))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Fatal error: %s", e)
        sys.exit(1)
