#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: plot_forcing_ensemble.py
Author: Franz Wagner (+ ChatGPT assist)
Date: 2025-10-09 (Europe/Vienna)

Description
-----------
Compare unperturbed 'open_loop' meteorology with perturbed ensemble members created by
`build_forcing_ensemble.py`.

What's in here
--------------
- Time-windowed plots (inclusive PLOT_START .. PLOT_END).
- Precip shown as hydrological-year cumulative (reset Oct 1 by default).
- Distinct member colors; ensemble mean & 5–95% band.
- Legend title carries variable + units; member entries show perturbations (ΔT, f_p).
- Legend placed below the plot in 4 columns (won’t cover lines).
- Open-loop line is always black/thick/top.
- Skips figures where all values in window are NaN.
- Adds station altitude (from stations.csv) to the plot subtitle.

Dependencies: pandas, numpy, matplotlib
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

# Path to stations metadata CSV.
# If None, defaults to "<OUTPUT_ROOT>/open_loop/meteo/stations.csv"
STATIONS_CSV_PATH: Optional[str] = None

# Column names (set to None to autodetect)
TIME_COL:   Optional[str] = "date"
TEMP_COL:   Optional[str] = "temp"
PRECIP_COL: Optional[str] = "precip"

# Autodetection patterns (case-insensitive)
TIME_PATTERNS  = [r"time", r"date", r"datetime", r"timestamp"]
TEMP_PATTERNS  = [r"^temp(erature)?$", r"^ta$", r"^t$", r"^t2m$", r"air.?temp", r"^tt$", r"^tg$"]
PREC_PATTERNS  = [r"^precip(it(a|)tion)?$", r"^psum$", r"^rr$", r"^rrr$", r"^pr(cp)?$", r"^p$", r"^rf$", r"niederschlag", r"^ppt$", r"^rain$"]

# Plot window (inclusive); None -> full span
PLOT_START: Optional[str] = "2017-10-01 00:00:00"
PLOT_END:   Optional[str] = "2018-09-30 23:59:59"

# Hydrological year for cumulative precipitation
HYDRO_START_MONTH: int = 10
HYDRO_START_DAY:   int = 1
PRECIP_CUM_SUFFIX: str = "_cum"  # internal only

# Optional resampling / smoothing
RESAMPLE_RULE: Optional[str] = None  # e.g. 'D', 'W', None
AGG_TEMP: str = "mean"
AGG_PREC: str = "sum"
ROLLING_WINDOW: Optional[int] = None  # e.g. 24 for 24-step rolling

# Plot style
FIGSIZE = (14, 7)
ALPHA_MEMBERS = 0.28
LW_OPEN = 1.4
LW_MEMBER = 1.0
LW_MEAN = 1.4
SHOW_ENS_BAND = True
BAND_ALPHA = 0.7          # 70% opacity for ensemble band
BAND_COLOR = "#5A96E8"    # ensemble band color (your blue)
ANNOTATE_PERTURBATIONS_IN_LEGEND = True  # include ΔT/f_p next to each member in legend

# Distinct member colors
COLOR_CYCLE = [
    "#00429d", "#4771b2", "#73a2c6", "#94c4d9", "#ffa600",
    "#ff6361", "#bc5090", "#58508d", "#003f5c", "#2f4b7c",
    "#7a5195", "#ef5675", "#ffa600", "#d45087", "#1f77b4",
    "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
]

# Limits / selection
MAX_STATIONS: Optional[int] = None

# Logging
LOG_NAME = "plot_forcing_ensemble.log"


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

def maybe_resample_and_smooth(df: pd.DataFrame,
                              tcol: Optional[str],
                              pcol: Optional[str]) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    if RESAMPLE_RULE:
        agg: Dict[str, str] = {}
        if tcol and tcol in df.columns: agg[tcol] = AGG_TEMP
        if pcol and pcol in df.columns: agg[pcol] = AGG_PREC
        if agg:
            df = df.resample(RESAMPLE_RULE).agg(agg)
    if ROLLING_WINDOW and ROLLING_WINDOW > 1:
        for c in [tcol, pcol]:
            if c and c in df.columns:
                df[c] = df[c].rolling(ROLLING_WINDOW, min_periods=1).mean()
    return df

def _parse_ts(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s: return None
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        logging.warning("Invalid timestamp for plot window: %s", s)
        return None
    return ts

def apply_window(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    if start is not None: df = df[df.index >= start]
    if end   is not None: df = df[df.index <= end]
    return df

def _series_has_data(s: pd.Series) -> bool:
    return pd.to_numeric(s, errors="coerce").notna().any() if s is not None else False

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

def envelope(dfs: List[pd.DataFrame], col: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if not dfs: return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    aligned = pd.concat([d[[col]] for d in dfs if col in d.columns], axis=1, join="inner")
    if aligned.empty: return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    return aligned.mean(axis=1), aligned.quantile(0.05, axis=1), aligned.quantile(0.95, axis=1)

def _member_id(folder: str) -> str:
    m = re.match(r"member_(\d{3})$", folder)
    return f"m{m.group(1)}" if m else folder

def read_member_perturb_info_once(members: List[str]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Return {member_dir: (delta_t, f_p)} from INFO.txt; missing -> (None, None)."""
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for md in members:
        info_path = os.path.join(OUTPUT_ROOT, md, "INFO.txt")
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

def _ensure_unique_dir(base_dir: str) -> str:
    if not os.path.exists(base_dir): return base_dir
    k = 2
    while True:
        cand = f"{base_dir}_v{k}"
        if not os.path.exists(cand): return cand
        k += 1

def _window_suffix(start_ts: Optional[pd.Timestamp], end_ts: Optional[pd.Timestamp]) -> str:
    if start_ts is None and end_ts is None: return "full-span"
    fmt = lambda ts: ts.strftime("%m-%d-%y") if ts is not None else "..."
    return f"{fmt(start_ts)}_{fmt(end_ts)}"

def _apply_xlim(ax: plt.Axes, idx: Optional[pd.DatetimeIndex], start_ts: Optional[pd.Timestamp], end_ts: Optional[pd.Timestamp]) -> None:
    if start_ts is None and end_ts is None: return
    xmin = start_ts if start_ts is not None else (idx.min() if idx is not None and len(idx) else None)
    xmax = end_ts   if end_ts   is not None else (idx.max() if idx is not None and len(idx) else None)
    try: ax.set_xlim(xmin, xmax)
    except Exception as ex: logging.warning("Failed to set x-limits: %s", ex)

def _color_cycler() -> cycler.Cycler:
    return cycler(color=COLOR_CYCLE)

def _any_data_in_window(ol_df: pd.DataFrame, mem_dfs: List[pd.DataFrame], col: str) -> bool:
    series = []
    if col in ol_df.columns: series.append(ol_df[col])
    for d in mem_dfs:
        if col in d.columns: series.append(d[col])
    return any(_series_has_data(s) for s in series)

def _legend_below(ax: plt.Axes, title: str, ncol: int = 4) -> None:
    """
    Place legend below the plot in n columns, with a title.
    Tighter layout version: less whitespace below the legend.
    """
    # Reserve slightly less space below the figure
    plt.subplots_adjust(bottom=0.14)

    # Draw legend a bit closer to the x-axis
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),   # smaller negative offset = closer to axis
        ncol=ncol,
        frameon=True,
        title=title,
        columnspacing=1.2,
        handletextpad=0.5
    )

    # Align legend items to the left edge of the figure
    if leg:
        leg._legend_box.align = "left"


# ======== STATIONS / ALTITUDE LOOKUP (NEW) ========

def _resolve_stations_csv_path() -> Optional[str]:
    """Return explicit STATIONS_CSV_PATH or default to <OUTPUT_ROOT>/open_loop/meteo/stations.csv."""
    if STATIONS_CSV_PATH:
        return STATIONS_CSV_PATH
    cand = os.path.join(OUTPUT_ROOT, "open_loop", "meteo", "stations.csv")
    return cand if os.path.isfile(cand) else None

def _load_stations(csv_path: Optional[str]) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Read stations CSV and return mapping by both ID and NAME for robust lookup.

    Expected columns:
      - Either ['id','name','alt'] OR ['name','alt'] (case-insensitive).
    Keys are lowercased (id and/or name). Values: {'name': <str>, 'alt': <float or None>}.
    """
    mapping: Dict[str, Dict[str, Optional[float]]] = {}
    if not csv_path:
        logging.info("No stations CSV found; titles will show file token only.")
        return mapping
    try:
        df = pd.read_csv(csv_path)
    except Exception as ex:
        logging.warning("Failed to read stations CSV '%s': %s", csv_path, ex)
        return mapping

    cols = {c.lower().strip(): c for c in df.columns}
    has_id = "id" in cols
    has_name = "name" in cols
    has_alt = "alt" in cols or "elev" in cols or "altitude" in cols

    if not has_alt:
        logging.warning("Stations CSV lacks an altitude column; looked for 'alt', 'elev', or 'altitude'.")
        return mapping

    alt_key = cols.get("alt") or cols.get("elev") or cols.get("altitude")

    for _, row in df.iterrows():
        name_val = str(row[cols["name"]]).strip() if has_name else None
        try:
            alt_val = float(row[alt_key]) if pd.notna(row[alt_key]) else None
        except Exception:
            alt_val = None
        rec = {"name": name_val, "alt": alt_val}

        if has_name and name_val:
            mapping[name_val.lower()] = rec
        if has_id and pd.notna(row[cols["id"]]):
            key_id = str(row[cols["id"]]).strip().lower()
            mapping[key_id] = rec

    logging.info("Loaded %d station metadata entries from %s", len(mapping), csv_path)
    return mapping

def _lookup_station_info(stations_map: Dict[str, Dict[str, Optional[float]]],
                         station_token: str) -> Dict[str, Optional[str | float]]:
    """
    Try by id first, then by name (case-insensitive).
    station_token is derived from filename '<token>.csv'.
    """
    key = station_token.strip().lower()
    rec = stations_map.get(key)
    if rec:
        return rec
    return {"name": station_token, "alt": None}


# ============= PLOTTING PER STATION (TEMP + PRECIP) =============

def plot_station(station_file: str,
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

    # Station meta (name + altitude)
    st = _lookup_station_info(stations_map, station_token)
    station_name = st.get("name") or station_token
    station_alt = st.get("alt")
    alt_str = f" ({int(round(station_alt))} m)" if isinstance(station_alt, (int, float)) else ""

    # resample/smooth
    ol_proc = maybe_resample_and_smooth(ol_df.copy(), temp_col, prec_col)
    mem_proc = [maybe_resample_and_smooth(d.copy(), temp_col, prec_col) for d in mem_dfs]

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
                    d[temp_col].plot(ax=ax, linewidth=LW_MEMBER, alpha=ALPHA_MEMBERS, label=lab)

            # Ensemble mean & band
            mean, p05, p95 = envelope(mem_proc, temp_col)
            if not mean.empty:
                if SHOW_ENS_BAND and not p05.empty and not p95.empty:
                    ax.fill_between(p05.index, p05.values, p95.values,
                                    alpha=BAND_ALPHA, color=BAND_COLOR, label="ensemble 5–95%")
                mean.plot(ax=ax, linewidth=LW_MEAN, label="ensemble mean")

            # Open-loop (black)
            ol_proc[temp_col].plot(ax=ax, linewidth=LW_OPEN, color="black", label="open loop", zorder=10)

            # Titles / labels / legend
            fig.suptitle("Comparison of Open-Loop and Perturbed Forcing Ensembles",
                         fontsize=14, fontweight="bold", y=0.97)  # slightly tighter gap
            ax.set_title(f"{station_name}{alt_str} — Air Temperature (K)", fontsize=11, pad=3)

            ax.set_xlabel("Time")
            ax.set_ylabel("Air temperature (K)")
            _legend_below(ax, title="Air temperature (K)", ncol=4)
            ax.grid(True)

            _apply_xlim(ax, ol_proc.index if isinstance(ol_proc.index, pd.DatetimeIndex) else None, start_ts, end_ts)
            fig.tight_layout(rect=[0, 0.002, 1, 1])  # tighter bottom margin
            out_path = os.path.join(outdir, f"{station_token}_temp.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            logging.info("Wrote %s", out_path)
    else:
        logging.warning("Skipping temperature plot for %s: temp column not found.", station_file)

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
                    d[cum_col].plot(ax=ax, linewidth=LW_MEMBER, alpha=ALPHA_MEMBERS, label=lab)

            # Ensemble mean & band
            mean, p05, p95 = envelope(mem_cum_only, cum_col)
            if not mean.empty:
                if SHOW_ENS_BAND and not p05.empty and not p95.empty:
                    ax.fill_between(p05.index, p05.values, p95.values,
                                    alpha=BAND_ALPHA, color=BAND_COLOR, label="ensemble 5–95%")
                mean.plot(ax=ax, linewidth=LW_MEAN, label="ensemble mean")

            # Open-loop cumulative (black)
            if cum_col in ol_cum.columns:
                ol_cum[cum_col].plot(ax=ax, linewidth=LW_OPEN, color="black", label="open loop", zorder=10)

            # Titles / labels / legend
            fig.suptitle("Comparison of Open-Loop and Perturbed Forcing Ensembles",
                         fontsize=14, fontweight="bold", y=0.97)
            ax.set_title(f"{station_name}{alt_str} — Cumulative precipitation (mm)", fontsize=11, pad=3)
            ax.set_xlabel("Time")
            ax.set_ylabel("Cumulative precipitation (mm)")
            _legend_below(ax, title="Cumulative precipitation (mm)", ncol=4)
            ax.grid(True)

            _apply_xlim(ax, ol_cum.index if isinstance(ol_cum.index, pd.DatetimeIndex) else None, start_ts, end_ts)
            fig.tight_layout(rect=[0, 0.06, 1, 1])  # tighter bottom margin
            out_path = os.path.join(outdir, f"{station_token}_precip_cum.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            logging.info("Wrote %s", out_path)
    else:
        logging.warning("Skipping precipitation plot for %s: precip column not found.", station_file)


# ============================
# ====== MAIN EXECUTION ======
# ============================

def main() -> None:
    # Resolve window + output folder (non-overwriting)
    start_ts = _parse_ts(PLOT_START)
    end_ts   = _parse_ts(PLOT_END)
    win_suffix = _window_suffix(start_ts, end_ts)
    base_plots_dir = os.path.join(OUTPUT_ROOT, f"plots_{win_suffix}")
    plots_dir = _ensure_unique_dir(base_plots_dir)
    os.makedirs(plots_dir, exist_ok=True)

    _setup_logging(plots_dir)
    logging.info("OUTPUT_ROOT: %s", os.path.abspath(OUTPUT_ROOT))
    logging.info("Plots will be written to: %s", os.path.abspath(plots_dir))
    logging.info("Plot window (validated): [%s .. %s]",
                 start_ts.isoformat() if start_ts else "-",
                 end_ts.isoformat() if end_ts else "-")

    # Discover members + INFO.txt metrics once
    members_dirs = list_member_dirs(OUTPUT_ROOT)
    logging.info("Found %d member folders.", len(members_dirs))
    member_info = read_member_perturb_info_once(members_dirs)

    # Discover station files
    station_files = list_open_loop_stations(OUTPUT_ROOT, stations_csv_name="stations.csv")
    if MAX_STATIONS is not None:
        station_files = station_files[:MAX_STATIONS]
        logging.info("Processing first %d stations due to MAX_STATIONS.", len(station_files))
    logging.info("Found %d station CSVs in open_loop/meteo.", len(station_files))

    # Load stations CSV for altitude/name
    stations_csv_path = _resolve_stations_csv_path()
    if stations_csv_path is None:
        logging.info("Stations CSV not found automatically; set STATIONS_CSV_PATH if available.")
    stations_map = _load_stations(stations_csv_path)

    # Process each station
    for fname in station_files:
        ol_path = os.path.join(OUTPUT_ROOT, "open_loop", "meteo", fname)
        ol_df, _, temp_col, prec_col = read_station_series(ol_path, TIME_COL, TEMP_COL, PRECIP_COL)
        if not isinstance(ol_df.index, pd.DatetimeIndex):
            logging.warning("Skipping %s: cannot parse time index.", fname)
            continue

        mem_dfs = collect_member_series(fname, members_dirs)
        if not mem_dfs:
            logging.warning("No member dataframes parsed for %s; plotting open loop only.", fname)

        plot_station(
            station_file=fname,
            ol_df=ol_df,
            mem_dfs=mem_dfs,
            members_dirs=members_dirs,
            member_info=member_info,
            temp_col=temp_col,
            prec_col=prec_col,
            outdir=plots_dir,
            start_ts=start_ts,
            end_ts=end_ts,
            stations_map=stations_map
        )

    logging.info("DONE. Compare PNGs in %s", os.path.abspath(plots_dir))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Fatal error: %s", e)
        sys.exit(1)
