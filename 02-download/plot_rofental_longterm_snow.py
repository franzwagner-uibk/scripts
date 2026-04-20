from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


DEFAULT_DATA_ROOT = Path(
    r"C:\Users\franz\Nextcloud\PhD\openamundsen_da\openamundsen-examples\data\rofental"
)
DEFAULT_OUT_DIR = Path(r"C:\Users\franz\Nextcloud\PhD\02-Daten\rofental_longterm_snow_plots")
DEFAULT_STATIONS = ("bellavista", "latschbloder", "proviantdepot")
DEFAULT_HIGHLIGHT_SEASON = "2024-2025"


@dataclass(frozen=True)
class StationSeries:
    depth_daily: dict[date, float]
    swe_daily: dict[date, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create long-term snow depth/SWE plots for Rofental validation data, "
            "including an Oct-Jul season overlay."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stations", nargs="+", default=list(DEFAULT_STATIONS))
    parser.add_argument("--highlight-season", type=str, default=DEFAULT_HIGHLIGHT_SEASON)
    return parser.parse_args()


def _to_float(value: str) -> float | None:
    txt = str(value).strip()
    if txt == "":
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _snow_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    depth_cols: list[str] = []
    swe_cols: list[str] = []
    for col in columns:
        low = col.lower()
        if "snow depth" in low:
            depth_cols.append(col)
        if ("snow water equivalent" in low) or ("swe" in low):
            swe_cols.append(col)
    return depth_cols, swe_cols


def _read_station_validation(path: Path) -> StationSeries:
    depth_by_day: dict[date, list[float]] = defaultdict(list)
    swe_by_day: dict[date, list[float]] = defaultdict(list)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        depth_cols, swe_cols = _snow_columns(columns)
        for row in reader:
            ts = str(row.get("Date and time", "")).strip()
            if ts == "":
                continue
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            d = dt.date()

            depth_vals = [_to_float(row.get(col, "")) for col in depth_cols]
            depth_vals = [v for v in depth_vals if v is not None]
            if depth_vals:
                depth_by_day[d].append(sum(depth_vals) / len(depth_vals))

            swe_vals = [_to_float(row.get(col, "")) for col in swe_cols]
            swe_vals = [v for v in swe_vals if v is not None]
            if swe_vals:
                swe_by_day[d].append(sum(swe_vals) / len(swe_vals))

    depth_daily = {d: (sum(vals) / len(vals)) for d, vals in depth_by_day.items() if vals}
    swe_daily = {d: (sum(vals) / len(vals)) for d, vals in swe_by_day.items() if vals}
    return StationSeries(depth_daily=depth_daily, swe_daily=swe_daily)


def _season_label(d: date) -> str:
    if d.month >= 10:
        return f"{d.year}-{d.year + 1}"
    return f"{d.year - 1}-{d.year}"


def _season_day_idx(d: date) -> int:
    start_year = d.year if d.month >= 10 else d.year - 1
    season_start = date(start_year, 10, 1)
    return (d - season_start).days + 1


def _network_daily_median(series_by_station: dict[str, StationSeries], key: str) -> dict[date, float]:
    # key is one of {"depth_daily", "swe_daily"}
    per_day_vals: dict[date, list[float]] = defaultdict(list)
    for station_series in series_by_station.values():
        daily = getattr(station_series, key)
        for d, v in daily.items():
            per_day_vals[d].append(v)

    out: dict[date, float] = {}
    for d, vals in per_day_vals.items():
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        if n == 0:
            continue
        if n % 2 == 1:
            out[d] = vals_sorted[n // 2]
        else:
            out[d] = 0.5 * (vals_sorted[n // 2 - 1] + vals_sorted[n // 2])
    return out


def _plot_longterm_station_series(
    *,
    series_by_station: dict[str, StationSeries],
    variable: str,
    ylabel: str,
    out_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    for station, station_series in series_by_station.items():
        daily = station_series.depth_daily if variable == "depth" else station_series.swe_daily
        dates = sorted(daily.keys())
        vals = [daily[d] for d in dates]
        if not dates:
            continue
        x = [datetime(d.year, d.month, d.day) for d in dates]
        ax.plot(x, vals, label=station, linewidth=0.9, alpha=0.9)

    ax.set_title(f"Rofental Long-Term {ylabel} (validation daily mean)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def _plot_season_overlay(
    *,
    network_daily: dict[date, float],
    ylabel: str,
    highlight_season: str,
    out_file: Path,
) -> None:
    by_season: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for d, v in network_daily.items():
        s = _season_label(d)
        by_season[s].append((_season_day_idx(d), v))

    for s in list(by_season.keys()):
        by_season[s] = sorted(by_season[s], key=lambda x: x[0])

    seasons = sorted(by_season.keys())
    fig, ax = plt.subplots(figsize=(12, 5))

    for s in seasons:
        points = by_season[s]
        if len(points) < 50:
            continue
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        if s == highlight_season:
            ax.plot(x, y, color="crimson", linewidth=2.0, label=f"{s} (highlight)")
        else:
            ax.plot(x, y, color="0.6", linewidth=0.8, alpha=0.5)

    ax.set_title(f"{ylabel} Season Overlay (Oct-Jul network median)")
    ax.set_xlabel("Day of hydrological season (Oct 1 = day 1)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, 304)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stations = [str(s).strip() for s in args.stations if str(s).strip()]
    series_by_station: dict[str, StationSeries] = {}
    for station in stations:
        valid_file = Path(args.data_root) / "validation" / f"{station}_valid.csv"
        series_by_station[station] = _read_station_validation(valid_file)

    _plot_longterm_station_series(
        series_by_station=series_by_station,
        variable="depth",
        ylabel="Snow Depth [mm]",
        out_file=out_dir / "rofental_longterm_snow_depth_daily.png",
    )
    _plot_longterm_station_series(
        series_by_station=series_by_station,
        variable="swe",
        ylabel="SWE [mm]",
        out_file=out_dir / "rofental_longterm_swe_daily.png",
    )

    network_depth = _network_daily_median(series_by_station, "depth_daily")
    network_swe = _network_daily_median(series_by_station, "swe_daily")

    _plot_season_overlay(
        network_daily=network_depth,
        ylabel="Snow Depth [mm]",
        highlight_season=str(args.highlight_season),
        out_file=out_dir / "rofental_depth_overlay_oct_jul.png",
    )
    _plot_season_overlay(
        network_daily=network_swe,
        ylabel="SWE [mm]",
        highlight_season=str(args.highlight_season),
        out_file=out_dir / "rofental_swe_overlay_oct_jul.png",
    )

    print(f"Done. Plots written to: {out_dir}")


if __name__ == "__main__":
    main()
