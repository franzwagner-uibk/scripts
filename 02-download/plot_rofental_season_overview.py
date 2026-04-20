from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import boto3
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


DEFAULT_DATA_ROOT = Path(
    r"C:\Users\franz\Nextcloud\PhD\openamundsen_da\openamundsen-examples\data\rofental"
)
DEFAULT_OUT_DIR = Path(r"C:\Users\franz\Nextcloud\PhD\02-Daten\rofental_2024_2025_plots")
DEFAULT_START = "2024-10-01"
DEFAULT_END = "2025-07-31"
DEFAULT_STATIONS = ("bellavista", "latschbloder", "proviantdepot")
DEFAULT_TILES = ("32TPS", "32TPT")
DEFAULT_PRODUCTS = ("FSC", "SWS")

DEFAULT_ENDPOINT = "https://s3.WAW3-2.cloudferro.com"
DEFAULT_BUCKET = "HRWSI"
DEFAULT_ACCESS_KEY = "c4ae60af7b144053803c618a8860f7c9"
DEFAULT_SECRET_KEY = "dcb3ba1f6eab45aaaec5802feef5e2e4"


@dataclass(frozen=True)
class SeasonWindow:
    start_dt: datetime
    end_dt: datetime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Rofental season plots for meteo, snow validation, and Copernicus availability."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=DEFAULT_END, help="YYYY-MM-DD")
    parser.add_argument("--stations", nargs="+", default=list(DEFAULT_STATIONS))
    parser.add_argument("--tiles", nargs="+", default=list(DEFAULT_TILES))
    parser.add_argument("--products", nargs="+", default=list(DEFAULT_PRODUCTS))
    return parser.parse_args()


def _season_window(start_date: str, end_date: str) -> SeasonWindow:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23)
    if start > end:
        raise ValueError("start_date is after end_date")
    return SeasonWindow(start_dt=start, end_dt=end)


def _to_float(value: str) -> float | None:
    txt = str(value).strip()
    if txt == "":
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _read_meteo_station(path: Path, window: SeasonWindow) -> tuple[list[datetime], list[float], list[tuple[datetime, float]]]:
    temps_by_day: dict[date, list[float]] = defaultdict(list)
    precip_hourly: list[tuple[datetime, float]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = str(row.get("Date and time", "")).strip()
            if ts == "":
                continue
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            if dt < window.start_dt or dt > window.end_dt:
                continue

            temp = _to_float(row.get("temp", ""))
            if temp is not None:
                temps_by_day[dt.date()].append(temp)

            precip = _to_float(row.get("precip", ""))
            if precip is not None:
                precip_hourly.append((dt, precip))

    daily_dates = sorted(temps_by_day.keys())
    daily_means = [sum(vals) / len(vals) for d in daily_dates if (vals := temps_by_day[d])]
    # Rebuild dates to stay aligned after comprehension filtering.
    daily_dates = [d for d in daily_dates if len(temps_by_day[d]) > 0]

    precip_hourly.sort(key=lambda x: x[0])
    return (
        [datetime(d.year, d.month, d.day) for d in daily_dates],
        daily_means,
        precip_hourly,
    )


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


def _read_validation_station(
    path: Path,
    window: SeasonWindow,
) -> tuple[list[datetime], list[float], list[datetime], list[float]]:
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
            if dt < window.start_dt or dt > window.end_dt:
                continue

            depth_vals = [_to_float(row.get(col, "")) for col in depth_cols]
            depth_vals = [v for v in depth_vals if v is not None]
            if depth_vals:
                depth_by_day[dt.date()].append(sum(depth_vals) / len(depth_vals))

            swe_vals = [_to_float(row.get(col, "")) for col in swe_cols]
            swe_vals = [v for v in swe_vals if v is not None]
            if swe_vals:
                swe_by_day[dt.date()].append(sum(swe_vals) / len(swe_vals))

    depth_dates = sorted(depth_by_day.keys())
    depth_series = [sum(depth_by_day[d]) / len(depth_by_day[d]) for d in depth_dates]
    swe_dates = sorted(swe_by_day.keys())
    swe_series = [sum(swe_by_day[d]) / len(swe_by_day[d]) for d in swe_dates]

    return (
        [datetime(d.year, d.month, d.day) for d in depth_dates],
        depth_series,
        [datetime(d.year, d.month, d.day) for d in swe_dates],
        swe_series,
    )


def _month_iter(start: datetime, end: datetime) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    y = start.year
    m = start.month
    while (y < end.year) or (y == end.year and m <= end.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _copernicus_dates(
    *,
    window: SeasonWindow,
    tiles: list[str],
    products: list[str],
) -> dict[str, list[datetime]]:
    client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("HRWSI_ACCESS_KEY", DEFAULT_ACCESS_KEY),
        aws_secret_access_key=os.getenv("HRWSI_SECRET_KEY", DEFAULT_SECRET_KEY),
        endpoint_url=os.getenv("HRWSI_ENDPOINT_URL", DEFAULT_ENDPOINT),
    )

    found: dict[str, set[date]] = {p: set() for p in products}
    months = _month_iter(window.start_dt, window.end_dt)

    for product in products:
        for tile in tiles:
            for year, month in months:
                prefix = f"{product}/{tile}/{year:04d}/{month:02d}/"
                paginator = client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=DEFAULT_BUCKET, Prefix=prefix):
                    for obj in page.get("Contents", []):
                        parts = obj["Key"].split("/")
                        if len(parts) < 5:
                            continue
                        try:
                            dt = date(int(parts[2]), int(parts[3]), int(parts[4]))
                        except ValueError:
                            continue
                        if window.start_dt.date() <= dt <= window.end_dt.date():
                            found[product].add(dt)

    return {
        p: [datetime(d.year, d.month, d.day) for d in sorted(found[p])]
        for p in products
    }


def _plot_temperature(
    *,
    station_series: dict[str, tuple[list[datetime], list[float], list[tuple[datetime, float]]]],
    window: SeasonWindow,
    out_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    for station, (dates, temps, _precip) in station_series.items():
        ax.plot(dates, temps, label=station, linewidth=1.2)
    ax.set_title("Rofental Daily Mean Temperature")
    ax.set_ylabel("Temperature [degC]")
    ax.set_xlim(window.start_dt, window.end_dt)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def _plot_cumulative_precip(
    *,
    station_series: dict[str, tuple[list[datetime], list[float], list[tuple[datetime, float]]]],
    window: SeasonWindow,
    out_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    for station, (_dates, _temps, precip_hourly) in station_series.items():
        cum_dates: list[datetime] = []
        cum_vals: list[float] = []
        total = 0.0
        for dt, val in precip_hourly:
            total += val
            cum_dates.append(dt)
            cum_vals.append(total)
        if cum_dates:
            ax.plot(cum_dates, cum_vals, label=station, linewidth=1.2)

    ax.set_title("Rofental Cumulative Precipitation")
    ax.set_ylabel("Cumulative Precipitation [mm]")
    ax.set_xlim(window.start_dt, window.end_dt)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def _plot_snow_depth_swe(
    *,
    valid_series: dict[str, tuple[list[datetime], list[float], list[datetime], list[float]]],
    window: SeasonWindow,
    out_file: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax_depth, ax_swe = axes

    for station, (depth_dates, depth_vals, swe_dates, swe_vals) in valid_series.items():
        if depth_dates:
            ax_depth.plot(depth_dates, depth_vals, label=station, linewidth=1.2)
        if swe_dates:
            ax_swe.plot(swe_dates, swe_vals, label=station, linewidth=1.2)

    ax_depth.set_title("Rofental Snow Depth (Validation, Daily Mean)")
    ax_depth.set_ylabel("Snow Depth [mm]")
    ax_depth.grid(alpha=0.3)
    ax_depth.legend()

    ax_swe.set_title("Rofental SWE (Validation, Daily Mean)")
    ax_swe.set_ylabel("SWE [mm]")
    ax_swe.set_xlim(window.start_dt, window.end_dt)
    ax_swe.grid(alpha=0.3)
    ax_swe.legend()
    ax_swe.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_swe.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def _plot_copernicus_observation_dates(
    *,
    obs_dates: dict[str, list[datetime]],
    window: SeasonWindow,
    out_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    products = list(obs_dates.keys())
    y_levels = {p: idx for idx, p in enumerate(products)}

    for product in products:
        dates = obs_dates[product]
        ys = [y_levels[product]] * len(dates)
        ax.scatter(dates, ys, s=18, label=product)

    ax.set_title("Copernicus Observation Dates (Rofental Tiles)")
    ax.set_xlim(window.start_dt, window.end_dt)
    ax.set_yticks(list(y_levels.values()))
    ax.set_yticklabels(products)
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    window = _season_window(args.start_date, args.end_date)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stations = [str(s).strip() for s in args.stations if str(s).strip()]
    station_series: dict[str, tuple[list[datetime], list[float], list[tuple[datetime, float]]]] = {}
    valid_series: dict[str, tuple[list[datetime], list[float], list[datetime], list[float]]] = {}

    for station in stations:
        meteo_path = Path(args.data_root) / "meteo" / "csv" / f"{station}.csv"
        valid_path = Path(args.data_root) / "validation" / f"{station}_valid.csv"
        station_series[station] = _read_meteo_station(meteo_path, window)
        valid_series[station] = _read_validation_station(valid_path, window)

    tiles = [str(t).strip().upper() for t in args.tiles if str(t).strip()]
    products = [str(p).strip().upper() for p in args.products if str(p).strip()]
    obs_dates = _copernicus_dates(window=window, tiles=tiles, products=products)

    _plot_temperature(
        station_series=station_series,
        window=window,
        out_file=out_dir / "rofental_2024_2025_temperature.png",
    )
    _plot_cumulative_precip(
        station_series=station_series,
        window=window,
        out_file=out_dir / "rofental_2024_2025_cumulative_precip.png",
    )
    _plot_snow_depth_swe(
        valid_series=valid_series,
        window=window,
        out_file=out_dir / "rofental_2024_2025_snow_depth_swe.png",
    )
    _plot_copernicus_observation_dates(
        obs_dates=obs_dates,
        window=window,
        out_file=out_dir / "rofental_2024_2025_copernicus_obs_dates.png",
    )

    for product, dates in obs_dates.items():
        print(f"{product}: {len(dates)} dates")
    print(f"Done. Plots written to: {out_dir}")


if __name__ == "__main__":
    main()
