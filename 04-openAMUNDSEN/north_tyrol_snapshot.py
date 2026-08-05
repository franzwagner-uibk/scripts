"""Build a provenance-complete, event-neutral North Tyrol data snapshot."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence


PINNED_IMAGE = (
    "ghcr.io/openamundsen/openamundsen-da:0.9.4@"
    "sha256:f3834a701e116b9ab11c50677d94236bffcd5d9adb045ae6b871b3ccf2c98723"
)
SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_VERSION = 1
DOMAIN = "north_tyrol_subdomains"
CRS = "EPSG:25832"
FORCING_BUFFER_M = 10_000.0
GRID_BUFFER_M = 10_000.0
FSC_PAD_M = 100.0
EXPECTED_SUBDOMAINS = 8
EXPECTED_FORCING_STATIONS = 161
EXPECTED_SNOW_STATIONS = 35
EXPECTED_FSC_COUNTS = {
    2017: 114,
    2018: 121,
    2019: 116,
    2020: 119,
    2021: 145,
    2022: 123,
}
REQUIRED_FORCING_VARIABLES = ("temp", "precip", "rel_hum", "sw_in", "wind_speed")
FSC_VALID_MIN = 0.0
FSC_VALID_MAX = 100.0
FSC_CLOUD_CLASSES = (205.0, 255.0)
FSC_WATER_CLASSES = (210.0,)
FSC_NODATA_CLASSES = (215.0,)


SOURCE_RELATIVE_PATHS = {
    "roi": Path("01-data/01-aoi/TESTSITES/Testsite_North_Tyrol.gpkg"),
    "raw_subregions": Path("01-data/01-aoi/SUBREGIONS/raw/subregions_avalanche_report_raw_25832.gpkg"),
    "forcing": Path("01-data/02-meteo/01-data/01-initial/openamundsen-v2"),
    "forcing_meta": Path("01-data/02-meteo/02-meta/gpkg/meta-all.gpkg"),
    "snow": Path("01-data/02-meteo/01-data/02-snow_obs/Tirol_snow_depth"),
    "fsc": Path("50-eurac/SCF_Eurac_v3/SCF_Eurac_v3"),
    "dem": Path("01-data/05-dem/euregio/dem_euregio_100.asc"),
    "landcover": Path(
        "01-data/03-landcover/lc_eusalp/openAMUNDSEN-euregio/lc_euregio_100_eusalp.asc"
    ),
    "srf": Path("01-data/06-srf/euregio/srf_euregio_100.asc"),
}


@dataclass(frozen=True)
class Season:
    """One complete October--September hydrological year."""

    start_year: int
    start: datetime
    end: datetime

    @property
    def name(self) -> str:
        """Return the project directory name."""

        return f"project_{self.start_year}_{self.start_year + 1}"


@dataclass(frozen=True)
class SourcePaths:
    """Resolved Fram3S source contracts."""

    root: Path
    roi: Path
    raw_subregions: Path
    forcing: Path
    forcing_meta: Path
    snow: Path
    fsc: Path
    dem: Path
    landcover: Path
    srf: Path


@dataclass(frozen=True)
class SnapshotOptions:
    """Validated snapshot build options."""

    source_root: Path
    target_root: Path
    start_year: int
    end_year: int
    resolution: int
    image: str

    @property
    def snapshot_name(self) -> str:
        """Return the deterministic versioned snapshot name."""

        return (
            f"north_tyrol_hydro_{self.start_year}_{self.end_year + 1}_"
            f"snapshot_v{SNAPSHOT_VERSION}"
        )

    @property
    def final_path(self) -> Path:
        """Return the final snapshot path."""

        return self.target_root / self.snapshot_name


@dataclass(frozen=True)
class AsciiGridHeader:
    """Six-line ESRI ASCII grid metadata."""

    ncols: int
    nrows: int
    xllcorner: float
    yllcorner: float
    cellsize: float
    nodata_value: str


def hydrological_seasons(start_year: int, end_year: int) -> tuple[Season, ...]:
    """Return inclusive complete October--September project windows."""

    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")
    return tuple(
        Season(
            start_year=year,
            start=datetime(year, 10, 1, 0, 0),
            end=datetime(year + 1, 9, 30, 21, 0),
        )
        for year in range(start_year, end_year + 1)
    )


def overall_window(seasons: Sequence[Season]) -> tuple[datetime, datetime]:
    """Return the complete data window represented by seasons."""

    if not seasons:
        raise ValueError("At least one season is required")
    return seasons[0].start, seasons[-1].end.replace(hour=23)


def resolve_sources(source_root: Path) -> SourcePaths:
    """Resolve and validate every fixed Fram3S source contract."""

    root = source_root.expanduser().resolve()
    resolved = {name: root / relative for name, relative in SOURCE_RELATIVE_PATHS.items()}
    missing = [str(path) for path in resolved.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required source paths:\n" + "\n".join(missing))
    for directory_name in ("forcing", "snow", "fsc"):
        if not resolved[directory_name].is_dir():
            raise NotADirectoryError(resolved[directory_name])
    for file_name in ("roi", "raw_subregions", "forcing_meta", "dem", "landcover", "srf"):
        if not resolved[file_name].is_file():
            raise FileNotFoundError(resolved[file_name])
    return SourcePaths(root=root, **resolved)


def validate_options(options: SnapshotOptions) -> tuple[Season, ...]:
    """Validate immutable CLI and target safety contracts."""

    if options.resolution != 100:
        raise ValueError("Only the native 100 m snapshot is supported")
    if options.image != PINNED_IMAGE:
        raise ValueError(f"Image must be pinned exactly to {PINNED_IMAGE}")
    seasons = hydrological_seasons(options.start_year, options.end_year)
    if options.final_path.exists():
        raise FileExistsError(f"Final snapshot already exists: {options.final_path}")
    resolve_sources(options.source_root)
    return seasons


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    """Copy one raw source file and verify byte identity."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256_file(source)
    destination_hash = sha256_file(destination)
    if source_hash != destination_hash:
        raise RuntimeError(f"Raw-copy hash mismatch: {source} -> {destination}")
    return {
        "source": str(source),
        "raw_copy": str(destination),
        "bytes": source.stat().st_size,
        "sha256": source_hash,
    }


def longest_missing_run(values: Sequence[bool]) -> int:
    """Return the longest consecutive True run."""

    longest = 0
    current = 0
    for missing in values:
        if missing:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def classify_fsc(values: Any) -> dict[str, Any]:
    """Return mutually exclusive FSC class masks for a NumPy array."""

    import numpy as np

    array = np.asarray(values)
    finite = np.isfinite(array)
    valid = finite & (array >= FSC_VALID_MIN) & (array <= FSC_VALID_MAX)
    cloud = finite & np.isin(array, FSC_CLOUD_CLASSES)
    water = finite & np.isin(array, FSC_WATER_CLASSES)
    nodata = (~finite) | np.isin(array, FSC_NODATA_CLASSES)
    classified = valid | cloud | water | nodata
    unknown = ~classified
    return {
        "valid": valid,
        "cloud": cloud,
        "water": water,
        "nodata": nodata,
        "unknown": unknown,
    }


def read_ascii_header(path: Path) -> AsciiGridHeader:
    """Read and validate an ESRI ASCII grid header."""

    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="strict") as file_obj:
        for _ in range(6):
            line = file_obj.readline()
            if not line:
                raise ValueError(f"Incomplete ASCII header: {path}")
            key, value = line.split(maxsplit=1)
            values[key.lower()] = value.strip()
    required = {"ncols", "nrows", "xllcorner", "yllcorner", "cellsize", "nodata_value"}
    if required - values.keys():
        raise ValueError(f"Unsupported ASCII header in {path}: {sorted(values)}")
    return AsciiGridHeader(
        ncols=int(float(values["ncols"])),
        nrows=int(float(values["nrows"])),
        xllcorner=float(values["xllcorner"]),
        yllcorner=float(values["yllcorner"]),
        cellsize=float(values["cellsize"]),
        nodata_value=values["nodata_value"],
    )


def ascii_crop_window(
    header: AsciiGridHeader,
    bounds: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    """Return a native-cell-aligned crop window."""

    import math

    min_x, min_y, max_x, max_y = bounds
    top = header.yllcorner + header.nrows * header.cellsize
    col_start = max(0, math.floor((min_x - header.xllcorner) / header.cellsize))
    col_stop = min(header.ncols, math.ceil((max_x - header.xllcorner) / header.cellsize))
    row_start = max(0, math.floor((top - max_y) / header.cellsize))
    row_stop = min(header.nrows, math.ceil((top - min_y) / header.cellsize))
    if row_stop <= row_start or col_stop <= col_start:
        raise ValueError(f"Empty ASCII crop window for bounds {bounds}")
    return row_start, row_stop, col_start, col_stop


def crop_ascii_grid(
    source: Path,
    destination: Path,
    bounds: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Crop an ASCII grid by selecting original cells without resampling."""

    header = read_ascii_header(source)
    row_start, row_stop, col_start, col_stop = ascii_crop_window(header, bounds)
    destination.parent.mkdir(parents=True, exist_ok=True)
    new_x = header.xllcorner + col_start * header.cellsize
    new_y = header.yllcorner + (header.nrows - row_stop) * header.cellsize
    with source.open("r", encoding="utf-8", errors="strict") as input_file:
        for _ in range(6):
            input_file.readline()
        with destination.open("w", encoding="utf-8") as output_file:
            output_file.write(f"ncols        {col_stop - col_start}\n")
            output_file.write(f"nrows        {row_stop - row_start}\n")
            output_file.write(f"xllcorner    {new_x:.10f}\n")
            output_file.write(f"yllcorner    {new_y:.10f}\n")
            output_file.write(f"cellsize     {header.cellsize:.10f}\n")
            output_file.write(f"NODATA_value {header.nodata_value}\n")
            written_rows = 0
            for row_index, line in enumerate(input_file):
                if row_index < row_start:
                    continue
                if row_index >= row_stop:
                    break
                cells = line.split()
                if len(cells) != header.ncols:
                    raise ValueError(
                        f"ASCII row {row_index} has {len(cells)} cells, expected {header.ncols}: {source}"
                    )
                output_file.write(" ".join(cells[col_start:col_stop]) + "\n")
                written_rows += 1
    if written_rows != row_stop - row_start:
        raise ValueError(f"ASCII source ended early: {source}")
    return {
        "source": str(source),
        "working": str(destination),
        "resolution": header.cellsize,
        "window": [row_start, row_stop, col_start, col_stop],
        "source_sha256": sha256_file(source),
        "working_sha256": sha256_file(destination),
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write a deterministic CSV table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    """Write deterministic indented JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_domain(
    sources: SourcePaths,
    working_root: Path,
    raw_root: Path,
    raw_records: list[dict[str, Any]],
) -> tuple[Any, Any]:
    """Copy and load the fixed eight-subdomain North Tyrol geometry."""

    import geopandas as gpd
    from shapely.ops import unary_union

    raw_records.append(copy_verified(sources.roi, raw_root / "env" / sources.roi.name))
    raw_records.append(
        copy_verified(sources.raw_subregions, raw_root / "env" / sources.raw_subregions.name)
    )
    subdomains = gpd.read_file(sources.roi).to_crs(CRS)
    if len(subdomains) != EXPECTED_SUBDOMAINS or "id" not in subdomains.columns:
        raise ValueError(
            f"Expected {EXPECTED_SUBDOMAINS} subdomains with an id field in {sources.roi}, "
            f"got {len(subdomains)}"
        )
    subdomains = subdomains[["id", "geometry"]].copy()
    subdomains["id"] = subdomains["id"].astype(str)
    subdomains["geometry"] = subdomains.geometry.buffer(0)
    if subdomains.crs.to_epsg() != 25832 or subdomains.geometry.is_empty.any():
        raise ValueError("North Tyrol subdomains must be non-empty EPSG:25832 geometries")
    env_dir = working_root / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    subdomains.sort_values("id").to_file(env_dir / "subdomains.gpkg", driver="GPKG")
    roi_geometry = unary_union(list(subdomains.geometry))
    roi = gpd.GeoDataFrame({"id": [DOMAIN]}, geometry=[roi_geometry], crs=CRS)
    roi.to_file(env_dir / "roi.gpkg", driver="GPKG")
    return subdomains.sort_values("id").reset_index(drop=True), roi_geometry


def _timestamp_column(frame: Any, source: Path) -> str:
    """Return the supported time column in a station CSV."""

    for candidate in ("date", "datetime", "time"):
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"No date/datetime/time column in {source}")


def _forcing_source_extent(source: Path) -> tuple[datetime, datetime]:
    """Return the first and last valid timestamps in one forcing CSV."""

    import pandas as pd

    header = pd.read_csv(source, nrows=0)
    time_column = _timestamp_column(header, source)
    values = pd.read_csv(source, usecols=[time_column])[time_column]
    if values.empty:
        raise ValueError(f"No forcing timestamps: {source}")
    timestamps = pd.to_datetime(values, errors="raise")
    if timestamps.isna().any():
        raise ValueError(f"Invalid forcing timestamps: {source}")
    return timestamps.min().to_pydatetime(), timestamps.max().to_pydatetime()


def _forcing_data_window(seasons: Sequence[Season]) -> tuple[datetime, datetime]:
    """Return the forcing selection window, including the model lookback day."""

    data_start, data_end = overall_window(seasons)
    return data_start - timedelta(days=1), data_end


def prepare_forcing(
    sources: SourcePaths,
    raw_root: Path,
    working_root: Path,
    inventory_root: Path,
    roi_geometry: Any,
    seasons: Sequence[Season],
    raw_records: list[dict[str, Any]],
) -> Any:
    """Select, preserve, subset and inventory the single forcing source."""

    import pandas as pd

    metadata_raw = raw_root / "meteo" / sources.forcing_meta.name
    raw_records.append(copy_verified(sources.forcing_meta, metadata_raw))
    selected, _ = _selected_forcing_metadata(sources, roi_geometry, seasons)
    required_metadata = {"stn_name_orig", "elev"}
    missing_metadata = required_metadata - set(selected.columns)
    if missing_metadata:
        raise ValueError(f"Forcing metadata missing columns: {sorted(missing_metadata)}")
    metadata = selected
    metadata["name"] = metadata["stn_name_orig"].fillna(metadata["stn_name"]).astype(str)
    metadata["x"] = metadata.geometry.x
    metadata["y"] = metadata.geometry.y
    metadata["alt"] = pd.to_numeric(metadata["elev"], errors="raise")
    if len(selected) != EXPECTED_FORCING_STATIONS:
        raise ValueError(
            f"Expected {EXPECTED_FORCING_STATIONS} forcing stations within ROI + 10 km "
            f"and the forcing window, got {len(selected)}"
        )

    data_start, _ = overall_window(seasons)
    forcing_start, data_end = _forcing_data_window(seasons)
    model_times = pd.date_range(data_start, seasons[-1].end, freq="3h")
    network_support = {
        variable: pd.Series(0, index=model_times, dtype="int64")
        for variable in REQUIRED_FORCING_VARIABLES
    }
    coverage_rows: list[dict[str, Any]] = []
    station_rows: list[dict[str, Any]] = []
    working_dir = working_root / "meteo"
    raw_dir = raw_root / "meteo" / "stations"
    working_dir.mkdir(parents=True, exist_ok=True)

    for station in selected.itertuples(index=False):
        station_id = str(station.id)
        source = sources.forcing / f"{station_id}.csv"
        raw_destination = raw_dir / source.name
        raw_records.append(copy_verified(source, raw_destination))
        frame = pd.read_csv(source)
        time_column = _timestamp_column(frame, source)
        timestamps = pd.to_datetime(frame[time_column], errors="raise")
        if timestamps.duplicated().any():
            raise ValueError(f"Duplicate forcing timestamps: {source}")
        source_variables = set(frame.columns)
        for variable in set(REQUIRED_FORCING_VARIABLES) - source_variables:
            frame[variable] = float("nan")
        frame = frame.assign(date=timestamps)
        frame = frame.loc[(frame["date"] >= forcing_start) & (frame["date"] <= data_end)].copy()
        frame = frame.drop(columns=[time_column]) if time_column != "date" else frame
        if frame.empty:
            raise ValueError(f"No forcing data in snapshot window: {source}")
        frame = frame.sort_values("date")
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        frame.to_csv(working_dir / source.name, index=False, lineterminator="\n")

        indexed = pd.read_csv(working_dir / source.name, parse_dates=["date"]).set_index("date")
        expected_hourly = pd.date_range(forcing_start, data_end, freq="h")
        indexed = indexed.reindex(expected_hourly)
        for variable in REQUIRED_FORCING_VARIABLES:
            numeric = pd.to_numeric(indexed[variable], errors="coerce")
            in_model_window = numeric.loc[data_start:data_end]
            valid = in_model_window.notna()
            valid_values = in_model_window.loc[valid]
            coverage_rows.append(
                {
                    "station_id": station_id,
                    "variable": variable,
                    "source_variable_present": variable in source_variables,
                    "expected_hourly_count": len(in_model_window),
                    "valid_count": int(valid.sum()),
                    "missing_count": int((~valid).sum()),
                    "coverage_fraction": float(valid.mean()),
                    "first_valid": valid_values.index.min().isoformat() if not valid_values.empty else "",
                    "last_valid": valid_values.index.max().isoformat() if not valid_values.empty else "",
                    "longest_gap_hours": longest_missing_run((~valid).tolist()),
                }
            )
            at_model_times = numeric.reindex(model_times).notna().astype("int64")
            network_support[variable] = network_support[variable].add(at_model_times, fill_value=0).astype("int64")
        station_rows.append(
            {
                "id": station_id,
                "name": str(station.name),
                "x": float(station.x),
                "y": float(station.y),
                "alt": float(station.alt),
            }
        )

    pd.DataFrame(station_rows).to_csv(working_dir / "stations.csv", index=False, lineterminator="\n")
    coverage_fields = (
        "station_id",
        "variable",
        "source_variable_present",
        "expected_hourly_count",
        "valid_count",
        "missing_count",
        "coverage_fraction",
        "first_valid",
        "last_valid",
        "longest_gap_hours",
    )
    write_csv(inventory_root / "forcing_station_variable_coverage.csv", coverage_rows, coverage_fields)
    support_rows = []
    unsupported = []
    for variable in REQUIRED_FORCING_VARIABLES:
        for timestamp, count in network_support[variable].items():
            row = {
                "timestamp": timestamp.isoformat(),
                "variable": variable,
                "active_station_count": int(count),
            }
            support_rows.append(row)
            if count < 1:
                unsupported.append(row)
    write_csv(
        inventory_root / "forcing_network_support_3h.csv",
        support_rows,
        ("timestamp", "variable", "active_station_count"),
    )
    if unsupported:
        raise ValueError(f"Mandatory forcing network has {len(unsupported)} unsupported 3 h values")
    return pd.DataFrame(station_rows)


def _snow_source(snow_root: Path, station_id: str) -> Path:
    """Resolve one station file using the canonical Tirol naming variants."""

    for candidate in (snow_root / f"{station_id}.csv", snow_root / f"{station_id}_SH.csv"):
        if candidate.is_file():
            return candidate
    matches = sorted(
        path
        for path in snow_root.glob(f"{station_id}*.csv")
        if path.name != "stations_snow_depth.csv"
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one Tirol snow-depth file for {station_id}, got {matches}")
    return matches[0]


def prepare_snow_observations(
    sources: SourcePaths,
    raw_root: Path,
    working_root: Path,
    inventory_root: Path,
    subdomains: Any,
    roi_geometry: Any,
    seasons: Sequence[Season],
    raw_records: list[dict[str, Any]],
) -> Any:
    """Preserve and normalize all Tirol snow-depth stations in the ROI."""

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely.geometry import Point

    metadata_source = sources.snow / "stations_snow_depth.csv"
    raw_records.append(copy_verified(metadata_source, raw_root / "obs" / "stations" / metadata_source.name))
    metadata = pd.read_csv(metadata_source)
    required = {"id", "name", "lat", "lon", "alt", "x", "y"}
    if required - set(metadata.columns):
        raise ValueError(f"Snow metadata missing columns: {sorted(required - set(metadata.columns))}")
    stations = gpd.GeoDataFrame(
        metadata.copy(),
        geometry=gpd.points_from_xy(metadata["x"], metadata["y"]),
        crs=CRS,
    )
    stations = stations.loc[stations.geometry.within(roi_geometry)].copy()
    stations["id"] = stations["id"].astype(str).str.strip()
    stations = stations.sort_values("id").reset_index(drop=True)
    if len(stations) != EXPECTED_SNOW_STATIONS:
        raise ValueError(f"Expected {EXPECTED_SNOW_STATIONS} Tirol snow stations, got {len(stations)}")
    data_start, data_end = overall_window(seasons)
    all_days = pd.date_range(data_start.date(), data_end.date(), freq="D")
    working_dir = working_root / "obs" / "stations"
    working_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []

    for station in stations.itertuples(index=False):
        station_id = str(station.id)
        source = _snow_source(sources.snow, station_id)
        raw_records.append(copy_verified(source, raw_root / "obs" / "stations" / source.name))
        frame = pd.read_csv(source)
        if {"time", "snow_depth"}.issubset(frame.columns):
            normalized = frame[["time", "snow_depth"]].copy()
            normalized["time"] = pd.to_datetime(normalized["time"], errors="raise")
            normalized["snow_depth"] = pd.to_numeric(normalized["snow_depth"], errors="coerce")
        elif {"date", "snow_height"}.issubset(frame.columns):
            normalized = pd.DataFrame(
                {
                    "time": pd.to_datetime(frame["date"], errors="raise"),
                    "snow_depth": pd.to_numeric(frame["snow_height"], errors="coerce") / 100.0,
                }
            )
        else:
            raise ValueError(f"Unsupported Tirol snow-depth format: {source}")
        normalized = normalized.loc[
            (normalized["time"] >= data_start) & (normalized["time"] <= data_end)
        ].copy()
        if normalized["time"].duplicated().any():
            raise ValueError(f"Duplicate snow observation timestamps: {source}")
        normalized["snow_depth"] = normalized["snow_depth"].clip(lower=0.0)
        normalized["swe"] = np.nan
        normalized = normalized.dropna(subset=["snow_depth"]).sort_values("time")
        daily_groups = normalized.set_index("time")["snow_depth"].groupby(pd.Grouper(freq="D"))
        daily_count = daily_groups.count().reindex(all_days, fill_value=0)
        first = normalized.set_index("time").groupby(pd.Grouper(freq="D")).apply(
            lambda values: values.index.min() if len(values) else pd.NaT
        ).reindex(all_days)
        last = normalized.set_index("time").groupby(pd.Grouper(freq="D")).apply(
            lambda values: values.index.max() if len(values) else pd.NaT
        ).reindex(all_days)
        point = Point(float(station.x), float(station.y))
        matches = subdomains.loc[subdomains.geometry.covers(point), "id"].astype(str).tolist()
        if len(matches) != 1:
            raise ValueError(f"Snow station {station_id} maps to {len(matches)} subdomains")
        subdomain_id = matches[0]
        for day in all_days:
            inventory_rows.append(
                {
                    "date": day.date().isoformat(),
                    "subdomain_id": subdomain_id,
                    "station_id": station_id,
                    "valid_observation_count": int(daily_count.loc[day]),
                    "first_observation": first.loc[day].isoformat() if not pd.isna(first.loc[day]) else "",
                    "last_observation": last.loc[day].isoformat() if not pd.isna(last.loc[day]) else "",
                }
            )
        normalized["time"] = normalized["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        normalized.to_csv(working_dir / f"{station_id}.csv", index=False, lineterminator="\n")
        role_rows.append(
            {
                "station_id": station_id,
                "station_uncertainty_pct": 25.0,
                "hs_sigma_abs_min": 0.10,
                "swe_sigma_abs_min": 20.0,
                "use_for_da": True,
                "use_for_benchmark": True,
                "status": "provisional_pending_event_selection",
            }
        )

    station_columns = [column for column in stations.columns if column != "geometry"]
    stations[station_columns].to_csv(
        working_dir / "stations_snow_depth.csv",
        columns=station_columns,
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(role_rows).to_csv(
        working_dir / "stations_da_metadata.csv", index=False, lineterminator="\n"
    )
    write_csv(
        inventory_root / "snow_station_daily_support.csv",
        inventory_rows,
        (
            "date",
            "subdomain_id",
            "station_id",
            "valid_observation_count",
            "first_observation",
            "last_observation",
        ),
    )
    return stations


def _load_ascii_array(path: Path) -> tuple[AsciiGridHeader, Any]:
    """Load an ASCII grid for deterministic derived-field calculations."""

    import numpy as np

    header = read_ascii_header(path)
    array = np.loadtxt(path, skiprows=6, dtype="float64")
    if array.shape != (header.nrows, header.ncols):
        raise ValueError(f"Unexpected ASCII data shape in {path}: {array.shape}")
    return header, array


def _write_ascii_array(path: Path, header: AsciiGridHeader, values: Any, *, decimals: int) -> None:
    """Write a derived array on an existing ASCII grid geometry."""

    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        file_obj.write(f"ncols        {header.ncols}\n")
        file_obj.write(f"nrows        {header.nrows}\n")
        file_obj.write(f"xllcorner    {header.xllcorner:.10f}\n")
        file_obj.write(f"yllcorner    {header.yllcorner:.10f}\n")
        file_obj.write(f"cellsize     {header.cellsize:.10f}\n")
        file_obj.write(f"NODATA_value {header.nodata_value}\n")
        np.savetxt(file_obj, values, fmt=f"%.{decimals}f")


def prepare_grids(
    sources: SourcePaths,
    raw_root: Path,
    working_root: Path,
    roi_geometry: Any,
    raw_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve native grids, crop original cells and derive a real SVF."""

    import numpy as np
    from openamundsen import terrain

    grid_sources = {
        "dem": sources.dem,
        "lc": sources.landcover,
        "srf": sources.srf,
    }
    bounds = roi_geometry.buffer(GRID_BUFFER_M).bounds
    records = []
    for grid_name, source in grid_sources.items():
        raw_records.append(copy_verified(source, raw_root / "grids" / source.name))
        working_name = f"{grid_name}_{DOMAIN}_100.asc"
        record = crop_ascii_grid(source, working_root / "grids" / working_name, bounds)
        if float(record["resolution"]) != 100.0:
            raise ValueError(f"Grid is not native 100 m: {source}")
        records.append({"kind": grid_name, **record})

    dem_path = working_root / "grids" / f"dem_{DOMAIN}_100.asc"
    dem_header, dem = _load_ascii_array(dem_path)
    nodata = float(dem_header.nodata_value)
    valid = ~np.isclose(dem, nodata)
    dem_for_svf = dem.copy()
    if not valid.all():
        dem_for_svf[~valid] = np.nan
    svf = terrain.sky_view_factor(
        dem_for_svf,
        100.0,
        min_azim=0,
        max_azim=360,
        azim_step=10,
        elev_step=1,
        num_sweeps=1,
    )
    if np.any((svf[valid] < 0.0) | (svf[valid] > 1.0) | ~np.isfinite(svf[valid])):
        raise ValueError("Derived SVF contains invalid values")
    svf[~valid] = nodata
    svf_path = working_root / "grids" / f"svf_{DOMAIN}_100.asc"
    _write_ascii_array(svf_path, dem_header, svf, decimals=3)
    records.append(
        {
            "kind": "svf",
            "working": str(svf_path),
            "working_sha256": sha256_file(svf_path),
            "source_dem_sha256": sha256_file(dem_path),
            "method": "openamundsen.terrain.sky_view_factor",
            "parameters": {
                "resolution": 100.0,
                "min_azim": 0,
                "max_azim": 360,
                "azim_step": 10,
                "elev_step": 1,
                "num_sweeps": 1,
                "decimal_precision": 3,
            },
        }
    )
    return records


def _scene_date(path: Path) -> date:
    """Extract a SnowFLAKES acquisition date from its canonical filename."""

    match = re.search(r"SnowFLAKES_(\d{8})_v3_eurac", path.name)
    if not match:
        raise ValueError(f"Unexpected SCF filename: {path}")
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def _coordinate_slice(values: Any, minimum: float, maximum: float) -> slice:
    """Return an order-aware xarray coordinate slice."""

    if len(values) < 2:
        raise ValueError("Raster coordinate must contain at least two values")
    return slice(minimum, maximum) if values[0] < values[-1] else slice(maximum, minimum)


def discover_fsc_scenes(sources: SourcePaths, seasons: Sequence[Season]) -> list[tuple[date, Path]]:
    """Discover all unique SCF scenes in the requested hydrological windows."""

    start, end = overall_window(seasons)
    scenes = []
    for path in sorted(sources.fsc.rglob("*.nc")):
        scene_date = _scene_date(path)
        if start.date() <= scene_date <= end.date():
            scenes.append((scene_date, path))
    dates = [scene_date for scene_date, _ in scenes]
    duplicates = sorted({scene_date for scene_date in dates if dates.count(scene_date) > 1})
    if duplicates:
        raise ValueError(f"Duplicate SCF dates: {[value.isoformat() for value in duplicates]}")
    for season in seasons:
        count = sum(season.start.date() <= scene_date <= season.end.date() for scene_date in dates)
        expected = EXPECTED_FSC_COUNTS.get(season.start_year)
        if expected is not None and count != expected:
            raise ValueError(
                f"Expected {expected} SCF scenes for {season.start_year}/{season.start_year + 1}, got {count}"
            )
    return scenes


def prepare_fsc(
    sources: SourcePaths,
    raw_root: Path,
    working_root: Path,
    inventory_root: Path,
    subdomains: Any,
    roi_geometry: Any,
    seasons: Sequence[Season],
    raw_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve, crop and inventory every EURAC FSC scene without selection."""

    import numpy as np
    import pandas as pd
    import pyproj
    import xarray as xr
    from rasterio import features
    from rasterio.transform import from_origin
    from shapely.geometry import mapping

    scenes = discover_fsc_scenes(sources, seasons)
    rows: list[dict[str, Any]] = []
    scene_records: list[dict[str, Any]] = []
    masks: dict[str, Any] | None = None
    min_x, min_y, max_x, max_y = roi_geometry.bounds
    for index, (scene_date, source) in enumerate(scenes, start=1):
        print(f"FSC {index}/{len(scenes)} {scene_date.isoformat()}", flush=True)
        raw_destination = raw_root / "obs" / "snowcover" / source.name
        raw_record = copy_verified(source, raw_destination)
        raw_records.append(raw_record)
        with xr.open_dataset(source) as dataset:
            required = {"fsc", "uncertainty", "x", "y", "time", "spatial_ref"}
            if required - set(dataset.variables):
                raise ValueError(f"SCF scene missing variables {sorted(required - set(dataset.variables))}: {source}")
            time_value = pd.Timestamp(np.asarray(dataset["time"].values).reshape(-1)[0]).date()
            if time_value != scene_date:
                raise ValueError(f"SCF filename/time mismatch: {source} has {time_value}")
            spatial_attrs = dataset["spatial_ref"].attrs
            crs_text = spatial_attrs.get("crs_wkt") or spatial_attrs.get("spatial_ref")
            if not crs_text or pyproj.CRS.from_user_input(crs_text).to_epsg() != 25832:
                raise ValueError(f"SCF scene is not EPSG:25832: {source}")
            if "GeoTransform" not in spatial_attrs:
                raise ValueError(f"SCF scene lacks GeoTransform metadata: {source}")
            subset = dataset.sel(
                x=_coordinate_slice(dataset["x"].values, min_x - FSC_PAD_M, max_x + FSC_PAD_M),
                y=_coordinate_slice(dataset["y"].values, min_y - FSC_PAD_M, max_y + FSC_PAD_M),
            ).load()
            if subset.sizes.get("x", 0) < 2 or subset.sizes.get("y", 0) < 2:
                raise ValueError(f"Empty FSC crop: {source}")
            for variable in ("fsc", "uncertainty"):
                subset[variable].attrs["grid_mapping"] = "spatial_ref"
            output = working_root / "obs" / "snowcover" / source.name
            output.parent.mkdir(parents=True, exist_ok=True)
            encoding = {
                "fsc": {"zlib": True, "complevel": 4, "dtype": "float32"},
                "uncertainty": {"zlib": True, "complevel": 4, "dtype": "float32"},
            }
            subset.to_netcdf(output, engine="netcdf4", encoding=encoding)

        fsc_data = subset["fsc"]
        uncertainty_data = subset["uncertainty"]
        for dimension in tuple(fsc_data.dims):
            if dimension not in {"y", "x"}:
                fsc_data = fsc_data.isel({dimension: 0})
        for dimension in tuple(uncertainty_data.dims):
            if dimension not in {"y", "x"}:
                uncertainty_data = uncertainty_data.isel({dimension: 0})
        x = subset["x"].values
        y = subset["y"].values
        resolution_x = float(abs(x[1] - x[0]))
        resolution_y = float(abs(y[1] - y[0]))
        if resolution_x != 50.0 or resolution_y != 50.0:
            raise ValueError(f"EURAC FSC is not on its native 50 m grid: {source}")
        transform = from_origin(float(x.min() - 25.0), float(y.max() + 25.0), 50.0, 50.0)
        if masks is None:
            masks = {}
            for subdomain in subdomains.itertuples(index=False):
                mask = features.rasterize(
                    [(mapping(subdomain.geometry), 1)],
                    out_shape=(len(y), len(x)),
                    transform=transform,
                    fill=0,
                    dtype="uint8",
                ).astype(bool)
                if not mask.any():
                    raise ValueError(f"No FSC pixels for subdomain {subdomain.id}")
                masks[str(subdomain.id)] = mask
        fsc_array = np.asarray(fsc_data.values)
        uncertainty_array = np.asarray(uncertainty_data.values)
        for subdomain_id, mask in masks.items():
            values = fsc_array[mask]
            classes = classify_fsc(values)
            if int(classes["unknown"].sum()) > 0:
                unknown_values = np.unique(values[classes["unknown"]])
                raise ValueError(f"Unknown FSC classes in {source}: {unknown_values.tolist()}")
            uncertainty_values = uncertainty_array[mask]
            uncertainty_valid = uncertainty_values[np.isfinite(uncertainty_values)]
            total = int(mask.sum())
            counts = {name: int(classes[name].sum()) for name in ("valid", "cloud", "water", "nodata")}
            rows.append(
                {
                    "date": scene_date.isoformat(),
                    "subdomain_id": subdomain_id,
                    "source_file": source.name,
                    "pixel_count": total,
                    **{f"{name}_count": count for name, count in counts.items()},
                    **{f"{name}_fraction": count / total for name, count in counts.items()},
                    "uncertainty_count": int(uncertainty_valid.size),
                    "uncertainty_min": float(np.min(uncertainty_valid)) if uncertainty_valid.size else "",
                    "uncertainty_mean": float(np.mean(uncertainty_valid)) if uncertainty_valid.size else "",
                    "uncertainty_median": float(np.median(uncertainty_valid)) if uncertainty_valid.size else "",
                    "uncertainty_p90": float(np.percentile(uncertainty_valid, 90)) if uncertainty_valid.size else "",
                    "uncertainty_max": float(np.max(uncertainty_valid)) if uncertainty_valid.size else "",
                }
            )
        scene_records.append(
            {
                "date": scene_date.isoformat(),
                "source": str(source),
                "source_sha256": raw_record["sha256"],
                "working": str(output),
                "working_sha256": sha256_file(output),
            }
        )
        subset.close()

    write_csv(
        inventory_root / "fsc_scene_subdomain_quality.csv",
        rows,
        (
            "date",
            "subdomain_id",
            "source_file",
            "pixel_count",
            "valid_count",
            "cloud_count",
            "water_count",
            "nodata_count",
            "valid_fraction",
            "cloud_fraction",
            "water_fraction",
            "nodata_fraction",
            "uncertainty_count",
            "uncertainty_min",
            "uncertainty_mean",
            "uncertainty_median",
            "uncertainty_p90",
            "uncertainty_max",
        ),
    )
    return scene_records


def _flow(values: Iterable[Any]) -> Any:
    """Return a ruamel sequence rendered in inline flow style."""

    from ruamel.yaml.comments import CommentedSeq

    sequence = CommentedSeq(values)
    sequence.fa.set_flow_style()
    return sequence


def dump_yaml(path: Path, value: dict[str, Any]) -> None:
    """Write deterministic human-readable YAML."""

    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        yaml.dump(value, file_obj)


def setup_configuration(forcing_stations: Any, seasons: Sequence[Season]) -> dict[str, Any]:
    """Return the shared six-season openAMUNDSEN setup configuration."""

    points = [
        {"x": float(row.x), "y": float(row.y), "name": str(row.id)}
        for row in forcing_stations.sort_values("id").itertuples(index=False)
    ]
    start = seasons[0].start
    end = seasons[-1].end
    return {
        "domain": DOMAIN,
        "resolution": 100,
        "timestep": "3H",
        "crs": "epsg:25832",
        "timezone": 1,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
        "input_data": {
            "grids": {"dir": "data_working/grids"},
            "meteo": {
                "dir": "data_working/meteo",
                "format": "csv",
                "crs": "epsg:25832",
                "bounds": "grid",
                "aggregate_when_downsampling": True,
            },
        },
        "output_data": {
            "timeseries": {
                "format": "csv",
                "write_freq": "D",
                "add_default_points": False,
                "add_default_variables": True,
                "points": points,
            },
            "grids": {
                "format": "netcdf",
                "compress": True,
                "variables": [{"var": "snow.depth", "name": "snowdepth_daily", "freq": "D"}],
            },
        },
        "meteo": {
            "interpolation": {
                "temperature": {
                    "trend_method": "fixed",
                    "extrapolate": True,
                    "lapse_rate": _flow(
                        [
                            -0.0026,
                            -0.0035,
                            -0.0047,
                            -0.0053,
                            -0.0052,
                            -0.0053,
                            -0.0049,
                            -0.0047,
                            -0.0042,
                            -0.0033,
                            -0.0035,
                            -0.0031,
                        ]
                    ),
                },
                "precipitation": {
                    "trend_method": "fractional",
                    "extrapolate": True,
                    "lapse_rate": _flow(
                        [
                            0.00048,
                            0.00046,
                            0.00041,
                            0.00033,
                            0.00028,
                            0.00025,
                            0.00024,
                            0.00025,
                            0.00028,
                            0.00033,
                            0.00041,
                            0.00046,
                        ]
                    ),
                },
                "humidity": {
                    "trend_method": "fixed",
                    "extrapolate": True,
                    "lapse_rate": _flow(
                        [
                            -0.0044,
                            -0.0046,
                            -0.0049,
                            -0.0048,
                            -0.0046,
                            -0.0047,
                            -0.0043,
                            -0.0042,
                            -0.0045,
                            -0.0044,
                            -0.0047,
                            -0.0046,
                        ]
                    ),
                },
                "cloudiness": {
                    "day_method": "clear_sky_fraction",
                    "night_method": "humidity",
                    "allow_fallback": True,
                },
                "wind_speed": {"trend_method": "regression", "extrapolate": False},
            },
            "precipitation_phase": {
                "method": "wet_bulb_temp",
                "threshold_temp": 273.65,
                "temp_range": 1.0,
            },
            "precipitation_correction": [{"method": "srf"}],
            "radiation": {
                "snow_emissivity": 0.99,
                "cloud_emissivity": 0.976,
                "rock_emission_factor": 0.01,
                "ozone_layer_thickness": 0.0035,
                "atmospheric_visibility": 25000.0,
                "single_scattering_albedo": 0.9,
                "clear_sky_albedo": 0.0685,
                "num_shadow_sweeps": 1,
            },
            "measurement_height": {"temperature": 2, "wind": 10},
            "stability_correction": False,
            "stability_adjustment_parameter": 5.0,
        },
        "snow": {
            "model": "multilayer",
            "thermal_conductivity": 0.24,
            "roughness_length": 0.01,
            "measurement_height_adjustment": False,
            "snow_cover_fraction_depth_scale": 1.0e-6,
            "albedo": {
                "method": "snow_age",
                "min": 0.55,
                "max": 0.85,
                "cold_snow_decay_timescale": 480,
                "melting_snow_decay_timescale": 200,
                "decay_timescale_determination_temperature": "surface",
                "refresh_snowfall": 0.5,
                "refresh_method": "binary",
                "firn": 0.4,
                "ice": 0.2,
            },
            "compaction": {"method": "anderson", "timescale": 200, "max_cold_density": 300},
        },
    }


def project_configuration(season: Season) -> dict[str, Any]:
    """Return one deliberately non-runnable ES30 pending project."""

    return {
        "run_mode": "subdomain",
        "start_date": season.start.strftime("%Y-%m-%d"),
        "end_date": season.end.strftime("%Y-%m-%d %H:%M:%S"),
        "obs": {
            "stations": {"dir": "data_working/obs/stations"},
            "snowcover": {
                "dir": "data_working/obs/snowcover",
                "format": "netcdf",
                "product_tag": "SNOWCOVER",
                "classes": {
                    "valid": _flow(range(0, 101)),
                    "cloud": _flow([205, 255]),
                    "water": _flow([210]),
                    "nodata": _flow([215]),
                },
            },
        },
        "data_assimilation": {
            "prior_forcing": {
                "ensemble_size": 30,
                "random_seed": 42,
                "sigma_t": 0.5,
                "mu_p": 0.0,
                "sigma_p": 0.5,
                "sigma_rh": 0.5,
                "sigma_sw": 0.05,
            },
            "h_of_x": {"method": "depth_threshold", "variable": "hs", "params": {"h0": 0.03, "k": 80}},
            "station": {
                "default_station_uncertainty_pct": 25,
                "min_station_uncertainty_pct": 10,
                "single_station_factor": 2.0,
            },
            "subdomain_event_filter": {
                "enabled": True,
                "drop_unavailable": True,
                "variables": {
                    "scf": {"max_cloud_fraction": 0.2},
                    "station_hs": {"min_active_stations": 1, "max_time_delta_hours": 36},
                },
                "subdomains": {"AT-07-20": {"variables": {"scf": {"max_cloud_fraction": 0.25}}}},
            },
            "landcover_mask": {"enabled": True, "classes_to_exclude": _flow([2, 3, 13])},
            "likelihood": {
                "scf": {
                    "obs_sigma": 0.1,
                    "use_binomial": True,
                    "sigma_floor": 0.05,
                    "sigma_cloud_scale": 0.1,
                    "min_sigma": 0.03,
                }
            },
            "uncertainty": {
                "scf": {
                    "enabled": True,
                    "input_dir": "data_working/obs/snowcover",
                    "ingest": {
                        "scf_variable": "fsc",
                        "time_variable": "time",
                        "uncertainty_source": "internal",
                    },
                    "assimilation": {"sigma_mode": "uncertainty_layer", "aggregate_metric": "unc_mean"},
                    "u_min": 5.0,
                    "u_max": 20.0,
                    "nodata_value": 255.0,
                    "penalties": [
                        {
                            "name": "forest",
                            "source": "landcover",
                            "enabled": True,
                            "classes": _flow([8, 9, 10, 11, 12]),
                            "penalty": 20.0,
                        }
                    ],
                }
            },
            "resampling": {"algorithm": "systematic", "ess_threshold_ratio": 0.7, "seed": 42},
            "rejuvenation": {
                "sigma_t": 0.5,
                "sigma_p": 0.5,
                "sigma_rh": 0.5,
                "sigma_sw": 0.05,
                "seed": 42,
                "rebase_open_loop": False,
            },
            "restart": {"dump_state": True, "state_pattern": "model_state.pickle.gz"},
            "output": {
                "retention": "full",
                "grids": {
                    "format": "netcdf",
                    "compress": True,
                    "dims": _flow(["x", "y", "time"]),
                    "variables": [
                        {
                            "var": "snowdepth_daily",
                            "name": "snowdepth_daily",
                            "metrics": _flow(
                                [
                                    "open_loop",
                                    "ens_mean",
                                    "ens_std",
                                    "ens_min",
                                    "ens_max",
                                    "increment",
                                    "analysis_mean",
                                    "analysis_increment",
                                ]
                            ),
                        },
                        {
                            "var": "swe_daily",
                            "name": "swe_daily",
                            "metrics": _flow(["open_loop", "ens_mean", "ens_std", "ens_min", "ens_max", "increment"]),
                        },
                    ],
                },
            },
            "benchmark": {
                "enabled": True,
                "variables": _flow(["scf", "station_hs"]),
                "independent_variables": _flow(["station_hs"]),
                "performance_scores_exclude_variables": _flow([]),
            },
            "assimilation_events": [],
        },
    }


MAPS_YAML = """maps:
  subdomain_example_setup_overview:
    title: Subdomain example setup overview
    output_name: setup_overview
    layout:
      nrows: 2
      ncols: 3
    defaults:
      show_scalebar: true
    panels:
      - {row: 0, col: 0, kind: overview, title: Overview, scale: 2500000, roi_label: Subdomain ROI}
      - row: 0
        col: 1
        kind: roi
        title: ROI and stations
        show_station_marker: true
        show_stations_name: false
        show_stations_elev: false
        below_items: [{kind: station_symbol, label: Meteo and snow stations}]
      - {row: 0, col: 2, kind: dem, title: Elevation}
      - {row: 1, col: 0, kind: landcover, title: Land cover}
      - {row: 1, col: 1, kind: hillshade, title: Terrain shading, show_station_marker: true, show_stations_name: false}
      - {row: 1, col: 2, kind: srf, title: Snow redistribution factor, show_hillshade: true}
"""

PLOTS_YAML = """panels:
  - panel: fSC
    show_obs: true
    title: Snow cover fraction subdomain ROI - openAMUNDSEN ensemble and satellite data
  - panel: roi-sd
    title: Mean snow depth subdomain ROI - openAMUNDSEN ensemble and open loop
  - panel: ess
  - panel: scores-crpss
"""


def write_pending_projects(root: Path, seasons: Sequence[Season]) -> None:
    """Write six pending project skeletons with explicit provisional status."""

    provisional_fields = [
        "data_assimilation.prior_forcing",
        "data_assimilation.h_of_x",
        "data_assimilation.station",
        "data_assimilation.subdomain_event_filter",
        "data_assimilation.likelihood",
        "data_assimilation.uncertainty",
        "data_assimilation.resampling",
        "data_assimilation.rejuvenation",
        "data_assimilation.benchmark",
        "data_working.obs.stations.stations_da_metadata.csv roles",
    ]
    for season in seasons:
        project_dir = root / "projects_pending_events" / season.name
        project_dir.mkdir(parents=True, exist_ok=False)
        dump_yaml(project_dir / f"{season.name}.yml", project_configuration(season))
        (project_dir / "maps.yml").write_text(MAPS_YAML, encoding="utf-8")
        (project_dir / "plots.yml").write_text(PLOTS_YAML, encoding="utf-8")
        (project_dir / "PENDING_EVENTS").write_text(
            "This project is deliberately non-runnable until assimilation events are reviewed and added.\n",
            encoding="utf-8",
        )
        write_json(
            project_dir / "PROJECT_STATUS.json",
            {
                "status": "PENDING_EVENTS",
                "runnable": False,
                "assimilation_events": [],
                "provisional_fields": provisional_fields,
                "promotion_target": f"projects/{season.name}",
            },
        )


def _selected_forcing_metadata(
    sources: SourcePaths,
    roi_geometry: Any,
    seasons: Sequence[Season],
) -> tuple[Any, int]:
    """Return spatially eligible forcing stations that overlap the data window."""

    import geopandas as gpd

    metadata = gpd.read_file(sources.forcing_meta).to_crs(CRS)
    required_metadata = {"provider", "stn_name", "geometry"}
    missing_metadata = required_metadata - set(metadata.columns)
    if missing_metadata:
        raise ValueError(f"Forcing metadata missing columns: {sorted(missing_metadata)}")
    metadata["id"] = metadata["provider"].astype(str) + "." + metadata["stn_name"].astype(str)
    spatial = metadata.loc[metadata.geometry.within(roi_geometry.buffer(FORCING_BUFFER_M))].copy()
    spatial = spatial.loc[
        spatial["id"].map(lambda value: (sources.forcing / f"{value}.csv").is_file())
    ]
    spatial = spatial.sort_values("id").reset_index(drop=True)
    forcing_start, forcing_end = _forcing_data_window(seasons)
    overlaps = []
    for station_id in spatial["id"].astype(str):
        source_start, source_end = _forcing_source_extent(sources.forcing / f"{station_id}.csv")
        overlaps.append(source_start <= forcing_end and source_end >= forcing_start)
    selected = spatial.loc[overlaps].reset_index(drop=True)
    return selected, len(spatial)


def preflight(options: SnapshotOptions) -> dict[str, Any]:
    """Perform read-only source discovery, coverage counts and size estimation."""

    import geopandas as gpd

    seasons = validate_options(options)
    sources = resolve_sources(options.source_root)
    subdomains = gpd.read_file(sources.roi).to_crs(CRS)
    if len(subdomains) != EXPECTED_SUBDOMAINS:
        raise ValueError(f"Expected {EXPECTED_SUBDOMAINS} subdomains, got {len(subdomains)}")
    roi_geometry = subdomains.geometry.buffer(0).unary_union
    forcing, forcing_spatial_file_count = _selected_forcing_metadata(
        sources, roi_geometry, seasons
    )
    if len(forcing) != EXPECTED_FORCING_STATIONS:
        raise ValueError(f"Expected {EXPECTED_FORCING_STATIONS} forcing stations, got {len(forcing)}")
    snow_metadata = __import__("pandas").read_csv(sources.snow / "stations_snow_depth.csv")
    snow_points = gpd.GeoDataFrame(
        snow_metadata,
        geometry=gpd.points_from_xy(snow_metadata["x"], snow_metadata["y"]),
        crs=CRS,
    )
    snow = snow_points.loc[snow_points.geometry.within(roi_geometry)].copy()
    if len(snow) != EXPECTED_SNOW_STATIONS:
        raise ValueError(f"Expected {EXPECTED_SNOW_STATIONS} snow stations, got {len(snow)}")
    scenes = discover_fsc_scenes(sources, seasons)
    raw_files = [sources.roi, sources.raw_subregions, sources.forcing_meta, sources.dem, sources.landcover, sources.srf]
    raw_files.extend(sources.forcing / f"{station_id}.csv" for station_id in forcing["id"].astype(str))
    raw_files.append(sources.snow / "stations_snow_depth.csv")
    raw_files.extend(_snow_source(sources.snow, str(station_id).strip()) for station_id in snow["id"])
    raw_files.extend(path for _, path in scenes)
    counts = {
        str(season.start_year): sum(
            season.start.date() <= scene_date <= season.end.date() for scene_date, _ in scenes
        )
        for season in seasons
    }
    return {
        "status": "PREFLIGHT_OK",
        "source_root": str(sources.root),
        "target": str(options.final_path),
        "image": options.image,
        "resolution": options.resolution,
        "seasons": [asdict(season) | {"name": season.name} for season in seasons],
        "subdomain_count": len(subdomains),
        "forcing_spatial_file_count": forcing_spatial_file_count,
        "forcing_station_count": len(forcing),
        "forcing_outside_window_count": forcing_spatial_file_count - len(forcing),
        "snow_station_count": len(snow),
        "fsc_scene_count": len(scenes),
        "fsc_counts_by_start_year": counts,
        "selected_raw_file_count": len(raw_files),
        "selected_raw_bytes": sum(path.stat().st_size for path in raw_files),
    }


def validate_snapshot(root: Path, seasons: Sequence[Season]) -> dict[str, Any]:
    """Validate setup parsing, reader compatibility and pending-project safety."""

    import openamundsen
    from openamundsen import fileio

    forbidden = [root / name for name in ("projects", "steps", "results") if (root / name).exists()]
    if forbidden:
        raise ValueError(f"Forbidden active/runtime directories exist: {forbidden}")
    setup_path = root / "subdomains.yml"
    configuration = openamundsen.parse_config(openamundsen.read_config(setup_path))
    reader_checks = []
    for season in seasons:
        dataset = fileio.read_meteo_data(
            "csv",
            root / "data_working" / "meteo",
            season.start,
            season.end,
            meteo_crs=CRS,
            grid_crs=CRS,
            freq="3h",
            aggregate=True,
        )
        missing = [variable for variable in REQUIRED_FORCING_VARIABLES if variable not in dataset]
        if missing:
            raise ValueError(f"Reader output missing forcing variables for {season.name}: {missing}")
        for variable in REQUIRED_FORCING_VARIABLES:
            if bool(dataset[variable].isnull().all(dim="station").any()):
                raise ValueError(f"No reader-level network support for {variable} in {season.name}")
        reader_checks.append(
            {
                "project": season.name,
                "time_count": int(dataset.sizes["time"]),
                "station_count": int(dataset.sizes["station"]),
            }
        )
        dataset.close()
        project_dir = root / "projects_pending_events" / season.name
        project_path = project_dir / f"{season.name}.yml"
        from ruamel.yaml import YAML

        project = YAML(typ="safe").load(project_path)
        if project["data_assimilation"]["assimilation_events"] != []:
            raise ValueError(f"Pending project unexpectedly has events: {project_path}")
        if not (project_dir / "PENDING_EVENTS").is_file():
            raise ValueError(f"Pending marker missing: {project_dir}")
    return {
        "openamundsen_version": getattr(openamundsen, "__version__", "unknown"),
        "setup_start": str(configuration.start_date),
        "setup_end": str(configuration.end_date),
        "reader_checks": reader_checks,
    }


def _hash_tree(root: Path, *, exclude: set[Path] | None = None) -> list[dict[str, Any]]:
    """Hash every regular file below a snapshot subtree."""

    excluded = exclude or set()
    rows = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path in excluded:
            continue
        rows.append(
            {
                "path": path.relative_to(root.parent).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _relativize_staging_paths(value: Any, staging: Path) -> Any:
    """Replace staging-root paths in provenance data with stable relative paths."""

    if isinstance(value, dict):
        return {key: _relativize_staging_paths(item, staging) for key, item in value.items()}
    if isinstance(value, list):
        return [_relativize_staging_paths(item, staging) for item in value]
    if isinstance(value, str):
        try:
            return Path(value).relative_to(staging).as_posix()
        except ValueError:
            return value
    return value


def build_snapshot(options: SnapshotOptions) -> Path:
    """Build, validate and atomically promote one snapshot."""

    seasons = validate_options(options)
    sources = resolve_sources(options.source_root)
    options.target_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    staging = options.target_root / f".{options.snapshot_name}.staging-{stamp}-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    raw_root = staging / "data_raw"
    working_root = staging / "data_working"
    inventory_root = staging / "inventories"
    provenance_root = staging / "provenance"
    raw_records: list[dict[str, Any]] = []
    try:
        subdomains, roi_geometry = load_domain(sources, working_root, raw_root, raw_records)
        forcing_stations = prepare_forcing(
            sources,
            raw_root,
            working_root,
            inventory_root,
            roi_geometry,
            seasons,
            raw_records,
        )
        prepare_snow_observations(
            sources,
            raw_root,
            working_root,
            inventory_root,
            subdomains,
            roi_geometry,
            seasons,
            raw_records,
        )
        grid_records = prepare_grids(sources, raw_root, working_root, roi_geometry, raw_records)
        fsc_records = prepare_fsc(
            sources,
            raw_root,
            working_root,
            inventory_root,
            subdomains,
            roi_geometry,
            seasons,
            raw_records,
        )
        dump_yaml(staging / "subdomains.yml", setup_configuration(forcing_stations, seasons))
        write_pending_projects(staging, seasons)

        normalized_raw_records = []
        for record in raw_records:
            raw_copy = Path(record["raw_copy"])
            normalized_raw_records.append(
                {**record, "raw_copy": raw_copy.relative_to(staging).as_posix()}
            )
        write_json(provenance_root / "raw_files.json", normalized_raw_records)
        write_json(
            provenance_root / "grid_derivation.json",
            _relativize_staging_paths(grid_records, staging),
        )
        write_json(
            provenance_root / "fsc_working_files.json",
            _relativize_staging_paths(fsc_records, staging),
        )
        validation = validate_snapshot(staging, seasons)
        working_hashes = _hash_tree(working_root)
        write_json(provenance_root / "working_files.json", working_hashes)
        try:
            builder_commit = subprocess.check_output(
                ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            builder_commit = "unknown"
        manifest = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "status": "READY_FOR_EVENT_SELECTION",
            "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source_root": str(sources.root),
            "source_contracts": {name: str(getattr(sources, name)) for name in SOURCE_RELATIVE_PATHS},
            "snapshot_name": options.snapshot_name,
            "image": options.image,
            "builder_commit": builder_commit,
            "resolution": options.resolution,
            "timezone_contract": {
                "source_timestamps": "preserved without shifting",
                "setup_timezone": 1,
                "forcing_lookback_start": (seasons[0].start - timedelta(days=1)).isoformat(),
                "acceptance_start": seasons[0].start.isoformat(),
                "acceptance_end": seasons[-1].end.isoformat(),
            },
            "counts": {
                "subdomains": EXPECTED_SUBDOMAINS,
                "forcing_stations": EXPECTED_FORCING_STATIONS,
                "snow_stations": EXPECTED_SNOW_STATIONS,
                "fsc_scenes": len(fsc_records),
            },
            "fsc_classes": {
                "valid": [0, 100],
                "cloud": [205, 255],
                "water": [210],
                "nodata": [215, "NaN"],
            },
            "seasons": [
                {
                    "project": season.name,
                    "start": season.start.isoformat(),
                    "end": season.end.isoformat(),
                    "fsc_scene_count": EXPECTED_FSC_COUNTS.get(season.start_year),
                }
                for season in seasons
            ],
            "validation": validation,
        }
        write_json(staging / "snapshot_manifest.json", manifest)
        write_json(
            staging / "READY_FOR_EVENT_SELECTION",
            {
                "status": "READY_FOR_EVENT_SELECTION",
                "next_phase": "review candidate inventories, select events, then promote configs to projects/",
            },
        )
        (staging / "README.md").write_text(
            "# North Tyrol six-season snapshot\n\n"
            "This snapshot is data-complete but deliberately not runnable. Review `inventories/` and add "
            "assimilation events before promoting any directory from `projects_pending_events/` to `projects/`.\n",
            encoding="utf-8",
        )
        staging.replace(options.final_path)
        return options.final_path
    except Exception as exc:
        write_json(
            staging / "INCOMPLETE.json",
            {
                "status": "INCOMPLETE",
                "failed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
