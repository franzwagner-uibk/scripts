#!/usr/bin/env python3
"""Build the shipped sub-domain example for openAMUNDSEN-DA.

All FRAMES-specific extraction, clipping, and copy logic intentionally lives in
this external helper, not in the openamundsen_da package.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import Point, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


WORKSPACE = Path("/home/franz/workspace")
OPENAMUNDSEN_DA = WORKSPACE / "repos" / "openamundsen_da"
FRAMES_DATA = WORKSPACE / "fram3s" / "01-data"
if not FRAMES_DATA.exists() and (WORKSPACE / "01-data").exists():
    FRAMES_DATA = WORKSPACE / "01-data"
EXAMPLE_DIR = OPENAMUNDSEN_DA / "examples" / "subdomains"
ARCHIVE_ROOT = WORKSPACE / "dev_examples" / "archive"
EUREGIO_EXAMPLE_DIR = WORKSPACE / "dev_examples" / "euregio_subdomains"

ROI_PATH = FRAMES_DATA / "01-aoi" / "TESTSITES" / "Testsite_North_Tyrol.gpkg"
RAW_SUBREGIONS_PATH = FRAMES_DATA / "01-aoi" / "SUBREGIONS" / "raw" / "subregions_avalanche_report_raw_25832.gpkg"
FORCING_DIR = FRAMES_DATA / "02-meteo" / "01-data" / "01-initial" / "openamundsen-v2"
FORCING_META = FRAMES_DATA / "02-meteo" / "02-meta" / "gpkg" / "meta-all.gpkg"
SNOW_OBS_DIR = FRAMES_DATA / "02-meteo" / "01-data" / "02-snow_obs" / "Tirol_snow_depth"
SCF_ZIP_SOURCE = WORKSPACE / "fram3s" / "50-eurac" / "SCF_Eurac_v3.zip"
SCF_ZIP_CACHE = WORKSPACE / ".cache" / "north_tyrol_sources" / "SCF_Eurac_v3.zip"

DEM_DIR = FRAMES_DATA / "05-dem" / "euregio"
LC_DIR = FRAMES_DATA / "03-landcover" / "lc_eusalp" / "openAMUNDSEN-euregio"
SRF_DIR = FRAMES_DATA / "06-srf" / "euregio"
GRID_SOURCE_CACHE = WORKSPACE / ".cache" / "north_tyrol_sources"

DOMAIN = "subdomain_example"
PROJECT_NAME = "project_2022_2023"
START_DATE = "2022-10-01"
END_DATE = "2023-06-30 21:00:00"
SEASON_START = date(2022, 10, 1)
SEASON_END = date(2023, 6, 30)
STATION_EVENT_CANDIDATES = [
    date(2022, 11, 17),
    date(2022, 12, 7),
    date(2023, 1, 1),
    date(2023, 1, 31),
    date(2023, 2, 10),
    date(2023, 2, 21),
    date(2023, 3, 3),
    date(2023, 3, 17),
    date(2023, 4, 20),
]
FSC_EVENT_CANDIDATES = [
    date(2022, 10, 5),
    date(2022, 10, 28),
    date(2022, 11, 12),
    date(2022, 11, 27),
    date(2022, 12, 17),
    date(2022, 12, 29),
    date(2023, 1, 6),
    date(2023, 2, 15),
    date(2023, 3, 7),
    date(2023, 3, 22),
    date(2023, 4, 3),
    date(2023, 4, 6),
    date(2023, 4, 26),
    date(2023, 5, 26),
    date(2023, 6, 2),
]
RESOLUTIONS = (50, 100, 250, 500)
BUILD_RESOLUTIONS = (100, 250, 500, 50)
DEFAULT_RESOLUTION = 100
EUREGIO_RESOLUTIONS = (100, 250)
EUREGIO_BUILD_RESOLUTIONS = (250, 100)
EUREGIO_DEFAULT_RESOLUTION = 250
FORCING_BUFFER_M = 10_000.0
GRID_CROP_BUFFER_M = 10_000.0
SUBDOMAIN_GRID_BUFFER_M = 10_000.0
FSC_MAX_CLOUD_FRACTION = 0.20
FSC_SUBDOMAIN_CLOUD_OVERRIDES = {"AT-07-20": 0.25}
FSC_CLOUD_CLASSES = (210.0, 215.0)
BIWEEKLY_SPACING_DAYS = 14
EVENT_SHIFT_DAYS = 4
HIGH_SEASON_MONTHS = {12, 1, 2, 3}


@dataclass(frozen=True)
class ScfCandidate:
    date: date
    zip_name: str
    cloud_by_subdomain: dict[str, float]
    selected_for_subdomains: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuildOptions:
    region_set: str
    example_dir: Path
    domain: str
    resolutions: tuple[int, ...]
    build_resolutions: tuple[int, ...]
    default_resolution: int
    compact_output: bool


def _flow(values: Iterable[object]):
    from ruamel.yaml.comments import CommentedSeq

    seq = CommentedSeq(values)
    seq.fa.set_flow_style()
    return seq


def _dump_yaml(path: Path, data: dict) -> None:
    from ruamel.yaml import YAML

    path.parent.mkdir(parents=True, exist_ok=True)
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)


def _archive_existing_example(example_dir: Path, archive_root: Path) -> Path | None:
    if not example_dir.exists():
        return None
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = archive_root / f"subdomains_{stamp}"
    shutil.move(str(example_dir), target)
    print(f"archived existing example -> {target}")
    return target


def _clean_output(example_dir: Path) -> None:
    if example_dir.exists():
        shutil.rmtree(example_dir)
    for name in ("grids", "meteo", "env", "obs", "projects"):
        (example_dir / name).mkdir(parents=True, exist_ok=True)


def _subdomain_roi_source(example_dir: Path = EXAMPLE_DIR) -> Path | None:
    roi_source = ROI_PATH if ROI_PATH.is_file() else example_dir / "env" / "subdomains.gpkg"
    if not roi_source.is_file():
        return None
    return roi_source


def _read_subdomains(region_set: str, example_dir: Path = EXAMPLE_DIR) -> gpd.GeoDataFrame:
    raw = gpd.read_file(RAW_SUBREGIONS_PATH).to_crs("EPSG:25832")
    raw["id"] = raw["id"].astype(str).astype(object)
    raw["geometry"] = raw.geometry.buffer(0)
    if region_set == "euregio":
        if len(raw) != 90:
            raise ValueError(f"Expected 90 Euregio avalanche-report subdomains in {RAW_SUBREGIONS_PATH}, got {len(raw)}")
        return _make_non_overlapping(raw[["id", "geometry"]].copy())

    roi_source = _subdomain_roi_source(example_dir)
    if roi_source is None:
        raise FileNotFoundError(f"Subdomain example ROI source not found: {ROI_PATH}")
    roi = gpd.read_file(roi_source).to_crs("EPSG:25832")
    roi["geometry"] = roi.geometry.buffer(0)
    if len(roi) != 8:
        raise ValueError(f"Expected 8 subdomains in {roi_source}, got {len(roi)}")
    if "id" not in roi.columns:
        raise ValueError(f"ROI layer missing id column: {roi_source}")
    roi["id"] = roi["id"].astype(str).astype(object)

    raw_ids = set(raw["id"].astype(str))
    missing = sorted(set(roi["id"].astype(str)) - raw_ids)
    if missing:
        raise ValueError(f"ROI subdomain ids missing in raw avalanche subregions: {missing}")
    return _make_non_overlapping(roi[["id", "geometry"]].copy())


def _make_non_overlapping(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign tiny source overlaps to the first subdomain in file order."""
    rows = []
    assigned = None
    for row in gdf.itertuples(index=False):
        geom = row.geometry.buffer(0)
        if assigned is not None and not assigned.is_empty:
            geom = geom.difference(assigned).buffer(0)
        if geom.is_empty:
            raise ValueError(f"Subdomain {row.id} became empty while resolving source overlaps")
        rows.append({"id": str(row.id), "geometry": geom})
        assigned = geom if assigned is None else unary_union([assigned, geom]).buffer(0)
    out = gpd.GeoDataFrame(rows, crs=gdf.crs)
    return out


def _write_vectors(example_dir: Path, subdomains: gpd.GeoDataFrame, *, domain: str = DOMAIN) -> BaseGeometry:
    env_dir = example_dir / "env"
    subdomains = subdomains.copy()
    subdomains["id"] = subdomains["id"].astype(str).astype(object)
    subdomains.to_file(env_dir / "subdomains.gpkg", driver="GPKG")
    roi_geom = unary_union(list(subdomains.geometry))
    roi_attrs = pd.DataFrame({"id": pd.Series([domain], dtype=object)})
    gpd.GeoDataFrame(roi_attrs, geometry=[roi_geom], crs=subdomains.crs).to_file(
        env_dir / "roi.gpkg",
        driver="GPKG",
    )
    return roi_geom


def _grid_source(prefix: str, resolution: int) -> Path:
    if prefix == "dem":
        source = DEM_DIR / f"dem_euregio_{resolution}.asc"
    elif prefix == "lc":
        source = LC_DIR / f"lc_euregio_{resolution}_eusalp.asc"
    elif prefix == "srf":
        source = SRF_DIR / f"srf_euregio_{resolution}.asc"
    else:
        raise ValueError(prefix)

    cached = GRID_SOURCE_CACHE / source.name
    if cached.exists():
        print(f"using cached grid source {cached.name}", flush=True)
        return cached
    return source


def _read_ascii_header(handle) -> dict[str, float | int | str]:
    header: dict[str, float | int | str] = {}
    for _ in range(6):
        line = handle.readline()
        if not line:
            raise ValueError("Unexpected end of file while reading ASCII grid header")
        key, value = line.strip().split(maxsplit=1)
        key = key.lower()
        if key in {"ncols", "nrows"}:
            header[key] = int(float(value))
        elif key == "nodata_value":
            header[key] = value
        else:
            header[key] = float(value)
    if "xllcorner" not in header or "yllcorner" not in header:
        raise ValueError("ASCII crop helper expects xllcorner/yllcorner grids")
    return header


def _ascii_window(header: dict[str, float | int | str], bounds: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    minx, miny, maxx, maxy = bounds
    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    cell = float(header["cellsize"])
    xll = float(header["xllcorner"])
    yll = float(header["yllcorner"])
    top = yll + nrows * cell
    col0 = max(0, int(np.floor((minx - xll) / cell)))
    col1 = min(ncols, int(np.ceil((maxx - xll) / cell)))
    row0 = max(0, int(np.floor((top - maxy) / cell)))
    row1 = min(nrows, int(np.ceil((top - miny) / cell)))
    if row1 <= row0 or col1 <= col0:
        raise ValueError(f"Empty ASCII crop window for bounds {bounds}")
    return row0, row1, col0, col1


def _write_ascii_header(handle, header: dict[str, float | int | str], row0: int, row1: int, col0: int, col1: int) -> None:
    cell = float(header["cellsize"])
    xll = float(header["xllcorner"]) + col0 * cell
    yll = float(header["yllcorner"]) + (int(header["nrows"]) - row1) * cell
    handle.write(f"ncols        {col1 - col0}\n")
    handle.write(f"nrows        {row1 - row0}\n")
    handle.write(f"xllcorner    {xll:.10f}\n")
    handle.write(f"yllcorner    {yll:.10f}\n")
    handle.write(f"cellsize     {cell:.10f}\n")
    handle.write(f"NODATA_value {header.get('nodata_value', '-9999')}\n")


def _load_ascii_data(src: Path, header: dict[str, float | int | str]) -> np.ndarray:
    data = np.loadtxt(
        src,
        dtype=np.float32,
        skiprows=6,
        max_rows=int(header["nrows"]),
    )
    if data.ndim == 1:
        data = data.reshape((int(header["nrows"]), int(header["ncols"])))
    return data


def _write_ascii_grid(
    dst: Path,
    header: dict[str, float | int | str],
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    data: np.ndarray,
) -> None:
    with dst.open("w", encoding="utf-8") as out:
        _write_ascii_header(out, header, row0, row1, col0, col1)
        np.savetxt(out, data, fmt="%.6f")


def _crop_ascii_grid(src: Path, dst: Path, bounds: tuple[float, float, float, float]) -> None:
    print(f"cropping {src.name} -> {dst.name}", flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", errors="ignore") as f:
        header = _read_ascii_header(f)
    row0, row1, col0, col1 = _ascii_window(header, bounds)
    data = _load_ascii_data(src, header)
    _write_ascii_grid(dst, header, row0, row1, col0, col1, data[row0:row1, col0:col1])


def _write_neutral_svf(dem_path: Path, dst: Path) -> None:
    print(f"writing neutral SVF -> {dst.name}", flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dem_path.open("r", encoding="utf-8", errors="ignore") as f:
        header = _read_ascii_header(f)
    data = _load_ascii_data(dem_path, header)
    nodata = float(header.get("nodata_value", -9999))
    svf = np.where(np.isclose(data, nodata), -9999.0, 1.0)
    _write_ascii_grid(
        dst,
        {**header, "nodata_value": "-9999"},
        0,
        int(header["nrows"]),
        0,
        int(header["ncols"]),
        svf,
    )


def _write_grids(
    example_dir: Path,
    roi_geom: BaseGeometry,
    resolutions: Iterable[int],
    *,
    domain: str = DOMAIN,
) -> dict[int, dict[str, Path]]:
    bounds = roi_geom.buffer(GRID_CROP_BUFFER_M).bounds
    out: dict[int, dict[str, Path]] = {}
    for res in resolutions:
        res_paths: dict[str, Path] = {}
        dem_dst = example_dir / "grids" / f"dem_{domain}_{res}.asc"
        _crop_ascii_grid(_grid_source("dem", res), dem_dst, bounds)
        res_paths["dem"] = dem_dst
        for key in ("lc", "srf"):
            dst = example_dir / "grids" / f"{key}_{domain}_{res}.asc"
            _crop_ascii_grid(_grid_source(key, res), dst, bounds)
            res_paths[key] = dst
        svf_dst = example_dir / "grids" / f"svf_{domain}_{res}.asc"
        _write_neutral_svf(dem_dst, svf_dst)
        res_paths["svf"] = svf_dst
        out[res] = res_paths
        print(f"wrote grids for {res} m", flush=True)
    return out


def _load_forcing_metadata() -> gpd.GeoDataFrame:
    meta = gpd.read_file(FORCING_META).to_crs("EPSG:25832")
    meta["id"] = meta["provider"].astype(str) + "." + meta["stn_name"].astype(str)
    meta["name"] = meta["stn_name_orig"].fillna(meta["stn_name"]).astype(str)
    meta["x"] = meta.geometry.x
    meta["y"] = meta.geometry.y
    meta["alt"] = pd.to_numeric(meta["elev"], errors="coerce")
    return meta


def _trim_timeseries(src: Path, dst: Path, *, start: str, end: str) -> bool:
    if not src.is_file():
        return False
    df = pd.read_csv(src)
    time_col = "date" if "date" in df.columns else "datetime" if "datetime" in df.columns else None
    if time_col is None:
        raise ValueError(f"{src} missing required time column 'date' or 'datetime'")
    times = pd.to_datetime(df[time_col], errors="coerce")
    mask = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
    out = df.loc[mask].copy()
    if out.empty:
        return False
    if time_col != "date":
        out = out.rename(columns={time_col: "date"})
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return True


def _write_forcing(example_dir: Path, roi_geom: BaseGeometry) -> pd.DataFrame:
    meta = _load_forcing_metadata()
    selected = meta.loc[meta.geometry.within(roi_geom.buffer(FORCING_BUFFER_M))].copy()
    available_files = {path.name for path in FORCING_DIR.glob("*.csv")}
    rows = []
    selected_sorted = selected.sort_values("id")
    print(
        f"trimming forcing candidates: {len(selected_sorted)} within ROI + {FORCING_BUFFER_M / 1000:.0f} km",
        flush=True,
    )
    for idx, row in enumerate(selected_sorted.itertuples(index=False), start=1):
        if idx == 1 or idx % 20 == 0:
            print(f"trimmed forcing candidates: {idx}/{len(selected_sorted)}", flush=True)
        sid = str(row.id)
        if f"{sid}.csv" not in available_files:
            continue
        src = FORCING_DIR / f"{sid}.csv"
        dst = example_dir / "meteo" / f"{sid}.csv"
        if _trim_timeseries(src, dst, start=START_DATE, end=END_DATE):
            rows.append({"id": sid, "name": row.name, "x": row.x, "y": row.y, "alt": row.alt})
    stations = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    if stations.empty:
        raise ValueError("No forcing stations with data were selected")
    stations.to_csv(example_dir / "meteo" / "stations.csv", index=False)
    print(f"wrote forcing stations: {len(stations)}", flush=True)
    return stations


def _normalize_station_id(raw: object) -> str:
    return str(raw).strip()


def _read_snow_station_file(src: Path) -> pd.DataFrame:
    df = pd.read_csv(src)
    if "time" in df.columns and "snow_depth" in df.columns:
        out = df[["time", "snow_depth"]].copy()
    elif "date" in df.columns and "snow_height" in df.columns:
        out = pd.DataFrame(
            {
                "time": pd.to_datetime(df["date"], errors="coerce"),
                "snow_depth": pd.to_numeric(df["snow_height"], errors="coerce") / 100.0,
            }
        )
    else:
        raise ValueError(f"Unsupported snow observation format: {src}")
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out["snow_depth"] = pd.to_numeric(out["snow_depth"], errors="coerce")
    out = out.dropna(subset=["time", "snow_depth"])
    out = out.loc[(out["time"] >= pd.Timestamp(START_DATE)) & (out["time"] <= pd.Timestamp(END_DATE))].copy()
    out["snow_depth"] = out["snow_depth"].clip(lower=0.0)
    out["swe"] = np.nan
    out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out[["time", "snow_depth", "swe"]]


def _station_obs_source(station_id: str) -> Path:
    candidates = [
        SNOW_OBS_DIR / f"{station_id}.csv",
        SNOW_OBS_DIR / f"{station_id}_SH.csv",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    matches = sorted(SNOW_OBS_DIR.glob(f"{station_id}*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No snow observation CSV found for station {station_id}")


def _write_snow_observations(
    example_dir: Path,
    subdomains: gpd.GeoDataFrame,
    roi_geom: BaseGeometry,
    *,
    require_expected_count: bool = True,
    require_every_subdomain: bool = True,
) -> tuple[pd.DataFrame, list[date]]:
    meta = pd.read_csv(SNOW_OBS_DIR / "stations_snow_depth.csv")
    required = {"id", "name", "lat", "lon", "alt", "x", "y"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"stations_snow_depth.csv missing columns: {sorted(missing)}")
    gdf = gpd.GeoDataFrame(
        meta.copy(),
        geometry=gpd.points_from_xy(meta["x"], meta["y"]),
        crs="EPSG:25832",
    )
    selected = gdf.loc[gdf.geometry.within(roi_geom)].copy()
    selected["id"] = selected["id"].map(_normalize_station_id)
    if require_expected_count and len(selected) != 35:
        raise ValueError(f"Expected 35 snow stations inside ROI, got {len(selected)}")

    obs_dir = example_dir / "obs" / "stations"
    obs_dir.mkdir(parents=True, exist_ok=True)
    coverage: dict[str, set[date]] = {}
    for row in selected.sort_values("id").itertuples(index=False):
        sid = str(row.id)
        obs = _read_snow_station_file(_station_obs_source(sid))
        obs.to_csv(obs_dir / f"{sid}.csv", index=False)
        coverage[sid] = set(pd.to_datetime(obs["time"]).dt.date)

    selected.drop(columns=["geometry"]).sort_values("id").to_csv(obs_dir / "stations_snow_depth.csv", index=False)
    metadata = _station_da_metadata(selected, subdomains, coverage)
    metadata.to_csv(obs_dir / "stations_da_metadata.csv", index=False)
    station_events = _select_station_events(
        subdomains,
        selected,
        metadata,
        coverage,
        require_every_subdomain=require_every_subdomain,
    )
    _write_station_support_table(example_dir, subdomains, selected, metadata)
    print(f"wrote snow stations: {len(selected)}; station events: {[d.isoformat() for d in station_events]}", flush=True)
    return selected.drop(columns=["geometry"]).reset_index(drop=True), station_events


def _station_subdomain_lookup(subdomains: gpd.GeoDataFrame, stations: gpd.GeoDataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for station in stations.itertuples(index=False):
        point = Point(float(station.x), float(station.y))
        matches = subdomains.loc[subdomains.geometry.covers(point), "id"].astype(str).tolist()
        if matches:
            lookup[str(station.id)] = matches[0]
    return lookup


def _station_da_metadata(
    stations: gpd.GeoDataFrame,
    subdomains: gpd.GeoDataFrame,
    coverage: dict[str, set[date]],
) -> pd.DataFrame:
    lookup = _station_subdomain_lookup(subdomains, stations)
    holdouts: set[str] = set()
    for sid, group in stations.groupby(stations["id"].map(lookup)):
        if sid is None or len(group) < 2:
            continue
        candidates = sorted(
            [str(station_id) for station_id in group["id"]],
            key=lambda station_id: len(coverage.get(station_id, set())),
            reverse=True,
        )
        for candidate in candidates:
            da_ids = set(str(station_id) for station_id in group["id"]) - {candidate}
            if all(any(event_date in coverage.get(station_id, set()) for station_id in da_ids) for event_date in STATION_EVENT_CANDIDATES):
                holdouts.add(candidate)
                break

    rows = []
    for station in stations.sort_values("id").itertuples(index=False):
        sid = str(station.id)
        rows.append(
            {
                "station_id": sid,
                "station_uncertainty_pct": 25.0,
                "hs_sigma_abs_min": 0.10,
                "swe_sigma_abs_min": 20.0,
                "use_for_da": sid not in holdouts,
                "use_for_benchmark": True,
            }
        )
    return pd.DataFrame(rows)


def _select_station_events(
    subdomains: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    metadata: pd.DataFrame,
    coverage: dict[str, set[date]],
    *,
    require_every_subdomain: bool = True,
) -> list[date]:
    lookup = _station_subdomain_lookup(subdomains, stations)
    da_enabled = set(metadata.loc[metadata["use_for_da"].astype(bool), "station_id"].astype(str))
    station_ids_by_subdomain = {
        sub_id: [
            station_id
            for station_id, station_sub_id in lookup.items()
            if station_sub_id == sub_id and station_id in da_enabled
        ]
        for sub_id in subdomains["id"].astype(str)
    }
    required_subdomains = [
        sub_id
        for sub_id, station_ids in station_ids_by_subdomain.items()
        if require_every_subdomain or station_ids
    ]
    if not required_subdomains:
        return []

    candidate_dates = STATION_EVENT_CANDIDATES
    if not require_every_subdomain:
        all_dates = sorted({dt for station_id in da_enabled for dt in coverage.get(station_id, set())})
        candidate_dates = [dt for dt in all_dates if SEASON_START <= dt <= SEASON_END and dt.month in HIGH_SEASON_MONTHS]

    events: list[date] = []
    for event_date in candidate_dates:
        ok = True
        for sub_id in required_subdomains:
            station_ids = station_ids_by_subdomain[sub_id]
            if not any(event_date in coverage.get(station_id, set()) for station_id in station_ids):
                ok = False
                break
        if ok:
            events.append(event_date)

    if not require_every_subdomain:
        events = _thin_dates_by_spacing(events, spacing_days=28)
    if len(events) < 5 and require_every_subdomain:
        raise ValueError(f"Too few station events with DA support in every subdomain: {events}")
    return events


def _write_station_support_table(
    example_dir: Path,
    subdomains: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    metadata: pd.DataFrame,
) -> None:
    lookup = _station_subdomain_lookup(subdomains, stations)
    da_enabled = set(metadata.loc[metadata["use_for_da"].astype(bool), "station_id"].astype(str))
    benchmark_enabled = set(metadata.loc[metadata["use_for_benchmark"].astype(bool), "station_id"].astype(str))
    rows = []
    for sub_id in subdomains["id"].astype(str):
        station_ids = sorted(station_id for station_id, station_sub_id in lookup.items() if station_sub_id == sub_id)
        rows.append(
            {
                "subdomain_id": sub_id,
                "station_count": len(station_ids),
                "da_station_count": sum(station_id in da_enabled for station_id in station_ids),
                "benchmark_station_count": sum(station_id in benchmark_enabled for station_id in station_ids),
            }
        )
    (example_dir / "obs" / "stations").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(example_dir / "obs" / "stations" / "station_support_by_subdomain.csv", index=False)


def _thin_dates_by_spacing(dates: Iterable[date], *, spacing_days: int) -> list[date]:
    selected: list[date] = []
    last: date | None = None
    for dt in sorted(set(dates)):
        if last is None or (dt - last).days >= spacing_days:
            selected.append(dt)
            last = dt
    return selected


def _scf_zip_entries() -> list[tuple[date, str]]:
    pattern = re.compile(r"SnowFLAKES_(\d{8})_v3_eurac\.nc$")
    out = []
    with zipfile.ZipFile(_scf_zip_path()) as zf:
        for name in zf.namelist():
            match = pattern.search(name)
            if not match:
                continue
            dt = datetime.strptime(match.group(1), "%Y%m%d").date()
            if SEASON_START <= dt <= SEASON_END:
                out.append((dt, name))
    return sorted(out)


def _scf_zip_path() -> Path:
    if SCF_ZIP_CACHE.exists() and (
        not SCF_ZIP_SOURCE.exists() or SCF_ZIP_CACHE.stat().st_size == SCF_ZIP_SOURCE.stat().st_size
    ):
        print(f"using cached SCF zip {SCF_ZIP_CACHE}", flush=True)
        return SCF_ZIP_CACHE
    if not SCF_ZIP_SOURCE.exists():
        raise FileNotFoundError(f"SCF zip source not found and cache is unavailable: {SCF_ZIP_SOURCE}")
    return SCF_ZIP_SOURCE


def _extract_zip_member(zf: zipfile.ZipFile, name: str, temp_dir: Path) -> Path:
    target = temp_dir / Path(name).name
    with zf.open(name) as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def _subset_scf_dataset(path: Path, bounds: tuple[float, float, float, float]) -> xr.Dataset:
    minx, miny, maxx, maxy = bounds
    pad = 100.0
    ds = xr.open_dataset(path)
    subset = ds.sel(x=slice(minx - pad, maxx + pad), y=slice(maxy + pad, miny - pad))
    if "spatial_ref" in subset:
        for var_name in ("fsc", "uncertainty"):
            if var_name in subset:
                subset[var_name].attrs["grid_mapping"] = "spatial_ref"
    return subset


def _scf_transform(ds: xr.Dataset):
    x = ds["x"].values
    y = ds["y"].values
    res_x = float(abs(x[1] - x[0]))
    res_y = float(abs(y[1] - y[0]))
    return from_origin(float(x.min() - res_x / 2.0), float(y.max() + res_y / 2.0), res_x, res_y)


def _subdomain_masks_for_scf(ds: xr.Dataset, subdomains: gpd.GeoDataFrame) -> dict[str, np.ndarray]:
    shape = (int(ds.sizes["y"]), int(ds.sizes["x"]))
    transform = _scf_transform(ds)
    masks: dict[str, np.ndarray] = {}
    for row in subdomains.itertuples(index=False):
        mask = features.rasterize(
            [(mapping(row.geometry), 1)],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)
        if not mask.any():
            raise ValueError(f"No SCF pixels for subdomain {row.id}")
        masks[str(row.id)] = mask
    return masks


def _scan_scf_candidates(
    subdomains: gpd.GeoDataFrame,
    roi_geom: BaseGeometry,
    *,
    excluded_dates: set[date],
) -> list[ScfCandidate]:
    entries = _scf_zip_entries()
    print(f"checking SCF candidates: {len(entries)}", flush=True)
    candidates: list[ScfCandidate] = []
    masks: dict[str, np.ndarray] | None = None
    with tempfile.TemporaryDirectory(prefix="subdomain_example_scf_") as tmp, zipfile.ZipFile(_scf_zip_path()) as zf:
        temp_dir = Path(tmp)
        checked = 0
        for dt, name in entries:
            if dt in excluded_dates:
                continue
            checked += 1
            if checked == 1 or checked % 20 == 0:
                print(f"checked SCF candidates: {checked}", flush=True)
            extracted = _extract_zip_member(zf, name, temp_dir)
            ds = None
            try:
                ds = _subset_scf_dataset(extracted, roi_geom.bounds)
                if masks is None:
                    masks = _subdomain_masks_for_scf(ds, subdomains)
                arr = ds["fsc"].isel(band=0).values
                cloud_by_subdomain: dict[str, float] = {}
                for sub_id, mask in masks.items():
                    vals = arr[mask]
                    valid = np.isfinite(vals) & (vals >= 0.0) & (vals <= 100.0)
                    cloud = np.isin(vals, FSC_CLOUD_CLASSES)
                    denominator = int(np.count_nonzero(valid | cloud))
                    cloud_by_subdomain[sub_id] = (
                        float(np.count_nonzero(cloud) / denominator) if denominator > 0 else 1.0
                    )
                candidates.append(ScfCandidate(dt, name, cloud_by_subdomain))
            finally:
                if ds is not None:
                    try:
                        ds.close()
                    except Exception:
                        pass
                extracted.unlink(missing_ok=True)
    print(f"screened SCF dates: {len(candidates)}", flush=True)
    return candidates


def _scf_limit_for_subdomain(subdomain_id: str) -> float:
    return float(FSC_SUBDOMAIN_CLOUD_OVERRIDES.get(subdomain_id, FSC_MAX_CLOUD_FRACTION))


def _select_scf_events_by_subdomain(
    candidates: list[ScfCandidate],
    subdomains: gpd.GeoDataFrame,
) -> list[ScfCandidate]:
    by_date = {cand.date: cand for cand in candidates}
    missing = [dt for dt in FSC_EVENT_CANDIDATES if dt not in by_date]
    if missing:
        raise ValueError(f"Configured FSC dates missing from SnowFLAKES source: {[dt.isoformat() for dt in missing]}")
    overlap = sorted(set(FSC_EVENT_CANDIDATES) & set(STATION_EVENT_CANDIDATES))
    if overlap:
        raise ValueError(f"FSC and station DA events must not share dates: {[dt.isoformat() for dt in overlap]}")

    selected: list[ScfCandidate] = []
    for dt in FSC_EVENT_CANDIDATES:
        cand = by_date[dt]
        selected_for = tuple(
            sorted(
                subdomain_id
                for subdomain_id in subdomains["id"].astype(str)
                if cand.cloud_by_subdomain[subdomain_id] <= _scf_limit_for_subdomain(subdomain_id)
            )
        )
        if not selected_for:
            raise ValueError(f"Configured FSC date {dt.isoformat()} is too cloudy in every subdomain")
        selected.append(
            ScfCandidate(
                cand.date,
                cand.zip_name,
                cand.cloud_by_subdomain,
                selected_for_subdomains=selected_for,
            )
        )

    print("selected project-level FSC events:", flush=True)
    for cand in selected:
        print(
            f"  {cand.date.isoformat()}: "
            + ", ".join(
                f"{sub_id}={cand.cloud_by_subdomain[sub_id]:.0%}"
                for sub_id in sorted(cand.cloud_by_subdomain)
            )
            + f" (eligible: {', '.join(cand.selected_for_subdomains)})",
            flush=True,
        )
    return [
        cand
        for cand in sorted(selected, key=lambda cand: cand.date)
    ]


def _target_slots(*, spacing_days: int = BIWEEKLY_SPACING_DAYS) -> list[date]:
    slots: list[date] = []
    current = SEASON_START
    while current <= SEASON_END:
        slots.append(current)
        current += timedelta(days=spacing_days)
    return slots


def _select_biweekly_scf_events_by_subdomain(
    candidates: list[ScfCandidate],
    subdomains: gpd.GeoDataFrame,
    *,
    shift_days: int = EVENT_SHIFT_DAYS,
) -> list[ScfCandidate]:
    selected: list[ScfCandidate] = []
    used_dates: set[date] = set()
    subdomain_ids = sorted(subdomains["id"].astype(str))
    for target in _target_slots():
        window_start = target - timedelta(days=shift_days)
        window_end = target + timedelta(days=shift_days)
        window = [
            cand
            for cand in candidates
            if window_start <= cand.date <= window_end and cand.date not in used_dates
        ]
        if not window:
            continue

        scored: list[tuple[int, float, int, ScfCandidate, tuple[str, ...]]] = []
        for cand in window:
            selected_for = tuple(
                sorted(
                    subdomain_id
                    for subdomain_id in subdomain_ids
                    if cand.cloud_by_subdomain[subdomain_id] <= _scf_limit_for_subdomain(subdomain_id)
                )
            )
            if not selected_for:
                continue
            mean_cloud = float(np.mean([cand.cloud_by_subdomain[sub_id] for sub_id in selected_for]))
            distance = abs((cand.date - target).days)
            scored.append((len(selected_for), -mean_cloud, -distance, cand, selected_for))
        if not scored:
            continue
        _, _, _, cand, selected_for = max(scored, key=lambda item: (item[0], item[1], item[2]))
        selected.append(
            ScfCandidate(
                cand.date,
                cand.zip_name,
                cand.cloud_by_subdomain,
                selected_for_subdomains=selected_for,
            )
        )
        used_dates.add(cand.date)

    if not selected:
        raise ValueError("No biweekly FSC events could be selected from SnowFLAKES source")
    coverage = {
        sub_id: sum(sub_id in cand.selected_for_subdomains for cand in selected)
        for sub_id in subdomain_ids
    }
    missing = [sub_id for sub_id, count in coverage.items() if count == 0]
    if missing:
        raise ValueError(f"Biweekly FSC selection leaves subdomains without FSC support: {missing}")

    print("selected biweekly FSC events:", flush=True)
    for cand in selected:
        print(
            f"  {cand.date.isoformat()}: eligible {len(cand.selected_for_subdomains)}/{len(subdomain_ids)}",
            flush=True,
        )
    return sorted(selected, key=lambda cand: cand.date)


def _write_scf_files(
    example_dir: Path,
    selected: list[ScfCandidate],
    roi_geom: BaseGeometry,
    *,
    domain: str = DOMAIN,
) -> None:
    out_dir = example_dir / "obs" / "snowcover"
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="subdomain_example_scf_clip_") as tmp, zipfile.ZipFile(_scf_zip_path()) as zf:
        temp_dir = Path(tmp)
        for cand in selected:
            extracted = _extract_zip_member(zf, cand.zip_name, temp_dir)
            subset = None
            try:
                subset = _subset_scf_dataset(extracted, roi_geom.bounds)
                out = out_dir / f"SnowFLAKES_{cand.date.strftime('%Y%m%d')}_v3_eurac_{domain}.nc"
                encoding = {
                    "fsc": {"zlib": True, "complevel": 4, "dtype": "float32"},
                    "uncertainty": {"zlib": True, "complevel": 4, "dtype": "float32"},
                }
                subset.to_netcdf(out, engine="netcdf4", encoding=encoding)
                print(f"wrote SCF {cand.date.isoformat()} -> {out.name}", flush=True)
            finally:
                if subset is not None:
                    try:
                        subset.close()
                    except Exception:
                        pass
                extracted.unlink(missing_ok=True)


def _write_setup_yaml(
    example_dir: Path,
    stations: pd.DataFrame,
    *,
    domain: str = DOMAIN,
    default_resolution: int = DEFAULT_RESOLUTION,
) -> None:
    points = [
        {"x": float(row.x), "y": float(row.y), "name": str(row.id)}
        for row in stations.sort_values("id").itertuples(index=False)
    ]
    setup = {
        "domain": domain,
        "resolution": default_resolution,
        "timestep": "3H",
        "crs": "epsg:25832",
        "timezone": 1,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "input_data": {
            "grids": {"dir": "grids"},
            "meteo": {
                "dir": "meteo",
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
                "variables": [
                    {"var": "snow.depth", "name": "snowdepth_daily", "freq": "D"},
                ],
            },
        },
        "meteo": {
            "interpolation": {
                "temperature": {
                    "trend_method": "fixed",
                    "extrapolate": True,
                    "lapse_rate": _flow([-0.0026, -0.0035, -0.0047, -0.0053, -0.0052, -0.0053, -0.0049, -0.0047, -0.0042, -0.0033, -0.0035, -0.0031]),
                },
                "precipitation": {
                    "trend_method": "fractional",
                    "extrapolate": True,
                    "lapse_rate": _flow([0.00048, 0.00046, 0.00041, 0.00033, 0.00028, 0.00025, 0.00024, 0.00025, 0.00028, 0.00033, 0.00041, 0.00046]),
                },
                "humidity": {
                    "trend_method": "fixed",
                    "extrapolate": True,
                    "lapse_rate": _flow([-0.0044, -0.0046, -0.0049, -0.0048, -0.0046, -0.0047, -0.0043, -0.0042, -0.0045, -0.0044, -0.0047, -0.0046]),
                },
                "cloudiness": {"day_method": "clear_sky_fraction", "night_method": "humidity", "allow_fallback": True},
                "wind_speed": {"trend_method": "regression", "extrapolate": False},
            },
            "precipitation_phase": {"method": "wet_bulb_temp", "threshold_temp": 273.65, "temp_range": 1.0},
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
    _dump_yaml(example_dir / "subdomains.yml", setup)


def _write_project_yaml(
    example_dir: Path,
    station_events: list[date],
    scf_events: list[ScfCandidate],
    *,
    compact_output: bool = False,
    fsc_unc_min: float = 5.0,
) -> None:
    events = [
        *({"date": dt.isoformat(), "variable": "station_hs", "product": "STATION"} for dt in station_events),
        *({"date": cand.date.isoformat(), "variable": "scf", "product": "SNOWCOVER"} for cand in scf_events),
    ]
    events = sorted(events, key=lambda item: (str(item["date"]), str(item["variable"])))
    project = {
        "run_mode": "subdomain",
        "start_date": START_DATE,
        "end_date": END_DATE,
        "obs": {
            "stations": {"dir": "obs/stations"},
            "snowcover": {
                "dir": "obs/snowcover",
                "product_tag": "SNOWCOVER",
                "classes": {
                    "valid": _flow(range(0, 101)),
                    "cloud": _flow([int(value) for value in FSC_CLOUD_CLASSES]),
                    "water": _flow([]),
                    "nodata": _flow([]),
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
                    "scf": {"max_cloud_fraction": FSC_MAX_CLOUD_FRACTION},
                    "station_hs": {"min_active_stations": 1, "max_time_delta_hours": 36},
                },
                "subdomains": {
                    subdomain_id: {"variables": {"scf": {"max_cloud_fraction": threshold}}}
                    for subdomain_id, threshold in FSC_SUBDOMAIN_CLOUD_OVERRIDES.items()
                },
            },
            "landcover_mask": {"enabled": True, "classes_to_exclude": _flow([2, 3, 13])},
            "likelihood": {
                "scf": {
                    "obs_sigma": 0.10,
                    "use_binomial": True,
                    "sigma_floor": 0.05,
                    "sigma_cloud_scale": 0.10,
                    "min_sigma": 0.03,
                }
            },
            "uncertainty": {
                "scf": {
                    "enabled": True,
                    "input_dir": "obs/snowcover",
                    "ingest": {"scf_variable": "fsc", "uncertainty_source": "internal", "time_variable": "time"},
                    "assimilation": {"sigma_mode": "uncertainty_layer", "aggregate_metric": "unc_mean"},
                    "u_min": float(fsc_unc_min),
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
            "restart": {"use_state": True, "dump_state": True, "state_pattern": "model_state.pickle.gz"},
            "output": {
                "retention": "compact" if compact_output else "full",
                "grids": {
                    "format": "netcdf",
                    "compress": True,
                    "dims": _flow(["x", "y", "time"]),
                    "variables": [
                        {
                            "var": "snowdepth_daily",
                            "name": "snowdepth_daily",
                            "metrics": _flow(["open_loop", "ens_mean", "increment", "analysis_mean", "analysis_increment"]),
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
            "assimilation_events": events,
        },
    }
    project_dir = example_dir / "projects" / PROJECT_NAME
    _dump_yaml(project_dir / f"{PROJECT_NAME}.yml", project)


def _write_maps_yaml(example_dir: Path) -> None:
    maps = {
        "maps": {
            "subdomain_example_setup_overview": {
                "title": "Subdomain example setup overview",
                "output_name": "setup_overview",
                "layout": {"nrows": 2, "ncols": 3},
                "defaults": {"show_scalebar": True},
                "panels": [
                    {
                        "row": 0,
                        "col": 0,
                        "kind": "overview",
                        "title": "overview",
                        "scale": 2_500_000,
                        "roi_label": "Subdomain ROI",
                    },
                    {
                        "row": 0,
                        "col": 1,
                        "kind": "roi",
                        "title": "ROI and stations",
                        "show_station_marker": True,
                        "show_stations_name": False,
                        "show_stations_elev": False,
                        "below_items": [{"kind": "station_symbol", "label": "Meteo and snow stations"}],
                    },
                    {"row": 0, "col": 2, "kind": "dem", "title": "elevation"},
                    {"row": 1, "col": 0, "kind": "landcover", "title": "land cover"},
                    {
                        "row": 1,
                        "col": 1,
                        "kind": "hillshade",
                        "title": "terrain shading",
                        "show_station_marker": True,
                        "show_stations_name": False,
                    },
                    {
                        "row": 1,
                        "col": 2,
                        "kind": "srf",
                        "title": "snow redistribution factor",
                        "show_hillshade": True,
                    },
                ],
            },
        }
    }
    _dump_yaml(example_dir / "projects" / PROJECT_NAME / "maps.yml", maps)


def _write_plots_yaml(example_dir: Path) -> None:
    text = """# Custom result overview configuration for subdomain reports.
#
# This mirrors the shipped Rofental custom overview intent but omits station
# panels and wet-snow panels that are not part of the shipped subdomain setup.
panels:
  - panel: fSC
    show_obs: true
    title: snow cover fraction subdomain ROI - openAMUNDSEN ensemble and satellite data

  - panel: roi-sd
    title: mean snow depth subdomain ROI - openAMUNDSEN ensemble and open loop

  - panel: ess

  - panel: scores-crpss
"""
    (example_dir / "projects" / PROJECT_NAME / "plots.yml").write_text(text, encoding="utf-8")


def _write_readme(
    example_dir: Path,
    station_count: int,
    forcing_count: int,
    scf_events: list[ScfCandidate],
    *,
    region_set: str = "north-tyrol",
    resolutions: tuple[int, ...] = RESOLUTIONS,
    default_resolution: int = DEFAULT_RESOLUTION,
    compact_output: bool = False,
) -> None:
    domain_text = "the full Euregio avalanche-report region set" if region_set == "euregio" else "a larger alpine ROI as 8 avalanche-report subdomains"
    station_text = (
        f"{station_count} North Tyrol station snow-depth series in `obs/stations`; station DA/benchmarking is only available in subdomains with local station support"
        if region_set == "euregio"
        else f"{station_count} ROI stations in `obs/stations`, with `use_for_da` and `use_for_benchmark` role flags"
    )
    text = f"""# Subdomain Example

This setup covers {domain_text}.

- Spatial domain: `env/subdomains.gpkg` and `env/roi.gpkg`, EPSG:25832.
- Temporal domain: `{START_DATE}` to `{END_DATE}`.
- Default resolution: `{default_resolution} m`; available grid resolutions: `{', '.join(map(str, resolutions))} m`.
- Forcing: {forcing_count} `openamundsen-v2` stations within the ROI plus {int(FORCING_BUFFER_M / 1000)} km buffer, trimmed to the project window.
- Station snow depth: {station_text}.
- FSC: {len(scf_events)} clipped SnowFLAKES NetCDF files in `obs/snowcover`; configured as project-level event candidates. The subdomain event filter drops cloudy subdomain/date combinations above {FSC_MAX_CLOUD_FRACTION:.0%} cloud cover, except documented per-subdomain overrides in the project YAML.
- DA config: 30 ensemble members, ESS threshold ratio 0.7, {"compact" if compact_output else "full"} output retention, and four-variable forcing/rejuvenation perturbations for temperature, precipitation, humidity, and shortwave radiation.
- Maps: `projects/{PROJECT_NAME}/maps.yml` adds a setup overview. Generated DA-event maps are rendered automatically from the configured assimilation events.

Run the example with:

```bash
oa-da-subdomain pipeline --setup-dir {example_dir} --project-dir {example_dir}/projects/{PROJECT_NAME} --regions {example_dir}/env/subdomains.gpkg --station-buffer-km 10 --grid-buffer-m {int(SUBDOMAIN_GRID_BUFFER_M)} --max-workers 8 --inner-max-workers 3 --overwrite
```

The FRAMES-specific build logic is external to `openamundsen_da`:
`/home/franz/workspace/repos/scripts/04-openAMUNDSEN/buildSubdomainExample.py`.
"""
    (example_dir / "README.md").write_text(text, encoding="utf-8")

    agents = """# Subdomains Example Notes

Inherit `examples/AGENTS.md`. This file adds local rules for the canonical sub-domain baseline.

- `examples/subdomains` is the shipped sub-domain regression setup and must stay compatible with the CI sub-domain pipelines.
- Keep `env/subdomains.gpkg`, project layout, and expected merged outputs aligned with `scripts/ci/run_integration_tests_subdomain.sh` and `scripts/ci/run_integration_tests_model_subdomain.sh`.
- If sub-domain reports, manifests, event filtering, or output paths change, update validators, tests, and docs in the same work.
- Preserve deterministic region handling and merged project-level outputs because CI validates them as a contract.
- Prefer explicit project-level event candidates plus generic sub-domain filtering over hidden per-region special cases.
- Avoid large data churn unless it is necessary for the behavior under test.
"""
    (example_dir / "AGENTS.md").write_text(agents, encoding="utf-8")


def _write_manifest(
    example_dir: Path,
    archive_path: Path | None,
    station_events: list[date],
    scf_events: list[ScfCandidate],
    forcing_count: int,
    station_count: int,
    *,
    region_set: str = "north-tyrol",
    domain: str = DOMAIN,
    resolutions: tuple[int, ...] = RESOLUTIONS,
    default_resolution: int = DEFAULT_RESOLUTION,
    compact_output: bool = False,
) -> None:
    manifest = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "archive_path": str(archive_path) if archive_path else None,
        "region_set": region_set,
        "source_paths": {
            "roi": str(_subdomain_roi_source(example_dir) or ROI_PATH),
            "raw_subregions": str(RAW_SUBREGIONS_PATH),
            "forcing": str(FORCING_DIR),
            "forcing_meta": str(FORCING_META),
            "snow_obs": str(SNOW_OBS_DIR),
            "scf_zip": str(SCF_ZIP_SOURCE),
            "scf_zip_cache": str(SCF_ZIP_CACHE) if SCF_ZIP_CACHE.exists() else None,
        },
        "domain": domain,
        "project": PROJECT_NAME,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "resolutions": list(resolutions),
        "default_resolution": default_resolution,
        "compact_output": compact_output,
        "forcing_station_count": forcing_count,
        "snow_station_count": station_count,
        "station_events": [d.isoformat() for d in station_events],
        "scf_events": [
            {
                "date": cand.date.isoformat(),
                "source": cand.zip_name,
                "selected_for_subdomains": list(cand.selected_for_subdomains),
                "max_cloud_fraction": max(cand.cloud_by_subdomain.values()),
                "cloud_by_subdomain": cand.cloud_by_subdomain,
            }
            for cand in scf_events
        ],
    }
    (example_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = []
    for cand in scf_events:
        for sub_id, cloud_fraction in cand.cloud_by_subdomain.items():
            rows.append(
                {
                    "date": cand.date.isoformat(),
                    "subdomain_id": sub_id,
                    "cloud_fraction": cloud_fraction,
                    "selected_for_subdomain": sub_id in cand.selected_for_subdomains,
                }
            )
    pd.DataFrame(rows).to_csv(example_dir / "obs" / "snowcover" / "selected_scf_quality.csv", index=False)
    pd.DataFrame(rows).to_csv(example_dir / "obs" / "snowcover" / "fsc_quality_by_subdomain_date.csv", index=False)


def _build_options(region_set: str, output_dir: Path | None) -> BuildOptions:
    if region_set == "euregio":
        return BuildOptions(
            region_set=region_set,
            example_dir=(output_dir or EUREGIO_EXAMPLE_DIR),
            domain="euregio_subdomain_example",
            resolutions=EUREGIO_RESOLUTIONS,
            build_resolutions=EUREGIO_BUILD_RESOLUTIONS,
            default_resolution=EUREGIO_DEFAULT_RESOLUTION,
            compact_output=True,
        )
    return BuildOptions(
        region_set="north-tyrol",
        example_dir=(output_dir or EXAMPLE_DIR),
        domain=DOMAIN,
        resolutions=RESOLUTIONS,
        build_resolutions=BUILD_RESOLUTIONS,
        default_resolution=DEFAULT_RESOLUTION,
        compact_output=False,
    )


def build(*, no_archive: bool = False, region_set: str = "north-tyrol", output_dir: Path | None = None) -> None:
    options = _build_options(region_set, output_dir)
    subdomains = _read_subdomains(options.region_set, options.example_dir)
    archive_path = None if no_archive else _archive_existing_example(options.example_dir, ARCHIVE_ROOT)
    _clean_output(options.example_dir)
    roi_geom = _write_vectors(options.example_dir, subdomains, domain=options.domain)
    _write_grids(options.example_dir, roi_geom, options.build_resolutions, domain=options.domain)
    forcing = _write_forcing(options.example_dir, roi_geom)
    stations, station_events = _write_snow_observations(
        options.example_dir,
        subdomains,
        roi_geom,
        require_expected_count=(options.region_set == "north-tyrol"),
        require_every_subdomain=(options.region_set == "north-tyrol"),
    )
    scf_candidates = _scan_scf_candidates(subdomains, roi_geom, excluded_dates=set(station_events))
    scf_events = (
        _select_biweekly_scf_events_by_subdomain(scf_candidates, subdomains)
        if options.region_set == "euregio"
        else _select_scf_events_by_subdomain(scf_candidates, subdomains)
    )
    if len(scf_events) < 8:
        raise ValueError(f"Too few FSC events selected: {len(scf_events)}")
    _write_scf_files(options.example_dir, scf_events, roi_geom, domain=options.domain)
    _write_setup_yaml(options.example_dir, stations, domain=options.domain, default_resolution=options.default_resolution)
    _write_project_yaml(
        options.example_dir,
        station_events,
        scf_events,
        compact_output=options.compact_output,
        fsc_unc_min=5.0,
    )
    _write_maps_yaml(options.example_dir)
    _write_plots_yaml(options.example_dir)
    _write_readme(
        options.example_dir,
        station_count=len(stations),
        forcing_count=len(forcing),
        scf_events=scf_events,
        region_set=options.region_set,
        resolutions=options.resolutions,
        default_resolution=options.default_resolution,
        compact_output=options.compact_output,
    )
    _write_manifest(
        options.example_dir,
        archive_path=archive_path,
        station_events=station_events,
        scf_events=scf_events,
        forcing_count=len(forcing),
        station_count=len(stations),
        region_set=options.region_set,
        domain=options.domain,
        resolutions=options.resolutions,
        default_resolution=options.default_resolution,
        compact_output=options.compact_output,
    )
    print(f"built example -> {options.example_dir}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-archive", action="store_true", help="Do not archive the existing examples/subdomains directory first")
    parser.add_argument(
        "--region-set",
        choices=("north-tyrol", "euregio"),
        default="north-tyrol",
        help="Subdomain region set to build (default: north-tyrol shipped example)",
    )
    parser.add_argument("--output-dir", type=Path, help="Output setup directory (default depends on --region-set)")
    args = parser.parse_args()
    build(no_archive=bool(args.no_archive), region_set=args.region_set, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
