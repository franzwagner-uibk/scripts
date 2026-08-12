#!/usr/bin/env python3
"""Finalize the accepted six-season North Tyrol snapshot without propagation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from da_event_scheduler import (
    SchedulePolicy,
    ScheduleResult,
    StationRoleResult,
    load_policy,
    log_selected_station_interpolations,
    match_station_support,
    parse_date,
    read_csv_records,
    schedule_with_adaptive_roles,
)


EXPECTED_IMAGE = (
    "ghcr.io/openamundsen/openamundsen-da:0.9.4@"
    "sha256:f3834a701e116b9ab11c50677d94236bffcd5d9adb045ae6b871b3ccf2c98723"
)
EXPECTED_BUILDER_COMMIT = "fbb9eceb6c075583deff54892cd8c2794d347b99"
EXPECTED_PROJECTS = tuple(f"project_{year}_{year + 1}" for year in range(2017, 2023))
EXPECTED_SUBDOMAINS = 8
FINALIZER_SCHEMA_VERSION = 2
LEGACY_LAYOUT = "legacy_snapshot"
CANONICAL_LAYOUT = "canonical_setup"
EXPECTED_FORCING_SOURCE_TIMESTEP = timedelta(hours=1)
FORCING_FLATLINE_MINIMUM_DURATION = timedelta(hours=24)
FORCING_FLATLINE_SEVERE_DURATION = timedelta(days=7)
MODEL_FORCING_VARIABLES = (
    "precip",
    "rel_hum",
    "sw_in",
    "temp",
    "wind_dir",
    "wind_speed",
)


def parse_args() -> argparse.Namespace:
    """Parse the documented North Tyrol finalizer interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-root", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--discard-runtime-artifacts",
        action="store_true",
        help="Explicitly allow a canonical refresh to replace completed results/restart artifacts",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image_reference(image: str) -> None:
    """Require an immutable container reference without pinning one release."""

    if not re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("Image must be an immutable reference pinned by a sha256 digest")


def detect_setup_layout(root: Path) -> str:
    """Distinguish the legacy snapshot from the documentation-shaped setup."""

    root = Path(root)
    legacy = (root / "snapshot_manifest.json").is_file() and (root / "data_working").is_dir()
    canonical = all((root / name).is_dir() for name in ("env", "grids", "meteo", "obs", "projects", "raw"))
    setup_yamls = sorted(root.glob("*.yml"))
    if legacy and canonical:
        raise ValueError("Setup mixes legacy snapshot and canonical layout markers")
    if legacy:
        return LEGACY_LAYOUT
    if canonical and len(setup_yamls) == 1:
        return CANONICAL_LAYOUT
    raise ValueError(f"Cannot identify North Tyrol setup layout: {root}")


def tree_inventory(root: Path, *, excluded: Iterable[Path] = ()) -> list[dict[str, Any]]:
    """Return deterministic path, size and hash records for a tree."""

    excluded_resolved = {Path(path).resolve() for path in excluded}
    rows: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in Path(root).rglob("*") if candidate.is_file()):
        if path.resolve() in excluded_resolved:
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def inventory_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash one normalized tree inventory."""

    payload = json.dumps(list(rows), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_recorded_data_hashes(root: Path) -> dict[str, int]:
    """Verify accepted raw and working files against their recorded hashes."""

    raw_records = _read_json(root / "provenance" / "raw_files.json")
    working_records = _read_json(root / "provenance" / "working_files.json")
    if not isinstance(raw_records, list) or not isinstance(working_records, list):
        raise ValueError("Snapshot provenance hash tables must be lists")
    for record in raw_records:
        copy_path = root / str(record["raw_copy"])
        if not copy_path.is_file() or sha256_file(copy_path) != str(record["sha256"]):
            raise ValueError(f"Raw-copy hash mismatch: {copy_path}")
    for record in working_records:
        working_path = root / str(record["path"])
        if not working_path.is_file() or sha256_file(working_path) != str(record["sha256"]):
            raise ValueError(f"Working-data hash mismatch: {working_path}")
    return {"raw_files": len(raw_records), "working_files": len(working_records)}


def validate_source_snapshot(root: Path, image: str) -> dict[str, Any]:
    """Validate the immutable accepted snapshot state required for finalization."""

    root = Path(root).resolve()
    validate_image_reference(image)
    if image != EXPECTED_IMAGE:
        raise ValueError(f"Image must be pinned exactly to {EXPECTED_IMAGE}")
    manifest_path = root / "snapshot_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "READY_FOR_EVENT_SELECTION":
        raise ValueError("Snapshot manifest is not READY_FOR_EVENT_SELECTION")
    if manifest.get("builder_commit") != EXPECTED_BUILDER_COMMIT:
        raise ValueError("Unexpected snapshot builder commit")
    if manifest.get("image") != image:
        raise ValueError("Snapshot image differs from the requested pinned image")
    if not (root / "READY_FOR_EVENT_SELECTION").is_file():
        raise FileNotFoundError(root / "READY_FOR_EVENT_SELECTION")
    for forbidden in ("projects", "steps", "results", "READY_TO_RUN"):
        if (root / forbidden).exists():
            raise ValueError(f"Unexpected active artifact in accepted snapshot: {forbidden}")
    pending_root = root / "projects_pending_events"
    actual_projects = tuple(sorted(path.name for path in pending_root.iterdir() if path.is_dir()))
    if actual_projects != EXPECTED_PROJECTS:
        raise ValueError(f"Unexpected pending projects: {actual_projects}")
    for project_name in EXPECTED_PROJECTS:
        project_dir = pending_root / project_name
        project = _read_yaml(project_dir / f"{project_name}.yml")
        events = project.get("data_assimilation", {}).get("assimilation_events")
        if events != [] or not (project_dir / "PENDING_EVENTS").is_file():
            raise ValueError(f"Project is not in the accepted pending state: {project_name}")
    return {
        "root": str(root),
        "parent_manifest_sha256": sha256_file(manifest_path),
        "pending_tree_sha256": inventory_digest(tree_inventory(pending_root)),
        "builder_commit": manifest["builder_commit"],
        "image": image,
    }


def build_schedules(
    root: Path,
    policy_path: Path,
) -> tuple[StationRoleResult, dict[str, ScheduleResult]]:
    """Build all six schedules using one station split in either supported layout."""

    if detect_setup_layout(root) == CANONICAL_LAYOUT:
        return _build_canonical_schedules(root, policy_path)

    manifest = _read_json(root / "snapshot_manifest.json")
    timestep_inventory = root / "inventories" / "snow_station_timestep_support.csv"
    if not timestep_inventory.is_file():
        raise FileNotFoundError(
            "Legacy snapshot lacks inventories/snow_station_timestep_support.csv; "
            "daily summaries are insufficient for half-timestep station matching"
        )
    windows = tuple(
        (
            str(item["project"]),
            parse_date(item["start"], field="project start"),
            parse_date(item["end"], field="project end"),
        )
        for item in manifest["seasons"]
    )
    return schedule_with_adaptive_roles(
        policy=load_policy(policy_path),
        fsc_rows=_read_legacy_fsc_inventory(root),
        snow_rows=read_csv_records(timestep_inventory),
        station_rows=read_csv_records(root / "data_working" / "obs" / "stations" / "stations_snow_depth.csv"),
        windows=windows,
    )


def _read_legacy_fsc_inventory(root: Path) -> list[dict[str, str]]:
    """Require FSC evidence produced from stable archive masks and uncertainties."""

    path = root / "inventories" / "fsc_scene_subdomain_quality.csv"
    rows = read_csv_records(path)
    required = {
        "reference_count",
        "permanent_nodata_count",
        "water_mask_stable",
        "valid_count",
        "cloud_count",
        "nodata_count",
        "water_count",
        "cloud_reference_fraction",
        "invalid_reference_fraction",
        "uncertainty_valid_fsc_count",
        "uncertainty_mean",
        "uncertainty_p90",
    }
    columns = set(rows[0]) if rows else set()
    missing = sorted(required - columns)
    if missing:
        raise ValueError(
            "Legacy FSC inventory predates the stable-reference quality schema; "
            "rebuild it from retained NetCDF scenes before scheduling. Missing columns: "
            + ", ".join(missing)
        )
    if any(str(row["water_mask_stable"]).strip().lower() not in {"true", "1"} for row in rows):
        raise ValueError("Legacy FSC inventory does not prove one stable archive-wide water mask")
    return rows


def _build_canonical_schedules(
    root: Path,
    policy_path: Path,
) -> tuple[StationRoleResult, dict[str, ScheduleResult]]:
    setup_path = _canonical_setup_yaml(root)
    setup = _read_yaml(setup_path)
    policy = load_policy(policy_path)
    project_start_time = datetime.fromisoformat(str(setup["start_date"])).time()
    if project_start_time != policy.station_observation_time:
        raise ValueError(
            "Policy station_hs.observation_time must equal the setup start_date time-of-day"
        )
    windows = []
    for project_name in EXPECTED_PROJECTS:
        project = _read_yaml(root / "projects" / project_name / f"{project_name}.yml")
        windows.append(
            (
                project_name,
                parse_date(project["start_date"], field=f"{project_name} start"),
                parse_date(project["end_date"], field=f"{project_name} end"),
            )
        )
    station_rows = _canonical_station_rows_with_subdomains(
        root,
        read_csv_records(root / "obs" / "stations" / "stations_da_metadata.csv"),
    )
    return schedule_with_adaptive_roles(
        policy=policy,
        fsc_rows=_canonical_fsc_inventory(root),
        snow_rows=_canonical_snow_inventory(root, station_rows),
        station_rows=station_rows,
        windows=tuple(windows),
    )


def _log_selected_interpolations_for_root(
    root: Path,
    schedules: Mapping[str, ScheduleResult],
    policy_path: Path,
) -> int:
    """Log selected midpoint interpolations without changing audit schemas."""

    policy = load_policy(policy_path)
    if detect_setup_layout(root) == CANONICAL_LAYOUT:
        station_rows = _canonical_station_rows_with_subdomains(
            root,
            read_csv_records(root / "obs" / "stations" / "stations_da_metadata.csv"),
        )
        snow_rows = _canonical_snow_inventory(root, station_rows)
    else:
        snow_rows = read_csv_records(
            root / "inventories" / "snow_station_timestep_support.csv"
        )
    return log_selected_station_interpolations(schedules, snow_rows, policy)


def _canonical_setup_yaml(root: Path) -> Path:
    paths = sorted(Path(root).glob("*.yml"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one setup YAML, got {len(paths)}")
    return paths[0]


def _canonical_station_rows_with_subdomains(
    root: Path,
    station_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return canonical station rows with strict polygon-derived membership."""

    import geopandas as gpd
    from shapely.geometry import Point

    regions_path = Path(root) / "env" / "subdomains.gpkg"
    if not regions_path.is_file():
        raise FileNotFoundError(regions_path)
    regions = gpd.read_file(regions_path)
    if "id" not in regions.columns:
        raise ValueError(f"Canonical subdomains lack required 'id' field: {regions_path}")
    if regions.crs is None or regions.crs.to_epsg() != 25832:
        raise ValueError(f"Canonical subdomains must use EPSG:25832, got {regions.crs}")

    identifiers = [str(value).strip() for value in regions["id"]]
    if any(not value for value in identifiers):
        raise ValueError("Canonical subdomain IDs must be non-empty")
    duplicate_ids = sorted(
        identifier
        for identifier, count in Counter(identifiers).items()
        if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"Canonical subdomain IDs must be unique: {duplicate_ids}")
    invalid_geometries = sorted(
        identifiers[index]
        for index, geometry in enumerate(regions.geometry)
        if geometry is None or geometry.is_empty or not geometry.is_valid
    )
    if invalid_geometries:
        raise ValueError(
            "Canonical subdomain geometries must be non-empty and valid: "
            f"{invalid_geometries}"
        )
    ordered_regions = sorted(
        zip(identifiers, regions.geometry, strict=True),
        key=lambda item: item[0],
    )

    normalized: list[dict[str, Any]] = []
    for station in station_rows:
        station_id = str(station.get("station_id", station.get("id", ""))).strip()
        if not station_id:
            raise ValueError("Canonical station metadata contains an empty station ID")
        try:
            x = float(station["x"])
            y = float(station["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Canonical station has invalid EPSG:25832 coordinates: {station_id}"
            ) from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(
                f"Canonical station has invalid EPSG:25832 coordinates: {station_id}"
            )
        point = Point(x, y)
        matches = [
            subdomain_id
            for subdomain_id, geometry in ordered_regions
            if geometry.covers(point)
        ]
        if len(matches) != 1:
            raise ValueError(
                "Canonical station must be covered by exactly one subdomain: "
                f"station_id={station_id!r}, matches={matches}"
            )
        normalized.append({**station, "subdomain_id": matches[0]})
    return tuple(normalized)


def _canonical_snow_inventory(
    root: Path,
    station_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Read exact station timesteps from active observations."""

    rows: list[dict[str, Any]] = []
    for station in sorted(station_rows, key=lambda item: str(item["station_id"])):
        station_id = str(station["station_id"]).strip()
        subdomain_id = str(station.get("subdomain_id", "")).strip()
        if not subdomain_id:
            raise ValueError(f"Normalized station metadata lacks subdomain_id: {station_id}")
        series_path = root / "obs" / "stations" / f"{station_id}.csv"
        if not series_path.is_file():
            raise FileNotFoundError(series_path)
        for observation in read_csv_records(series_path):
            timestamp = str(observation.get("time", "")).strip()
            snow_depth = str(observation.get("snow_depth", "")).strip()
            if not timestamp or not snow_depth:
                continue
            try:
                value = float(snow_depth)
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    "station_id": station_id,
                    "subdomain_id": subdomain_id,
                    "valid_observation_count": 1,
                    "observation_value": value,
                }
            )
    return rows


def _canonical_fsc_inventory(root: Path) -> list[dict[str, Any]]:
    """Rebuild FSC quality rows from retained native scenes and subdomains."""

    import numpy as np
    import pyproj
    import xarray as xr
    from rasterio import features
    from rasterio.transform import from_origin

    from north_tyrol_snapshot import (
        summarize_fsc_quality,
        update_fsc_archive_support,
        update_fsc_archive_water_mask,
    )

    import geopandas as gpd

    regions = gpd.read_file(root / "env" / "subdomains.gpkg")[["id", "geometry"]]
    regions["id"] = regions["id"].astype(str)
    paths = tuple(sorted((root / "obs" / "snowcover").glob("*.nc")))
    if not paths:
        raise ValueError("Canonical setup contains no FSC scenes")
    masks: dict[str, Any] | None = None
    reference_x: Any | None = None
    reference_y: Any | None = None
    archive_support: Any | None = None
    archive_water_mask: Any | None = None
    for path in paths:
        with xr.open_dataset(path) as dataset:
            required = {"fsc", "uncertainty", "x", "y", "time"}
            if required - set(dataset.variables):
                raise ValueError(f"FSC scene lacks variables {sorted(required - set(dataset.variables))}: {path}")
            fsc = dataset["fsc"]
            for dimension in tuple(fsc.dims):
                if dimension not in {"y", "x"}:
                    fsc = fsc.isel({dimension: 0})
            x = np.asarray(dataset["x"].values)
            y = np.asarray(dataset["y"].values)
            if len(x) < 2 or len(y) < 2:
                raise ValueError(f"FSC scene has insufficient raster coordinates: {path}")
            if not np.allclose(np.diff(x), 50.0) or not np.allclose(np.diff(y), -50.0):
                raise ValueError(f"FSC scene is not on the expected native 50 m orientation: {path}")
            spatial_ref = dataset.get("spatial_ref")
            crs_text = None if spatial_ref is None else (
                spatial_ref.attrs.get("crs_wkt") or spatial_ref.attrs.get("spatial_ref")
            )
            if not crs_text or pyproj.CRS.from_user_input(crs_text).to_epsg() != 25832:
                raise ValueError(f"FSC scene is not EPSG:25832: {path}")
            if masks is None:
                reference_x = x.copy()
                reference_y = y.copy()
                transform = from_origin(
                    float(x.min() - abs(x[1] - x[0]) / 2),
                    float(y.max() + abs(y[1] - y[0]) / 2),
                    float(abs(x[1] - x[0])),
                    float(abs(y[1] - y[0])),
                )
                masks = {
                    str(region.id): features.rasterize(
                        [(region.geometry, 1)],
                        out_shape=(len(y), len(x)),
                        transform=transform,
                        fill=0,
                        dtype="uint8",
                    ).astype(bool)
                    for region in regions.itertuples(index=False)
                }
            elif not np.array_equal(x, reference_x) or not np.array_equal(y, reference_y):
                raise ValueError(f"FSC scene grid differs from the archive reference grid: {path}")
            archive_support = update_fsc_archive_support(archive_support, np.asarray(fsc.values))
            archive_water_mask = update_fsc_archive_water_mask(
                archive_water_mask,
                np.asarray(fsc.values),
            )
    if masks is None or archive_support is None or archive_water_mask is None:
        raise ValueError("Unable to derive FSC archive support")

    rows: list[dict[str, Any]] = []
    for path in paths:
        with xr.open_dataset(path) as dataset:
            fsc = dataset["fsc"]
            uncertainty = dataset["uncertainty"]
            for dimension in tuple(fsc.dims):
                if dimension not in {"y", "x"}:
                    fsc = fsc.isel({dimension: 0})
            for dimension in tuple(uncertainty.dims):
                if dimension not in {"y", "x"}:
                    uncertainty = uncertainty.isel({dimension: 0})
            scene_date = str(np.asarray(dataset["time"].values).reshape(-1)[0])[:10]
            fsc_values = np.asarray(fsc.values)
            uncertainty_values = np.asarray(uncertainty.values)
        for subdomain_id, mask in masks.items():
            quality = summarize_fsc_quality(
                fsc_values,
                uncertainty_values,
                mask,
                archive_support,
                archive_water_mask,
            )
            rows.append(
                {
                    "date": scene_date,
                    "subdomain_id": subdomain_id,
                    "source_file": path.name,
                    **quality,
                    "uncertainty_count": quality["uncertainty_valid_fsc_count"],
                }
            )
    return rows


def preflight(root: Path, policy_path: Path, image: str, *, verify_hashes: bool = True) -> dict[str, Any]:
    """Read and validate all finalization inputs without writing."""

    layout = detect_setup_layout(root)
    if layout == LEGACY_LAYOUT:
        source = validate_source_snapshot(root, image)
        hash_counts = verify_recorded_data_hashes(root) if verify_hashes else {"raw_files": 0, "working_files": 0}
    else:
        source = validate_canonical_setup(root, image)
        hash_counts = {}
    roles, schedules = build_schedules(root, policy_path)
    _log_selected_interpolations_for_root(root, schedules, policy_path)
    return {
        "status": "PREFLIGHT_OK",
        "layout": layout,
        **source,
        **hash_counts,
        "policy_sha256": sha256_file(policy_path),
        "station_roles": _role_counts(roles.roles),
        "station_role_exceptions": list(roles.exceptions),
        "projects": {name: result.summary for name, result in schedules.items()},
    }


def validate_canonical_setup(root: Path, image: str) -> dict[str, Any]:
    """Validate the docs-shaped source without requiring it to be result-free."""

    root = Path(root).resolve()
    validate_image_reference(image)
    if detect_setup_layout(root) != CANONICAL_LAYOUT:
        raise ValueError("Expected canonical documentation-shaped setup")
    projects = tuple(sorted(path.name for path in (root / "projects").iterdir() if path.is_dir()))
    if projects != EXPECTED_PROJECTS:
        raise ValueError(f"Unexpected canonical projects: {projects}")
    station_dir = root / "obs" / "stations"
    metadata = read_csv_records(station_dir / "stations_da_metadata.csv")
    station_ids = [str(row["station_id"]).strip() for row in metadata]
    normalized_station_ids = {station_id.lower() for station_id in station_ids}
    if len(station_ids) != 35 or len(normalized_station_ids) != 35 or any(not value for value in station_ids):
        raise ValueError("Canonical setup must contain 35 unique snow stations")
    missing_series = [station_id for station_id in station_ids if not (station_dir / f"{station_id}.csv").is_file()]
    if missing_series:
        raise ValueError(f"Snow station series are missing: {missing_series}")
    grid_count = sum(path.is_file() for path in (root / "grids").iterdir())
    meteo_count = sum(path.is_file() for path in (root / "meteo").iterdir())
    fsc_count = len(list((root / "obs" / "snowcover").glob("*.nc")))
    station_series_count = len(
        [
            path
            for path in station_dir.glob("*.csv")
            if path.name != "stations_da_metadata.csv"
        ]
    )
    if (grid_count, meteo_count, fsc_count, station_series_count) != (6, 162, 738, 35):
        raise ValueError(
            "Canonical data counts differ from the accepted setup: "
            f"grids={grid_count}, meteo={meteo_count}, FSC={fsc_count}, station_series={station_series_count}"
        )
    point_counts = _validate_output_point_identities(_read_yaml(_canonical_setup_yaml(root)), station_ids)
    return {
        "root": str(root),
        "image": image,
        "setup_yaml": _canonical_setup_yaml(root).name,
        "project_count": len(projects),
        "station_count": len(station_ids),
        "grid_files": grid_count,
        "meteo_files": meteo_count,
        "fsc_scene_count": fsc_count,
        "station_series": station_series_count,
        **point_counts,
    }


def _validate_output_point_identities(
    setup: Mapping[str, Any],
    snow_station_ids: Sequence[str],
) -> dict[str, int]:
    """Require the accepted 161 forcing and 35 independent snow output points."""

    points = setup.get("output_data", {}).get("timeseries", {}).get("points", [])
    if not isinstance(points, list):
        raise ValueError("Setup output_data.timeseries.points must be a list")
    names = [str(point.get("name", "")).strip() for point in points if isinstance(point, Mapping)]
    normalized = [name.lower() for name in names]
    if len(names) != len(points) or len(names) != 196 or len(set(normalized)) != 196:
        raise ValueError("Setup must contain 196 uniquely named output points")
    snow_normalized = {str(station_id).strip().lower() for station_id in snow_station_ids}
    missing = sorted(snow_normalized - set(normalized))
    if missing:
        raise ValueError(f"Snow-station output points are missing: {missing}")
    if len(names) - len(snow_normalized) != 161:
        raise ValueError("Setup must retain exactly 161 forcing output points")
    return {"forcing_output_points": 161, "snow_output_points": 35, "output_points": 196}


def finalize(
    root: Path,
    policy_path: Path,
    image: str,
    *,
    discard_runtime_artifacts: bool = False,
) -> Path:
    """Finalize in sibling staging, atomically promote and rollback on failure."""

    root = Path(root).resolve()
    if detect_setup_layout(root) == CANONICAL_LAYOUT:
        return _refresh_canonical_setup(
            root,
            policy_path,
            image,
            discard_runtime_artifacts=discard_runtime_artifacts,
        )
    if discard_runtime_artifacts:
        raise ValueError("--discard-runtime-artifacts is only valid for a canonical setup refresh")
    initial = preflight(root, policy_path, image)
    commit = _finalizer_commit()
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    staging = root.parent / f".{root.name}.finalizing-{stamp}-{os.getpid()}"
    backup = root.parent / f".{root.name}.pre-finalization-{stamp}"
    if staging.exists() or backup.exists():
        raise FileExistsError(f"Finalization sibling already exists: {staging} or {backup}")
    shutil.copytree(root, staging, copy_function=shutil.copy2)
    promoted = False
    try:
        roles, schedules = build_schedules(staging, policy_path)
        _preserve_parent_provenance(staging, initial)
        _write_station_roles(staging, roles)
        _promote_project_configs(staging, schedules)
        _write_scheduler_inventories(staging, schedules, roles)
        _prepare_partitioned_regions(staging, image)
        _prepare_all_projects(staging, image, schedules)
        _relativize_internal_symlinks(staging)
        acceptance = validate_final_snapshot(staging, image=image, verify_data_hashes=True)
        _set_ready_state(staging, commit)
        _write_finalization_manifest(
            staging,
            initial=initial,
            acceptance=acceptance,
            schedules=schedules,
            roles=roles,
            policy_path=policy_path,
            image=image,
            commit=commit,
        )
        validate_final_snapshot(staging, image=image, verify_data_hashes=True)
        root.replace(backup)
        try:
            staging.replace(root)
            promoted = True
            validate_final_snapshot(root, image=image, verify_data_hashes=True)
        except BaseException:
            if root.exists():
                failed = root.parent / f".{root.name}.failed-{stamp}"
                root.replace(failed)
                _write_incomplete(failed, "Post-swap validation failed")
            backup.replace(root)
            raise
        shutil.rmtree(backup)
        return root
    except BaseException as exc:
        if not promoted and staging.exists():
            _write_incomplete(staging, str(exc))
        raise


def _refresh_canonical_setup(
    root: Path,
    policy_path: Path,
    image: str,
    *,
    discard_runtime_artifacts: bool,
) -> Path:
    """Replace a canonical setup from a fully validated lightweight staging tree."""

    _assert_canonical_runtime_safe(
        root,
        discard_runtime_artifacts=discard_runtime_artifacts,
    )
    runtime_artifacts = _canonical_runtime_artifacts(root)
    runtime_artifact_paths = [path.relative_to(root).as_posix() for path in runtime_artifacts]
    source = validate_canonical_setup(root, image)
    roles, schedules = build_schedules(root, policy_path)
    _log_selected_interpolations_for_root(root, schedules, policy_path)
    commit = _finalizer_commit()
    parent_configs = {
        _canonical_setup_yaml(root).relative_to(root).as_posix(): sha256_file(_canonical_setup_yaml(root)),
        **{
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted((root / "projects").glob("project_*/project_*.yml"))
        },
    }
    parent_transaction = root / "raw" / "metadata" / "canonical_refresh_manifest.json"
    parent_manifest_sha256 = sha256_file(parent_transaction) if parent_transaction.is_file() else None
    initial = {
        "status": "PREFLIGHT_OK",
        "layout": CANONICAL_LAYOUT,
        **source,
        "policy_sha256": sha256_file(policy_path),
        "station_roles": _role_counts(roles.roles),
        "station_role_exceptions": list(roles.exceptions),
        "projects": {name: result.summary for name, result in schedules.items()},
    }
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    staging = root.parent / f".{root.name}.refreshing-{stamp}-{os.getpid()}"
    backup = root.parent / f".{root.name}.pre-refresh-{stamp}"
    if staging.exists() or backup.exists():
        raise FileExistsError(f"Refresh sibling already exists: {staging} or {backup}")
    shutil.copytree(root, staging, copy_function=shutil.copy2, ignore=_canonical_copy_ignore)
    swapped = False
    try:
        _write_canonical_station_roles(staging, roles)
        _refresh_canonical_configs(staging, schedules)
        _write_canonical_audits(staging, schedules, roles, policy_path, image)
        _prepare_all_projects(
            staging,
            image,
            schedules,
            regions_path="/setup/env/subdomains.gpkg",
        )
        _validate_all_leaf_core_requirements(staging, image)
        _relativize_internal_symlinks(staging)
        _write_canonical_refresh_manifest(
            staging,
            schedules=schedules,
            roles=roles,
            policy_path=policy_path,
            image=image,
            commit=commit,
            parent_root=root,
            parent_manifest_sha256=parent_manifest_sha256,
            parent_configs=parent_configs,
            discarded_runtime_artifacts=runtime_artifact_paths,
            promotion_result="staging_validated",
        )
        validate_canonical_refresh(
            staging,
            schedules,
            expected_promotion_result="staging_validated",
        )
        root.replace(backup)
        try:
            staging.replace(root)
            swapped = True
            _write_canonical_refresh_manifest(
                root,
                schedules=schedules,
                roles=roles,
                policy_path=policy_path,
                image=image,
                commit=commit,
                parent_root=root,
                parent_manifest_sha256=parent_manifest_sha256,
                parent_configs=parent_configs,
                discarded_runtime_artifacts=runtime_artifact_paths,
                promotion_result="promoted",
            )
            validate_canonical_refresh(
                root,
                schedules,
                expected_promotion_result="promoted",
            )
        except BaseException:
            if root.exists():
                failed = root.parent / f".{root.name}.failed-refresh-{stamp}"
                root.replace(failed)
                _write_incomplete(failed, "Post-swap canonical validation failed")
            backup.replace(root)
            raise
        shutil.rmtree(backup)
        return root
    except BaseException as exc:
        if not swapped and staging.exists():
            _write_incomplete(staging, str(exc))
        raise


def _assert_canonical_runtime_safe(
    root: Path,
    *,
    discard_runtime_artifacts: bool,
) -> None:
    """Refuse live work and require acknowledgement before replacing runtime data."""

    root = Path(root).resolve()
    lock_names = {"RUNNING", ".RUNNING", "run.lock", "project.lock"}
    locks = sorted(path for path in root.rglob("*") if path.is_file() and path.name in lock_names)
    if locks:
        raise RuntimeError(f"Canonical setup contains active runtime lock markers: {locks[:10]}")
    active = _active_model_references(root)
    if active:
        raise RuntimeError("Canonical setup is referenced by active model work:\n- " + "\n- ".join(active))
    artifacts = _canonical_runtime_artifacts(root)
    if artifacts and not discard_runtime_artifacts:
        raise RuntimeError(
            "Canonical setup contains runtime artifacts that refresh would replace. "
            "Review them, then rerun with --discard-runtime-artifacts: "
            + ", ".join(str(path.relative_to(root)) for path in artifacts[:10])
        )


def _canonical_runtime_artifacts(root: Path) -> list[Path]:
    patterns = (
        "results",
        "restart",
        "model_state",
        "model_state*.pickle*",
        "state_pointer.json",
        "*.restart*",
        "*.log",
        "run_manifest.json",
        "subdomain_run_manifest.json",
    )
    return sorted(
        {
            path
            for pattern in patterns
            for path in Path(root).rglob(pattern)
            if path.exists()
        }
    )


def _active_model_references(root: Path) -> list[str]:
    """Return host processes or containers whose commands/mounts reference the setup."""

    root = Path(root).resolve()
    references: list[str] = []
    process_table = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        text=True,
        capture_output=True,
        check=False,
    )
    if process_table.returncode:
        raise RuntimeError(f"Cannot inspect host processes: {process_table.stderr.strip()}")
    root_text = str(root)
    model_markers = (
        "openamundsen-da run",
        "openamundsen-da subdomains run",
        "openamundsen_da.pipeline",
    )
    for line in process_table.stdout.splitlines():
        stripped = line.strip()
        pid_text, separator, arguments = stripped.partition(" ")
        if not separator or not pid_text.isdigit():
            continue
        lowered = arguments.lower()
        if not any(marker in lowered for marker in model_markers):
            continue
        cwd: Path | None = None
        try:
            cwd = Path(f"/proc/{pid_text}/cwd").resolve(strict=True)
        except OSError:
            pass
        cwd_inside_setup = cwd is not None and _path_is_within(cwd, root)
        if root_text in arguments or cwd_inside_setup:
            suffix = f" cwd={cwd}" if cwd_inside_setup else ""
            references.append(f"host process {stripped}{suffix}")

    container_ids = subprocess.run(
        ["docker", "ps", "-q"],
        text=True,
        capture_output=True,
        check=False,
    )
    if container_ids.returncode:
        raise RuntimeError(f"Cannot inspect running Docker containers: {container_ids.stderr.strip()}")
    ids = container_ids.stdout.split()
    if ids:
        inspected = subprocess.run(
            ["docker", "inspect", *ids],
            text=True,
            capture_output=True,
            check=False,
        )
        if inspected.returncode:
            raise RuntimeError(f"Cannot inspect running Docker containers: {inspected.stderr.strip()}")
        for container in json.loads(inspected.stdout):
            container_name = str(container.get("Name", "")).lstrip("/")
            for mount in container.get("Mounts") or []:
                source_text = str(mount.get("Source", "")).strip()
                if not source_text:
                    continue
                source = Path(source_text)
                overlaps = _path_is_within(source, root) or _path_is_within(root, source)
                if overlaps:
                    references.append(f"Docker container {container_name or container.get('Id', '')}")
                    break
    return sorted(set(references))


def _path_is_within(path: Path, parent: Path) -> bool:
    """Return whether a resolved path is equal to or below a resolved parent."""

    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except (OSError, ValueError):
        return False
    return True


def _canonical_copy_ignore(directory: str, names: list[str]) -> set[str]:
    """Omit derived and heavy runtime trees while retaining source inputs/configs."""

    path = Path(directory)
    ignored: set[str] = set()
    if path.name.startswith("project_") and path.parent.name == "projects":
        ignored.update(name for name in names if name in {"results", "steps", "subdomains"})
    ignored.update(
        name
        for name in names
        if name in {"restart", "model_state", "INCOMPLETE.json"}
        or name.startswith("model_state")
        or name.endswith(".restart")
        or name.endswith(".log")
        or name in {"run_manifest.json", "subdomain_run_manifest.json"}
    )
    return ignored


def _write_canonical_station_roles(staging: Path, roles: StationRoleResult) -> None:
    metadata_path = staging / "obs" / "stations" / "stations_da_metadata.csv"
    existing = read_csv_records(metadata_path)
    role_by_id = {str(row["station_id"]): row for row in roles.roles}
    if set(role_by_id) != {str(row["station_id"]) for row in existing}:
        raise ValueError("Shared station-role IDs differ from canonical station metadata")
    updated = []
    for row in existing:
        role = role_by_id[str(row["station_id"])]
        updated.append(
            {
                **row,
                "use_for_da": role["use_for_da"],
                "use_for_benchmark": role["use_for_benchmark"],
                "status": f"final_{role['role']}",
            }
        )
    _write_csv(metadata_path, updated)


def _refresh_canonical_configs(staging: Path, schedules: Mapping[str, ScheduleResult]) -> None:
    setup_path = _canonical_setup_yaml(staging)
    setup = _read_yaml(setup_path)
    _ensure_openamundsen_snow_outputs(setup)
    station_ids = [
        str(row["station_id"]).strip()
        for row in read_csv_records(staging / "obs" / "stations" / "stations_da_metadata.csv")
    ]
    _validate_output_point_identities(setup, station_ids)
    _write_yaml(setup_path, setup)

    for project_name in EXPECTED_PROJECTS:
        path = staging / "projects" / project_name / f"{project_name}.yml"
        project = _read_yaml(path)
        da = project["data_assimilation"]
        if int(da["prior_forcing"]["ensemble_size"]) != 50:
            raise ValueError(f"Canonical project is not ES50: {project_name}")
        if str(da["output"]["retention"]) != "compact":
            raise ValueError(f"Canonical project does not use compact retention: {project_name}")
        da.pop("subdomain_event_filter", None)
        _ensure_compact_snow_outputs(da, project_name)
        da["assimilation_events"] = [
            {
                "date": event["selected_date"],
                "variable": event["variable"],
                **({"product": "SNOWCOVER"} if event["variable"] == "scf" else {}),
            }
            for event in schedules[project_name].events
        ]
        _write_yaml(path, project)


def _ensure_openamundsen_snow_outputs(setup: dict[str, Any]) -> None:
    """Ensure maintained daily output names map to the correct model variables."""

    grids = setup.setdefault("output_data", {}).setdefault("grids", {})
    variables = grids.setdefault("variables", [])
    if not isinstance(variables, list):
        raise ValueError("openAMUNDSEN grid variables must be a list")
    required = {
        "snowdepth_daily": "snow.depth",
        "swe_daily": "snow.swe",
    }
    for name, variable in required.items():
        existing = [item for item in variables if isinstance(item, Mapping) and str(item.get("name", "")) == name]
        if len(existing) > 1:
            raise ValueError(f"Duplicate openAMUNDSEN grid output name: {name}")
        if existing and str(existing[0].get("var", "")) != variable:
            raise ValueError(
                f"openAMUNDSEN grid output {name} maps to {existing[0].get('var')!r}, not {variable!r}"
            )
        if existing and str(existing[0].get("freq", "")) != "D":
            raise ValueError(f"openAMUNDSEN grid output {name} must use freq: D")
        if not existing:
            variables.append({"var": variable, "name": name, "freq": "D"})


def _ensure_compact_snow_outputs(data_assimilation: dict[str, Any], project_name: str) -> None:
    """Ensure both maintained daily snow products are requested by DA output."""

    grids = data_assimilation.setdefault("output", {}).setdefault("grids", {})
    variables = grids.setdefault("variables", [])
    if not isinstance(variables, list):
        raise ValueError(f"Compact grid variables must be a list: {project_name}")
    by_name: dict[str, list[Mapping[str, Any]]] = {}
    for item in variables:
        if not isinstance(item, Mapping):
            raise ValueError(f"Compact grid variable entries must be mappings: {project_name}")
        by_name.setdefault(str(item.get("name", item.get("var", ""))), []).append(item)
    for name in ("snowdepth_daily", "swe_daily"):
        if len(by_name.get(name, [])) > 1:
            raise ValueError(f"Duplicate compact grid output name {name}: {project_name}")
        if by_name.get(name) and str(by_name[name][0].get("var", "")) != name:
            raise ValueError(
                f"Compact grid output {name} maps to {by_name[name][0].get('var')!r}: {project_name}"
            )
        if by_name.get(name):
            metrics = by_name[name][0].get("metrics")
            if not isinstance(metrics, list) or not metrics:
                raise ValueError(f"Compact grid output {name} requires non-empty metrics: {project_name}")
    if "snowdepth_daily" not in by_name:
        variables.append(
            {
                "var": "snowdepth_daily",
                "name": "snowdepth_daily",
                "metrics": [
                    "open_loop",
                    "ens_mean",
                    "ens_std",
                    "ens_min",
                    "ens_max",
                    "increment",
                    "analysis_mean",
                    "analysis_increment",
                ],
            }
        )
    if "swe_daily" not in by_name:
        variables.append(
            {
                "var": "swe_daily",
                "name": "swe_daily",
                "metrics": ["open_loop", "ens_mean", "ens_std", "ens_min", "ens_max", "increment"],
            }
        )


def _write_canonical_audits(
    staging: Path,
    schedules: Mapping[str, ScheduleResult],
    roles: StationRoleResult,
    policy_path: Path,
    image: str,
) -> None:
    records: list[dict[str, Any]] = []
    subdomain_ids = _canonical_subdomain_ids(staging)
    for project_name in EXPECTED_PROJECTS:
        result = schedules[project_name]
        records.extend({"record_type": "target", "project": project_name, **row} for row in result.targets)
        records.extend({"record_type": "event", "project": project_name, **row} for row in result.events)
        records.extend({"record_type": "quality", "project": project_name, **row} for row in result.quality)
        records.extend({"record_type": "exception", "project": project_name, **row} for row in result.exceptions)
        for event in result.events:
            for subdomain_id in subdomain_ids:
                selected_for_leaf = subdomain_id in event["supported_subdomains"]
                records.append(
                    {
                        "record_type": "leaf_event_selection",
                        "project": project_name,
                        "subdomain_id": subdomain_id,
                        "selected_date": event["selected_date"],
                        "variable": event["variable"],
                        "selected_for_leaf": selected_for_leaf,
                        "reason": "supported" if selected_for_leaf else "no_qualified_observation_support",
                    }
                )
    records.extend({"record_type": "station_role", "project": "shared", **row} for row in roles.roles)
    records.extend({"record_type": "exception", "project": "shared", **row} for row in roles.exceptions)
    metadata = staging / "raw" / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    _write_csv(metadata / "da_selection_audit.csv", records)
    _write_json(
        metadata / "da_selection_summary.json",
        {
            "schema_version": FINALIZER_SCHEMA_VERSION,
            "policy_sha256": sha256_file(policy_path),
            "image": image,
            "projects": {name: schedules[name].summary for name in EXPECTED_PROJECTS},
            "station_roles": _role_counts(roles.roles),
            "station_role_exceptions": list(roles.exceptions),
        },
    )
    _write_forcing_flatline_inventory(staging)
    _write_station_fsc_audit(staging, schedules, load_policy(policy_path))
    _write_fsc_areal_strata_audit(
        staging,
        schedules,
        elevation_band_width_m=250,
    )


def _write_canonical_refresh_manifest(
    root: Path,
    *,
    schedules: Mapping[str, ScheduleResult],
    roles: StationRoleResult,
    policy_path: Path,
    image: str,
    commit: str,
    parent_root: Path,
    parent_manifest_sha256: str | None,
    parent_configs: Mapping[str, str],
    discarded_runtime_artifacts: Sequence[str],
    promotion_result: str,
) -> None:
    """Record the canonical refresh transaction without hashing scientific data."""

    if promotion_result not in {"staging_validated", "promoted"}:
        raise ValueError(f"Invalid canonical promotion result: {promotion_result}")
    payload = {
        "schema_version": FINALIZER_SCHEMA_VERSION,
        "scheduler_commit": commit,
        "policy_sha256": sha256_file(policy_path),
        "image": image,
        "parent_root": str(Path(parent_root).resolve()),
        "parent_manifest_sha256": parent_manifest_sha256,
        "parent_config_sha256": dict(sorted(parent_configs.items())),
        "discarded_runtime_artifacts": list(discarded_runtime_artifacts),
        "promotion_result": promotion_result,
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "core_leaf_requirement_validations": 48,
        "projects": {
            name: {
                "event_count": len(schedules[name].events),
                "by_variable": schedules[name].summary["by_variable"],
                "by_subdomain": schedules[name].summary["by_subdomain"],
            }
            for name in EXPECTED_PROJECTS
        },
        "station_roles": _role_counts(roles.roles),
        "station_role_exceptions": list(roles.exceptions),
    }
    _write_json(root / "raw" / "metadata" / "canonical_refresh_manifest.json", payload)


def _write_forcing_flatline_inventory(root: Path) -> None:
    """Inventory native-cadence forcing plateaus as non-mutating QC evidence."""

    project_windows = _canonical_project_windows(root)
    source_runs: list[dict[str, Any]] = []
    cadence_rows: list[dict[str, Any]] = []
    forcing_paths = sorted(path for path in (root / "meteo").glob("*.csv") if path.name != "stations.csv")
    if len(forcing_paths) != 161:
        raise ValueError(f"Expected 161 forcing time-series files, got {len(forcing_paths)}")
    for path in forcing_paths:
        records = read_csv_records(path)
        timestamps, cadence = forcing_source_timestamps(
            records,
            station_file=path.name,
            expected_timestep=EXPECTED_FORCING_SOURCE_TIMESTEP,
        )
        cadence_rows.append(
            {
                "station_file": path.name,
                "station_id": path.stem,
                "inferred_source_cadence_hours": cadence.total_seconds() / 3600.0,
                "timestamp_count": len(timestamps),
                "first_timestamp": timestamps[0].isoformat(sep=" "),
                "last_timestamp": timestamps[-1].isoformat(sep=" "),
                "gap_count": sum(
                    right - left != cadence for left, right in zip(timestamps, timestamps[1:])
                ),
            }
        )
        source_runs.extend(
            forcing_flatline_runs(
                records,
                station_file=path.name,
                expected_timestep=EXPECTED_FORCING_SOURCE_TIMESTEP,
                minimum_duration=FORCING_FLATLINE_MINIMUM_DURATION,
            )
        )
    rows = clip_forcing_flatline_runs(
        source_runs,
        project_windows=project_windows,
        minimum_duration=FORCING_FLATLINE_MINIMUM_DURATION,
    )
    overlaps = forcing_multivariable_overlaps(rows)
    _validate_eissee_2017_2018_flatline(rows)
    metadata = root / "raw" / "metadata"
    _write_csv(metadata / "forcing_source_cadence.csv", cadence_rows)
    _write_csv(metadata / "forcing_flatline_runs.csv", rows)
    _write_csv(metadata / "forcing_flatline_multivariable_overlaps.csv", overlaps)
    _write_json(
        metadata / "forcing_flatline_summary.json",
        forcing_flatline_summary(
            rows,
            overlaps=overlaps,
            cadence_rows=cadence_rows,
            project_windows=project_windows,
        ),
    )


def forcing_flatline_runs(
    records: Sequence[Mapping[str, Any]],
    *,
    station_file: str,
    expected_timestep: timedelta = EXPECTED_FORCING_SOURCE_TIMESTEP,
    minimum_duration: timedelta = FORCING_FLATLINE_MINIMUM_DURATION,
) -> list[dict[str, Any]]:
    """Return exact constant-value runs at one forcing table's native cadence."""

    timestamps, source_timestep = forcing_source_timestamps(
        records,
        station_file=station_file,
        expected_timestep=expected_timestep,
    )
    rows: list[dict[str, Any]] = []
    variables = [variable for variable in MODEL_FORCING_VARIABLES if variable in records[0]]
    for variable in variables:
        run_start = 0
        run_value: float | None = None
        for index in range(len(records) + 1):
            value = _finite_or_none(records[index].get(variable)) if index < len(records) else None
            continues = (
                index > run_start
                and value is not None
                and run_value is not None
                and value == run_value
                and timestamps[index] - timestamps[index - 1] == source_timestep
            ) if index < len(records) else False
            if index == run_start:
                run_value = value
                continue
            if continues:
                continue
            duration = timestamps[index - 1] - timestamps[run_start]
            if run_value is not None and duration >= minimum_duration:
                classification = (
                    "dry_zero_precip"
                    if variable == "precip" and run_value == 0.0
                    else "candidate_stuck_sensor"
                )
                rows.append(
                    {
                        "station_file": station_file,
                        "station_id": Path(station_file).stem,
                        "variable": variable,
                        "value": run_value,
                        "start_timestamp": timestamps[run_start].isoformat(sep=" "),
                        "end_timestamp": timestamps[index - 1].isoformat(sep=" "),
                        "sample_count": index - run_start,
                        "duration_hours": duration.total_seconds() / 3600.0,
                        "source_timestep_hours": source_timestep.total_seconds() / 3600.0,
                        "zero_value": run_value == 0.0,
                        "classification": classification,
                        "severity": (
                            "severe" if duration >= FORCING_FLATLINE_SEVERE_DURATION else "plateau"
                        ),
                    }
                )
            run_start = index
            run_value = value
    return rows


def forcing_source_timestamps(
    records: Sequence[Mapping[str, Any]],
    *,
    station_file: str,
    expected_timestep: timedelta,
) -> tuple[list[datetime], timedelta]:
    """Parse timestamps and require the inferred native forcing cadence."""

    if len(records) < 2:
        raise ValueError(f"Forcing table needs at least two timestamps: {station_file}")
    timestamp_field = next(
        (name for name in ("date", "datetime", "time") if name in records[0]),
        None,
    )
    if timestamp_field is None:
        raise ValueError(f"Forcing table has no timestamp column: {station_file}")
    timestamps = [
        _naive_datetime(record[timestamp_field], field=f"{station_file}:{timestamp_field}")
        for record in records
    ]
    deltas = [right - left for left, right in zip(timestamps, timestamps[1:])]
    if any(delta <= timedelta(0) for delta in deltas):
        raise ValueError(f"Forcing timestamps are not strictly increasing: {station_file}")
    delta_counts = Counter(deltas)
    inferred_timestep = min(
        delta_counts,
        key=lambda delta: (-delta_counts[delta], delta),
    )
    if inferred_timestep != expected_timestep:
        raise ValueError(
            f"Forcing source cadence must be {expected_timestep}, got {inferred_timestep}: {station_file}"
        )
    expected_seconds = expected_timestep.total_seconds()
    if any(delta.total_seconds() % expected_seconds != 0 for delta in deltas):
        raise ValueError(f"Forcing timestamp gap is not aligned to the hourly source cadence: {station_file}")
    return timestamps, inferred_timestep


def clip_forcing_flatline_runs(
    source_runs: Sequence[Mapping[str, Any]],
    *,
    project_windows: Mapping[str, tuple[datetime, datetime]],
    minimum_duration: timedelta,
) -> list[dict[str, Any]]:
    """Clip native-source flatlines to every overlapping project window."""

    rows: list[dict[str, Any]] = []
    for source_row in source_runs:
        source_start = _naive_datetime(source_row["start_timestamp"], field="flatline start")
        source_end = _naive_datetime(source_row["end_timestamp"], field="flatline end")
        source_timestep = timedelta(hours=float(source_row["source_timestep_hours"]))
        for project, (project_start, project_end) in sorted(project_windows.items()):
            start = max(source_start, project_start)
            end = min(source_end, project_end)
            duration = end - start
            if duration < minimum_duration:
                continue
            row = dict(source_row)
            row.update(
                {
                    "project": project,
                    "source_start_timestamp": source_start.isoformat(sep=" "),
                    "source_end_timestamp": source_end.isoformat(sep=" "),
                    "source_sample_count": source_row["sample_count"],
                    "start_timestamp": start.isoformat(sep=" "),
                    "end_timestamp": end.isoformat(sep=" "),
                    "sample_count": int(duration / source_timestep) + 1,
                    "duration_hours": duration.total_seconds() / 3600.0,
                    "severity": (
                        "severe" if duration >= FORCING_FLATLINE_SEVERE_DURATION else "plateau"
                    ),
                }
            )
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            str(row["project"]),
            str(row["station_id"]),
            str(row["variable"]),
            str(row["start_timestamp"]),
        ),
    )


def forcing_multivariable_overlaps(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize intervals with two or more candidate stuck forcing fields."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["classification"] != "candidate_stuck_sensor":
            continue
        key = (str(row["project"]), str(row["station_file"]), str(row["station_id"]))
        grouped.setdefault(key, []).append(row)
    overlaps: list[dict[str, Any]] = []
    for (project, station_file, station_id), station_rows in sorted(grouped.items()):
        boundaries = sorted(
            {
                _naive_datetime(row[field], field=f"{station_file}:{field}")
                for row in station_rows
                for field in ("start_timestamp", "end_timestamp")
            }
        )
        segments: list[tuple[datetime, datetime, tuple[str, ...]]] = []
        for start, end in zip(boundaries, boundaries[1:]):
            variables = tuple(
                sorted(
                    str(row["variable"])
                    for row in station_rows
                    if _naive_datetime(row["start_timestamp"], field="flatline start") <= start
                    and _naive_datetime(row["end_timestamp"], field="flatline end") >= end
                )
            )
            if len(variables) < 2 or end <= start:
                continue
            if segments and segments[-1][1] == start and segments[-1][2] == variables:
                segments[-1] = (segments[-1][0], end, variables)
            else:
                segments.append((start, end, variables))
        for start, end, variables in segments:
            duration = end - start
            overlaps.append(
                {
                    "project": project,
                    "station_file": station_file,
                    "station_id": station_id,
                    "start_timestamp": start.isoformat(sep=" "),
                    "end_timestamp": end.isoformat(sep=" "),
                    "duration_hours": duration.total_seconds() / 3600.0,
                    "variables": variables,
                    "variable_count": len(variables),
                    "severity": (
                        "severe" if duration >= FORCING_FLATLINE_SEVERE_DURATION else "overlap"
                    ),
                }
            )
    return overlaps


def forcing_flatline_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    overlaps: Sequence[Mapping[str, Any]],
    cadence_rows: Sequence[Mapping[str, Any]],
    project_windows: Mapping[str, tuple[datetime, datetime]],
) -> dict[str, Any]:
    """Build deterministic aggregate forcing-QC evidence."""

    projects: dict[str, Any] = {}
    for project, (start, end) in sorted(project_windows.items()):
        project_rows = [row for row in rows if row["project"] == project]
        project_overlaps = [row for row in overlaps if row["project"] == project]
        projects[project] = {
            "start_timestamp": start.isoformat(sep=" "),
            "end_timestamp": end.isoformat(sep=" "),
            "run_count": len(project_rows),
            "severe_run_count": sum(row["severity"] == "severe" for row in project_rows),
            "candidate_stuck_sensor_count": sum(
                row["classification"] == "candidate_stuck_sensor" for row in project_rows
            ),
            "dry_zero_precip_count": sum(
                row["classification"] == "dry_zero_precip" for row in project_rows
            ),
            "station_count": len({str(row["station_id"]) for row in project_rows}),
            "variable_run_counts": dict(sorted(Counter(str(row["variable"]) for row in project_rows).items())),
            "multivariable_overlap_count": len(project_overlaps),
            "multivariable_station_count": len(
                {str(row["station_id"]) for row in project_overlaps}
            ),
            "maximum_duration_hours": max(
                (float(row["duration_hours"]) for row in project_rows),
                default=0.0,
            ),
        }
    return {
        "schema_version": 2,
        "definition": "exact equal finite values over consecutive native hourly source rows",
        "scientific_action": "warning_only_no_fill_mask_or_exclusion",
        "audited_variables": list(MODEL_FORCING_VARIABLES),
        "minimum_duration_hours": FORCING_FLATLINE_MINIMUM_DURATION.total_seconds() / 3600.0,
        "severe_duration_hours": FORCING_FLATLINE_SEVERE_DURATION.total_seconds() / 3600.0,
        "expected_source_cadence_hours": EXPECTED_FORCING_SOURCE_TIMESTEP.total_seconds() / 3600.0,
        "forcing_files": len(cadence_rows),
        "source_cadence_file_count": len(cadence_rows),
        "source_gap_count": sum(int(row["gap_count"]) for row in cadence_rows),
        "run_count": len(rows),
        "severe_run_count": sum(row["severity"] == "severe" for row in rows),
        "multivariable_overlap_count": len(overlaps),
        "projects": projects,
    }


def _canonical_project_windows(root: Path) -> dict[str, tuple[datetime, datetime]]:
    """Read the exact local-time window of every maintained project."""

    windows: dict[str, tuple[datetime, datetime]] = {}
    for project_name in EXPECTED_PROJECTS:
        project_path = root / "projects" / project_name / f"{project_name}.yml"
        project = _read_yaml(project_path)
        start = _naive_datetime(project["start_date"], field=f"{project_name} start_date")
        end = _naive_datetime(project["end_date"], field=f"{project_name} end_date")
        if end <= start:
            raise ValueError(f"Invalid forcing-QC project window: {project_name}")
        windows[project_name] = (start, end)
    return windows


def _validate_eissee_2017_2018_flatline(rows: Sequence[Mapping[str, Any]]) -> None:
    """Require the known 2017/18 Eissee multivariable forcing limitation."""

    expected_variables = {"temp", "rel_hum", "wind_speed", "wind_dir"}
    matched = {
        str(row["variable"])
        for row in rows
        if row["project"] == "project_2017_2018"
        and row["station_file"] == "AT_LWD.Eissee.csv"
        and row["start_timestamp"] == "2018-04-05 15:00:00"
        and row["end_timestamp"] == "2018-09-30 21:00:00"
        and row["severity"] == "severe"
    }
    if matched != expected_variables:
        raise ValueError(
            "Forcing-QC audit did not recover the confirmed 2017/18 Eissee "
            f"temperature/humidity/wind plateau; matched {sorted(matched)}"
        )


def _duration_value(value: object) -> timedelta:
    """Parse the hour-based timestep used by maintained North Tyrol setups."""

    match = re.fullmatch(r"\s*(\d+)\s*[Hh]\s*", str(value))
    if not match:
        raise ValueError(f"Unsupported North Tyrol timestep: {value!r}")
    return timedelta(hours=int(match.group(1)))


def _naive_datetime(value: object, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if parsed.tzinfo is not None:
        raise ValueError(f"{field} must be normalized to naive setup-local time")
    return parsed


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _write_station_fsc_audit(
    root: Path,
    schedules: Mapping[str, ScheduleResult],
    policy: SchedulePolicy,
) -> None:
    """Write co-temporal point and 3x3 station/FSC consistency evidence."""

    import numpy as np
    import xarray as xr

    from north_tyrol_snapshot import classify_fsc

    station_rows = _canonical_station_rows_with_subdomains(
        root,
        read_csv_records(root / "obs" / "stations" / "stations_da_metadata.csv"),
    )
    metadata = {str(row["station_id"]).strip(): row for row in station_rows}
    snow_rows = _canonical_snow_inventory(root, station_rows)
    station_support = match_station_support(snow_rows, policy)
    selected: dict[str, dict[str, Any]] = {}
    for project_name in EXPECTED_PROJECTS:
        for event in schedules[project_name].events:
            if event["variable"] != "scf":
                continue
            event_date = str(event["selected_date"])
            previous = selected.setdefault(event_date, {"project": project_name, **event})
            if previous["source_file"] != event["source_file"]:
                raise ValueError(f"One FSC date maps to multiple scenes: {event_date}")
    threshold = _station_snow_threshold(root)
    comparison_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for event_date, event in sorted(selected.items()):
        model_time = datetime.combine(parse_date(event_date, field="FSC event date"), policy.station_observation_time)
        matches = station_support.get(model_time, {})
        path = root / "obs" / "snowcover" / str(event["source_file"])
        if not path.is_file():
            raise FileNotFoundError(path)
        with xr.open_dataset(path) as dataset:
            fsc = dataset["fsc"]
            for dimension in tuple(fsc.dims):
                if dimension not in {"y", "x"}:
                    fsc = fsc.isel({dimension: 0})
            values = np.asarray(fsc.values)
            x = np.asarray(dataset["x"].values, dtype=float)
            y = np.asarray(dataset["y"].values, dtype=float)
            acquisition_time = str(np.asarray(dataset["time"].values).reshape(-1)[0])
        supported_domains = {str(value) for value in event["supported_subdomains"]}
        event_rows.append(
            {
                "project": event["project"],
                "event_date": event_date,
                "source_file": path.name,
                "acquisition_timestamp": acquisition_time,
                "matched_station_count": len(matches),
                "matched_station_ids": sorted(matches),
                "supported_subdomains": sorted(supported_domains),
            }
        )
        for station_id, match in sorted(matches.items()):
            station = metadata[station_id]
            x_index = int(np.argmin(np.abs(x - float(station["x"]))))
            y_index = int(np.argmin(np.abs(y - float(station["y"]))))
            x_spacing = float(np.min(np.abs(np.diff(x))))
            y_spacing = float(np.min(np.abs(np.diff(y))))
            if abs(x[x_index] - float(station["x"])) > x_spacing / 2 or abs(y[y_index] - float(station["y"])) > y_spacing / 2:
                raise ValueError(f"Snow station lies outside FSC grid: {station_id}/{path.name}")
            row_start = max(0, y_index - 1)
            row_stop = min(values.shape[0], y_index + 2)
            col_start = max(0, x_index - 1)
            col_stop = min(values.shape[1], x_index + 2)
            neighborhood = values[row_start:row_stop, col_start:col_stop]
            neighborhood_classes = classify_fsc(neighborhood)
            valid_values = neighborhood[neighborhood_classes["valid"]]
            point_value = float(values[y_index, x_index])
            point_classes = classify_fsc(np.asarray([point_value]))
            point_class = next(
                name
                for name in ("valid", "cloud", "water", "nodata", "unknown")
                if bool(point_classes[name][0])
            )
            comparison_rows.append(
                {
                    "project": event["project"],
                    "event_date": event_date,
                    "source_file": path.name,
                    "subdomain_id": match.subdomain_id,
                    "scheduled_fsc_support": match.subdomain_id in supported_domains,
                    "station_id": station_id,
                    "station_name": station.get("name", station.get("id", "")),
                    "station_x": station["x"],
                    "station_y": station["y"],
                    "station_elevation_m": station.get("alt", station.get("elevation_m", "")),
                    "model_timestamp": model_time.isoformat(sep=" "),
                    "station_observation_timestamp": match.observation_timestamp.isoformat(sep=" "),
                    "station_match_delta_minutes": match.delta_minutes,
                    "station_snow_depth_m": match.observation_value if match.observation_value is not None else "",
                    "station_snow_present_h0": (
                        match.observation_value >= threshold
                        if match.observation_value is not None
                        else ""
                    ),
                    "snow_depth_threshold_m": threshold,
                    "fsc_point_class": point_class,
                    "fsc_point_percent": point_value if point_class == "valid" else "",
                    "fsc_3x3_valid_count": int(valid_values.size),
                    "fsc_3x3_total_count": int(neighborhood.size),
                    "fsc_3x3_mean_percent": float(np.mean(valid_values)) if valid_values.size else "",
                    "fsc_3x3_median_percent": float(np.median(valid_values)) if valid_values.size else "",
                    "fsc_3x3_cloud_count": int(neighborhood_classes["cloud"].sum()),
                    "fsc_3x3_water_count": int(neighborhood_classes["water"].sum()),
                    "fsc_3x3_nodata_count": int(neighborhood_classes["nodata"].sum()),
                }
            )
    output_root = root / "raw" / "metadata"
    _write_csv(output_root / "station_fsc_cotemporal_point_3x3.csv", comparison_rows)
    _write_csv(output_root / "station_fsc_cotemporal_events.csv", event_rows)
    _write_json(
        output_root / "station_fsc_cotemporal_summary.json",
        {
            "selected_fsc_events": len(selected),
            "station_scene_comparisons": len(comparison_rows),
            "station_time_matching": policy.station_matching,
            "spatial_support": ["native FSC pixel containing nearest center", "bounded native 3x3 FSC neighborhood"],
            "interpretation": (
                "Semi-independent consistency evidence only. Point snow depth is local and must not be "
                "compared as if it were a subdomain mean; agreement or disagreement is not causal proof."
            ),
        },
    )


def _station_snow_threshold(root: Path) -> float:
    project_name = EXPECTED_PROJECTS[0]
    project = _read_yaml(root / "projects" / project_name / f"{project_name}.yml")
    data_assimilation = project["data_assimilation"]
    operator = data_assimilation.get("h_of_x")
    if not isinstance(operator, Mapping):
        operator = data_assimilation.get("likelihood", {}).get("scf", {}).get("h_of_x")
    try:
        value = operator["params"]["h0"]
    except (KeyError, TypeError) as exc:
        raise ValueError("North Tyrol project lacks the configured FSC snow-depth threshold") from exc
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError(f"Invalid FSC snow-depth threshold: {value!r}")
    return threshold


def _write_fsc_areal_strata_audit(
    root: Path,
    schedules: Mapping[str, ScheduleResult],
    *,
    elevation_band_width_m: int,
) -> None:
    """Write areal FSC context by DEM band and land-cover class without resampling."""

    import geopandas as gpd
    import numpy as np
    import xarray as xr
    from rasterio import features
    from rasterio.transform import from_origin

    from north_tyrol_snapshot import _load_ascii_array, classify_fsc

    if elevation_band_width_m <= 0:
        raise ValueError("elevation_band_width_m must be positive")
    dem_paths = sorted((root / "grids").glob("dem_*_100.asc"))
    landcover_paths = sorted((root / "grids").glob("lc_*_100.asc"))
    if len(dem_paths) != 1 or len(landcover_paths) != 1:
        raise ValueError("Expected one North Tyrol DEM and one land-cover grid")
    dem_header, dem = _load_ascii_array(dem_paths[0])
    landcover_header, landcover = _load_ascii_array(landcover_paths[0])
    if dem_header != landcover_header:
        raise ValueError("DEM and land-cover ASCII grids are not aligned")
    regions = gpd.read_file(root / "env" / "subdomains.gpkg")[["id", "geometry"]]
    regions["id"] = regions["id"].astype(str)
    selected = [
        {"project": project_name, **event}
        for project_name in EXPECTED_PROJECTS
        for event in schedules[project_name].events
        if event["variable"] == "scf"
    ]
    rows: list[dict[str, Any]] = []
    cached_grid: tuple[Any, Any, dict[str, Any], Any, Any] | None = None
    for event in sorted(selected, key=lambda item: (str(item["selected_date"]), str(item["source_file"]))):
        path = root / "obs" / "snowcover" / str(event["source_file"])
        with xr.open_dataset(path) as dataset:
            fsc = dataset["fsc"]
            for dimension in tuple(fsc.dims):
                if dimension not in {"y", "x"}:
                    fsc = fsc.isel({dimension: 0})
            values = np.asarray(fsc.values)
            x = np.asarray(dataset["x"].values, dtype=float)
            y = np.asarray(dataset["y"].values, dtype=float)
        if cached_grid is None or not np.array_equal(x, cached_grid[0]) or not np.array_equal(y, cached_grid[1]):
            transform = from_origin(
                float(x.min() - abs(x[1] - x[0]) / 2),
                float(y.max() + abs(y[1] - y[0]) / 2),
                float(abs(x[1] - x[0])),
                float(abs(y[1] - y[0])),
            )
            masks = {
                str(region.id): features.rasterize(
                    [(region.geometry, 1)],
                    out_shape=values.shape,
                    transform=transform,
                    fill=0,
                    dtype="uint8",
                ).astype(bool)
                for region in regions.itertuples(index=False)
            }
            x_columns = np.floor((x - dem_header.xllcorner) / dem_header.cellsize).astype(int)
            y_from_bottom = np.floor((y - dem_header.yllcorner) / dem_header.cellsize).astype(int)
            y_rows = dem_header.nrows - 1 - y_from_bottom
            inside_x = (x_columns >= 0) & (x_columns < dem_header.ncols)
            inside_y = (y_rows >= 0) & (y_rows < dem_header.nrows)
            mapped_dem = np.full(values.shape, np.nan, dtype=float)
            mapped_landcover = np.full(values.shape, np.nan, dtype=float)
            valid_grid = inside_y[:, None] & inside_x[None, :]
            safe_rows = np.clip(y_rows, 0, dem_header.nrows - 1)
            safe_columns = np.clip(x_columns, 0, dem_header.ncols - 1)
            sampled_dem = dem[safe_rows[:, None], safe_columns[None, :]]
            sampled_landcover = landcover[safe_rows[:, None], safe_columns[None, :]]
            mapped_dem[valid_grid] = sampled_dem[valid_grid]
            mapped_landcover[valid_grid] = sampled_landcover[valid_grid]
            cached_grid = (x.copy(), y.copy(), masks, mapped_dem, mapped_landcover)
        _, _, masks, mapped_dem, mapped_landcover = cached_grid
        classes = classify_fsc(values)
        for subdomain_id, mask in sorted(masks.items()):
            rows.extend(
                _summarize_fsc_strata(
                    values,
                    mapped_dem,
                    mapped_landcover,
                    mask,
                    elevation_band_width_m=elevation_band_width_m,
                    base={
                        "project": event["project"],
                        "event_date": event["selected_date"],
                        "source_file": event["source_file"],
                        "subdomain_id": subdomain_id,
                        "scheduled_fsc_support": subdomain_id in event["supported_subdomains"],
                    },
                    valid_fsc_mask=classes["valid"],
                    dem_nodata=float(dem_header.nodata_value),
                    landcover_nodata=float(landcover_header.nodata_value),
                )
            )
    metadata = root / "raw" / "metadata"
    _write_csv(metadata / "fsc_areal_elevation_landcover.csv", rows)
    _write_json(
        metadata / "fsc_areal_elevation_landcover_summary.json",
        {
            "selected_fsc_events": len(selected),
            "record_count": len(rows),
            "elevation_band_width_m": elevation_band_width_m,
            "sampling": "native 50 m FSC pixels assigned to containing native 100 m DEM/land-cover cells",
            "interpretation": "Areal observation-support context; not a station-to-subdomain-mean comparison.",
        },
    )


def _summarize_fsc_strata(
    fsc: Any,
    dem: Any,
    landcover: Any,
    domain_mask: Any,
    *,
    elevation_band_width_m: int,
    base: Mapping[str, Any],
    valid_fsc_mask: Any,
    dem_nodata: float,
    landcover_nodata: float,
) -> list[dict[str, Any]]:
    """Summarize retrieval-valid native FSC pixels by two support dimensions."""

    import numpy as np

    fsc = np.asarray(fsc, dtype=float)
    dem = np.asarray(dem, dtype=float)
    landcover = np.asarray(landcover, dtype=float)
    support = np.asarray(domain_mask, dtype=bool) & np.asarray(valid_fsc_mask, dtype=bool)
    rows: list[dict[str, Any]] = []
    elevation_support = support & np.isfinite(dem) & ~np.isclose(dem, dem_nodata)
    elevation_lower = np.floor(dem / elevation_band_width_m) * elevation_band_width_m
    for lower in sorted(set(elevation_lower[elevation_support].astype(int))):
        selected = elevation_support & (elevation_lower == lower)
        rows.append(
            _fsc_stratum_row(
                base,
                "elevation_band",
                f"{lower}-{lower + elevation_band_width_m}",
                fsc[selected],
            )
        )
    landcover_support = support & np.isfinite(landcover) & ~np.isclose(landcover, landcover_nodata)
    for class_code in sorted(set(landcover[landcover_support].astype(int))):
        selected = landcover_support & np.isclose(landcover, class_code)
        rows.append(_fsc_stratum_row(base, "landcover_class", str(class_code), fsc[selected]))
    return rows


def _fsc_stratum_row(
    base: Mapping[str, Any],
    stratum_type: str,
    stratum: str,
    values: Any,
) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(values, dtype=float)
    return {
        **base,
        "stratum_type": stratum_type,
        "stratum": stratum,
        "valid_pixel_count": int(values.size),
        "fsc_mean_percent": float(np.mean(values)),
        "fsc_median_percent": float(np.median(values)),
        "fsc_p10_percent": float(np.percentile(values, 10)),
        "fsc_p90_percent": float(np.percentile(values, 90)),
    }


def validate_canonical_refresh(
    root: Path,
    schedules: Mapping[str, ScheduleResult],
    *,
    expected_promotion_result: str,
) -> dict[str, Any]:
    """Validate refreshed projects and deterministic preparation without propagation."""

    _validate_internal_symlinks(root)
    transaction_path = root / "raw" / "metadata" / "canonical_refresh_manifest.json"
    if not transaction_path.is_file():
        raise FileNotFoundError(transaction_path)
    transaction = _read_json(transaction_path)
    required_transaction_fields = {
        "scheduler_commit",
        "policy_sha256",
        "image",
        "parent_root",
        "parent_manifest_sha256",
        "parent_config_sha256",
        "promotion_result",
    }
    missing_transaction_fields = sorted(required_transaction_fields - set(transaction))
    if missing_transaction_fields:
        raise ValueError(f"Canonical refresh manifest lacks fields: {missing_transaction_fields}")
    if transaction["promotion_result"] != expected_promotion_result:
        raise ValueError(
            "Canonical refresh promotion result differs: "
            f"{transaction['promotion_result']!r} != {expected_promotion_result!r}"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", str(transaction["scheduler_commit"])):
        raise ValueError("Canonical refresh manifest has an invalid scheduler commit")
    if not re.fullmatch(r"[0-9a-f]{64}", str(transaction["policy_sha256"])):
        raise ValueError("Canonical refresh manifest has an invalid policy digest")
    validate_image_reference(str(transaction["image"]))
    parent_manifest = transaction["parent_manifest_sha256"]
    if parent_manifest is not None and not re.fullmatch(r"[0-9a-f]{64}", str(parent_manifest)):
        raise ValueError("Canonical refresh manifest has an invalid parent-manifest digest")
    parent_configs = transaction["parent_config_sha256"]
    if not isinstance(parent_configs, Mapping) or not parent_configs or any(
        not re.fullmatch(r"[0-9a-f]{64}", str(digest))
        for digest in parent_configs.values()
    ):
        raise ValueError("Canonical refresh manifest has invalid parent-config digests")
    if int(transaction.get("core_leaf_requirement_validations", 0)) != 48:
        raise ValueError("Canonical refresh manifest does not record 48 core leaf validations")
    forbidden = [
        path
        for pattern in (
            "results",
            "restart",
            "model_state*.pickle*",
            "*.restart*",
            "*.log",
            "run_manifest.json",
            "subdomain_run_manifest.json",
        )
        for path in root.rglob(pattern)
        if path.exists()
    ]
    if forbidden:
        raise ValueError(f"Runtime artifacts remain in refreshed staging: {forbidden[:10]}")
    setup = _read_yaml(_canonical_setup_yaml(root))
    setup_variables = setup.get("output_data", {}).get("grids", {}).get("variables", [])
    required_setup_outputs = {
        ("snow.depth", "snowdepth_daily", "D"),
        ("snow.swe", "swe_daily", "D"),
    }
    actual_setup_outputs = {
        (str(item.get("var", "")), str(item.get("name", "")), str(item.get("freq", "")))
        for item in setup_variables
        if isinstance(item, Mapping)
    }
    if not required_setup_outputs <= actual_setup_outputs:
        raise ValueError("Refreshed setup lacks required daily snow-depth or SWE model output")
    combinations = 0
    for project_name in EXPECTED_PROJECTS:
        project_dir = root / "projects" / project_name
        project = _read_yaml(project_dir / f"{project_name}.yml")
        data_assimilation = project["data_assimilation"]
        if int(data_assimilation["prior_forcing"]["ensemble_size"]) != 50:
            raise ValueError(f"Refreshed project is not ES50: {project_name}")
        if str(data_assimilation["output"]["retention"]) != "compact":
            raise ValueError(f"Refreshed project does not use compact retention: {project_name}")
        compact_variables = data_assimilation["output"]["grids"]["variables"]
        required_compact_outputs = {
            ("snowdepth_daily", "snowdepth_daily"),
            ("swe_daily", "swe_daily"),
        }
        actual_compact_outputs = {
            (str(item.get("var", "")), str(item.get("name", item.get("var", ""))))
            for item in compact_variables
            if isinstance(item, Mapping)
        }
        if not required_compact_outputs <= actual_compact_outputs:
            raise ValueError(f"Refreshed project lacks required compact snow outputs: {project_name}")
        compact_by_name = {
            str(item.get("name", item.get("var", ""))): item
            for item in compact_variables
            if isinstance(item, Mapping)
        }
        if any(
            not isinstance(compact_by_name[name].get("metrics"), list)
            or not compact_by_name[name]["metrics"]
            for name in ("snowdepth_daily", "swe_daily")
        ):
            raise ValueError(f"Refreshed project has empty compact snow metrics: {project_name}")
        expected_events = [
            (str(event["selected_date"]), str(event["variable"]))
            for event in schedules[project_name].events
        ]
        actual_events = [
            (str(event["date"]), str(event["variable"]))
            for event in data_assimilation["assimilation_events"]
        ]
        if actual_events != expected_events or len(actual_events) != len(set(actual_events)):
            raise ValueError(f"Canonical event schedule differs after refresh: {project_name}")
        if "subdomain_event_filter" in data_assimilation:
            raise ValueError(f"Legacy event selection remains in top-level YAML: {project_name}")
        leaves = sorted(
            path
            for path in (project_dir / "subdomains").iterdir()
            if path.is_dir() and path.name.startswith("AT-")
        )
        if len(leaves) != EXPECTED_SUBDOMAINS:
            raise ValueError(f"Expected eight prepared leaves for {project_name}")
        for leaf in leaves:
            combinations += 1
            leaf_project = leaf / "projects" / project_name
            leaf_config = _read_yaml(leaf_project / f"{project_name}.yml")
            leaf_da = leaf_config["data_assimilation"]
            if int(leaf_da["prior_forcing"]["ensemble_size"]) != 50 or str(
                leaf_da["output"]["retention"]
            ) != "compact":
                raise ValueError(f"Leaf ES50/compact contract failed: {project_name}/{leaf.name}")
            leaf_compact_outputs = {
                (str(item.get("var", "")), str(item.get("name", item.get("var", ""))))
                for item in leaf_da["output"]["grids"]["variables"]
                if isinstance(item, Mapping)
            }
            if not required_compact_outputs <= leaf_compact_outputs:
                raise ValueError(f"Leaf compact snow outputs are incomplete: {project_name}/{leaf.name}")
            leaf_compact_by_name = {
                str(item.get("name", item.get("var", ""))): item
                for item in leaf_da["output"]["grids"]["variables"]
                if isinstance(item, Mapping)
            }
            if any(
                not isinstance(leaf_compact_by_name[name].get("metrics"), list)
                or not leaf_compact_by_name[name]["metrics"]
                for name in ("snowdepth_daily", "swe_daily")
            ):
                raise ValueError(f"Leaf compact snow metrics are empty: {project_name}/{leaf.name}")
            if "subdomain_event_filter" in leaf_da:
                raise ValueError(f"Legacy event selection remains in leaf YAML: {project_name}/{leaf.name}")
            expected_leaf_events = [
                (str(event["selected_date"]), str(event["variable"]))
                for event in schedules[project_name].events
                if leaf.name in event["supported_subdomains"]
            ]
            actual_leaf_events = [
                (str(event["date"]), str(event["variable"]))
                for event in leaf_da["assimilation_events"]
            ]
            if actual_leaf_events != expected_leaf_events:
                raise ValueError(f"Leaf event schedule differs from external selection: {project_name}/{leaf.name}")
            retained = len(leaf_config["data_assimilation"]["assimilation_events"])
            steps = list((leaf_project / "steps").glob("step_*"))
            if len(steps) != retained + 1:
                raise ValueError(f"Step count mismatch: {project_name}/{leaf.name}")
    if combinations != 48:
        raise ValueError(f"Expected 48 project/subdomain combinations, got {combinations}")
    return {"project_subdomain_combinations": combinations}


def _preserve_parent_provenance(staging: Path, initial: Mapping[str, Any]) -> None:
    provenance = staging / "provenance" / "pre_finalization"
    provenance.mkdir(parents=True, exist_ok=False)
    shutil.copy2(staging / "snapshot_manifest.json", provenance / "snapshot_manifest.json")
    pending_inventory = tree_inventory(staging / "projects_pending_events")
    _write_json(provenance / "pending_tree_inventory.json", pending_inventory)
    _write_json(provenance / "parent_acceptance.json", dict(initial))


def _write_station_roles(staging: Path, roles: StationRoleResult) -> None:
    source_dir = staging / "data_working" / "obs" / "stations"
    source_metadata = source_dir / "stations_da_metadata.csv"
    source_coordinates = source_dir / "stations_snow_depth.csv"
    finalized_dir = staging / "data_finalized" / "obs" / "stations"
    finalized_dir.mkdir(parents=True, exist_ok=False)
    for source in sorted(path for path in source_dir.iterdir() if path.is_file()):
        if source.name == source_metadata.name:
            continue
        target = finalized_dir / source.name
        target.symlink_to(os.path.relpath(source, start=target.parent))
    existing = read_csv_records(source_metadata)
    coordinates_by_id = {str(row["id"]): row for row in read_csv_records(source_coordinates)}
    role_by_id = {str(row["station_id"]): row for row in roles.roles}
    if set(role_by_id) != {str(row["station_id"]) for row in existing}:
        raise ValueError("Station role IDs differ from stations_da_metadata.csv")
    if set(role_by_id) != set(coordinates_by_id):
        raise ValueError("Station coordinate IDs differ from stations_da_metadata.csv")
    updated = []
    for row in existing:
        station_id = str(row["station_id"])
        role = role_by_id[station_id]
        coordinates = coordinates_by_id[station_id]
        updated.append(
            {
                **row,
                "id": coordinates["id"],
                "name": coordinates["name"],
                "x": coordinates["x"],
                "y": coordinates["y"],
                "alt": coordinates["alt"],
                "use_for_da": role["use_for_da"],
                "use_for_benchmark": role["use_for_benchmark"],
                "status": f"final_{role['role']}",
            }
        )
    _write_csv(finalized_dir / source_metadata.name, updated)


def _promote_project_configs(staging: Path, schedules: Mapping[str, ScheduleResult]) -> None:
    pending_root = staging / "projects_pending_events"
    projects_root = staging / "projects"
    projects_root.mkdir(exist_ok=False)
    for project_name in EXPECTED_PROJECTS:
        source = pending_root / project_name
        target = projects_root / project_name
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("PENDING_EVENTS", "PROJECT_STATUS.json"))
        yaml_path = target / f"{project_name}.yml"
        project = _read_yaml(yaml_path)
        da = project["data_assimilation"]
        project["obs"]["stations"]["dir"] = "data_finalized/obs/stations"
        da["prior_forcing"]["ensemble_size"] = 50
        da["output"]["retention"] = "compact"
        da.pop("subdomain_event_filter", None)
        _ensure_compact_snow_outputs(da, project_name)
        da["assimilation_events"] = [
            {
                "date": event["selected_date"],
                "variable": event["variable"],
                **({"product": "SNOWCOVER"} if event["variable"] == "scf" else {}),
            }
            for event in schedules[project_name].events
        ]
        _write_yaml(yaml_path, project)
        _write_json(
            target / "PROJECT_STATUS.json",
            {
                "status": "SCHEDULED_PENDING_SUBDOMAIN_PREPARATION",
                "runnable": False,
                "ensemble_size": 50,
                "retention": "compact",
                "event_count": len(schedules[project_name].events),
            },
        )
    shutil.rmtree(pending_root)


def _write_scheduler_inventories(
    staging: Path,
    schedules: Mapping[str, ScheduleResult],
    roles: StationRoleResult,
) -> None:
    inventory_root = staging / "inventories" / "da_event_scheduler"
    inventory_root.mkdir(parents=True, exist_ok=False)
    event_rows = []
    slot_rows = []
    exception_rows = []
    for project_name in EXPECTED_PROJECTS:
        result = schedules[project_name]
        event_rows.extend({"project": project_name, **event} for event in result.events)
        slot_rows.extend({"project": project_name, **target} for target in result.targets)
        exception_rows.extend({"project": project_name, **row} for row in result.exceptions)
    exception_rows.extend({"project": "shared_station_roles", **row} for row in roles.exceptions)
    _write_csv(inventory_root / "events.csv", event_rows)
    _write_csv(inventory_root / "target_slots.csv", slot_rows)
    _write_csv(inventory_root / "station_roles.csv", roles.roles)
    _write_csv(inventory_root / "exceptions.csv", exception_rows)
    _write_json(
        inventory_root / "summary.json",
        {
            "projects": {name: schedules[name].summary for name in EXPECTED_PROJECTS},
            "station_roles": _role_counts(roles.roles),
            "station_role_exceptions": list(roles.exceptions),
        },
    )


def _prepare_all_projects(
    staging: Path,
    image: str,
    schedules: Mapping[str, ScheduleResult],
    *,
    regions_path: str = "/setup/provenance/finalization_subdomains.gpkg",
) -> None:
    uid_gid = f"{os.getuid()}:{os.getgid()}"
    for project_name in EXPECTED_PROJECTS:
        container_project = f"/setup/projects/{project_name}"
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            uid_gid,
            "--volume",
            f"{staging}:/setup:rw",
            image,
            "openamundsen-da",
            "subdomains",
            "prepare",
            container_project,
            "--regions",
            regions_path,
            "--station-buffer-km",
            "50",
            "--grid-buffer-m",
            "0",
            "--overwrite",
            "--json",
        ]
        _run(command)
        _normalize_subdomain_project_paths(staging, project_name)
        _write_leaf_event_schedules(staging, project_name, schedules[project_name])
        script = (
            "from pathlib import Path; "
            "from openamundsen_da.subdomain.manifest import SubdomainManifest; "
            "from openamundsen_da.subdomain.run import _prepare_obs_for_subdomain; "
            f"m=SubdomainManifest.load(Path('{container_project}/subdomains/subdomain_manifest.json')); "
            "[_prepare_obs_for_subdomain(m.subdomains[s],m,overwrite=True) for s in sorted(m.subdomains)]"
        )
        _run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--user",
                uid_gid,
                "--volume",
                f"{staging}:/setup:rw",
                image,
                "python",
                "-c",
                script,
            ]
        )


def _write_leaf_event_schedules(
    staging: Path,
    project_name: str,
    schedule: ScheduleResult,
) -> None:
    """Materialize the externally selected final event list in every leaf YAML."""

    subdomain_root = staging / "projects" / project_name / "subdomains"
    leaf_dirs = sorted(
        path for path in subdomain_root.iterdir() if path.is_dir() and path.name.startswith("AT-")
    )
    if len(leaf_dirs) != EXPECTED_SUBDOMAINS:
        raise ValueError(f"Expected eight prepared leaves before event handoff: {project_name}")
    for leaf_dir in leaf_dirs:
        project_path = leaf_dir / "projects" / project_name / f"{project_name}.yml"
        project = _read_yaml(project_path)
        data_assimilation = project["data_assimilation"]
        data_assimilation.pop("subdomain_event_filter", None)
        data_assimilation["assimilation_events"] = [
            {
                "date": event["selected_date"],
                "variable": event["variable"],
                **({"product": "SNOWCOVER"} if event["variable"] == "scf" else {}),
            }
            for event in schedule.events
            if leaf_dir.name in event["supported_subdomains"]
        ]
        if not data_assimilation["assimilation_events"]:
            raise ValueError(f"Externally selected leaf event list is empty: {project_name}/{leaf_dir.name}")
        _write_yaml(project_path, project)


def _validate_all_leaf_core_requirements(staging: Path, image: str) -> None:
    """Run the pinned core pre-run requirement validator for every leaf."""

    script = """
from pathlib import Path

from openamundsen_da.io.paths import list_steps_sorted
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.validation import validate_assimilation_requirements

root = Path('/setup')
leaf_projects = sorted(root.glob('projects/project_*/subdomains/AT-*/projects/project_*'))
if len(leaf_projects) != 48:
    raise ValueError(f'Expected 48 leaf projects for core validation, got {len(leaf_projects)}')
for project_dir in leaf_projects:
    setup_dir = project_dir.parents[1]
    steps = list_steps_sorted(project_dir)
    events = load_assimilation_events(project_dir)
    if len(steps) != len(events) + 1:
        raise ValueError(f'Step/event mismatch before core validation: {project_dir}')
    validate_assimilation_requirements(
        setup_dir=setup_dir,
        project_dir=project_dir,
        steps=steps,
        events=events,
    )
print('CORE_REQUIREMENTS_OK=48')
"""
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--read-only",
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=1g",
            "--volume",
            f"{staging}:/setup:ro",
            image,
            "python",
            "-c",
            script,
        ]
    )


def _canonical_subdomain_ids(root: Path) -> tuple[str, ...]:
    import geopandas as gpd

    frame = gpd.read_file(root / "env" / "subdomains.gpkg")
    identifiers = tuple(sorted(str(value) for value in frame["id"]))
    if len(identifiers) != EXPECTED_SUBDOMAINS or len(set(identifiers)) != EXPECTED_SUBDOMAINS:
        raise ValueError(f"Expected eight unique subdomain IDs, got {identifiers}")
    return identifiers


def _prepare_partitioned_regions(staging: Path, image: str) -> None:
    """Derive a union-preserving, non-overlapping preparation vector."""

    script = """
import hashlib
import json
from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union

source = Path('/setup/data_working/env/subdomains.gpkg')
output = Path('/setup/provenance/finalization_subdomains.gpkg')
report = Path('/setup/provenance/finalization_regions.json')
frame = gpd.read_file(source)[['id', 'geometry']].copy()
frame['id'] = frame['id'].astype(str)
frame = frame.sort_values('id').reset_index(drop=True)
if frame.crs is None or frame.crs.to_epsg() != 25832:
    raise ValueError(f'Expected EPSG:25832 subdomains, got {frame.crs}')

occupied = None
partitioned = []
records = []
for row in frame.itertuples(index=False):
    original = row.geometry.buffer(0)
    geometry = original if occupied is None else original.difference(occupied).buffer(0)
    if geometry.is_empty:
        raise ValueError(f'Partition removed all geometry for {row.id}')
    removed_area = max(0.0, float(original.area - geometry.area))
    records.append({'subdomain_id': str(row.id), 'removed_overlap_area_m2': removed_area})
    partitioned.append(geometry)
    occupied = original if occupied is None else unary_union([occupied, original])

result = gpd.GeoDataFrame({'id': frame['id']}, geometry=partitioned, crs=frame.crs)
for index, left in enumerate(result.geometry):
    for right in result.geometry.iloc[index + 1:]:
        if left.intersection(right).area > 1.0e-6:
            raise ValueError('Derived preparation regions still overlap')
source_union = unary_union(list(frame.geometry))
result_union = unary_union(list(result.geometry))
union_difference = float(source_union.symmetric_difference(result_union).area)
if union_difference > 1.0e-6:
    raise ValueError(f'Derived regions changed the union by {union_difference} m2')
result.to_file(output, driver='GPKG')

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    'method': 'sorted_id_priority_difference',
    'crs': 'EPSG:25832',
    'source': 'data_working/env/subdomains.gpkg',
    'source_sha256': sha256(source),
    'output': 'provenance/finalization_subdomains.gpkg',
    'output_sha256': sha256(output),
    'source_union_area_m2': float(source_union.area),
    'union_symmetric_difference_area_m2': union_difference,
    'total_removed_overlap_area_m2': sum(item['removed_overlap_area_m2'] for item in records),
    'subdomains': records,
}
report.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
"""
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--read-only",
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=1g",
            "--volume",
            f"{staging}:/setup:rw",
            image,
            "python",
            "-c",
            script,
        ]
    )


def _normalize_subdomain_project_paths(staging: Path, project_name: str) -> None:
    """Point copied subdomain projects at their spatially filtered station data."""

    subdomain_root = staging / "projects" / project_name / "subdomains"
    subdomain_dirs = sorted(
        path for path in subdomain_root.iterdir() if path.is_dir() and path.name.startswith("AT-")
    )
    if len(subdomain_dirs) != EXPECTED_SUBDOMAINS:
        raise ValueError(f"Expected eight subdomain directories before preparation: {project_name}")
    for subdomain_dir in subdomain_dirs:
        project_path = subdomain_dir / "projects" / project_name / f"{project_name}.yml"
        project = _read_yaml(project_path)
        project["obs"]["stations"]["dir"] = "obs/stations"
        _write_yaml(project_path, project)


def _run(command: Sequence[str]) -> None:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command[:8])}\n"
            f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}"
        )


def _relativize_internal_symlinks(root: Path) -> None:
    """Make preparation-created links survive the staging-directory rename."""

    root = root.resolve()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_symlink()):
        raw_target = Path(os.readlink(path))
        if raw_target.is_absolute():
            try:
                target = root / raw_target.relative_to("/setup")
            except ValueError as exc:
                raise ValueError(f"Prepared symlink has an unknown absolute root: {path} -> {raw_target}") from exc
            target = target.resolve(strict=True)
        else:
            target = (path.parent / raw_target).resolve(strict=True)
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Prepared symlink escapes snapshot: {path} -> {target}") from exc
        relative_target = os.path.relpath(target, start=path.parent)
        path.unlink()
        path.symlink_to(relative_target)


def validate_final_snapshot(
    root: Path,
    *,
    image: str,
    verify_data_hashes: bool,
) -> dict[str, Any]:
    """Validate the complete prepared state without executing the model."""

    root = Path(root)
    _validate_internal_symlinks(root)
    if (root / "projects_pending_events").exists():
        raise ValueError("Pending project tree remains")
    if not (root / "projects").is_dir():
        raise FileNotFoundError(root / "projects")
    roles = read_csv_records(root / "inventories" / "da_event_scheduler" / "station_roles.csv")
    for role in roles:
        da = str(role["use_for_da"]).lower() == "true"
        benchmark = str(role["use_for_benchmark"]).lower() == "true"
        if da == benchmark:
            raise ValueError(f"Station role is not mutually exclusive: {role['station_id']}")
    _validate_fsc_event_links(root)
    per_domain_dropped: dict[str, int] = {}
    combinations = 0
    step_counts: dict[str, int] = {}
    for project_name in EXPECTED_PROJECTS:
        project_dir = root / "projects" / project_name
        project = _read_yaml(project_dir / f"{project_name}.yml")
        da = project["data_assimilation"]
        if da["prior_forcing"]["ensemble_size"] != 50 or da["output"]["retention"] != "compact":
            raise ValueError(f"ES50/compact contract failed: {project_name}")
        event_dates = [str(event["date"]) for event in da["assimilation_events"]]
        if len(event_dates) != len(set(event_dates)):
            raise ValueError(f"Duplicate event dates: {project_name}")
        subdomain_root = project_dir / "subdomains"
        subdomain_dirs = sorted(
            path for path in subdomain_root.iterdir() if path.is_dir() and path.name.startswith("AT-")
        )
        if len(subdomain_dirs) != EXPECTED_SUBDOMAINS:
            raise ValueError(f"Expected eight subdomains in {project_name}, got {len(subdomain_dirs)}")
        for subdomain_dir in subdomain_dirs:
            combinations += 1
            sub_project = subdomain_dir / "projects" / project_name
            sub_yaml = _read_yaml(sub_project / f"{project_name}.yml")
            retained = len(sub_yaml["data_assimilation"]["assimilation_events"])
            steps = sorted(path for path in (sub_project / "steps").glob("step_*") if path.is_dir())
            if len(steps) != retained + 1:
                raise ValueError(
                    f"Step count mismatch {project_name}/{subdomain_dir.name}: {len(steps)} != {retained + 1}"
                )
            step_counts[f"{project_name}/{subdomain_dir.name}"] = len(steps)
            dropped_path = subdomain_dir / "subdomain_dropped_events.csv"
            dropped = len(read_csv_records(dropped_path)) if dropped_path.is_file() else 0
            per_domain_dropped[f"{project_name}/{subdomain_dir.name}"] = dropped
    if combinations != 48:
        raise ValueError(f"Expected 48 project/subdomain combinations, got {combinations}")
    forbidden = [
        path
        for pattern in ("results", "restart", "model_state*.pickle*", "*.restart*")
        for path in root.rglob(pattern)
        if path.exists()
    ]
    if forbidden:
        raise ValueError(f"Runtime artifacts exist: {forbidden[:10]}")
    hash_counts = verify_recorded_data_hashes(root) if verify_data_hashes else {}
    _verify_finalization_hash(root)
    return {
        "image": image,
        "project_subdomain_combinations": combinations,
        "step_counts": step_counts,
        "per_domain_dropped_events": per_domain_dropped,
        **hash_counts,
    }


def _validate_internal_symlinks(root: Path) -> None:
    resolved_root = root.resolve()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_symlink()):
        if Path(os.readlink(path)).is_absolute():
            raise ValueError(f"Snapshot symlink is absolute: {path}")
        target = path.resolve(strict=True)
        try:
            target.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"Snapshot symlink escapes root: {path} -> {target}") from exc


def _verify_finalization_hash(root: Path) -> None:
    manifest_path = root / "provenance" / "finalization_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = _read_json(manifest_path)
    excluded = {
        root / "snapshot_manifest.json",
        manifest_path,
        root / "READY_FOR_EVENT_SELECTION",
        root / "READY_TO_RUN",
    }
    actual = inventory_digest(tree_inventory(root, excluded=excluded))
    if actual != manifest.get("final_hash"):
        raise ValueError(f"Finalization tree hash mismatch: {actual} != {manifest.get('final_hash')}")


def _validate_fsc_event_links(root: Path) -> None:
    working_records = _read_json(root / "provenance" / "fsc_working_files.json")
    available = {
        (str(record["date"]), Path(str(record["working"])).name): record
        for record in working_records
    }
    event_rows = read_csv_records(root / "inventories" / "da_event_scheduler" / "events.csv")
    quality_rows = read_csv_records(root / "inventories" / "fsc_scene_subdomain_quality.csv")
    uncertainty_counts: dict[tuple[str, str], int] = {}
    for row in quality_rows:
        key = (str(row["date"]), str(row["source_file"]))
        uncertainty_counts[key] = max(
            uncertainty_counts.get(key, 0),
            int(float(row["uncertainty_valid_fsc_count"])),
        )
    for event in event_rows:
        if event["variable"] != "scf":
            continue
        key = (str(event["selected_date"]), str(event["source_file"]))
        record = available.get(key)
        if record is None:
            raise ValueError(f"Selected FSC event is not linked to a working scene: {key}")
        working_path = root / str(record["working"])
        if not working_path.is_file() or uncertainty_counts.get(key, 0) < 1:
            raise ValueError(f"Selected FSC event lacks its scene or uncertainty layer: {key}")


def _write_finalization_manifest(
    staging: Path,
    *,
    initial: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    schedules: Mapping[str, ScheduleResult],
    roles: StationRoleResult,
    policy_path: Path,
    image: str,
    commit: str,
) -> None:
    excluded = {
        staging / "snapshot_manifest.json",
        staging / "provenance" / "finalization_manifest.json",
        staging / "READY_FOR_EVENT_SELECTION",
        staging / "READY_TO_RUN",
    }
    final_inventory = tree_inventory(staging, excluded=excluded)
    manifest = {
        "schema_version": FINALIZER_SCHEMA_VERSION,
        "status": "READY_TO_RUN",
        "finalized_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scheduler_commit": commit,
        "policy_sha256": sha256_file(policy_path),
        "image": image,
        "parent_manifest_sha256": initial["parent_manifest_sha256"],
        "pending_tree_sha256": initial["pending_tree_sha256"],
        "preparation_regions": _read_json(staging / "provenance" / "finalization_regions.json"),
        "final_hash_scope": "all regular files except status markers and mutable top-level manifests",
        "final_hash": inventory_digest(final_inventory),
        "final_file_count": len(final_inventory),
        "projects": {name: schedules[name].summary for name in EXPECTED_PROJECTS},
        "station_roles": _role_counts(roles.roles),
        "station_role_exceptions": list(roles.exceptions),
        "per_domain_dropped_events": acceptance["per_domain_dropped_events"],
        "acceptance": dict(acceptance),
    }
    _write_json(staging / "provenance" / "finalization_manifest.json", manifest)


def _set_ready_state(staging: Path, commit: str) -> None:
    marker = staging / "READY_FOR_EVENT_SELECTION"
    if marker.exists():
        marker.unlink()
    snapshot_manifest = _read_json(staging / "snapshot_manifest.json")
    snapshot_manifest["status"] = "READY_TO_RUN"
    snapshot_manifest["finalizer_commit"] = commit
    snapshot_manifest["finalization_manifest"] = "provenance/finalization_manifest.json"
    _write_json(staging / "snapshot_manifest.json", snapshot_manifest)
    _write_json(
        staging / "READY_TO_RUN",
        {
            "status": "READY_TO_RUN",
            "ensemble_size": 50,
            "model_executed": False,
            "next_phase": "separate explicitly approved ES50 compute plan",
        },
    )
    (staging / "README.md").write_text(
        "# North Tyrol six-project snapshot\n\n"
        "This snapshot is prepared for event-reviewed ES50 execution. No model propagation has run.\n",
        encoding="utf-8",
    )
    for status_path in sorted((staging / "projects").rglob("PROJECT_STATUS.json")):
        status = _read_json(status_path)
        status.update({"status": "READY_TO_RUN", "runnable": True, "model_executed": False})
        _write_json(status_path, status)


def _role_counts(roles: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in roles:
        role = str(row["role"])
        counts[role] = counts.get(role, 0) + 1
    return dict(sorted(counts.items()))


def _finalizer_commit() -> str:
    supplied = os.environ.get("NORTH_TYROL_FINALIZER_COMMIT", "")
    if supplied:
        commit = supplied
    else:
        commit = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Finalizer commit must be a full lowercase Git SHA")
    return commit


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _read_yaml(path: Path) -> dict[str, Any]:
    from ruamel.yaml import YAML

    payload = YAML().load(Path(path))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 120
    with Path(path).open("w", encoding="utf-8") as file_obj:
        yaml.dump(payload, file_obj)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fieldnames = sorted({key for row in materialized for key in row})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as file_obj:
        if not fieldnames:
            return
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in materialized:
            writer.writerow(
                {
                    key: ";".join(str(item) for item in value) if isinstance(value, (list, tuple)) else value
                    for key, value in ((key, row.get(key, "")) for key in fieldnames)
                }
            )


def _write_incomplete(root: Path, error: str) -> None:
    _write_json(
        root / "INCOMPLETE.json",
        {
            "status": "INCOMPLETE",
            "failed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "error": error,
        },
    )


def main() -> int:
    """Run read-only preflight or rollback-safe finalization."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.preflight:
        print(json.dumps(preflight(args.setup_root, args.policy, args.image), indent=2, sort_keys=True))
        return 0
    output = finalize(
        args.setup_root,
        args.policy,
        args.image,
        discard_runtime_artifacts=args.discard_runtime_artifacts,
    )
    print(f"READY_TO_RUN: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
