#!/usr/bin/env python3
"""Finalize the accepted six-season North Tyrol snapshot without propagation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from da_event_scheduler import (
    ScheduleResult,
    StationRoleResult,
    load_policy,
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
FINALIZER_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    """Parse the documented North Tyrol finalizer interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-root", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """Build all six schedules using one station split."""

    manifest = _read_json(root / "snapshot_manifest.json")
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
        fsc_rows=read_csv_records(root / "inventories" / "fsc_scene_subdomain_quality.csv"),
        snow_rows=read_csv_records(root / "inventories" / "snow_station_daily_support.csv"),
        station_rows=read_csv_records(root / "data_working" / "obs" / "stations" / "stations_snow_depth.csv"),
        windows=windows,
    )


def preflight(root: Path, policy_path: Path, image: str, *, verify_hashes: bool = True) -> dict[str, Any]:
    """Read and validate all finalization inputs without writing."""

    source = validate_source_snapshot(root, image)
    roles, schedules = build_schedules(root, policy_path)
    hash_counts = verify_recorded_data_hashes(root) if verify_hashes else {"raw_files": 0, "working_files": 0}
    return {
        "status": "PREFLIGHT_OK",
        **source,
        **hash_counts,
        "policy_sha256": sha256_file(policy_path),
        "station_roles": _role_counts(roles.roles),
        "station_role_exceptions": list(roles.exceptions),
        "projects": {name: result.summary for name, result in schedules.items()},
    }


def finalize(root: Path, policy_path: Path, image: str) -> Path:
    """Finalize in sibling staging, atomically promote and rollback on failure."""

    root = Path(root).resolve()
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
        _prepare_all_projects(staging, image)
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


def _preserve_parent_provenance(staging: Path, initial: Mapping[str, Any]) -> None:
    provenance = staging / "provenance" / "pre_finalization"
    provenance.mkdir(parents=True, exist_ok=False)
    shutil.copy2(staging / "snapshot_manifest.json", provenance / "snapshot_manifest.json")
    pending_inventory = tree_inventory(staging / "projects_pending_events")
    _write_json(provenance / "pending_tree_inventory.json", pending_inventory)
    _write_json(provenance / "parent_acceptance.json", dict(initial))


def _write_station_roles(staging: Path, roles: StationRoleResult) -> None:
    path = staging / "data_working" / "obs" / "stations" / "stations_da_metadata.csv"
    existing = read_csv_records(path)
    role_by_id = {str(row["station_id"]): row for row in roles.roles}
    if set(role_by_id) != {str(row["station_id"]) for row in existing}:
        raise ValueError("Station role IDs differ from stations_da_metadata.csv")
    updated = []
    for row in existing:
        station_id = str(row["station_id"])
        role = role_by_id[station_id]
        updated.append(
            {
                **row,
                "use_for_da": role["use_for_da"],
                "use_for_benchmark": role["use_for_benchmark"],
                "status": f"final_{role['role']}",
            }
        )
    _write_csv(path, updated)


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
        da["prior_forcing"]["ensemble_size"] = 50
        da["output"]["retention"] = "compact"
        event_filter = da["subdomain_event_filter"]
        event_filter["variables"]["scf"]["max_cloud_fraction"] = 0.20
        event_filter.pop("subdomains", None)
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
        slot_rows.extend(
            {
                "project": project_name,
                "slot_index": slot.index,
                "target_date": slot.target_date,
                "variable": slot.variable,
            }
            for slot in result.slots
        )
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


def _prepare_all_projects(staging: Path, image: str) -> None:
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
            "/setup/data_working/env/subdomains.gpkg",
            "--station-buffer-km",
            "50",
            "--grid-buffer-m",
            "0",
            "--overwrite",
            "--json",
        ]
        _run(command)
        _normalize_subdomain_project_paths(staging, project_name)
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
        target = path.resolve(strict=True)
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
    uncertainty_counts = {
        (str(row["date"]), str(row["source_file"])): int(float(row["uncertainty_count"]))
        for row in quality_rows
    }
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

    args = parse_args()
    if args.preflight:
        print(json.dumps(preflight(args.setup_root, args.policy, args.image), indent=2, sort_keys=True))
        return 0
    output = finalize(args.setup_root, args.policy, args.image)
    print(f"READY_TO_RUN: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
