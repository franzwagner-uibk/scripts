"""Focused synthetic contracts for scheduling and rollback-safe finalization."""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import finalizeNorthTyrolProjects as finalizer  # noqa: E402
from da_event_scheduler import (  # noqa: E402
    SchedulePolicy,
    StationRoleResult,
    adapt_station_roles_for_support,
    assign_station_roles,
    generate_slots,
    schedule_events,
    write_schedule_outputs,
)


def _policy(**changes: object) -> SchedulePolicy:
    base = SchedulePolicy(
        schema_version=1,
        target_spacing_days=6,
        sequence=("scf", "station_hs"),
        interval_start=(10, 7),
        interval_end=(7, 31),
        fsc_search_days=4,
        station_hs_search_days=4,
        minimum_gap_days=5,
        maximum_gap_days=7,
        maximum_cloud_fraction=0.2,
        minimum_fulfillment=0.85,
    )
    return replace(base, **changes)


def _fsc(day: str, cloud_a: float, cloud_b: float, *, uncertainty: float = 10.0) -> list[dict[str, object]]:
    return [
        {
            "date": day,
            "subdomain_id": subdomain,
            "source_file": f"scene_{day}.nc",
            "cloud_fraction": cloud,
            "uncertainty_count": 10,
            "uncertainty_mean": uncertainty,
        }
        for subdomain, cloud in (("A", cloud_a), ("B", cloud_b))
    ]


def _snow(day: str, station_id: str, subdomain: str, count: int = 1) -> dict[str, object]:
    return {
        "date": day,
        "station_id": station_id,
        "subdomain_id": subdomain,
        "valid_observation_count": count,
    }


def _roles() -> tuple[dict[str, object], ...]:
    return (
        {"station_id": "a", "subdomain_id": "A", "role": "da"},
        {"station_id": "b", "subdomain_id": "B", "role": "holdout"},
        {"station_id": "c", "subdomain_id": "B", "role": "da"},
    )


def test_slots_boundaries_alternation_and_deterministic_matching() -> None:
    policy = _policy()
    slots = generate_slots(date(2022, 10, 1), date(2023, 9, 30), policy)
    assert len(slots) == 50
    assert [(slot.target_date.isoformat(), slot.variable) for slot in slots[:3]] == [
        ("2022-10-07", "scf"),
        ("2022-10-13", "station_hs"),
        ("2022-10-19", "scf"),
    ]
    assert slots[-1].target_date <= date(2023, 7, 31)

    short_policy = replace(policy, interval_end=(10, 25))
    fsc_rows = [
        *_fsc("2022-10-06", 0.1, 0.1, uncertainty=20),
        *_fsc("2022-10-07", 0.05, 0.4, uncertainty=5),
        *_fsc("2022-10-18", 0.1, 0.1),
        *_fsc("2022-10-19", 0.3, 0.4),
    ]
    snow_rows = [
        _snow("2022-10-12", "a", "A"),
        _snow("2022-10-12", "b", "B"),
        _snow("2022-10-12", "c", "B"),
        _snow("2022-10-24", "a", "A"),
    ]
    result = schedule_events(
        policy=short_policy,
        fsc_rows=fsc_rows,
        snow_rows=snow_rows,
        station_roles=_roles(),
        start=date(2022, 10, 1),
        end=date(2023, 9, 30),
    )
    assert [event["selected_date"] for event in result.events] == [
        "2022-10-06",
        "2022-10-12",
        "2022-10-18",
        "2022-10-24",
    ]
    assert len({event["selected_date"] for event in result.events}) == len(result.events)
    assert result.events[0]["supported_subdomain_count"] == 2


def test_skips_unavailable_or_bad_gap_slots_and_enforces_fulfillment() -> None:
    policy = _policy(interval_end=(10, 25), minimum_fulfillment=0.5)
    result = schedule_events(
        policy=policy,
        fsc_rows=[*_fsc("2022-10-07", 0.1, 0.1), *_fsc("2022-10-19", 0.1, 0.1)],
        snow_rows=[_snow("2022-10-15", "a", "A"), _snow("2022-10-25", "a", "A")],
        station_roles=_roles(),
        start=date(2022, 10, 1),
        end=date(2022, 10, 31),
    )
    assert [event["variable"] for event in result.events] == ["scf", "scf", "station_hs"]
    assert result.exceptions[0]["variable"] == "station_hs"

    with pytest.raises(ValueError, match="fulfillment"):
        schedule_events(
            policy=replace(policy, minimum_fulfillment=0.85),
            fsc_rows=[*_fsc("2022-10-07", 0.3, 0.4), *_fsc("2022-10-19", 0.3, 0.4)],
            snow_rows=[_snow("2022-10-13", "a", "A"), _snow("2022-10-25", "a", "A")],
            station_roles=_roles(),
            start=date(2022, 10, 1),
            end=date(2022, 10, 31),
        )


def test_shared_elevation_aware_holdouts_reduce_only_for_da_support() -> None:
    metadata = [
        {"id": "a", "alt": 1000},
        {"id": "b", "alt": 1500},
        {"id": "c", "alt": 2000},
        {"id": "d", "alt": 3000},
    ]
    snow_rows = [
        _snow(day, station_id, "A", count)
        for station_id, count in (("a", 1), ("b", 1), ("c", 1), ("d", 1))
        for day in ("2022-10-07", "2022-10-13")
    ]
    roles = assign_station_roles(snow_rows, metadata)
    holdouts = {row["station_id"] for row in roles.roles if row["role"] == "holdout"}
    assert holdouts == {"a", "d"}
    assert all(bool(row["use_for_da"]) != bool(row["use_for_benchmark"]) for row in roles.roles)

    only_holdout_active = [_snow("2022-10-19", "a", "A")]
    adjusted = adapt_station_roles_for_support(roles, only_holdout_active, [date(2022, 10, 19)])
    row_a = next(row for row in adjusted.roles if row["station_id"] == "a")
    assert row_a["role"] == "da"
    assert adjusted.exceptions[0]["exception"] == "holdout_reduced_for_da_support"


def test_preflight_output_is_read_only_and_nonempty_output_is_refused(tmp_path: Path) -> None:
    result = schedule_events(
        policy=_policy(interval_end=(10, 13)),
        fsc_rows=_fsc("2022-10-07", 0.1, 0.1),
        snow_rows=[_snow("2022-10-13", "a", "A")],
        station_roles=_roles(),
        start=date(2022, 10, 1),
        end=date(2022, 10, 31),
    )
    output = tmp_path / "output"
    summary = write_schedule_outputs(output, result, StationRoleResult(_roles(), ()), preflight=True)
    assert summary["schedule"]["retained_count"] == 2
    assert not output.exists()
    output.mkdir()
    (output / "owned.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_schedule_outputs(output, result, StationRoleResult(_roles(), ()))
    assert (output / "owned.txt").read_text(encoding="utf-8") == "keep"


def test_source_state_refuses_existing_projects(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    (root / "projects").mkdir(parents=True)
    (root / "snapshot_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="READY_FOR_EVENT_SELECTION"):
        finalizer.validate_source_snapshot(root, finalizer.EXPECTED_IMAGE)


def test_container_rooted_symlinks_become_internal_relative_links(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    target = root / "data_working" / "meteo" / "station.csv"
    target.parent.mkdir(parents=True)
    target.write_text("time,temp\n", encoding="utf-8")
    link = root / "projects" / "project" / "subdomains" / "AT-01" / "meteo" / "station.csv"
    link.parent.mkdir(parents=True)
    link.symlink_to("/setup/data_working/meteo/station.csv")

    finalizer._relativize_internal_symlinks(root)

    assert not Path(link.readlink()).is_absolute()
    assert link.resolve(strict=True) == target
    finalizer._validate_internal_symlinks(root)


def test_station_role_layer_preserves_recorded_working_metadata(tmp_path: Path) -> None:
    working = tmp_path / "data_working" / "obs" / "stations"
    working.mkdir(parents=True)
    source_metadata = working / "stations_da_metadata.csv"
    original = (
        "station_id,use_for_da,use_for_benchmark,status\n"
        "a,True,True,provisional_pending_event_selection\n"
    )
    source_metadata.write_text(original, encoding="utf-8")
    (working / "stations_snow_depth.csv").write_text(
        "id,name,x,y,alt\n"
        "a,Station A,100.0,200.0,1500.0\n",
        encoding="utf-8",
    )
    (working / "a.csv").write_text("time,snow_depth\n", encoding="utf-8")
    roles = StationRoleResult(
        (
            {
                "station_id": "a",
                "role": "holdout",
                "use_for_da": False,
                "use_for_benchmark": True,
            },
        ),
        (),
    )

    finalizer._write_station_roles(tmp_path, roles)

    finalized = tmp_path / "data_finalized" / "obs" / "stations"
    assert source_metadata.read_text(encoding="utf-8") == original
    assert (finalized / "a.csv").is_symlink()
    finalized_role = finalizer.read_csv_records(finalized / "stations_da_metadata.csv")[0]
    assert finalized_role["station_id"] == "a"
    assert finalized_role["status"] == "final_holdout"
    assert finalized_role["use_for_da"] == "False"
    assert finalized_role["use_for_benchmark"] == "True"
    assert finalized_role["id"] == "a"
    assert finalized_role["name"] == "Station A"
    assert finalized_role["x"] == "100.0"
    assert finalized_role["y"] == "200.0"
    assert finalized_role["alt"] == "1500.0"


def test_finalizer_failure_leaves_canonical_tree_and_marks_staging_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    sentinel = root / "accepted.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    fake_roles = StationRoleResult((), ())
    monkeypatch.setattr(
        finalizer,
        "preflight",
        lambda *_args, **_kwargs: {"parent_manifest_sha256": "a", "pending_tree_sha256": "b"},
    )
    monkeypatch.setattr(finalizer, "_finalizer_commit", lambda: "a" * 40)
    monkeypatch.setattr(finalizer, "build_schedules", lambda *_args: (fake_roles, {}))
    for function_name in (
        "_preserve_parent_provenance",
        "_write_station_roles",
        "_promote_project_configs",
        "_write_scheduler_inventories",
        "_prepare_partitioned_regions",
    ):
        monkeypatch.setattr(finalizer, function_name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        finalizer,
        "_prepare_all_projects",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic preparation failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic preparation failure"):
        finalizer.finalize(root, tmp_path / "policy.yml", finalizer.EXPECTED_IMAGE)
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    staging = next(tmp_path.glob(".snapshot.finalizing-*"))
    assert (staging / "INCOMPLETE.json").is_file()
