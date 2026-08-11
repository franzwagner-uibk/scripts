"""Focused contracts for the North Tyrol v2 scheduler and finalizer."""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pytest


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import finalizeNorthTyrolProjects as finalizer  # noqa: E402
from da_event_scheduler import (  # noqa: E402
    SchedulePolicy,
    ScheduleResult,
    StationRoleResult,
    adapt_station_roles_for_support,
    assign_station_roles,
    fsc_reference_metrics,
    generate_slots,
    load_policy,
    schedule_events,
    schedule_with_adaptive_roles,
    write_schedule_outputs,
)


def _policy(**changes: object) -> SchedulePolicy:
    base = SchedulePolicy(
        schema_version=2,
        target_spacing_days=6,
        sequence=("scf", "station_hs"),
        interval_start=(10, 7),
        interval_end=(7, 31),
        fsc_search_days=4,
        station_hs_search_days=4,
        minimum_gap_days=5,
        maximum_gap_days=7,
        maximum_cloud_fraction=0.20,
        maximum_invalid_fraction=0.20,
        minimum_fulfillment=0.85,
        station_observation_time=time(0),
        model_timestep_hours=3,
        fulfillment_denominator="feasible_slots",
    )
    return replace(base, **changes)


def _fsc(day: str, *, cloud: int = 10, invalid: int = 10, water: int = 20) -> list[dict[str, object]]:
    return [
        {
            "date": day,
            "subdomain_id": domain,
            "source_file": f"scene_{day}.nc",
            "pixel_count": 100 + water,
            "valid_count": 100 - cloud - invalid,
            "cloud_count": cloud,
            "nodata_count": invalid,
            "water_count": water,
            "uncertainty_count": 80,
            "uncertainty_valid_fsc_count": 100 - cloud - invalid,
            "uncertainty_mean": 10.0,
            "uncertainty_p90": 12.0,
        }
        for domain in ("A", "B")
    ]


def _snow(timestamp: str, station_id: str, domain: str) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "station_id": station_id,
        "subdomain_id": domain,
        "valid_observation_count": 1,
    }


def _roles() -> tuple[dict[str, object], ...]:
    return (
        {"station_id": "a", "subdomain_id": "A", "role": "da"},
        {"station_id": "b", "subdomain_id": "B", "role": "da"},
    )


def test_versioned_policy_loads_the_fixed_v2_contract() -> None:
    policy = load_policy(MODULE_ROOT / "policies" / "north_tyrol_alternating_6day_v2.yml")
    assert policy.schema_version == 2
    assert policy.maximum_cloud_fraction == 0.20
    assert policy.maximum_invalid_fraction == 0.20
    assert policy.fulfillment_denominator == "feasible_slots"


def test_slots_quality_gates_and_exact_timestep_selection() -> None:
    policy = replace(_policy(), interval_end=(10, 25))
    slots = generate_slots(date(2022, 10, 1), date(2023, 9, 30), _policy())
    assert len(slots) == 50
    assert [(item.target_date.isoformat(), item.variable) for item in slots[:3]] == [
        ("2022-10-07", "scf"),
        ("2022-10-13", "station_hs"),
        ("2022-10-19", "scf"),
    ]
    fsc_rows = [*_fsc("2022-10-06"), *_fsc("2022-10-18")]
    snow_rows = [
        _snow("2022-10-12 01:29:00", "a", "A"),
        _snow("2022-10-12 06:00:00", "b", "B"),
        _snow("2022-10-24 00:00:00", "b", "B"),
    ]
    result = schedule_events(
        policy=policy,
        fsc_rows=fsc_rows,
        snow_rows=snow_rows,
        station_roles=_roles(),
        start=date(2022, 10, 1),
        end=date(2022, 10, 31),
    )
    assert [event["selected_timestamp"] for event in result.events] == [
        "2022-10-06 00:00:00",
        "2022-10-12 00:00:00",
        "2022-10-18 00:00:00",
        "2022-10-24 00:00:00",
    ]
    assert all(event["selected_date"] != "2022-10-12" or event["active_station_ids"] == ["a"] for event in result.events)
    hs_event = next(event for event in result.events if event["selected_date"] == "2022-10-12")
    assert hs_event["station_match_max_delta_minutes"] == 89.0


def test_partial_window_preserves_the_fixed_cadence_and_sequence_phase() -> None:
    slots = generate_slots(date(2022, 10, 14), date(2022, 10, 31), _policy())
    assert [(slot.target_date.isoformat(), slot.variable) for slot in slots] == [
        ("2022-10-19", "scf"),
        ("2022-10-25", "station_hs"),
        ("2022-10-31", "scf"),
    ]
    winter_slots = generate_slots(date(2023, 1, 3), date(2023, 1, 12), _policy())
    assert [(slot.target_date.isoformat(), slot.variable) for slot in winter_slots] == [
        ("2023-01-05", "station_hs"),
        ("2023-01-11", "scf"),
    ]


def test_fsc_reference_excludes_water_and_separates_cloud_from_invalid() -> None:
    metrics = fsc_reference_metrics(_fsc("2022-10-07", cloud=20, invalid=20, water=900)[0])
    assert metrics["reference_count"] == 100
    assert metrics["cloud_reference_fraction"] == pytest.approx(0.20)
    assert metrics["invalid_reference_fraction"] == pytest.approx(0.20)

    policy = replace(_policy(), interval_end=(10, 7))
    accepted = schedule_events(
        policy=policy,
        fsc_rows=_fsc("2022-10-07", cloud=20, invalid=20, water=900),
        snow_rows=[],
        station_roles=(),
        start=date(2022, 10, 1),
        end=date(2022, 10, 31),
    )
    assert len(accepted.events) == 1
    rejected = schedule_events(
        policy=policy,
        fsc_rows=_fsc("2022-10-07", cloud=20, invalid=21),
        snow_rows=[],
        station_roles=(),
        start=date(2022, 10, 1),
        end=date(2022, 10, 31),
    )
    assert not rejected.events
    assert rejected.summary["by_variable"]["scf"]["feasible_targets"] == 0

    incomplete = _fsc("2022-10-07")
    incomplete[0]["uncertainty_valid_fsc_count"] = int(incomplete[0]["valid_count"]) - 1
    partly_supported = schedule_events(
        policy=policy,
        fsc_rows=incomplete,
        snow_rows=[],
        station_roles=(),
        start=date(2022, 10, 1),
        end=date(2022, 10, 31),
    )
    assert partly_supported.events[0]["supported_subdomains"] == ["B"]
    rejected_row = next(
        row for row in partly_supported.quality if row.get("subdomain_id") == "A"
    )
    assert rejected_row["rejection_reasons"] == ["incomplete_uncertainty"]


def test_fsc_ranking_prefers_valid_support_then_uncertainty_and_offset() -> None:
    policy = replace(_policy(), interval_end=(10, 7))
    weak = _fsc("2022-10-06", cloud=15, invalid=15)
    strong = _fsc("2022-10-08", cloud=5, invalid=5)
    result = schedule_events(
        policy=policy,
        fsc_rows=[*weak, *strong],
        snow_rows=[],
        station_roles=(),
        start=date(2022, 10, 1),
        end=date(2022, 10, 31),
    )
    assert result.events[0]["selected_date"] == "2022-10-08"
    assert result.events[0]["valid_support_count"] == 180


def test_leaf_type_fulfillment_fails_when_no_common_schedule_can_reach_85_percent() -> None:
    policy = replace(_policy(), interval_end=(10, 7))
    first = _fsc("2022-10-06")
    second = _fsc("2022-10-08")
    first[1].update({"valid_count": 69, "cloud_count": 10, "nodata_count": 21, "uncertainty_valid_fsc_count": 69})
    second[0].update({"valid_count": 69, "cloud_count": 10, "nodata_count": 21, "uncertainty_valid_fsc_count": 69})
    with pytest.raises(ValueError, match=r"[AB] scf feasible-slot fulfillment"):
        schedule_events(
            policy=policy,
            fsc_rows=[*first, *second],
            snow_rows=[],
            station_roles=(),
            start=date(2022, 10, 1),
            end=date(2022, 10, 31),
        )


def test_station_half_timestep_tie_is_rejected() -> None:
    rows = [
        _snow("2022-10-12 23:30:00", "a", "A"),
        _snow("2022-10-13 00:30:00", "a", "A"),
    ]
    with pytest.raises(ValueError, match="Ambiguous station observations"):
        schedule_events(
            policy=replace(_policy(), interval_start=(10, 13), interval_end=(10, 13), sequence=("station_hs", "scf")),
            fsc_rows=[],
            snow_rows=rows,
            station_roles=_roles(),
            start=date(2022, 10, 1),
            end=date(2022, 10, 31),
        )


def test_fulfillment_uses_feasible_slots_and_dates_are_globally_unique() -> None:
    policy = replace(_policy(), interval_end=(10, 25))
    result = schedule_events(
        policy=policy,
        fsc_rows=_fsc("2022-10-07"),
        snow_rows=[_snow("2022-10-13 00:00:00", "a", "A")],
        station_roles=_roles(),
        start=date(2022, 10, 1),
        end=date(2022, 10, 31),
    )
    assert result.summary["by_variable"]["scf"] == {
        "targets": 2,
        "feasible_targets": 1,
        "unavailable_targets": 1,
        "retained": 1,
        "fulfillment_denominator": "feasible_targets",
        "fulfillment_fraction": 1.0,
    }
    assert len({event["selected_date"] for event in result.events}) == len(result.events)


def test_skipped_slot_never_allows_an_under_five_day_event_gap() -> None:
    policy = replace(_policy(), interval_end=(10, 19), minimum_fulfillment=0.50)
    result = schedule_events(
        policy=policy,
        fsc_rows=[*_fsc("2022-10-11"), *_fsc("2022-10-15")],
        snow_rows=[],
        station_roles=(),
        start=date(2022, 10, 1),
        end=date(2022, 10, 31),
    )
    assert len(result.events) == 1


def test_shared_sparse_roles_keep_feasible_event_support_in_da() -> None:
    metadata = [
        {"id": "a", "alt": 1000},
        {"id": "b", "alt": 1500},
        {"id": "c", "alt": 2000},
        {"id": "x", "alt": 1200},
        {"id": "y", "alt": 1800},
    ]
    station_dates = {"a": 7, "b": 13, "c": 19}
    operational_rows = [
        *[
            _snow(f"2022-10-{day:02d} 00:00:00", station, "A")
            for station, day in station_dates.items()
        ],
        *[
            _snow(f"2022-10-{day:02d} 00:00:00", "x", "B")
            for day in station_dates.values()
        ],
        _snow("2022-10-13 00:00:00", "y", "B"),
    ]
    initial = assign_station_roles(operational_rows, metadata, _policy())
    assert {row["station_id"] for row in initial.roles if row["subdomain_id"] == "B" and row["role"] == "da"} == {"x", "y"}
    holdout = next(row["station_id"] for row in initial.roles if row["subdomain_id"] == "A" and row["role"] == "holdout")
    event_day = station_dates[str(holdout)]
    roles, schedules = schedule_with_adaptive_roles(
        policy=replace(
            _policy(),
            interval_start=(10, event_day),
            interval_end=(10, event_day),
            sequence=("station_hs", "scf"),
        ),
        fsc_rows=[],
        snow_rows=operational_rows,
        station_rows=metadata,
        windows=(("project", date(2022, 10, 1), date(2022, 10, 31)),),
    )
    assert schedules["project"].events
    role_by_station = {str(row["station_id"]): str(row["role"]) for row in roles.roles}
    assert role_by_station[str(holdout)] == "da"
    assert not roles.exceptions


def test_holdout_selection_keeps_best_temporal_support_for_da() -> None:
    metadata = [
        {"id": station_id, "alt": altitude}
        for station_id, altitude in (("a", 1000), ("b", 1500), ("c", 2000), ("d", 2500))
    ]
    rows = [
        _snow(f"2022-10-{day:02d} 00:00:00", "a", "A")
        for day in (1, 2, 3, 4)
    ] + [
        _snow("2022-10-01 00:00:00", station_id, "A")
        for station_id in ("b", "c", "d")
    ]
    roles = assign_station_roles(rows, metadata, _policy())
    role_by_station = {str(row["station_id"]): str(row["role"]) for row in roles.roles}
    assert role_by_station["a"] == "da"
    assert {station_id for station_id, role in role_by_station.items() if role == "holdout"} == {"b", "d"}


def test_adaptive_reduction_promotes_strongest_active_holdout() -> None:
    roles = StationRoleResult(
        roles=(
            {
                "station_id": "a",
                "subdomain_id": "A",
                "role": "holdout",
                "use_for_da": False,
                "use_for_benchmark": True,
                "holdout_rank": 1,
                "valid_timestep_count": 20,
            },
            {
                "station_id": "b",
                "subdomain_id": "A",
                "role": "holdout",
                "use_for_da": False,
                "use_for_benchmark": True,
                "holdout_rank": 2,
                "valid_timestep_count": 5,
            },
        ),
        exceptions=(),
    )
    rows = [
        _snow("2022-10-13 00:00:00", station_id, "A")
        for station_id in ("a", "b")
    ]
    adjusted = adapt_station_roles_for_support(
        roles,
        rows,
        [datetime(2022, 10, 13)],
        _policy(),
    )
    role_by_station = {str(row["station_id"]): str(row["role"]) for row in adjusted.roles}
    assert role_by_station == {"a": "da", "b": "holdout"}


def test_preflight_output_is_read_only_and_quality_is_written(tmp_path: Path) -> None:
    result = schedule_events(
        policy=replace(_policy(), interval_end=(10, 7)),
        fsc_rows=_fsc("2022-10-07"),
        snow_rows=[],
        station_roles=(),
        start=date(2022, 10, 1),
        end=date(2022, 10, 31),
    )
    output = tmp_path / "output"
    summary = write_schedule_outputs(output, result, StationRoleResult((), ()), preflight=True)
    assert summary["schedule"]["retained_count"] == 1
    assert not output.exists()
    write_schedule_outputs(output, result, StationRoleResult((), ()))
    assert (output / "quality.csv").is_file()
    with pytest.raises(FileExistsError):
        write_schedule_outputs(output, result, StationRoleResult((), ()))

    second = tmp_path / "second"
    write_schedule_outputs(second, result, StationRoleResult((), ()))
    assert {
        path.name: path.read_bytes()
        for path in sorted(output.iterdir())
    } == {
        path.name: path.read_bytes()
        for path in sorted(second.iterdir())
    }


def test_image_requires_immutable_digest() -> None:
    finalizer.validate_image_reference("registry.example/oa@sha256:" + "a" * 64)
    with pytest.raises(ValueError, match="digest"):
        finalizer.validate_image_reference("registry.example/oa:latest")


def test_output_point_identity_contract_requires_161_plus_35() -> None:
    snow_ids = [f"snow_{index:02d}" for index in range(35)]
    points = [
        {"name": f"forcing_{index:03d}", "x": index, "y": 0}
        for index in range(161)
    ] + [{"name": station_id, "x": 0, "y": 0} for station_id in snow_ids]
    setup = {"output_data": {"timeseries": {"points": points}}}
    assert finalizer._validate_output_point_identities(setup, snow_ids) == {
        "forcing_output_points": 161,
        "snow_output_points": 35,
        "output_points": 196,
    }


def test_maintained_snow_output_mappings_are_added_and_mismatches_fail() -> None:
    setup: dict[str, object] = {"output_data": {"grids": {"variables": []}}}
    finalizer._ensure_openamundsen_snow_outputs(setup)
    variables = setup["output_data"]["grids"]["variables"]  # type: ignore[index]
    assert {item["name"]: item["var"] for item in variables} == {
        "snowdepth_daily": "snow.depth",
        "swe_daily": "snow.swe",
    }
    broken = {
        "output_data": {
            "grids": {
                "variables": [{"name": "swe_daily", "var": "snow.depth"}],
            }
        }
    }
    with pytest.raises(ValueError, match="swe_daily maps"):
        finalizer._ensure_openamundsen_snow_outputs(broken)


def test_forcing_flatline_inventory_respects_gaps_and_minimum_length() -> None:
    records = [
        {
            "date": f"2022-10-{1 + (index * 3) // 24:02d} {index * 3 % 24:02d}:00:00",
            "temp": "273.15" if index < 8 else str(273.15 + index),
            "precip": "0",
        }
        for index in range(10)
    ]
    rows = finalizer.forcing_flatline_runs(
        records,
        station_file="Eissee.csv",
        timestep=timedelta(hours=3),
        minimum_samples=8,
    )
    assert rows is not None
    temp = next(row for row in rows if row["variable"] == "temp")
    assert temp["station_id"] == "Eissee"
    assert temp["sample_count"] == 8
    assert temp["duration_hours"] == 21.0


def test_fsc_areal_strata_keep_point_and_areal_support_separate() -> None:
    import numpy as np

    rows = finalizer._summarize_fsc_strata(
        np.array([[0.0, 50.0], [100.0, 215.0]]),
        np.array([[1050.0, 1150.0], [1250.0, 1350.0]]),
        np.array([[1.0, 1.0], [2.0, 2.0]]),
        np.ones((2, 2), dtype=bool),
        elevation_band_width_m=100,
        base={"event_date": "2022-10-07"},
        valid_fsc_mask=np.array([[True, True], [True, False]]),
        dem_nodata=-9999.0,
        landcover_nodata=-9999.0,
    )
    elevation = [row for row in rows if row["stratum_type"] == "elevation_band"]
    landcover = [row for row in rows if row["stratum_type"] == "landcover_class"]
    assert [row["stratum"] for row in elevation] == ["1000-1100", "1100-1200", "1200-1300"]
    assert landcover[0]["valid_pixel_count"] == 2
    assert landcover[0]["fsc_mean_percent"] == 25.0


def test_container_rooted_symlinks_become_internal_relative_links(tmp_path: Path) -> None:
    root = tmp_path / "staging"
    target = root / "meteo" / "station.csv"
    target.parent.mkdir(parents=True)
    target.write_text("time,temp\n", encoding="utf-8")
    link = root / "projects" / "project" / "subdomains" / "AT-01" / "meteo" / "station.csv"
    link.parent.mkdir(parents=True)
    link.symlink_to("/setup/meteo/station.csv")
    finalizer._relativize_internal_symlinks(root)
    assert not Path(link.readlink()).is_absolute()
    assert link.resolve(strict=True) == target
    finalizer._validate_internal_symlinks(root)


def test_legacy_station_role_layer_preserves_source_metadata(tmp_path: Path) -> None:
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
        roles=(
            {
                "station_id": "a",
                "role": "holdout",
                "use_for_da": False,
                "use_for_benchmark": True,
            },
        ),
        exceptions=(),
    )

    finalizer._write_station_roles(tmp_path, roles)

    finalized = tmp_path / "data_finalized" / "obs" / "stations"
    assert source_metadata.read_text(encoding="utf-8") == original
    assert (finalized / "a.csv").is_symlink()
    finalized_role = finalizer.read_csv_records(finalized / "stations_da_metadata.csv")[0]
    assert finalized_role == {
        "alt": "1500.0",
        "id": "a",
        "name": "Station A",
        "station_id": "a",
        "status": "final_holdout",
        "use_for_benchmark": "True",
        "use_for_da": "False",
        "x": "100.0",
        "y": "200.0",
    }


def test_failed_staging_keeps_accepted_tree_and_marks_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    sentinel = root / "accepted.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    fake_roles = StationRoleResult((), ())
    monkeypatch.setattr(finalizer, "detect_setup_layout", lambda _root: finalizer.LEGACY_LAYOUT)
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


def test_leaf_yaml_receives_only_externally_selected_supported_events(tmp_path: Path) -> None:
    project_name = "project_2017_2018"
    leaf_ids = tuple(f"AT-{index:02d}" for index in range(8))
    for leaf_id in leaf_ids:
        project_dir = (
            tmp_path
            / "projects"
            / project_name
            / "subdomains"
            / leaf_id
            / "projects"
            / project_name
        )
        project_dir.mkdir(parents=True)
        finalizer._write_yaml(
            project_dir / f"{project_name}.yml",
            {
                "data_assimilation": {
                    "subdomain_event_filter": {"enabled": True},
                    "assimilation_events": [],
                }
            },
        )
    events = (
        {
            "selected_date": "2017-10-07",
            "variable": "scf",
            "supported_subdomains": list(leaf_ids),
        },
        {
            "selected_date": "2017-10-13",
            "variable": "station_hs",
            "supported_subdomains": [leaf_ids[0]],
        },
    )
    schedule = ScheduleResult((), (), events, (), (), {})

    finalizer._write_leaf_event_schedules(tmp_path, project_name, schedule)

    first = finalizer._read_yaml(
        tmp_path
        / "projects"
        / project_name
        / "subdomains"
        / leaf_ids[0]
        / "projects"
        / project_name
        / f"{project_name}.yml"
    )["data_assimilation"]
    second = finalizer._read_yaml(
        tmp_path
        / "projects"
        / project_name
        / "subdomains"
        / leaf_ids[1]
        / "projects"
        / project_name
        / f"{project_name}.yml"
    )["data_assimilation"]
    assert "subdomain_event_filter" not in first
    assert [event["date"] for event in first["assimilation_events"]] == ["2017-10-07", "2017-10-13"]
    assert [event["date"] for event in second["assimilation_events"]] == ["2017-10-07"]
