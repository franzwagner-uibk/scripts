"""Focused contracts for the North Tyrol v2 scheduler and finalizer."""

from __future__ import annotations

import subprocess
import sys
import logging
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from time import perf_counter

import pytest


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import da_event_scheduler as scheduler  # noqa: E402
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
    log_selected_station_interpolations,
    match_station_support,
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
        station_matching="unique_nearest_within_half_timestep",
        symmetric_tie_max_span_hours=None,
        fulfillment_denominator="feasible_slots",
    )
    return replace(base, **changes)


def _policy_v3(**changes: object) -> SchedulePolicy:
    return replace(
        _policy(),
        schema_version=3,
        station_matching="unique_nearest_or_symmetric_mean_within_half_timestep",
        symmetric_tie_max_span_hours=24,
        **changes,
    )


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
            "water_mask_stable": True,
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


def _large_search_inputs(
    *,
    center_supports_all: bool,
) -> tuple[
    SchedulePolicy,
    date,
    date,
    list[dict[str, object]],
    list[dict[str, object]],
    tuple[dict[str, object], ...],
]:
    policy = _policy()
    start = date(2022, 10, 1)
    end = date(2023, 9, 30)
    slots = generate_slots(start, end, policy)
    domains = tuple(f"D{index}" for index in range(8))
    fsc_rows: list[dict[str, object]] = []
    snow_rows: list[dict[str, object]] = []
    roles: dict[str, dict[str, object]] = {}
    for slot in slots:
        for offset, label in ((-1, "left"), (0, "center"), (1, "right")):
            candidate_date = slot.target_date + timedelta(days=offset)
            excluded = (
                None
                if center_supports_all and offset == 0
                else domains[(slot.index + offset + 1) % len(domains)]
            )
            if slot.variable == "scf":
                for domain in domains:
                    supported = domain != excluded
                    if supported and offset:
                        valid, cloud, invalid = 85, 10, 5
                    elif supported:
                        valid, cloud, invalid = 70, 15, 15
                    else:
                        valid, cloud, invalid = 60, 10, 30
                    fsc_rows.append(
                        {
                            "date": candidate_date.isoformat(),
                            "subdomain_id": domain,
                            "source_file": f"scene_{candidate_date}.nc",
                            "pixel_count": 100,
                            "valid_count": valid,
                            "cloud_count": cloud,
                            "nodata_count": invalid,
                            "water_count": 0,
                            "water_mask_stable": True,
                            "uncertainty_valid_fsc_count": valid,
                            "uncertainty_mean": 10.0,
                            "uncertainty_p90": 12.0,
                        }
                    )
                continue
            for domain in domains:
                if domain == excluded:
                    continue
                for copy in range(2 if offset else 1):
                    station_id = f"{label}_{domain}_{copy}"
                    roles[station_id] = {
                        "station_id": station_id,
                        "subdomain_id": domain,
                        "role": "da",
                    }
                    snow_rows.append(
                        _snow(
                            f"{candidate_date} 00:00:00",
                            station_id,
                            domain,
                        )
                    )
    return policy, start, end, fsc_rows, snow_rows, tuple(roles.values())


def test_versioned_policy_loads_the_fixed_v2_contract() -> None:
    policy = load_policy(MODULE_ROOT / "policies" / "north_tyrol_alternating_6day_v2.yml")
    assert policy.schema_version == 2
    assert policy.maximum_cloud_fraction == 0.20
    assert policy.maximum_invalid_fraction == 0.20
    assert policy.fulfillment_denominator == "feasible_slots"


def test_versioned_policy_v3_adds_fixed_symmetric_tie_contract() -> None:
    policy = load_policy(MODULE_ROOT / "policies" / "north_tyrol_alternating_6day_v3.yml")
    assert policy.schema_version == 3
    assert policy.station_matching == "unique_nearest_or_symmetric_mean_within_half_timestep"
    assert policy.symmetric_tie_max_span_hours == 24
    assert policy.minimum_fulfillment == 0.80


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

    legacy = dict(_fsc("2022-10-07")[0])
    legacy.pop("uncertainty_valid_fsc_count")
    with pytest.raises(ValueError, match="uncertainty_valid_fsc_count"):
        fsc_reference_metrics(legacy)


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


def test_fsc_source_uniqueness_backtracks_to_the_available_path() -> None:
    rows = [
        *_fsc("2022-10-07", cloud=5, invalid=5),
        *_fsc("2022-10-08", cloud=10, invalid=10),
        *_fsc("2022-10-13", cloud=5, invalid=5),
        *_fsc("2022-10-19", cloud=5, invalid=5),
    ]
    for row in rows:
        day = str(row["date"])
        row["source_file"] = (
            "shared_scene.nc"
            if day in {"2022-10-07", "2022-10-13"}
            else f"unique_{day}.nc"
        )

    result = schedule_events(
        policy=replace(
            _policy(),
            sequence=("scf",),
            interval_end=(10, 19),
        ),
        fsc_rows=rows,
        snow_rows=[],
        station_roles=(),
        start=date(2022, 10, 1),
        end=date(2022, 10, 31),
    )

    assert [event["selected_date"] for event in result.events] == [
        "2022-10-08",
        "2022-10-13",
        "2022-10-19",
    ]
    sources = [str(event["source_file"]) for event in result.events]
    assert len(sources) == len(set(sources)) == 3


def test_leaf_type_fulfillment_fails_when_no_common_schedule_can_reach_85_percent() -> None:
    policy = replace(_policy(), interval_end=(10, 7))
    first = _fsc("2022-10-06")
    second = _fsc("2022-10-08")
    first[1].update({"valid_count": 69, "cloud_count": 10, "nodata_count": 21, "uncertainty_valid_fsc_count": 69})
    second[0].update({"valid_count": 69, "cloud_count": 10, "nodata_count": 21, "uncertainty_valid_fsc_count": 69})
    with pytest.raises(ValueError, match=r"feasible-slot fulfillment.*date and gap constraints"):
        schedule_events(
            policy=policy,
            fsc_rows=[*first, *second],
            snow_rows=[],
            station_roles=(),
            start=date(2022, 10, 1),
            end=date(2022, 10, 31),
        )


def test_scheduler_preserves_leaf_fulfillment_counterexample() -> None:
    def scene(day: str, supported: set[str], *, stronger_a: bool = False) -> list[dict[str, object]]:
        rows = []
        for domain in ("A", "B", "C"):
            passes = domain in supported
            valid = 90 if passes and stronger_a and domain == "A" else (80 if passes else 69)
            invalid = 5 if valid == 90 else (10 if passes else 21)
            rows.append(
                {
                    "date": day,
                    "subdomain_id": domain,
                    "source_file": f"scene_{day}.nc",
                    "pixel_count": 100,
                    "valid_count": valid,
                    "cloud_count": 100 - valid - invalid,
                    "nodata_count": invalid,
                    "water_count": 0,
                    "water_mask_stable": True,
                    "uncertainty_valid_fsc_count": valid,
                    "uncertainty_mean": 10.0,
                    "uncertainty_p90": 12.0,
                }
            )
        return rows

    # The higher-quality Nov 5 choice yields A=7/7, B=4/5, C=3/3 and is
    # infeasible. The Nov 7 choice yields A=6/7, B=5/5, C=3/3 and satisfies
    # every 85% leaf/type constraint. Paths converge again on Nov 12.
    rows = [
        *scene("2022-10-07", {"A", "B", "C"}),
        *scene("2022-10-13", {"A", "B", "C"}),
        *scene("2022-10-19", {"A", "B", "C"}),
        *scene("2022-10-25", {"A", "B"}),
        *scene("2022-10-31", {"A"}),
        *scene("2022-11-05", {"A"}, stronger_a=True),
        *scene("2022-11-07", {"B"}),
        *scene("2022-11-12", {"A"}),
    ]
    result = schedule_events(
        policy=replace(
            _policy(),
            sequence=("scf",),
            interval_end=(11, 12),
        ),
        fsc_rows=rows,
        snow_rows=[],
        station_roles=(),
        start=date(2022, 10, 1),
        end=date(2022, 11, 30),
    )

    assert "2022-11-07" in {event["selected_date"] for event in result.events}
    assert "2022-11-05" not in {event["selected_date"] for event in result.events}
    assert {
        domain: result.summary["by_subdomain"][domain]["scf"]["retained"]
        for domain in ("A", "B", "C")
    } == {"A": 6, "B": 5, "C": 3}


def test_alternating_50_slot_eight_leaf_search_remains_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, start, end, fsc_rows, snow_rows, roles = _large_search_inputs(
        center_supports_all=True,
    )
    slots = generate_slots(start, end, policy)

    diagnostics: dict[str, int] = {}
    original = scheduler._select_schedule_path

    def select_with_diagnostics(**kwargs: object) -> tuple[tuple[object, object], ...]:
        kwargs["diagnostics"] = diagnostics
        return original(**kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(scheduler, "_select_schedule_path", select_with_diagnostics)
    result = schedule_events(
        policy=policy,
        fsc_rows=fsc_rows,
        snow_rows=snow_rows,
        station_roles=roles,
        start=start,
        end=end,
    )

    assert len(slots) == len(result.events) == 50
    assert diagnostics["constraint_states"] <= 100
    assert diagnostics["temporal_states"] <= 5_000


def test_infeasible_50_slot_eight_leaf_search_rejects_within_two_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, start, end, fsc_rows, snow_rows, roles = _large_search_inputs(
        center_supports_all=False,
    )
    diagnostics: dict[str, int] = {}
    original = scheduler._select_schedule_path

    def select_with_diagnostics(**kwargs: object) -> tuple[tuple[object, object], ...]:
        kwargs["diagnostics"] = diagnostics
        return original(**kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(scheduler, "_select_schedule_path", select_with_diagnostics)

    started = perf_counter()
    with pytest.raises(ValueError, match="feasible-slot fulfillment"):
        schedule_events(
            policy=policy,
            fsc_rows=fsc_rows,
            snow_rows=snow_rows,
            station_roles=roles,
            start=start,
            end=end,
        )
    elapsed = perf_counter() - started

    assert elapsed < 2.0
    assert diagnostics["support_prunes"] >= 1
    assert diagnostics["constraint_states"] <= 10


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


def test_policy_v3_interpolates_one_symmetric_tie_and_keeps_source_offset() -> None:
    rows = [
        {**_snow("2022-10-12 23:00:00", "a", "A"), "observation_value": 0.4},
        {**_snow("2022-10-13 01:00:00", "a", "A"), "observation_value": 0.8},
    ]
    matches = match_station_support(rows, _policy_v3())
    match = matches[datetime(2022, 10, 13)]["a"]

    assert match.observation_timestamp == datetime(2022, 10, 13)
    assert match.observation_value == pytest.approx(0.6)
    assert match.delta_minutes == 60.0
    assert match.interpolated is True
    assert match.source_timestamps == (
        datetime(2022, 10, 12, 23),
        datetime(2022, 10, 13, 1),
    )

    result = schedule_events(
        policy=replace(
            _policy_v3(),
            interval_start=(10, 13),
            interval_end=(10, 13),
            sequence=("station_hs", "scf"),
        ),
        fsc_rows=[],
        snow_rows=rows,
        station_roles=_roles(),
        start=date(2022, 10, 1),
        end=date(2022, 10, 31),
    )
    assert result.events[0]["station_match_max_delta_minutes"] == 60.0


def test_policy_v3_accepts_the_inclusive_24_hour_pair_span() -> None:
    policy = _policy_v3(
        station_observation_time=time(0),
        model_timestep_hours=24,
    )
    matches = match_station_support(
        [
            {**_snow("2022-10-12 12:00:00", "a", "A"), "observation_value": 0.2},
            {**_snow("2022-10-13 12:00:00", "a", "A"), "observation_value": 1.0},
        ],
        policy,
    )
    match = matches[datetime(2022, 10, 13)]["a"]
    assert match.observation_value == pytest.approx(0.6)
    assert match.delta_minutes == 12 * 60


def test_policy_v3_rejects_duplicate_timestamps() -> None:
    rows = [
        {**_snow("2022-10-12 23:00:00", "a", "A"), "observation_value": 0.2},
        {**_snow("2022-10-12 23:00:00", "a", "A"), "observation_value": 1.0},
    ]
    with pytest.raises(ValueError, match="Duplicate station observation timestamp"):
        match_station_support(rows, _policy_v3())


def test_policy_v3_does_not_match_a_pair_wider_than_24_hours() -> None:
    rows = [
        {**_snow("2022-10-12 11:00:00", "a", "A"), "observation_value": 0.2},
        {**_snow("2022-10-13 13:00:00", "a", "A"), "observation_value": 1.0},
    ]
    matches = match_station_support(rows, _policy_v3(model_timestep_hours=24))
    assert datetime(2022, 10, 13) not in matches


def test_policy_v3_prefers_exact_support_over_an_interpolated_candidate() -> None:
    policy = replace(
        _policy_v3(),
        interval_start=(10, 13),
        interval_end=(10, 13),
        sequence=("station_hs", "scf"),
    )
    rows = [
        {**_snow("2022-10-12 23:00:00", "a", "A"), "observation_value": 0.4},
        {**_snow("2022-10-13 01:00:00", "a", "A"), "observation_value": 0.8},
        {**_snow("2022-10-14 00:00:00", "a", "A"), "observation_value": 0.9},
    ]
    result = schedule_events(
        policy=policy,
        fsc_rows=[],
        snow_rows=rows,
        station_roles=_roles(),
        start=date(2022, 10, 1),
        end=date(2022, 10, 31),
    )
    assert result.events[0]["selected_date"] == "2022-10-14"
    assert result.events[0]["station_match_max_delta_minutes"] == 0.0


def test_policy_v3_interpolation_counts_as_role_support() -> None:
    rows = [
        {**_snow("2022-10-12 23:00:00", "a", "A"), "observation_value": 0.4},
        {**_snow("2022-10-13 01:00:00", "a", "A"), "observation_value": 0.8},
    ]
    roles = assign_station_roles(
        rows,
        [{"station_id": "a", "subdomain_id": "A", "alt": 1500}],
        _policy_v3(),
        support_times=(datetime(2022, 10, 13),),
    )
    assert roles.roles[0]["role"] == "da"
    assert roles.roles[0]["valid_timestep_count"] == 1


def test_selected_interpolation_logging_excludes_unselected_candidates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    policy = replace(
        _policy_v3(),
        interval_start=(10, 13),
        interval_end=(10, 13),
        sequence=("station_hs", "scf"),
    )
    rows = [
        {**_snow("2022-10-12 23:00:00", "a", "A"), "observation_value": 0.4},
        {**_snow("2022-10-13 01:00:00", "a", "A"), "observation_value": 0.8},
        {**_snow("2022-10-13 23:00:00", "a", "A"), "observation_value": 1.0},
        {**_snow("2022-10-14 01:00:00", "a", "A"), "observation_value": 1.2},
    ]
    result = schedule_events(
        policy=policy,
        fsc_rows=[],
        snow_rows=rows,
        station_roles=_roles(),
        start=date(2022, 10, 1),
        end=date(2022, 10, 31),
    )
    caplog.set_level(logging.INFO, logger="da_event_scheduler")

    count = log_selected_station_interpolations({"schedule": result}, rows, policy)

    assert count == 1
    assert "target=2022-10-13 00:00:00" in caplog.text
    assert "source_times=(2022-10-12 23:00:00, 2022-10-13 01:00:00)" in caplog.text
    assert "source_values=(0.4, 0.8) mean=0.6" in caplog.text
    assert "2022-10-14 00:00:00" not in caplog.text


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


def _write_test_subdomains(
    root: Path,
    identifiers: list[str],
    geometries: list[object],
    *,
    crs: str = "EPSG:25832",
) -> None:
    import geopandas as gpd

    env = root / "env"
    env.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"id": identifiers},
        geometry=geometries,
        crs=crs,
    ).to_file(env / "subdomains.gpkg", driver="GPKG")


def test_canonical_station_membership_is_derived_without_mutating_metadata(
    tmp_path: Path,
) -> None:
    from shapely.geometry import box

    _write_test_subdomains(
        tmp_path,
        ["B", "A"],
        [box(20, 0, 30, 10), box(0, 0, 10, 10)],
    )
    metadata_path = tmp_path / "obs" / "stations" / "stations_da_metadata.csv"
    metadata_path.parent.mkdir(parents=True)
    original = (
        "station_id,x,y,use_for_da,use_for_benchmark\n"
        "inside_b,25,5,True,False\n"
        "boundary_a,0,5,False,True\n"
    )
    metadata_path.write_text(original, encoding="utf-8")

    rows = finalizer._canonical_station_rows_with_subdomains(
        tmp_path,
        finalizer.read_csv_records(metadata_path),
    )

    assert [(row["station_id"], row["subdomain_id"]) for row in rows] == [
        ("inside_b", "B"),
        ("boundary_a", "A"),
    ]
    assert metadata_path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("identifiers", "geometries", "station", "message"),
    [
        (["A"], [(0, 0, 10, 10)], {"station_id": "outside", "x": 20, "y": 20}, "matches=\\[\\]"),
        (
            ["A", "B"],
            [(0, 0, 10, 10), (5, 0, 15, 10)],
            {"station_id": "overlap", "x": 7, "y": 5},
            "matches=\\['A', 'B'\\]",
        ),
    ],
)
def test_canonical_station_membership_rejects_non_unique_coverage(
    tmp_path: Path,
    identifiers: list[str],
    geometries: list[tuple[int, int, int, int]],
    station: dict[str, object],
    message: str,
) -> None:
    from shapely.geometry import box

    _write_test_subdomains(
        tmp_path,
        identifiers,
        [box(*bounds) for bounds in geometries],
    )

    with pytest.raises(ValueError, match=message):
        finalizer._canonical_station_rows_with_subdomains(tmp_path, [station])


def test_canonical_station_membership_rejects_invalid_spatial_contracts(
    tmp_path: Path,
) -> None:
    from shapely.geometry import Polygon, box

    wrong_crs = tmp_path / "wrong_crs"
    _write_test_subdomains(wrong_crs, ["A"], [box(0, 0, 1, 1)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="EPSG:25832"):
        finalizer._canonical_station_rows_with_subdomains(
            wrong_crs,
            [{"station_id": "a", "x": 0.5, "y": 0.5}],
        )

    duplicate = tmp_path / "duplicate"
    _write_test_subdomains(
        duplicate,
        ["A", "A"],
        [box(0, 0, 1, 1), box(2, 0, 3, 1)],
    )
    with pytest.raises(ValueError, match="IDs must be unique"):
        finalizer._canonical_station_rows_with_subdomains(
            duplicate,
            [{"station_id": "a", "x": 0.5, "y": 0.5}],
        )

    invalid_geometry = tmp_path / "invalid_geometry"
    _write_test_subdomains(
        invalid_geometry,
        ["A"],
        [Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])],
    )
    with pytest.raises(ValueError, match="geometries must be non-empty and valid"):
        finalizer._canonical_station_rows_with_subdomains(
            invalid_geometry,
            [{"station_id": "a", "x": 0.5, "y": 0.5}],
        )

    valid = tmp_path / "valid"
    _write_test_subdomains(valid, ["A"], [box(0, 0, 1, 1)])
    with pytest.raises(ValueError, match="invalid EPSG:25832 coordinates"):
        finalizer._canonical_station_rows_with_subdomains(
            valid,
            [{"station_id": "a", "x": "nan", "y": 0.5}],
        )


def test_canonical_schedule_normalizes_station_membership_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shapely.geometry import box

    project_name = "project_2017_2018"
    project = tmp_path / "projects" / project_name
    project.mkdir(parents=True)
    (tmp_path / "north.yml").write_text(
        "start_date: '2017-10-01 00:00:00'\n",
        encoding="utf-8",
    )
    (project / f"{project_name}.yml").write_text(
        "start_date: '2017-10-01'\nend_date: '2018-09-30'\n",
        encoding="utf-8",
    )
    _write_test_subdomains(tmp_path, ["A"], [box(0, 0, 10, 10)])
    station_dir = tmp_path / "obs" / "stations"
    station_dir.mkdir(parents=True)
    metadata_path = station_dir / "stations_da_metadata.csv"
    original = "station_id,x,y,alt\n001,5,5,1500\n"
    metadata_path.write_text(original, encoding="utf-8")
    (station_dir / "001.csv").write_text(
        "time,snow_depth\n2017-10-13 00:00:00,0.5\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def capture_schedule(**kwargs: object) -> tuple[StationRoleResult, dict[str, ScheduleResult]]:
        captured.update(kwargs)
        return StationRoleResult((), ()), {}

    monkeypatch.setattr(finalizer, "EXPECTED_PROJECTS", (project_name,))
    monkeypatch.setattr(finalizer, "_canonical_fsc_inventory", lambda _root: [])
    monkeypatch.setattr(finalizer, "schedule_with_adaptive_roles", capture_schedule)

    finalizer._build_canonical_schedules(
        tmp_path,
        MODULE_ROOT / "policies" / "north_tyrol_alternating_6day_v2.yml",
    )

    station_rows = captured["station_rows"]
    snow_rows = captured["snow_rows"]
    assert isinstance(station_rows, tuple)
    assert station_rows[0]["subdomain_id"] == "A"
    assert isinstance(snow_rows, list)
    assert snow_rows[0]["subdomain_id"] == "A"
    assert metadata_path.read_text(encoding="utf-8") == original


def test_legacy_fsc_inventory_requires_stable_reference_schema(tmp_path: Path) -> None:
    inventory = tmp_path / "inventories"
    inventory.mkdir()
    (inventory / "fsc_scene_subdomain_quality.csv").write_text(
        "date,subdomain_id,source_file,cloud_fraction,uncertainty_count\n"
        "2017-10-07,A,scene.nc,0.1,10\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rebuild it from retained NetCDF"):
        finalizer._read_legacy_fsc_inventory(tmp_path)


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

    wrong_frequency = {
        "output_data": {
            "grids": {
                "variables": [{"name": "swe_daily", "var": "snow.swe", "freq": "3H"}],
            }
        }
    }
    with pytest.raises(ValueError, match="freq: D"):
        finalizer._ensure_openamundsen_snow_outputs(wrong_frequency)

    with pytest.raises(ValueError, match="non-empty metrics"):
        finalizer._ensure_compact_snow_outputs(
            {
                "output": {
                    "grids": {
                        "variables": [
                            {"name": "snowdepth_daily", "var": "snowdepth_daily", "metrics": []},
                        ]
                    }
                }
            },
            "project",
        )


def test_forcing_flatline_inventory_uses_hourly_source_cadence_and_ignores_hs() -> None:
    start = datetime(2022, 10, 1)
    records = []
    for index in range(50):
        timestamp = start + timedelta(hours=index + (1 if index >= 25 else 0))
        records.append(
            {
                "date": timestamp.isoformat(sep=" "),
                "temp": "273.15",
                "precip": "0",
                "hs": "999",
            }
        )
    rows = finalizer.forcing_flatline_runs(
        records,
        station_file="Eissee.csv",
    )
    assert [(row["variable"], row["duration_hours"]) for row in rows] == [
        ("precip", 24.0),
        ("precip", 24.0),
        ("temp", 24.0),
        ("temp", 24.0),
    ]
    assert {row["classification"] for row in rows if row["variable"] == "precip"} == {
        "dry_zero_precip"
    }
    assert all(row["source_timestep_hours"] == 1.0 for row in rows)
    assert not any(row["variable"] == "hs" for row in rows)


def test_forcing_flatline_inventory_classifies_severe_and_rejects_nonhourly_cadence() -> None:
    start = datetime(2022, 10, 1)
    records = [
        {
            "date": (start + timedelta(hours=index)).isoformat(sep=" "),
            "temp": "271.25",
            "rel_hum": "84.8",
        }
        for index in range(169)
    ]
    rows = finalizer.forcing_flatline_runs(records, station_file="Eissee.csv")
    assert {row["severity"] for row in rows} == {"severe"}
    assert {row["duration_hours"] for row in rows} == {168.0}

    with pytest.raises(ValueError, match="source cadence must be"):
        finalizer.forcing_flatline_runs(
            [
                {"date": (start + timedelta(hours=3 * index)).isoformat(sep=" "), "temp": "1"}
                for index in range(10)
            ],
            station_file="three_hourly.csv",
        )


def test_forcing_flatline_inventory_clips_projects_and_summarizes_overlaps() -> None:
    source_rows = [
        {
            "station_file": "AT_LWD.Eissee.csv",
            "station_id": "AT_LWD.Eissee",
            "variable": variable,
            "value": value,
            "start_timestamp": "2018-04-05 15:00:00",
            "end_timestamp": "2018-10-31 10:00:00",
            "sample_count": 5012,
            "duration_hours": 5011.0,
            "source_timestep_hours": 1.0,
            "zero_value": False,
            "classification": "candidate_stuck_sensor",
            "severity": "severe",
        }
        for variable, value in (
            ("temp", 271.25),
            ("rel_hum", 84.8),
            ("wind_speed", 3.74),
            ("wind_dir", 188.0),
        )
    ]
    windows = {
        "project_2017_2018": (datetime(2017, 10, 1), datetime(2018, 9, 30, 21)),
        "project_2018_2019": (datetime(2018, 10, 1), datetime(2019, 9, 30, 21)),
    }
    rows = finalizer.clip_forcing_flatline_runs(
        source_rows,
        project_windows=windows,
        minimum_duration=timedelta(hours=24),
    )
    first_project = [row for row in rows if row["project"] == "project_2017_2018"]
    assert len(first_project) == 4
    assert {row["end_timestamp"] for row in first_project} == {"2018-09-30 21:00:00"}
    assert {row["sample_count"] for row in first_project} == {4279}
    overlaps = finalizer.forcing_multivariable_overlaps(rows)
    assert overlaps[0]["variables"] == ("rel_hum", "temp", "wind_dir", "wind_speed")
    assert overlaps[0]["severity"] == "severe"
    finalizer._validate_eissee_2017_2018_flatline(rows)

    summary = finalizer.forcing_flatline_summary(
        rows,
        overlaps=overlaps,
        cadence_rows=[{"gap_count": 2}],
        project_windows=windows,
    )
    assert summary["expected_source_cadence_hours"] == 1.0
    assert summary["projects"]["project_2017_2018"]["station_count"] == 1
    assert summary["projects"]["project_2017_2018"]["multivariable_overlap_count"] == 1


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


def test_canonical_audits_use_the_approved_250_m_elevation_bands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(finalizer, "EXPECTED_PROJECTS", ())
    monkeypatch.setattr(finalizer, "_canonical_subdomain_ids", lambda _root: ())
    monkeypatch.setattr(finalizer, "_write_forcing_flatline_inventory", lambda _root: None)
    monkeypatch.setattr(
        finalizer,
        "_write_station_fsc_audit",
        lambda _root, _schedules, _policy: None,
    )

    def capture_areal(
        root: Path,
        schedules: dict[str, ScheduleResult],
        *,
        elevation_band_width_m: int,
    ) -> None:
        captured.update(
            root=root,
            schedules=schedules,
            elevation_band_width_m=elevation_band_width_m,
        )

    monkeypatch.setattr(finalizer, "_write_fsc_areal_strata_audit", capture_areal)
    finalizer._write_canonical_audits(
        tmp_path,
        {},
        StationRoleResult((), ()),
        MODULE_ROOT / "policies" / "north_tyrol_alternating_6day_v3.yml",
        "registry.example/openamundsen-da@sha256:" + "a" * 64,
    )

    assert captured == {
        "root": tmp_path,
        "schedules": {},
        "elevation_band_width_m": 250,
    }


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


def test_canonical_runtime_replacement_requires_ack_and_never_ignores_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "projects" / "project_2017_2018" / "results"
    results.mkdir(parents=True)
    monkeypatch.setattr(finalizer, "_active_model_references", lambda _root: [])

    with pytest.raises(RuntimeError, match="--discard-runtime-artifacts"):
        finalizer._assert_canonical_runtime_safe(
            tmp_path,
            discard_runtime_artifacts=False,
        )
    finalizer._assert_canonical_runtime_safe(
        tmp_path,
        discard_runtime_artifacts=True,
    )

    lock = tmp_path / "project.lock"
    lock.write_text("active\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="lock markers"):
        finalizer._assert_canonical_runtime_safe(
            tmp_path,
            discard_runtime_artifacts=True,
        )
    lock.unlink()
    monkeypatch.setattr(
        finalizer,
        "_active_model_references",
        lambda _root: ["host process 123 openamundsen-da run /setup"],
    )
    with pytest.raises(RuntimeError, match="active model work"):
        finalizer._assert_canonical_runtime_safe(
            tmp_path,
            discard_runtime_artifacts=True,
        )


def test_active_model_discovery_resolves_process_cwd_for_relative_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_root = tmp_path / "setup"
    setup_root.mkdir()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            "openamundsen-da",
            "run",
        ],
        cwd=setup_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def run_without_docker(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if command[:2] == ["ps", "-eo"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"{process.pid} python relative-command openamundsen-da run\n",
                stderr="",
            )
        if command[:3] == ["docker", "ps", "-q"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(finalizer.subprocess, "run", run_without_docker)
    try:
        references = finalizer._active_model_references(setup_root)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    assert any(f"host process {process.pid}" in reference for reference in references)
    assert any(f"cwd={setup_root.resolve()}" in reference for reference in references)


def test_failed_canonical_refresh_keeps_source_and_marks_staging_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "north_tyrol_subdomains_100m"
    root.mkdir()
    (root / "setup.yml").write_text("domain: north_tyrol\n", encoding="utf-8")
    policy_path = tmp_path / "policy.yml"
    policy_path.write_text("schema_version: 2\n", encoding="utf-8")
    sentinel = root / "accepted.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    roles = StationRoleResult((), ())
    monkeypatch.setattr(finalizer, "_assert_canonical_runtime_safe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(finalizer, "validate_canonical_setup", lambda *_args: {"root": str(root)})
    monkeypatch.setattr(finalizer, "build_schedules", lambda *_args: (roles, {}))
    monkeypatch.setattr(finalizer, "_log_selected_interpolations_for_root", lambda *_args: 0)
    monkeypatch.setattr(finalizer, "_finalizer_commit", lambda: "a" * 40)
    for function_name in (
        "_write_canonical_station_roles",
        "_refresh_canonical_configs",
        "_write_canonical_audits",
    ):
        monkeypatch.setattr(finalizer, function_name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        finalizer,
        "_prepare_all_projects",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic canonical failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic canonical failure"):
        finalizer._refresh_canonical_setup(
            root,
            policy_path,
            "registry.example/openamundsen-da@sha256:" + "a" * 64,
            discard_runtime_artifacts=True,
        )
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    staging = next(tmp_path.glob(".north_tyrol_subdomains_100m.refreshing-*"))
    assert (staging / "INCOMPLETE.json").is_file()


def test_post_swap_validation_failure_restores_canonical_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "north_tyrol_subdomains_100m"
    root.mkdir()
    (root / "north_tyrol_subdomains_100m.yml").write_text("domain: north_tyrol\n", encoding="utf-8")
    policy_path = tmp_path / "policy.yml"
    policy_path.write_text("schema_version: 2\n", encoding="utf-8")
    sentinel = root / "accepted.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    roles = StationRoleResult((), ())
    schedules: dict[str, ScheduleResult] = {}
    monkeypatch.setattr(finalizer, "_assert_canonical_runtime_safe", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(finalizer, "validate_canonical_setup", lambda *_args: {"root": str(root)})
    monkeypatch.setattr(finalizer, "build_schedules", lambda *_args: (roles, schedules))
    monkeypatch.setattr(finalizer, "_log_selected_interpolations_for_root", lambda *_args: 0)
    monkeypatch.setattr(finalizer, "_finalizer_commit", lambda: "a" * 40)
    for function_name in (
        "_write_canonical_station_roles",
        "_refresh_canonical_configs",
        "_write_canonical_audits",
        "_prepare_all_projects",
        "_validate_all_leaf_core_requirements",
        "_relativize_internal_symlinks",
        "_write_canonical_refresh_manifest",
    ):
        monkeypatch.setattr(finalizer, function_name, lambda *_args, **_kwargs: None)
    validations = 0

    def validate(*_args: object, **_kwargs: object) -> None:
        nonlocal validations
        validations += 1
        if validations == 2:
            raise RuntimeError("synthetic post-swap failure")

    monkeypatch.setattr(finalizer, "validate_canonical_refresh", validate)

    with pytest.raises(RuntimeError, match="synthetic post-swap failure"):
        finalizer._refresh_canonical_setup(
            root,
            policy_path,
            "registry.example/openamundsen-da@sha256:" + "a" * 64,
            discard_runtime_artifacts=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    failed = next(tmp_path.glob(".north_tyrol_subdomains_100m.failed-refresh-*"))
    assert (failed / "INCOMPLETE.json").is_file()


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


def test_canonical_acceptance_invokes_pinned_core_validator_for_all_leaves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(finalizer, "_run", lambda command: commands.append(list(command)))
    image = "registry.example/openamundsen-da@sha256:" + "a" * 64

    finalizer._validate_all_leaf_core_requirements(tmp_path, image)

    assert len(commands) == 1
    command = commands[0]
    assert image in command
    assert "--network" in command and command[command.index("--network") + 1] == "none"
    assert f"{tmp_path}:/setup:ro" in command
    script = command[-1]
    assert "list_steps_sorted" in script
    assert "validate_assimilation_requirements" in script
    assert "CORE_REQUIREMENTS_OK=48" in script


def test_canonical_transaction_manifest_records_reviewed_inputs(tmp_path: Path) -> None:
    policy = tmp_path / "policy.yml"
    policy.write_text("schema_version: 2\n", encoding="utf-8")
    schedule = ScheduleResult(
        slots=(),
        targets=(),
        events=(),
        quality=(),
        exceptions=(),
        summary={"by_variable": {}, "by_subdomain": {}},
    )
    schedules = {name: schedule for name in finalizer.EXPECTED_PROJECTS}
    image = "registry.example/openamundsen-da@sha256:" + "a" * 64

    finalizer._write_canonical_refresh_manifest(
        tmp_path,
        schedules=schedules,
        roles=StationRoleResult((), ()),
        policy_path=policy,
        image=image,
        commit="b" * 40,
        parent_root=tmp_path / "parent",
        parent_manifest_sha256="d" * 64,
        parent_configs={"setup.yml": "c" * 64},
        discarded_runtime_artifacts=["projects/project_2017_2018/results"],
        promotion_result="promoted",
    )

    manifest = finalizer._read_json(
        tmp_path / "raw" / "metadata" / "canonical_refresh_manifest.json"
    )
    assert manifest["scheduler_commit"] == "b" * 40
    assert manifest["image"] == image
    assert manifest["parent_manifest_sha256"] == "d" * 64
    assert manifest["parent_config_sha256"] == {"setup.yml": "c" * 64}
    assert manifest["promotion_result"] == "promoted"
