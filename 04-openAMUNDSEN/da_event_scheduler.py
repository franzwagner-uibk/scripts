"""Deterministic, source-independent data-assimilation event scheduling."""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class SchedulePolicy:
    """Validated fixed-day scheduling and station-role policy."""

    schema_version: int
    target_spacing_days: int
    sequence: tuple[str, ...]
    interval_start: tuple[int, int]
    interval_end: tuple[int, int]
    fsc_search_days: int
    station_hs_search_days: int
    minimum_gap_days: int
    maximum_gap_days: int
    maximum_cloud_fraction: float
    maximum_invalid_fraction: float
    minimum_fulfillment: float
    station_observation_time: time
    model_timestep_hours: int
    fulfillment_denominator: str


@dataclass(frozen=True)
class Slot:
    """One immutable target in the fixed cadence."""

    index: int
    target_date: date
    variable: str


@dataclass(frozen=True)
class Candidate:
    """One normalized observation candidate for a target slot."""

    selected_timestamp: datetime
    variable: str
    source_file: str
    supported_subdomains: tuple[str, ...]
    active_holdouts: int = 0
    active_da_stations: int = 0
    valid_support_count: int = 0
    cloud_fraction_mean: float = 0.0
    invalid_fraction_mean: float = 0.0
    uncertainty_mean: float = 0.0
    uncertainty_p90: float = 0.0
    station_match_max_delta_minutes: float = 0.0
    active_station_ids: tuple[str, ...] = ()

    @property
    def selected_date(self) -> date:
        """Return the calendar date used by the project event contract."""

        return self.selected_timestamp.date()


@dataclass(frozen=True)
class ScheduleResult:
    """Deterministic retained events, targets, quality and exceptions."""

    slots: tuple[Slot, ...]
    targets: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]
    quality: tuple[dict[str, Any], ...]
    exceptions: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class StationRoleResult:
    """Shared station roles plus adaptive-reduction exceptions."""

    roles: tuple[dict[str, Any], ...]
    exceptions: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class StationMatch:
    """One observation matched uniquely to a configured daily DA timestep."""

    station_id: str
    subdomain_id: str
    observation_timestamp: datetime
    delta_minutes: float
    observation_value: float | None


def parse_date(value: object, *, field: str) -> date:
    """Parse one ISO date and name its source field in errors."""

    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc


def load_policy(path: Path) -> SchedulePolicy:
    """Load and strictly validate the versioned YAML policy."""

    from ruamel.yaml import YAML

    payload = YAML(typ="safe").load(Path(path))
    if not isinstance(payload, dict):
        raise ValueError("Policy root must be a mapping")
    cadence = payload.get("cadence") or {}
    fsc = payload.get("fsc") or {}
    station = payload.get("station_hs") or {}
    fulfillment = payload.get("fulfillment") or {}
    observation_time = _time_value(
        station.get("observation_time", "00:00:00"),
        field="station_hs.observation_time",
    )
    policy = SchedulePolicy(
        schema_version=int(payload.get("schema_version", 0)),
        target_spacing_days=int(cadence.get("target_spacing_days", 0)),
        sequence=tuple(str(item) for item in cadence.get("sequence", ())),
        interval_start=_month_day(cadence.get("interval_start"), field="cadence.interval_start"),
        interval_end=_month_day(cadence.get("interval_end"), field="cadence.interval_end"),
        fsc_search_days=int(fsc.get("search_window_days", -1)),
        station_hs_search_days=int(station.get("search_window_days", -1)),
        minimum_gap_days=int(cadence.get("accepted_adjacent_gap_days", {}).get("min", 0)),
        maximum_gap_days=int(cadence.get("accepted_adjacent_gap_days", {}).get("max", 0)),
        maximum_cloud_fraction=float(fsc.get("maximum_cloud_fraction", -1.0)),
        maximum_invalid_fraction=float(fsc.get("maximum_invalid_fraction", -1.0)),
        minimum_fulfillment=float(fulfillment.get("minimum_fraction_per_type", -1.0)),
        station_observation_time=observation_time,
        model_timestep_hours=int(station.get("model_timestep_hours", 0)),
        fulfillment_denominator=str(fulfillment.get("denominator", "")).strip(),
    )
    if policy.schema_version != 2:
        raise ValueError("Only scheduler policy schema_version 2 is supported")
    if policy.target_spacing_days < 1 or policy.sequence != ("scf", "station_hs"):
        raise ValueError("Policy must use positive fixed-day spacing and sequence [scf, station_hs]")
    if policy.minimum_gap_days < 1 or policy.maximum_gap_days < policy.minimum_gap_days:
        raise ValueError("Invalid accepted adjacent-event gap")
    if policy.fsc_search_days < 0 or policy.station_hs_search_days < 0:
        raise ValueError("Search windows must be non-negative")
    if not 0.0 <= policy.maximum_cloud_fraction <= 1.0:
        raise ValueError("maximum_cloud_fraction must be in [0, 1]")
    if not 0.0 <= policy.maximum_invalid_fraction <= 1.0:
        raise ValueError("maximum_invalid_fraction must be in [0, 1]")
    if not 0.0 < policy.minimum_fulfillment <= 1.0:
        raise ValueError("minimum_fulfillment must be in (0, 1]")
    if policy.model_timestep_hours < 1 or 24 % policy.model_timestep_hours:
        raise ValueError("station_hs.model_timestep_hours must divide 24")
    if observation_time.hour % policy.model_timestep_hours or observation_time.minute or observation_time.second:
        raise ValueError("station_hs.observation_time must align with the model timestep")
    if policy.fulfillment_denominator != "feasible_slots":
        raise ValueError("fulfillment.denominator must be feasible_slots")
    fixed_contract = (
        (cadence.get("mode"), "fixed_day", "cadence.mode"),
        (cadence.get("retain_skipped_slots"), True, "cadence.retain_skipped_slots"),
        (fsc.get("reference_footprint"), "valid_cloud_nodata", "fsc.reference_footprint"),
        (fsc.get("exclude_from_reference"), ["water"], "fsc.exclude_from_reference"),
        (fsc.get("admission"), "at_least_one_subdomain", "fsc.admission"),
        (fsc.get("filter_observed_snow_fraction"), False, "fsc.filter_observed_snow_fraction"),
        (fsc.get("holdouts"), False, "fsc.holdouts"),
        (
            station.get("matching"),
            "unique_nearest_within_half_timestep",
            "station_hs.matching",
        ),
        (
            station.get("shared_split_across_projects"),
            True,
            "station_hs.shared_split_across_projects",
        ),
    )
    for actual, expected, field in fixed_contract:
        if actual != expected:
            raise ValueError(f"{field} must be {expected!r}")
    holdouts = station.get("holdouts") or {}
    expected_holdouts = {
        "domains_with_at_least_4_stations": 2,
        "domains_with_3_stations": 1,
        "domains_with_at_most_2_stations": 0,
        "priority": ["remaining_da_event_support", "elevation_spread", "station_id"],
        "preserve_da_support": True,
        "allow_recorded_reduction": True,
    }
    for field, expected in expected_holdouts.items():
        if holdouts.get(field) != expected:
            raise ValueError(f"station_hs.holdouts.{field} must be {expected!r}")
    return policy


def _month_day(value: object, *, field: str) -> tuple[int, int]:
    if not isinstance(value, str) or len(value.split("-")) != 2:
        raise ValueError(f"{field} must be MM-DD")
    month, day = (int(item) for item in value.split("-"))
    date(2000, month, day)
    return month, day


def _time_value(value: object, *, field: str) -> time:
    try:
        return time.fromisoformat(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc


def generate_slots(start: date, end: date, policy: SchedulePolicy) -> tuple[Slot, ...]:
    """Generate alternating fixed-day targets within one project window."""

    crosses_year = policy.interval_end < policy.interval_start
    interval_start_year = start.year - int(
        crosses_year and (start.month, start.day) <= policy.interval_end
    )
    interval_start = date(interval_start_year, *policy.interval_start)
    interval_end_year = interval_start_year + crosses_year
    interval_end = date(interval_end_year, *policy.interval_end)
    last = min(end, interval_end)
    if last < max(start, interval_start):
        return ()
    slots: list[Slot] = []
    target = interval_start
    cadence_index = 0
    while target < start:
        target += timedelta(days=policy.target_spacing_days)
        cadence_index += 1
    while target <= last:
        slots.append(Slot(len(slots), target, policy.sequence[cadence_index % len(policy.sequence)]))
        target += timedelta(days=policy.target_spacing_days)
        cadence_index += 1
    return tuple(slots)


def assign_station_roles(
    snow_rows: Sequence[Mapping[str, Any]],
    station_rows: Sequence[Mapping[str, Any]],
    policy: SchedulePolicy | None = None,
    *,
    support_times: Iterable[datetime] | None = None,
) -> StationRoleResult:
    """Assign deterministic shared DA and elevation-aware holdout roles."""

    metadata_ids = [_station_id(row) for row in station_rows]
    if any(not station_id for station_id in metadata_ids) or len(set(metadata_ids)) != len(metadata_ids):
        raise ValueError("Station metadata requires unique non-empty station identifiers")
    station_metadata = {
        station_id: row
        for station_id, row in zip(metadata_ids, station_rows, strict=True)
    }
    station_domains: dict[str, str] = {}
    active_times: dict[str, set[datetime]] = defaultdict(set)
    for station_id, row in station_metadata.items():
        subdomain_id = str(row.get("subdomain_id", "")).strip()
        if station_id and subdomain_id:
            station_domains[station_id] = subdomain_id
    matched_support = match_station_support(snow_rows, policy) if policy else None
    eligible_times = set(support_times) if support_times is not None else None
    for row in snow_rows:
        station_id = _station_id(row)
        subdomain_id = str(row.get("subdomain_id", "")).strip()
        if not station_id or not subdomain_id:
            raise ValueError("Snow inventory requires station_id and subdomain_id")
        previous = station_domains.setdefault(station_id, subdomain_id)
        if previous != subdomain_id:
            raise ValueError(f"Station {station_id} maps to multiple subdomains")
        if matched_support is None and _row_has_observation(row):
            active_times[station_id].add(_observation_timestamp(row))
    if matched_support is not None:
        for model_time, station_matches in matched_support.items():
            if eligible_times is not None and model_time not in eligible_times:
                continue
            for station_id in station_matches:
                active_times[station_id].add(model_time)
    missing_metadata = sorted(set(station_domains) - set(station_metadata))
    if missing_metadata:
        raise ValueError(f"Station metadata missing IDs: {missing_metadata}")
    missing_domains = sorted(set(station_metadata) - set(station_domains))
    if missing_domains:
        raise ValueError(f"Station metadata IDs lack a subdomain mapping: {missing_domains}")

    by_domain: dict[str, list[str]] = defaultdict(list)
    for station_id, subdomain_id in station_domains.items():
        by_domain[subdomain_id].append(station_id)
    role_by_station: dict[str, str] = {}
    rank_by_station: dict[str, int] = {}
    for subdomain_id in sorted(by_domain):
        station_ids = sorted(by_domain[subdomain_id])
        target = 2 if len(station_ids) >= 4 else 1 if len(station_ids) == 3 else 0
        holdouts = _best_holdout_combination(station_ids, target, station_metadata, active_times)
        ordered_holdouts = sorted(
            holdouts,
            key=lambda station_id: (
                -len(active_times[station_id]),
                -_altitude(station_metadata[station_id]),
                station_id,
            ),
        )
        for station_id in station_ids:
            role_by_station[station_id] = "holdout" if station_id in holdouts else "da"
            rank_by_station[station_id] = ordered_holdouts.index(station_id) + 1 if station_id in holdouts else 0

    roles = tuple(
        {
            "station_id": station_id,
            "subdomain_id": station_domains[station_id],
            "role": role_by_station[station_id],
            "use_for_da": role_by_station[station_id] == "da",
            "use_for_benchmark": role_by_station[station_id] == "holdout",
            "valid_timestep_count": len(active_times[station_id]),
            "elevation_m": _altitude(station_metadata[station_id]),
            "holdout_rank": rank_by_station[station_id],
            "selection_reason": "da_support_then_holdout_coverage_then_elevation_spread",
        }
        for station_id in sorted(station_domains)
    )
    return StationRoleResult(roles=roles, exceptions=())


def _best_holdout_combination(
    station_ids: Sequence[str],
    target: int,
    metadata: Mapping[str, Mapping[str, Any]],
    active_times: Mapping[str, set[datetime]],
) -> set[str]:
    if target == 0:
        return set()
    combinations = itertools.combinations(station_ids, target)
    all_times = set().union(*(active_times[station_id] for station_id in station_ids))

    def score(combo: tuple[str, ...]) -> tuple[Any, ...]:
        da_stations = set(station_ids) - set(combo)
        da_supported_times = sum(
            any(model_time in active_times[station_id] for station_id in da_stations)
            for model_time in all_times
        )
        da_observations = sum(len(active_times[station_id]) for station_id in da_stations)
        elevations = [_altitude(metadata[station_id]) for station_id in combo]
        spread = max(elevations) - min(elevations) if len(elevations) > 1 else 0.0
        holdout_coverage = [len(active_times[station_id]) for station_id in combo]
        return (
            da_supported_times,
            da_observations,
            spread,
            min(holdout_coverage),
            sum(holdout_coverage),
            tuple(reversed(combo)),
        )

    return set(max(combinations, key=score))


def adapt_station_roles_for_support(
    role_result: StationRoleResult,
    snow_rows: Sequence[Mapping[str, Any]],
    station_event_timestamps: Iterable[datetime],
    policy: SchedulePolicy,
) -> StationRoleResult:
    """Reduce holdouts when an active leaf event would otherwise lack DA support."""

    roles = {str(row["station_id"]): dict(row) for row in role_result.roles}
    active_by_domain_time: dict[tuple[str, datetime], set[str]] = defaultdict(set)
    station_domains = {_station_id(row): str(row["subdomain_id"]) for row in snow_rows}
    for model_time, matches in match_station_support(snow_rows, policy).items():
        for station_id in matches:
            active_by_domain_time[(station_domains[station_id], model_time)].add(station_id)
    exceptions = [dict(row) for row in role_result.exceptions]
    domains = sorted({str(row["subdomain_id"]) for row in roles.values()})
    event_times = sorted(set(station_event_timestamps))
    for subdomain_id in domains:
        for event_time in event_times:
            active = active_by_domain_time.get((subdomain_id, event_time), set())
            if not active or any(roles[station_id]["role"] == "da" for station_id in active):
                continue
            candidates = sorted(
                (station_id for station_id in active if roles[station_id]["role"] == "holdout"),
                key=lambda station_id: (
                    -int(roles[station_id]["valid_timestep_count"]),
                    int(roles[station_id]["holdout_rank"]),
                    station_id,
                ),
            )
            if not candidates:
                raise ValueError(
                    f"No DA-capable station for {subdomain_id} at {event_time.isoformat()}"
                )
            station_id = candidates[0]
            roles[station_id]["role"] = "da"
            roles[station_id]["use_for_da"] = True
            roles[station_id]["use_for_benchmark"] = False
            roles[station_id]["selection_reason"] = "holdout_reduced_for_da_support"
            exceptions.append(
                {
                    "exception": "holdout_reduced_for_da_support",
                    "subdomain_id": subdomain_id,
                    "station_id": station_id,
                    "trigger_timestamp": event_time.isoformat(sep=" "),
                }
            )
    return StationRoleResult(
        roles=tuple(roles[station_id] for station_id in sorted(roles)),
        exceptions=tuple(sorted(exceptions, key=lambda row: tuple(str(row.get(key, "")) for key in sorted(row)))),
    )


def schedule_events(
    *,
    policy: SchedulePolicy,
    fsc_rows: Sequence[Mapping[str, Any]],
    snow_rows: Sequence[Mapping[str, Any]],
    station_roles: Sequence[Mapping[str, Any]],
    start: date,
    end: date,
) -> ScheduleResult:
    """Select an optimal deterministic event path from normalized tables."""

    slots = generate_slots(start, end, policy)
    fsc_candidates = _fsc_candidates(fsc_rows, policy)
    station_candidates = _station_candidates(snow_rows, station_roles, policy)
    candidates = {"scf": fsc_candidates, "station_hs": station_candidates}
    options_by_slot = {
        slot.index: tuple(
            candidate
            for candidate in candidates[slot.variable]
            if abs((candidate.selected_date - slot.target_date).days)
            <= (policy.fsc_search_days if slot.variable == "scf" else policy.station_hs_search_days)
        )
        for slot in slots
    }
    domains = sorted(
        {
            subdomain_id
            for options in options_by_slot.values()
            for candidate in options
            for subdomain_id in candidate.supported_subdomains
        }
    )
    feasible_counts: dict[tuple[str, str], int] = {}
    for variable in policy.sequence:
        feasible_counts[("__project__", variable)] = sum(
            slot.variable == variable and bool(options_by_slot[slot.index])
            for slot in slots
        )
    for subdomain_id in domains:
        for variable in policy.sequence:
            feasible_counts[(subdomain_id, variable)] = sum(
                slot.variable == variable
                and any(
                    subdomain_id in candidate.supported_subdomains
                    for candidate in options_by_slot[slot.index]
                )
                for slot in slots
            )
    coverage_keys = tuple(key for key in sorted(feasible_counts) if feasible_counts[key] > 0)
    score_size = len(coverage_keys) + 12
    states: dict[tuple[int | None, int | None], tuple[tuple[int, ...], tuple[tuple[Slot, Candidate], ...]]] = {
        (None, None): ((0,) * score_size, ())
    }
    for slot in slots:
        next_states: dict[
            tuple[int | None, int | None],
            tuple[tuple[int, ...], tuple[tuple[Slot, Candidate], ...]],
        ] = {}
        for (last_ordinal, last_slot), (score, path) in states.items():
            _keep_best(next_states, (last_ordinal, last_slot), score, path, len(coverage_keys))
            for candidate in options_by_slot[slot.index]:
                delta = abs((candidate.selected_date - slot.target_date).days)
                ordinal = candidate.selected_date.toordinal()
                if last_ordinal is not None and ordinal <= last_ordinal:
                    continue
                if last_ordinal is not None:
                    gap = ordinal - last_ordinal
                    if gap < policy.minimum_gap_days:
                        continue
                    if last_slot == slot.index - 1 and gap > policy.maximum_gap_days:
                        continue
                candidate_score = _candidate_score(
                    candidate,
                    delta,
                    coverage_keys,
                    feasible_counts,
                )
                combined = tuple(left + right for left, right in zip(score, candidate_score))
                _keep_best(
                    next_states,
                    (ordinal, slot.index),
                    combined,
                    path + ((slot, candidate),),
                    len(coverage_keys),
                )
        states = next_states
    _, selected_path = max(
        states.values(),
        key=lambda item: (_score_rank(item[0], len(coverage_keys)), _path_tie_key(item[1])),
    )
    selected_by_slot = {slot.index: candidate for slot, candidate in selected_path}
    events: list[dict[str, Any]] = []
    quality = _quality_records(fsc_rows, snow_rows, station_roles, policy, slots)
    exceptions: list[dict[str, Any]] = []
    for slot in slots:
        candidate = selected_by_slot.get(slot.index)
        if candidate is None:
            exceptions.append(
                {
                    "exception": "unavailable_slot" if not options_by_slot[slot.index] else "optimization_conflict",
                    "slot_index": slot.index,
                    "target_date": slot.target_date.isoformat(),
                    "variable": slot.variable,
                    "feasible": bool(options_by_slot[slot.index]),
                }
            )
            continue
        event = {
            "slot_index": slot.index,
            "target_date": slot.target_date.isoformat(),
            "selected_date": candidate.selected_date.isoformat(),
            "selected_timestamp": candidate.selected_timestamp.isoformat(sep=" "),
            "variable": slot.variable,
            "date_delta_days": (candidate.selected_date - slot.target_date).days,
            "source_file": candidate.source_file,
            "supported_subdomains": list(candidate.supported_subdomains),
            "supported_subdomain_count": len(candidate.supported_subdomains),
            "active_holdouts": candidate.active_holdouts,
            "active_da_stations": candidate.active_da_stations,
            "valid_support_count": candidate.valid_support_count,
            "cloud_reference_fraction_mean": candidate.cloud_fraction_mean,
            "invalid_reference_fraction_mean": candidate.invalid_fraction_mean,
            "uncertainty_mean": candidate.uncertainty_mean,
            "uncertainty_p90": candidate.uncertainty_p90,
            "station_match_max_delta_minutes": candidate.station_match_max_delta_minutes,
            "active_station_ids": list(candidate.active_station_ids),
        }
        events.append(event)
    targets = tuple(
        {
            "slot_index": slot.index,
            "target_date": slot.target_date.isoformat(),
            "variable": slot.variable,
            "feasible": bool(options_by_slot[slot.index]),
            "candidate_count": len(options_by_slot[slot.index]),
            "selected": slot.index in selected_by_slot,
            "selected_date": (
                selected_by_slot[slot.index].selected_date.isoformat()
                if slot.index in selected_by_slot
                else ""
            ),
        }
        for slot in slots
    )
    summary = _schedule_summary(slots, events, options_by_slot, policy)
    return ScheduleResult(slots, targets, tuple(events), tuple(quality), tuple(exceptions), summary)


def _keep_best(
    states: dict[tuple[int | None, int | None], tuple[tuple[int, ...], tuple[tuple[Slot, Candidate], ...]]],
    key: tuple[int | None, int | None],
    score: tuple[int, ...],
    path: tuple[tuple[Slot, Candidate], ...],
    coverage_count: int,
) -> None:
    current = states.get(key)
    if current is None or (_score_rank(score, coverage_count), _path_tie_key(path)) > (
        _score_rank(current[0], coverage_count),
        _path_tie_key(current[1]),
    ):
        states[key] = (score, path)


def _score_rank(score: tuple[int, ...], coverage_count: int) -> tuple[int, ...]:
    """Prioritize balanced per-type fulfillment before quality tie breakers."""

    coverage = score[:coverage_count]
    quality = score[coverage_count:]
    return (
        min(coverage, default=0),
        *sorted(coverage),
        min(quality[1], quality[2]),
        quality[0],
        quality[1],
        quality[2],
        *quality[3:],
    )


def _candidate_score(
    candidate: Candidate,
    delta: int,
    coverage_keys: Sequence[tuple[str, str]],
    feasible_counts: Mapping[tuple[str, str], int],
) -> tuple[int, ...]:
    is_fsc = int(candidate.variable == "scf")
    is_station = int(candidate.variable == "station_hs")
    return (
        *(
            1_000_000 // feasible_counts[(subdomain_id, variable)]
            if candidate.variable == variable
            and (subdomain_id == "__project__" or subdomain_id in candidate.supported_subdomains)
            else 0
            for subdomain_id, variable in coverage_keys
        ),
        1,
        is_fsc,
        is_station,
        candidate.valid_support_count,
        candidate.active_da_stations,
        len(candidate.active_station_ids),
        len(candidate.supported_subdomains),
        -round(candidate.uncertainty_p90 * 1_000),
        -round(candidate.uncertainty_mean * 1_000),
        -round(candidate.station_match_max_delta_minutes * 1_000),
        -delta,
        -candidate.selected_date.toordinal(),
    )


def _path_tie_key(path: Sequence[tuple[Slot, Candidate]]) -> tuple[str, ...]:
    return tuple(f"{candidate.selected_timestamp.isoformat()}:{candidate.source_file}" for _, candidate in path)


def _fsc_candidates(rows: Sequence[Mapping[str, Any]], policy: SchedulePolicy) -> tuple[Candidate, ...]:
    grouped: dict[date, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[parse_date(row.get("date"), field="FSC date")].append(row)
    candidates: list[Candidate] = []
    for scene_date in sorted(grouped):
        scene_rows = grouped[scene_date]
        subdomain_ids = [str(row.get("subdomain_id", "")).strip() for row in scene_rows]
        if any(not subdomain_id for subdomain_id in subdomain_ids):
            raise ValueError(f"FSC date {scene_date} contains an empty subdomain_id")
        source_files = {str(row.get("source_file", "")).strip() for row in scene_rows}
        if len(source_files) != 1 or not next(iter(source_files)):
            raise ValueError(f"FSC date {scene_date} must identify exactly one source_file")
        if len(set(subdomain_ids)) != len(scene_rows):
            raise ValueError(f"Duplicate FSC date/subdomain rows for {scene_date}")
        metrics = {str(row["subdomain_id"]): fsc_reference_metrics(row) for row in scene_rows}
        supported = tuple(sorted(
            subdomain_id
            for subdomain_id, values in metrics.items()
            if values["cloud_reference_fraction"] <= policy.maximum_cloud_fraction
            and values["invalid_reference_fraction"] <= policy.maximum_invalid_fraction
            and values["valid_count"] > 0
            and values["uncertainty_complete"]
        ))
        if not supported:
            continue
        supported_rows = [row for row in scene_rows if str(row["subdomain_id"]) in supported]
        candidates.append(
            Candidate(
                selected_timestamp=datetime.combine(scene_date, time()),
                variable="scf",
                source_file=next(iter(source_files)),
                supported_subdomains=supported,
                valid_support_count=sum(
                    int(metrics[str(row["subdomain_id"])]["valid_count"])
                    for row in supported_rows
                ),
                cloud_fraction_mean=sum(
                    metrics[str(row["subdomain_id"])]["cloud_reference_fraction"] for row in supported_rows
                )
                / len(supported_rows),
                invalid_fraction_mean=sum(
                    metrics[str(row["subdomain_id"])]["invalid_reference_fraction"] for row in supported_rows
                )
                / len(supported_rows),
                uncertainty_mean=sum(
                    _number(row.get("uncertainty_mean"), field="uncertainty_mean")
                    * int(metrics[str(row["subdomain_id"])]["valid_count"])
                    for row in supported_rows
                )
                / sum(int(metrics[str(row["subdomain_id"])]["valid_count"]) for row in supported_rows),
                uncertainty_p90=max(
                    _number(row.get("uncertainty_p90"), field="uncertainty_p90")
                    for row in supported_rows
                ),
            )
        )
    return tuple(candidates)


def fsc_reference_metrics(row: Mapping[str, Any]) -> dict[str, float | int | bool]:
    """Return FSC quality fractions over valid+cloud+nodata, excluding water."""

    valid = _integer(row.get("valid_count"), field="valid_count")
    cloud = _integer(row.get("cloud_count"), field="cloud_count")
    nodata = _integer(row.get("nodata_count"), field="nodata_count")
    water = _integer(row.get("water_count", 0), field="water_count")
    if min(valid, cloud, nodata, water) < 0:
        raise ValueError("FSC class counts must be non-negative")
    reference = valid + cloud + nodata
    if reference < 1:
        raise ValueError("FSC reference footprint is empty after excluding water")
    pixel_count = row.get("pixel_count")
    if pixel_count not in (None, "") and _integer(pixel_count, field="pixel_count") != reference + water:
        permanent = _integer(row.get("permanent_nodata_count", 0), field="permanent_nodata_count")
        if _integer(pixel_count, field="pixel_count") != reference + water + permanent:
            raise ValueError("FSC class counts do not sum to pixel_count")
    has_explicit_uncertainty_support = row.get("uncertainty_valid_fsc_count") not in (None, "")
    uncertainty_valid_count = _integer(
        row.get("uncertainty_valid_fsc_count", row.get("uncertainty_count", 0)),
        field="uncertainty_valid_fsc_count",
    )
    return {
        "valid_count": valid,
        "cloud_count": cloud,
        "nodata_count": nodata,
        "water_count": water,
        "reference_count": reference,
        "valid_reference_fraction": valid / reference,
        "cloud_reference_fraction": cloud / reference,
        "invalid_reference_fraction": nodata / reference,
        "uncertainty_valid_fsc_count": uncertainty_valid_count,
        "uncertainty_complete": (
            uncertainty_valid_count == valid
            if has_explicit_uncertainty_support
            else uncertainty_valid_count >= valid
        ),
    }


def _station_candidates(
    snow_rows: Sequence[Mapping[str, Any]],
    station_roles: Sequence[Mapping[str, Any]],
    policy: SchedulePolicy,
) -> tuple[Candidate, ...]:
    role_ids = [str(row["station_id"]).strip() for row in station_roles]
    if any(not station_id for station_id in role_ids) or len(set(role_ids)) != len(role_ids):
        raise ValueError("Station roles require unique non-empty station_id values")
    roles = {
        str(row["station_id"]).strip(): str(row["role"]).strip()
        for row in station_roles
    }
    invalid_roles = sorted(
        station_id for station_id, role in roles.items() if role not in {"da", "holdout"}
    )
    if invalid_roles:
        raise ValueError(f"Stations have invalid roles: {invalid_roles}")
    grouped = match_station_support(snow_rows, policy)
    support_ids = {station_id for matches in grouped.values() for station_id in matches}
    missing_roles = sorted(support_ids - set(roles))
    if missing_roles:
        raise ValueError(f"Station observations lack assigned roles: {missing_roles}")
    candidates: list[Candidate] = []
    for observation_time in sorted(grouped):
        matches = grouped[observation_time]
        station_ids = sorted(matches)
        active_da = [station_id for station_id in station_ids if roles.get(station_id) == "da"]
        active_holdout = [station_id for station_id in station_ids if roles.get(station_id) == "holdout"]
        supported = tuple(sorted({match.subdomain_id for match in matches.values()}))
        candidates.append(
            Candidate(
                selected_timestamp=observation_time,
                variable="station_hs",
                source_file="stations_da_metadata.csv",
                supported_subdomains=supported,
                active_holdouts=len(active_holdout),
                active_da_stations=len(active_da),
                active_station_ids=tuple(sorted(station_ids)),
                station_match_max_delta_minutes=max(
                    (match.delta_minutes for match in matches.values()),
                    default=0.0,
                ),
            )
        )
    return tuple(candidates)


def match_station_support(
    snow_rows: Sequence[Mapping[str, Any]],
    policy: SchedulePolicy,
) -> dict[datetime, dict[str, StationMatch]]:
    """Match observations to daily DA timesteps within half one model timestep."""

    tolerance = timedelta(hours=policy.model_timestep_hours / 2)
    matches: dict[datetime, dict[str, StationMatch]] = defaultdict(dict)
    domains: dict[str, str] = {}
    for row in snow_rows:
        if not _row_has_observation(row):
            continue
        station_id = _station_id(row)
        subdomain_id = str(row.get("subdomain_id", "")).strip()
        if not station_id or not subdomain_id:
            raise ValueError("Snow inventory requires station_id and subdomain_id")
        previous_domain = domains.setdefault(station_id, subdomain_id)
        if previous_domain != subdomain_id:
            raise ValueError(f"Station {station_id} maps to multiple subdomains")
        observation = _observation_timestamp(row)
        day_targets = tuple(
            datetime.combine(observation.date() + timedelta(days=offset), policy.station_observation_time)
            for offset in (-1, 0, 1)
        )
        deltas = tuple(abs(observation - target) for target in day_targets)
        minimum = min(deltas)
        if minimum > tolerance:
            continue
        nearest = tuple(target for target, delta in zip(day_targets, deltas) if delta == minimum)
        if len(nearest) != 1:
            raise ValueError(
                f"Ambiguous half-timestep match for station {station_id} at {observation.isoformat(sep=' ')}"
            )
        model_time = nearest[0]
        match = StationMatch(
            station_id=station_id,
            subdomain_id=subdomain_id,
            observation_timestamp=observation,
            delta_minutes=minimum.total_seconds() / 60.0,
            observation_value=_optional_observation_value(row),
        )
        existing = matches[model_time].get(station_id)
        if existing is None or match.delta_minutes < existing.delta_minutes:
            matches[model_time][station_id] = match
        elif match.delta_minutes == existing.delta_minutes:
            raise ValueError(
                f"Ambiguous station observations for {station_id} at model timestep "
                f"{model_time.isoformat(sep=' ')}"
            )
    return {model_time: dict(sorted(station_matches.items())) for model_time, station_matches in sorted(matches.items())}


def _quality_records(
    fsc_rows: Sequence[Mapping[str, Any]],
    snow_rows: Sequence[Mapping[str, Any]],
    station_roles: Sequence[Mapping[str, Any]],
    policy: SchedulePolicy,
    slots: Sequence[Slot],
) -> list[dict[str, Any]]:
    """Return candidate-level gate evidence, including rejected FSC domains."""

    records: list[dict[str, Any]] = []
    for row in sorted(
        fsc_rows,
        key=lambda item: (
            str(item.get("date", "")),
            str(item.get("source_file", "")),
            str(item.get("subdomain_id", "")),
        ),
    ):
        scene_date = parse_date(row.get("date"), field="FSC date")
        if not any(
            slot.variable == "scf"
            and abs((scene_date - slot.target_date).days) <= policy.fsc_search_days
            for slot in slots
        ):
            continue
        metrics = fsc_reference_metrics(row)
        reasons = []
        if metrics["cloud_reference_fraction"] > policy.maximum_cloud_fraction:
            reasons.append("cloud_fraction")
        if metrics["invalid_reference_fraction"] > policy.maximum_invalid_fraction:
            reasons.append("invalid_fraction")
        if metrics["valid_count"] < 1:
            reasons.append("no_valid_pixels")
        if not metrics["uncertainty_complete"]:
            reasons.append("incomplete_uncertainty")
        records.append(
            {
                "record_type": "fsc_subdomain_quality",
                "candidate_timestamp": f"{scene_date.isoformat()} 00:00:00",
                "source_file": str(row.get("source_file", "")),
                "subdomain_id": str(row.get("subdomain_id", "")),
                **metrics,
                "uncertainty_mean": row.get("uncertainty_mean", ""),
                "uncertainty_p90": row.get("uncertainty_p90", ""),
                "passes_quality": not reasons,
                "rejection_reasons": reasons,
            }
        )
    roles = {str(row["station_id"]): str(row["role"]) for row in station_roles}
    for model_time, matches in match_station_support(snow_rows, policy).items():
        if not any(
            slot.variable == "station_hs"
            and abs((model_time.date() - slot.target_date).days) <= policy.station_hs_search_days
            for slot in slots
        ):
            continue
        for station_id, match in matches.items():
            records.append(
                {
                    "record_type": "station_timestep_support",
                    "candidate_timestamp": model_time.isoformat(sep=" "),
                    "station_id": station_id,
                    "subdomain_id": match.subdomain_id,
                    "observation_timestamp": match.observation_timestamp.isoformat(sep=" "),
                    "match_delta_minutes": match.delta_minutes,
                    "role": roles.get(station_id, "unassigned"),
                    "passes_quality": roles.get(station_id) in {"da", "holdout"},
                }
            )
    return records


def _schedule_summary(
    slots: Sequence[Slot],
    events: Sequence[Mapping[str, Any]],
    options_by_slot: Mapping[int, Sequence[Candidate]],
    policy: SchedulePolicy,
) -> dict[str, Any]:
    counts: dict[str, dict[str, Any]] = {}
    for variable in policy.sequence:
        targets = sum(slot.variable == variable for slot in slots)
        feasible = sum(slot.variable == variable and bool(options_by_slot[slot.index]) for slot in slots)
        retained = sum(str(event["variable"]) == variable for event in events)
        fraction = retained / feasible if feasible else 1.0
        counts[variable] = {
            "targets": targets,
            "feasible_targets": feasible,
            "unavailable_targets": targets - feasible,
            "retained": retained,
            "fulfillment_denominator": "feasible_targets",
            "fulfillment_fraction": fraction,
        }
        if fraction + 1.0e-12 < policy.minimum_fulfillment:
            raise ValueError(
                f"{variable} feasible-slot fulfillment {retained}/{feasible} ({fraction:.1%}) is below "
                f"{policy.minimum_fulfillment:.1%}"
            )
    domains = sorted(
        {
            subdomain_id
            for options in options_by_slot.values()
            for candidate in options
            for subdomain_id in candidate.supported_subdomains
        }
    )
    by_subdomain: dict[str, dict[str, Any]] = {}
    for subdomain_id in domains:
        domain_counts: dict[str, Any] = {}
        for variable in policy.sequence:
            targets = sum(slot.variable == variable for slot in slots)
            feasible = sum(
                slot.variable == variable
                and any(subdomain_id in candidate.supported_subdomains for candidate in options_by_slot[slot.index])
                for slot in slots
            )
            retained = sum(
                str(event["variable"]) == variable
                and subdomain_id in event["supported_subdomains"]
                for event in events
            )
            fraction = retained / feasible if feasible else 1.0
            domain_counts[variable] = {
                "targets": targets,
                "feasible_targets": feasible,
                "unavailable_targets": targets - feasible,
                "retained": retained,
                "fulfillment_denominator": "feasible_targets",
                "fulfillment_fraction": fraction,
            }
            if feasible and fraction + 1.0e-12 < policy.minimum_fulfillment:
                raise ValueError(
                    f"{subdomain_id} {variable} feasible-slot fulfillment "
                    f"{retained}/{feasible} ({fraction:.1%}) is below {policy.minimum_fulfillment:.1%}"
                )
        by_subdomain[subdomain_id] = domain_counts
    dates = [parse_date(event["selected_date"], field="selected date") for event in events]
    if len(dates) != len(set(dates)):
        raise ValueError("Schedule contains duplicate selected dates")
    fsc_sources = [str(event["source_file"]) for event in events if event["variable"] == "scf"]
    if len(fsc_sources) != len(set(fsc_sources)):
        raise ValueError("Schedule selects one FSC source scene more than once")
    return {
        "target_count": len(slots),
        "retained_count": len(events),
        "by_variable": counts,
        "by_subdomain": by_subdomain,
        "first_selected_date": dates[0].isoformat() if dates else None,
        "last_selected_date": dates[-1].isoformat() if dates else None,
    }


def schedule_with_adaptive_roles(
    *,
    policy: SchedulePolicy,
    fsc_rows: Sequence[Mapping[str, Any]],
    snow_rows: Sequence[Mapping[str, Any]],
    station_rows: Sequence[Mapping[str, Any]],
    windows: Sequence[tuple[str, date, date]],
) -> tuple[StationRoleResult, dict[str, ScheduleResult]]:
    """Schedule multiple projects using one shared adaptive station split."""

    matched_times = tuple(match_station_support(snow_rows, policy))
    station_slots = tuple(
        slot
        for _, start, end in windows
        for slot in generate_slots(start, end, policy)
        if slot.variable == "station_hs"
    )
    eligible_station_times = {
        model_time
        for model_time in matched_times
        if any(abs((model_time.date() - slot.target_date).days) <= policy.station_hs_search_days for slot in station_slots)
    }
    roles = assign_station_roles(
        snow_rows,
        station_rows,
        policy,
        support_times=eligible_station_times,
    )
    for _ in range(len(roles.roles) + 1):
        schedules = {
            name: schedule_events(
                policy=policy,
                fsc_rows=fsc_rows,
                snow_rows=snow_rows,
                station_roles=roles.roles,
                start=start,
                end=end,
            )
            for name, start, end in windows
        }
        station_times = [
            datetime.fromisoformat(str(event["selected_timestamp"]))
            for result in schedules.values()
            for event in result.events
            if event["variable"] == "station_hs"
        ]
        adjusted = adapt_station_roles_for_support(roles, snow_rows, station_times, policy)
        if adjusted.roles == roles.roles:
            selected_dates = [
                str(event["selected_date"])
                for result in schedules.values()
                for event in result.events
            ]
            if len(selected_dates) != len(set(selected_dates)):
                raise ValueError("Schedules contain a duplicate selected date across projects or types")
            _validate_selected_station_support(roles, schedules, snow_rows, policy)
            return roles, schedules
        roles = adjusted
    raise RuntimeError("Station role adaptation did not converge")


def _validate_selected_station_support(
    roles: StationRoleResult,
    schedules: Mapping[str, ScheduleResult],
    snow_rows: Sequence[Mapping[str, Any]],
    policy: SchedulePolicy,
) -> None:
    """Prove every authored station leaf event has at least one active DA station."""

    role_by_station = {str(row["station_id"]): str(row["role"]) for row in roles.roles}
    support = match_station_support(snow_rows, policy)
    for project_name, schedule in schedules.items():
        for event in schedule.events:
            if event["variable"] != "station_hs":
                continue
            model_time = datetime.fromisoformat(str(event["selected_timestamp"]))
            matches = support.get(model_time, {})
            for subdomain_id in event["supported_subdomains"]:
                active_da = [
                    station_id
                    for station_id, match in matches.items()
                    if match.subdomain_id == subdomain_id and role_by_station.get(station_id) == "da"
                ]
                if not active_da:
                    raise ValueError(
                        "Selected station event lacks active DA support: "
                        f"{project_name}/{subdomain_id}/{model_time.isoformat(sep=' ')}"
                    )


def read_csv_records(path: Path) -> list[dict[str, str]]:
    """Read a normalized CSV table while preserving station identifiers."""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def write_schedule_outputs(
    output_dir: Path,
    result: ScheduleResult,
    roles: StationRoleResult,
    *,
    preflight: bool = False,
) -> dict[str, Any]:
    """Write deterministic scheduler tables or return their preflight summary."""

    summary = {"schedule": result.summary, "station_role_exceptions": len(roles.exceptions)}
    if preflight:
        return summary
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "target_slots.csv", result.targets)
    _write_csv(output_dir / "events.csv", result.events)
    _write_csv(output_dir / "quality.csv", result.quality)
    _write_csv(output_dir / "station_roles.csv", roles.roles)
    _write_csv(output_dir / "exceptions.csv", (*result.exceptions, *roles.exceptions))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fieldnames = sorted({key for row in materialized for key in row})
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        if not fieldnames:
            file_obj.write("")
            return
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _station_id(row: Mapping[str, Any]) -> str:
    return str(row.get("station_id", row.get("id", ""))).strip()


def _row_has_observation(row: Mapping[str, Any]) -> bool:
    value = row.get("valid_observation_count", 1)
    return _integer(value, field="valid_observation_count") > 0


def _observation_timestamp(row: Mapping[str, Any]) -> datetime:
    value = row.get("timestamp", row.get("time", ""))
    if not str(value).strip():
        first = str(row.get("first_observation", "")).strip()
        last = str(row.get("last_observation", "")).strip()
        if not first or first != last:
            raise ValueError(
                "Snow inventory requires an exact timestamp (timestamp/time or identical first/last observation)"
            )
        value = first
    try:
        timestamp = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid snow observation timestamp: {value!r}") from exc
    if timestamp.tzinfo is not None:
        raise ValueError("Snow inventory timestamps must be normalized to naive setup-local time")
    return timestamp


def _optional_observation_value(row: Mapping[str, Any]) -> float | None:
    value = row.get("observation_value", row.get("snow_depth", ""))
    if value in (None, ""):
        return None
    return _number(value, field="snow observation value")


def _altitude(row: Mapping[str, Any]) -> float:
    return _number(row.get("alt", row.get("elevation_m", 0.0)), field="station altitude")


def _number(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {field}: {value!r}")
    return number


def _integer(value: object, *, field: str) -> int:
    number = _number(value, field=field)
    if not number.is_integer():
        raise ValueError(f"{field} must be an integer: {value!r}")
    return int(number)
