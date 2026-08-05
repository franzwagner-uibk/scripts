"""Deterministic, source-independent data-assimilation event scheduling."""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
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
    minimum_fulfillment: float


@dataclass(frozen=True)
class Slot:
    """One immutable target in the fixed cadence."""

    index: int
    target_date: date
    variable: str


@dataclass(frozen=True)
class Candidate:
    """One normalized observation candidate for a target slot."""

    selected_date: date
    variable: str
    source_file: str
    supported_subdomains: tuple[str, ...]
    active_holdouts: int = 0
    active_da_stations: int = 0
    cloud_fraction_mean: float = 0.0
    uncertainty_mean: float = 0.0


@dataclass(frozen=True)
class ScheduleResult:
    """Deterministic retained events, targets, quality and exceptions."""

    slots: tuple[Slot, ...]
    events: tuple[dict[str, Any], ...]
    exceptions: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class StationRoleResult:
    """Shared station roles plus adaptive-reduction exceptions."""

    roles: tuple[dict[str, Any], ...]
    exceptions: tuple[dict[str, Any], ...]


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
        minimum_fulfillment=float(fulfillment.get("minimum_fraction_per_type", -1.0)),
    )
    if policy.schema_version != 1:
        raise ValueError("Only scheduler policy schema_version 1 is supported")
    if policy.target_spacing_days < 1 or policy.sequence != ("scf", "station_hs"):
        raise ValueError("Policy must use positive fixed-day spacing and sequence [scf, station_hs]")
    if policy.minimum_gap_days < 1 or policy.maximum_gap_days < policy.minimum_gap_days:
        raise ValueError("Invalid accepted adjacent-event gap")
    if policy.fsc_search_days < 0 or policy.station_hs_search_days < 0:
        raise ValueError("Search windows must be non-negative")
    if not 0.0 <= policy.maximum_cloud_fraction <= 1.0:
        raise ValueError("maximum_cloud_fraction must be in [0, 1]")
    if not 0.0 < policy.minimum_fulfillment <= 1.0:
        raise ValueError("minimum_fulfillment must be in (0, 1]")
    return policy


def _month_day(value: object, *, field: str) -> tuple[int, int]:
    if not isinstance(value, str) or len(value.split("-")) != 2:
        raise ValueError(f"{field} must be MM-DD")
    month, day = (int(item) for item in value.split("-"))
    date(2000, month, day)
    return month, day


def generate_slots(start: date, end: date, policy: SchedulePolicy) -> tuple[Slot, ...]:
    """Generate alternating fixed-day targets within one project window."""

    interval_start = date(start.year, *policy.interval_start)
    interval_end_year = start.year + (policy.interval_end < policy.interval_start)
    interval_end = date(interval_end_year, *policy.interval_end)
    first = max(start, interval_start)
    last = min(end, interval_end)
    if last < first:
        return ()
    slots: list[Slot] = []
    target = first
    while target <= last:
        slots.append(Slot(len(slots), target, policy.sequence[len(slots) % len(policy.sequence)]))
        target += timedelta(days=policy.target_spacing_days)
    return tuple(slots)


def assign_station_roles(
    snow_rows: Sequence[Mapping[str, Any]],
    station_rows: Sequence[Mapping[str, Any]],
) -> StationRoleResult:
    """Assign deterministic shared DA and elevation-aware holdout roles."""

    station_metadata = {_station_id(row): row for row in station_rows}
    station_domains: dict[str, str] = {}
    active_days: dict[str, set[date]] = defaultdict(set)
    for row in snow_rows:
        station_id = _station_id(row)
        subdomain_id = str(row.get("subdomain_id", "")).strip()
        if not station_id or not subdomain_id:
            raise ValueError("Snow inventory requires station_id and subdomain_id")
        previous = station_domains.setdefault(station_id, subdomain_id)
        if previous != subdomain_id:
            raise ValueError(f"Station {station_id} maps to multiple subdomains")
        if _integer(row.get("valid_observation_count"), field="valid_observation_count") > 0:
            active_days[station_id].add(parse_date(row.get("date"), field="snow date"))
    missing_metadata = sorted(set(station_domains) - set(station_metadata))
    if missing_metadata:
        raise ValueError(f"Station metadata missing IDs: {missing_metadata}")

    by_domain: dict[str, list[str]] = defaultdict(list)
    for station_id, subdomain_id in station_domains.items():
        by_domain[subdomain_id].append(station_id)
    role_by_station: dict[str, str] = {}
    rank_by_station: dict[str, int] = {}
    for subdomain_id in sorted(by_domain):
        station_ids = sorted(by_domain[subdomain_id])
        target = 2 if len(station_ids) >= 4 else 1 if len(station_ids) >= 2 else 0
        holdouts = _best_holdout_combination(station_ids, target, station_metadata, active_days)
        ordered_holdouts = sorted(
            holdouts,
            key=lambda station_id: (
                -len(active_days[station_id]),
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
            "valid_day_count": len(active_days[station_id]),
            "elevation_m": _altitude(station_metadata[station_id]),
            "holdout_rank": rank_by_station[station_id],
            "selection_reason": "temporal_coverage_then_elevation_spread",
        }
        for station_id in sorted(station_domains)
    )
    return StationRoleResult(roles=roles, exceptions=())


def _best_holdout_combination(
    station_ids: Sequence[str],
    target: int,
    metadata: Mapping[str, Mapping[str, Any]],
    active_days: Mapping[str, set[date]],
) -> set[str]:
    if target == 0:
        return set()
    combinations = itertools.combinations(station_ids, target)

    def score(combo: tuple[str, ...]) -> tuple[Any, ...]:
        coverage = [len(active_days[station_id]) for station_id in combo]
        elevations = [_altitude(metadata[station_id]) for station_id in combo]
        spread = max(elevations) - min(elevations) if len(elevations) > 1 else 0.0
        return (min(coverage), sum(coverage), spread, tuple(reversed(combo)))

    return set(max(combinations, key=score))


def adapt_station_roles_for_support(
    role_result: StationRoleResult,
    snow_rows: Sequence[Mapping[str, Any]],
    station_event_dates: Iterable[date],
) -> StationRoleResult:
    """Reduce holdouts only when they are the sole active support for an event."""

    roles = {str(row["station_id"]): dict(row) for row in role_result.roles}
    active_by_domain_date: dict[tuple[str, date], set[str]] = defaultdict(set)
    for row in snow_rows:
        if _integer(row.get("valid_observation_count"), field="valid_observation_count") > 0:
            active_by_domain_date[(str(row["subdomain_id"]), parse_date(row["date"], field="snow date"))].add(
                _station_id(row)
            )
    exceptions = [dict(row) for row in role_result.exceptions]
    domains = sorted({str(row["subdomain_id"]) for row in roles.values()})
    event_dates = sorted(set(station_event_dates))
    for subdomain_id in domains:
        active_by_date = [active_by_domain_date.get((subdomain_id, event_date), set()) for event_date in event_dates]
        if any(
            any(roles[station_id]["role"] == "da" for station_id in active)
            for active in active_by_date
        ):
            continue
        active_holdouts = {
            station_id
            for active in active_by_date
            for station_id in active
            if roles[station_id]["role"] == "holdout"
        }
        candidates = sorted(
            active_holdouts,
            key=lambda station_id: (
                roles[station_id]["valid_day_count"],
                roles[station_id]["holdout_rank"],
                station_id,
            ),
        )
        if not candidates:
            continue
        station_id = candidates[0]
        trigger_date = next(
            event_date
            for event_date, active in zip(event_dates, active_by_date)
            if station_id in active
        )
        roles[station_id]["role"] = "da"
        roles[station_id]["use_for_da"] = True
        roles[station_id]["use_for_benchmark"] = False
        roles[station_id]["selection_reason"] = "holdout_reduced_for_da_support"
        exceptions.append(
            {
                "exception": "holdout_reduced_for_da_support",
                "subdomain_id": subdomain_id,
                "station_id": station_id,
                "trigger_date": trigger_date.isoformat(),
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
    station_candidates = _station_candidates(snow_rows, station_roles)
    candidates = {"scf": fsc_candidates, "station_hs": station_candidates}
    states: dict[tuple[int | None, int | None], tuple[tuple[int, ...], tuple[tuple[Slot, Candidate], ...]]] = {
        (None, None): ((0,) * 10, ())
    }
    for slot in slots:
        next_states: dict[
            tuple[int | None, int | None],
            tuple[tuple[int, ...], tuple[tuple[Slot, Candidate], ...]],
        ] = {}
        for (last_ordinal, last_slot), (score, path) in states.items():
            _keep_best(next_states, (last_ordinal, last_slot), score, path)
            window = policy.fsc_search_days if slot.variable == "scf" else policy.station_hs_search_days
            for candidate in candidates[slot.variable]:
                delta = abs((candidate.selected_date - slot.target_date).days)
                ordinal = candidate.selected_date.toordinal()
                if delta > window or (last_ordinal is not None and ordinal <= last_ordinal):
                    continue
                if last_ordinal is not None and last_slot == slot.index - 1:
                    gap = ordinal - last_ordinal
                    if not policy.minimum_gap_days <= gap <= policy.maximum_gap_days:
                        continue
                candidate_score = _candidate_score(candidate, delta)
                combined = tuple(left + right for left, right in zip(score, candidate_score))
                _keep_best(next_states, (ordinal, slot.index), combined, path + ((slot, candidate),))
        states = next_states
    _, selected_path = max(states.values(), key=lambda item: (_score_rank(item[0]), _path_tie_key(item[1])))
    selected_by_slot = {slot.index: candidate for slot, candidate in selected_path}
    events: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    for slot in slots:
        candidate = selected_by_slot.get(slot.index)
        if candidate is None:
            exceptions.append(
                {
                    "exception": "unavailable_slot",
                    "slot_index": slot.index,
                    "target_date": slot.target_date.isoformat(),
                    "variable": slot.variable,
                }
            )
            continue
        event = {
            "slot_index": slot.index,
            "target_date": slot.target_date.isoformat(),
            "selected_date": candidate.selected_date.isoformat(),
            "variable": slot.variable,
            "date_delta_days": (candidate.selected_date - slot.target_date).days,
            "source_file": candidate.source_file,
            "supported_subdomains": list(candidate.supported_subdomains),
            "supported_subdomain_count": len(candidate.supported_subdomains),
            "active_holdouts": candidate.active_holdouts,
            "active_da_stations": candidate.active_da_stations,
            "cloud_fraction_mean": candidate.cloud_fraction_mean,
            "uncertainty_mean": candidate.uncertainty_mean,
        }
        events.append(event)
    summary = _schedule_summary(slots, events, policy)
    return ScheduleResult(slots, tuple(events), tuple(exceptions), summary)


def _keep_best(
    states: dict[tuple[int | None, int | None], tuple[tuple[int, ...], tuple[tuple[Slot, Candidate], ...]]],
    key: tuple[int | None, int | None],
    score: tuple[int, ...],
    path: tuple[tuple[Slot, Candidate], ...],
) -> None:
    current = states.get(key)
    if current is None or (_score_rank(score), _path_tie_key(path)) > (
        _score_rank(current[0]),
        _path_tie_key(current[1]),
    ):
        states[key] = (score, path)


def _score_rank(score: tuple[int, ...]) -> tuple[int, ...]:
    """Prioritize balanced per-type fulfillment before quality tie breakers."""

    return (min(score[1], score[2]), score[0], *score[1:])


def _candidate_score(candidate: Candidate, delta: int) -> tuple[int, ...]:
    is_fsc = int(candidate.variable == "scf")
    is_station = int(candidate.variable == "station_hs")
    return (
        1,
        is_fsc,
        is_station,
        len(candidate.supported_subdomains),
        candidate.active_holdouts,
        candidate.active_da_stations,
        -round(candidate.cloud_fraction_mean * 1_000_000),
        -round(candidate.uncertainty_mean * 1_000),
        -delta,
        -candidate.selected_date.toordinal(),
    )


def _path_tie_key(path: Sequence[tuple[Slot, Candidate]]) -> tuple[str, ...]:
    return tuple(f"{candidate.selected_date.isoformat()}:{candidate.source_file}" for _, candidate in path)


def _fsc_candidates(rows: Sequence[Mapping[str, Any]], policy: SchedulePolicy) -> tuple[Candidate, ...]:
    grouped: dict[date, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[parse_date(row.get("date"), field="FSC date")].append(row)
    candidates: list[Candidate] = []
    for scene_date in sorted(grouped):
        scene_rows = grouped[scene_date]
        source_files = {str(row.get("source_file", "")).strip() for row in scene_rows}
        if len(source_files) != 1 or not next(iter(source_files)):
            raise ValueError(f"FSC date {scene_date} must identify exactly one source_file")
        if len({str(row.get("subdomain_id", "")) for row in scene_rows}) != len(scene_rows):
            raise ValueError(f"Duplicate FSC date/subdomain rows for {scene_date}")
        for row in scene_rows:
            if _integer(row.get("uncertainty_count"), field="uncertainty_count") < 1:
                raise ValueError(f"FSC uncertainty layer is empty for {scene_date}")
        supported = tuple(
            sorted(
                str(row["subdomain_id"])
                for row in scene_rows
                if _number(row.get("cloud_fraction"), field="cloud_fraction")
                <= policy.maximum_cloud_fraction
            )
        )
        if not supported:
            continue
        supported_rows = [row for row in scene_rows if str(row["subdomain_id"]) in supported]
        candidates.append(
            Candidate(
                selected_date=scene_date,
                variable="scf",
                source_file=next(iter(source_files)),
                supported_subdomains=supported,
                cloud_fraction_mean=sum(
                    _number(row.get("cloud_fraction"), field="cloud_fraction") for row in supported_rows
                )
                / len(supported_rows),
                uncertainty_mean=sum(
                    _number(row.get("uncertainty_mean"), field="uncertainty_mean") for row in supported_rows
                )
                / len(supported_rows),
            )
        )
    return tuple(candidates)


def _station_candidates(
    snow_rows: Sequence[Mapping[str, Any]],
    station_roles: Sequence[Mapping[str, Any]],
) -> tuple[Candidate, ...]:
    roles = {str(row["station_id"]): str(row["role"]) for row in station_roles}
    grouped: dict[date, list[Mapping[str, Any]]] = defaultdict(list)
    for row in snow_rows:
        if _integer(row.get("valid_observation_count"), field="valid_observation_count") > 0:
            grouped[parse_date(row.get("date"), field="snow date")].append(row)
    candidates: list[Candidate] = []
    for observation_date in sorted(grouped):
        rows = grouped[observation_date]
        active_da = [row for row in rows if roles.get(_station_id(row)) == "da"]
        active_holdout = [row for row in rows if roles.get(_station_id(row)) == "holdout"]
        supported = tuple(sorted({str(row["subdomain_id"]) for row in active_da}))
        if not supported:
            continue
        candidates.append(
            Candidate(
                selected_date=observation_date,
                variable="station_hs",
                source_file="stations_snow_depth.csv",
                supported_subdomains=supported,
                active_holdouts=len(active_holdout),
                active_da_stations=len(active_da),
            )
        )
    return tuple(candidates)


def _schedule_summary(
    slots: Sequence[Slot],
    events: Sequence[Mapping[str, Any]],
    policy: SchedulePolicy,
) -> dict[str, Any]:
    counts: dict[str, dict[str, Any]] = {}
    for variable in policy.sequence:
        targets = sum(slot.variable == variable for slot in slots)
        retained = sum(str(event["variable"]) == variable for event in events)
        fraction = retained / targets if targets else 1.0
        counts[variable] = {"targets": targets, "retained": retained, "fulfillment_fraction": fraction}
        if fraction + 1.0e-12 < policy.minimum_fulfillment:
            raise ValueError(
                f"{variable} fulfillment {retained}/{targets} ({fraction:.1%}) is below "
                f"{policy.minimum_fulfillment:.1%}"
            )
    dates = [parse_date(event["selected_date"], field="selected date") for event in events]
    if len(dates) != len(set(dates)):
        raise ValueError("Schedule contains duplicate selected dates")
    return {
        "target_count": len(slots),
        "retained_count": len(events),
        "by_variable": counts,
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

    roles = assign_station_roles(snow_rows, station_rows)
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
        station_dates = [
            parse_date(event["selected_date"], field="selected date")
            for result in schedules.values()
            for event in result.events
            if event["variable"] == "station_hs"
        ]
        adjusted = adapt_station_roles_for_support(roles, snow_rows, station_dates)
        if adjusted.roles == roles.roles:
            return roles, schedules
        roles = adjusted
    raise RuntimeError("Station role adaptation did not converge")


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
    _write_csv(output_dir / "target_slots.csv", (asdict(slot) for slot in result.slots))
    _write_csv(output_dir / "events.csv", result.events)
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
