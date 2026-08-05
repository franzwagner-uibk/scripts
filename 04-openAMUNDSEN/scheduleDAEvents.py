#!/usr/bin/env python3
"""Schedule deterministic DA events from normalized observation inventories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from da_event_scheduler import (
    load_policy,
    parse_date,
    read_csv_records,
    schedule_with_adaptive_roles,
    write_schedule_outputs,
)


def parse_args() -> argparse.Namespace:
    """Parse the generic scheduler interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--fsc-inventory", required=True, type=Path)
    parser.add_argument("--snow-inventory", required=True, type=Path)
    parser.add_argument("--station-metadata", required=True, type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run preflight or materialize the generic scheduler outputs."""

    args = parse_args()
    start = parse_date(args.start, field="start")
    end = parse_date(args.end, field="end")
    roles, schedules = schedule_with_adaptive_roles(
        policy=load_policy(args.policy),
        fsc_rows=read_csv_records(args.fsc_inventory),
        snow_rows=read_csv_records(args.snow_inventory),
        station_rows=read_csv_records(args.station_metadata),
        windows=(("schedule", start, end),),
    )
    summary = write_schedule_outputs(
        args.output_dir,
        schedules["schedule"],
        roles,
        preflight=args.preflight,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
