#!/usr/bin/env python3
"""Build the event-neutral six-season North Tyrol snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from north_tyrol_snapshot import PINNED_IMAGE, SnapshotOptions, build_snapshot, preflight


def parse_args() -> argparse.Namespace:
    """Parse the documented snapshot-builder CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument("--start-year", required=True, type=int)
    parser.add_argument("--end-year", required=True, type=int)
    parser.add_argument("--resolution", required=True, type=int)
    parser.add_argument("--image", required=True, help=f"Must equal {PINNED_IMAGE}")
    parser.add_argument("--preflight", action="store_true", help="Inspect sources without writing anything")
    return parser.parse_args()


def main() -> int:
    """Run preflight or build mode."""

    args = parse_args()
    options = SnapshotOptions(
        source_root=args.source_root,
        target_root=args.target_root,
        start_year=args.start_year,
        end_year=args.end_year,
        resolution=args.resolution,
        image=args.image,
    )
    if args.preflight:
        print(json.dumps(preflight(options), indent=2, sort_keys=True, default=str))
        return 0
    output = build_snapshot(options)
    print(f"READY_FOR_EVENT_SELECTION: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
