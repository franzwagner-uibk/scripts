#!/usr/bin/env python3
"""Build a staged Fram3S collection; publication is a separate validated operation."""

import argparse
import json
import fcntl
import logging
import shutil
import time
from pathlib import Path

from fram3s_grids.common import RESOLUTIONS, save_json
from fram3s_grids.delivery import (
    deliver_resolution,
    finish_manifest,
    plot_overview,
    source_coverage_preflight,
    validate_resolution,
)
from fram3s_grids.geometry import build_geometry
from fram3s_grids.reporting import raster_diagnostics, write_collection_readmes
from fram3s_grids.processing import SOURCES, prepare_parent, prepare_sources, terrain_parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "resolution", "finish", "cleanup", "finalize"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--archive-stamp", help="Read archived originals after publication")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=RESOLUTIONS)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.work.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        source_root = args.source
        if args.archive_stamp:
            source_root = args.work / "input_cache"
            for relative in SOURCES.values():
                parts = Path(relative).parts
                original = args.source / parts[0] / "archive" / args.archive_stamp / Path(*parts[1:])
                target = source_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise FileExistsError(f"Use a fresh work directory: {target}")
                shutil.copy2(original, target)
        build_geometry(source_root, args.work / "output", args.work)
        manifest = prepare_sources(source_root, args.work)
        if args.archive_stamp:
            for kind, relative in SOURCES.items():
                parts = Path(relative).parts
                manifest[kind]["origin_relative"] = str(
                    Path(parts[0]) / "archive" / args.archive_stamp / Path(*parts[1:])
                )
            save_json(args.work / "source_manifest.json", manifest)
        for resolution in reversed(RESOLUTIONS):
            prepare_parent(args.work, resolution)
        missing = source_coverage_preflight(args.work)
        if missing:
            raise RuntimeError(f"Source coverage failed: {missing}")
    elif args.command == "resolution":
        if args.resolution is None:
            parser.error("resolution requires --resolution")
        import numba

        numba.set_num_threads(args.threads)
        lock = (args.work / f"resolution_{args.resolution}.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (args.work / f"status_{args.resolution}.json").exists():
            existing = json.loads((args.work / f"status_{args.resolution}.json").read_text())
            if not existing["failures"]:
                return
        started = time.time()
        prepare_parent(args.work, args.resolution)
        terrain_parent(args.work, args.resolution)
        deliver_resolution(args.work, args.resolution)
        failures = validate_resolution(args.work, args.resolution)
        save_json(
            args.work / f"status_{args.resolution}.json", {"seconds": time.time() - started, "failures": failures}
        )
        if failures:
            raise RuntimeError(json.dumps(failures))
    elif args.command == "cleanup":
        if args.source is None:
            parser.error("cleanup requires --source pointing to the completed previous work directory")
        from fram3s_grids.cleanup import cleanup_existing

        cleanup_existing(args.source, args.work)
    elif args.command == "finalize":
        from fram3s_grids.cleanup import finalize

        finalize(args.work)
    else:
        plot_overview(args.work / "output")
        raster_diagnostics(args.work, args.work / "diagnostics")
        write_collection_readmes(args.work)
        finish_manifest(args.work)


if __name__ == "__main__":
    main()
