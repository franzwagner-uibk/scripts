#!/usr/bin/env python3
"""Capture compact North Tyrol failure evidence without copying large ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


JSON_NAMES = {
    "leaf_finalization_manifest.json",
    "member_run.json",
    "retention_manifest.json",
    "run_manifest.json",
    "state_pointer.json",
    "storage_reservation.json",
    "subdomain_run_manifest.json",
}
COPY_NAMES = {
    "project_perf.png",
    "project_perf_metrics.csv",
    "project_perf_phases.csv",
    "project_perf_phases.png",
}
MAX_PARSE_BYTES = 64 * 1024 * 1024
MAX_COPY_BYTES = 32 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_summary(path: Path) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
    }
    if stat.st_size > MAX_PARSE_BYTES:
        result["parse_status"] = "identity_only_oversized"
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result.update(parse_status="invalid", error=str(exc))
        return result
    if not isinstance(payload, dict):
        result["parse_status"] = "non_object"
        return result
    result["parse_status"] = "parsed"
    for key in (
        "schema_version",
        "storage_reservation_schema_version",
        "retention_schema_version",
        "status",
        "phase",
        "member",
        "generation",
        "generation_id",
        "error",
    ):
        if key in payload and isinstance(payload[key], (str, int, float, bool, type(None))):
            result[key] = payload[key]
    batches = payload.get("batches")
    if isinstance(batches, list):
        result["batch_count"] = len(batches)
        result["batch_status_counts"] = dict(
            Counter(str(batch.get("status", "unknown")) for batch in batches if isinstance(batch, dict))
        )
        result["representative_failed_batches"] = [
            {
                key: batch.get(key)
                for key in ("id", "batch_id", "artifact_class", "status", "error")
                if key in batch
            }
            for batch in batches
            if isinstance(batch, dict) and str(batch.get("status", "")).lower() not in {"", "complete", "completed", "success"}
        ][:10]
    return result


def capture(project_root: Path, output_dir: Path) -> Path:
    """Write hashes, statuses and small diagnostics to a compact evidence tree."""
    project_root = project_root.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Failure evidence destination exists: {output_dir}")
    output_dir.mkdir(parents=True)
    records: list[dict[str, object]] = []
    copied: list[str] = []
    for path in sorted(project_root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(project_root).as_posix()
        if path.name in JSON_NAMES:
            records.append(
                {
                    "path": relative,
                    "kind": path.name,
                    **_json_summary(path),
                }
            )
        if path.name in COPY_NAMES and path.stat().st_size <= MAX_COPY_BYTES:
            destination = output_dir / "diagnostics" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied.append(relative)

    status_counts = Counter(
        str(record.get("status", "unknown")) for record in records
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "manifest_count": len(records),
        "manifest_status_counts": dict(status_counts),
        "copied_diagnostics": copied,
        "records": records,
    }
    summary_path = output_dir / "failure_evidence.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(capture(args.project_root, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
