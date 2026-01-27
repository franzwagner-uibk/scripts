"""
Convert meteo CSVs to openAMUNDSEN-ready schema:
- ensure time column is named 'date'
- strip timezone info so dates are naive ("YYYY-MM-DD HH:MM:SS")
- flatten all converted files into a single output folder.

Sources:  F:\\fram3s\\01-data\\02-meteo\\01-data\\01-initial\\long
Targets:  F:\\fram3s\\01-data\\02-meteo\\01-data\\01-initial\\openAMUNDSEN

Runs in parallel (tune MAX_WORKERS). Uses only the standard library.
"""

from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

SRC_ROOT = Path(r"F:\fram3s\01-data\02-meteo\01-data\01-initial\long")
DST_ROOT = Path(r"F:\fram3s\01-data\02-meteo\01-data\01-initial\openAMUNDSEN")
MAX_WORKERS = 24

DATE_COLUMN_CANDIDATES = ("date", "datetime", "timestamp", "time")
OUTPUT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _normalize_date(value: str) -> str:
    """Return a timezone-free timestamp string; fallback to sanitized original on parse errors."""
    if value is None:
        return ""
    raw = value.strip()
    if not raw:
        return ""

    # Replace trailing Z with UTC offset so fromisoformat can parse it.
    candidate = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
    # If we still have a bare T without offset, that's fine for fromisoformat.

    try:
        dt = datetime.fromisoformat(candidate)
    except Exception:
        # Try a few common patterns before giving up.
        for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                dt = datetime.strptime(candidate, pattern)
                break
            except Exception:
                dt = None
        if dt is None:
            # Final fallback: strip T and Z and return as-is (naive).
            return raw.replace("T", " ").replace("Z", "")

    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime(OUTPUT_DATE_FORMAT)


def _provider_name(src: Path) -> str:
    """Infer provider from the first directory under SRC_ROOT (best-effort)."""
    try:
        rel = src.relative_to(SRC_ROOT)
        # Expect: <provider>/<file>.csv
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except Exception:
        pass
    return "UNKNOWN"


def _output_name(src: Path) -> str:
    """Return output filename.

    Rules:
    - Always prefix with provider to guarantee uniqueness and match station ids.
    """
    return f"{_provider_name(src)}__{src.name}"


def _output_path(src: Path) -> Path:
    return DST_ROOT / _output_name(src)


def _pick_date_column(fieldnames: Iterable[str]) -> Tuple[str, list[str]]:
    """Find the date column name and return the final field order with 'date' first."""
    if not fieldnames:
        raise ValueError("CSV has no header")
    names = list(fieldnames)
    date_col = None
    for cand in DATE_COLUMN_CANDIDATES:
        if cand in names:
            date_col = cand
            break
    if date_col is None:
        raise ValueError(f"No date-like column found (looked for {DATE_COLUMN_CANDIDATES})")

    # Rebuild header with 'date' first (rename if necessary) and keep remaining columns in order.
    remaining = [n for n in names if n != date_col]
    return date_col, ["date", *remaining]


def _process_file(src: Path) -> Tuple[Path, str | None]:
    """Convert one CSV and return (path, error_message_or_None)."""
    dst = _output_path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        with src.open("r", newline="", encoding="utf-8") as f_in:
            reader = csv.DictReader(f_in)
            date_col, out_fieldnames = _pick_date_column(reader.fieldnames or [])

            rows = []
            for row in reader:
                raw_date = row.get(date_col, "")
                norm_date = _normalize_date(raw_date)
                # Remove original date column (if named differently) and set normalized 'date'
                if date_col != "date":
                    row.pop(date_col, None)
                row["date"] = norm_date
                rows.append(row)

        with dst.open("w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return src, None
    except Exception as exc:
        return src, f"{type(exc).__name__}: {exc}"


def main() -> int:
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(SRC_ROOT.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found under {SRC_ROOT}")
        return 1

    # Safety: ensure output filenames are unique (provider__<basename>.csv)
    out_counts: Dict[str, int] = {}
    for f in csv_files:
        out_name = _output_name(f)
        out_counts[out_name] = out_counts.get(out_name, 0) + 1
    dup_out = [n for n, c in out_counts.items() if c > 1]
    if dup_out:
        print("Error: duplicate output filenames detected; aborting to avoid overwrites:")
        for n in dup_out:
            print(f"  {n} (count={out_counts[n]})")
        return 1

    errors: list[Tuple[Path, str]] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(_process_file, src): src for src in csv_files}
        for i, fut in enumerate(as_completed(future_map), start=1):
            src = future_map[fut]
            path, err = fut.result()
            if err:
                errors.append((path, err))
            # Lightweight progress marker
            if i % 50 == 0 or i == len(csv_files):
                print(f"[{i}/{len(csv_files)}] processed")

    if errors:
        print("\nCompleted with errors on the following files:")
        for path, msg in errors:
            print(f"- {path}: {msg}")
        return 1

    print(f"Done. Converted {len(csv_files)} files into {DST_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
