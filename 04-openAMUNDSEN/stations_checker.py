"""
Clean stations.csv and check coverage against meteo files.

What it does:
- Reads the raw stations.csv (BOM-tolerant).
- Fixes rows with extra commas in station names by merging extra fields into `name`.
- Writes a cleaned copy (same columns) next to the raw file with suffix `_clean.csv`.
- Reports which meteo files have no station entry and which station ids have no meteo file.

Paths are set below; adjust if needed.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Input paths
# Adjust to your source stations CSV and meteo root.
STATIONS_PATH = Path(r"F:\fram3s\01-data\02-meteo\02-meta\stations.csv")
METEO_ROOT = Path(r"F:\fram3s\01-data\02-meteo\01-data\01-initial\long")

# Output (same folder as input)
CLEAN_PATH = STATIONS_PATH.with_name(STATIONS_PATH.stem + "_clean.csv")


def _read_and_fix_rows(path: Path) -> Tuple[List[str], List[List[str]], List[int]]:
    """Return (header, rows, fixed_row_indices)."""
    fixed_rows: List[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        expected = len(header)
        rows: List[List[str]] = []
        for idx, row in enumerate(reader, start=2):  # 1-based including header
            if not row:
                continue
            # If there are extra fields, merge them into the name column (index 1)
            while len(row) > expected:
                row[1] = row[1] + "," + row[2]
                del row[2]
                if idx not in fixed_rows:
                    fixed_rows.append(idx)
            if len(row) == expected:
                rows.append(row)
            else:
                # Keep malformed rows as-is but note them
                fixed_rows.append(idx)
                rows.append(row)
    return header, rows, fixed_rows


def _write_clean(header: List[str], rows: List[List[str]], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def _base_id(s: str) -> str:
    """Strip a trailing _digits from station id for fuzzy matching (e.g., Kolm_Saigurn_15431 -> Kolm_Saigurn)."""
    return re.sub(r"_[0-9]+$", "", s)


def _coverage(ids: List[str]) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    file_ids = {p.stem for p in METEO_ROOT.rglob("*.csv")}
    station_ids = set(ids)
    missing_by_id = sorted(file_ids - station_ids)
    extra_station_ids = sorted(station_ids - file_ids)

    # Fuzzy suggestions: match by base id (strip trailing _digits)
    station_base_map: Dict[str, List[str]] = {}
    for sid in extra_station_ids:
        b = _base_id(sid)
        station_base_map.setdefault(b, []).append(sid)

    suggestions: Dict[str, List[str]] = {}
    for fid in missing_by_id:
        cand = station_base_map.get(_base_id(fid), [])
        if cand:
            suggestions[fid] = cand
    return missing_by_id, extra_station_ids, suggestions


def main() -> int:
    header, rows, fixed_rows = _read_and_fix_rows(STATIONS_PATH)
    _write_clean(header, rows, CLEAN_PATH)

    ids = [r[0] for r in rows if r]
    missing_by_id, extra_station_ids, suggestions = _coverage(ids)

    print(f"Cleaned stations written to: {CLEAN_PATH}")
    print(f"Rows processed: {len(rows)}")
    print(f"Rows that needed name merge (extra commas): {len(fixed_rows)} -> lines {fixed_rows}")
    print(f"Meteo files without station entry: {len(missing_by_id)}")
    if missing_by_id:
        print("  missing by id (first 20):", missing_by_id[:20])
    if suggestions:
        print("  fuzzy matches (file -> station ids):")
        for fid, cands in suggestions.items():
            print(f"    {fid} -> {', '.join(cands)}")
    print(f"Station ids without meteo file: {len(extra_station_ids)}")
    if extra_station_ids:
        print("  extra station ids (first 20):", extra_station_ids[:20])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
