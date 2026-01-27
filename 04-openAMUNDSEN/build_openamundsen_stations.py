"""
Build an openAMUNDSEN-compatible stations CSV that matches the output filenames
produced by meteoConverter.py.

Why this exists
- openAMUNDSEN matches stations by `id` == meteo filename stem.
- meteoConverter.py writes all outputs as: PROVIDER__<original_basename>.csv
- Therefore the stations file must use ids: PROVIDER__<original_stem>

Inputs
- STATIONS_META: master station metadata (has extra columns, possible commas in names)
- METEO_SRC_ROOT: source meteo directory (provider subfolders with per-station CSVs)

Output
- STATIONS_OUT: stations file with header: id,name,x,y,alt
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

STATIONS_META = Path(r"F:\fram3s\01-data\02-meteo\02-meta\stations.csv")
METEO_SRC_ROOT = Path(r"F:\fram3s\01-data\02-meteo\01-data\01-initial\long")
STATIONS_OUT = Path(r"F:\fram3s\01-data\02-meteo\02-meta\stations_openamundsen.csv")


def _provider_name(src: Path) -> str:
    try:
        rel = src.relative_to(METEO_SRC_ROOT)
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except Exception:
        pass
    return "UNKNOWN"


def _load_meta_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        expected = len(header)
        rows: List[List[str]] = []
        for row in reader:
            if not row:
                continue
            # Fix comma-split names by merging extra fields into the name column (index 1)
            while len(row) > expected:
                row[1] = row[1] + "," + row[2]
                del row[2]
            if len(row) != expected:
                continue
            rows.append(row)
    return header, rows


def main() -> int:
    # If the metadata file already is an openAMUNDSEN stations file (id,name,x,y,alt
    # and ids are provider-prefixed), just copy it to the target and exit.
    try:
        with STATIONS_META.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header0 = next(reader)
            first_row = next(reader, None)
        header_norm = [h.strip() for h in header0]
        if header_norm == ["id", "name", "x", "y", "alt"] and first_row and "__" in first_row[0]:
            if STATIONS_META.resolve() != STATIONS_OUT.resolve():
                shutil.copyfile(STATIONS_META, STATIONS_OUT)
            print("Detected openAMUNDSEN stations.csv -> copied as-is")
            print("wrote:", STATIONS_OUT)
            return 0
    except Exception:
        pass

    # List all meteo files
    meteo_files = sorted(METEO_SRC_ROOT.rglob("*.csv"))

    # Load metadata
    header, rows = _load_meta_rows(STATIONS_META)
    idx = {name: i for i, name in enumerate(header)}
    for required in ("id", "name", "x", "y", "alt"):
        if required not in idx:
            raise ValueError(f"Missing required column '{required}' in {STATIONS_META}")

    # Index station rows by (id, provider) when available; also keep fallback by id
    idx_provider = idx.get("provider")
    by_id: Dict[str, List[List[str]]] = {}
    by_id_provider: Dict[Tuple[str, str], List[str]] = {}
    for r in rows:
        sid = r[idx["id"]]
        by_id.setdefault(sid, []).append(r)
        if idx_provider is not None:
            by_id_provider[(sid, r[idx_provider])] = r

    # Build output rows
    out_rows: List[List[str]] = []
    missing: List[Tuple[str, str]] = []
    for p in meteo_files:
        provider = _provider_name(p)
        stem = p.stem
        out_id = f"{provider}__{stem}"

        meta_row = None
        if idx_provider is not None:
            meta_row = by_id_provider.get((stem, provider))
        if meta_row is None:
            candidates = by_id.get(stem, [])
            if len(candidates) == 1:
                meta_row = candidates[0]
            elif len(candidates) > 1:
                # If multiple candidates exist (rare), prefer the one matching provider
                if idx_provider is not None:
                    provider_matches = [c for c in candidates if c[idx_provider] == provider]
                    if len(provider_matches) == 1:
                        meta_row = provider_matches[0]

        # Fuzzy fallback: if file stem is a base id, match meta ids like '<stem>_12345'
        if meta_row is None:
            if idx_provider is not None:
                provider_candidates = [
                    r
                    for (sid, prov), r in by_id_provider.items()
                    if prov == provider and sid.startswith(stem + "_")
                ]
            else:
                provider_candidates = []
            if provider_candidates:
                # Prefer the highest numeric suffix (e.g. pick ..._15431 over ..._15430)
                def _suffix_num(r: List[str]) -> int:
                    sid = r[idx["id"]]
                    try:
                        return int(sid.rsplit("_", 1)[-1])
                    except Exception:
                        return -1

                meta_row = sorted(provider_candidates, key=_suffix_num)[-1]

        if meta_row is None:
            missing.append((stem, provider))
            continue

        out_rows.append(
            [
                out_id,
                meta_row[idx["name"]],
                meta_row[idx["x"]],
                meta_row[idx["y"]],
                meta_row[idx["alt"]],
            ]
        )

    STATIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with STATIONS_OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "x", "y", "alt"])
        w.writerows(out_rows)

    print("meteo files:", len(meteo_files))
    print("stations written:", len(out_rows))
    print("missing metadata rows:", len(missing))
    if missing:
        print("missing sample:", missing[:20])
    print("wrote:", STATIONS_OUT)
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
