#!/usr/bin/env python3
"""
Standalone CLMS HR-WSI downloader for openamundsen_da-style workflows.

What this script does
1) Reads date window (default: from rofental example project YAML)
2) Uses rofental ROI vector by default
3) Ensures the CLMS client conda environment exists (optional auto-create)
4) Runs the upstream CLMS downloader for FSC + SWS
5) Collects likely assimilation-ready GeoTIFFs into snowcover/wetsnow folders
6) Writes a CSV manifest and a class-mapping notes file

This script is intentionally independent of openamundsen_da internals.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_PHD_ROOT = Path(r"C:\Users\franz\Nextcloud\PhD")
DEFAULT_CLMS_DIR = DEFAULT_PHD_ROOT / "clms-hrwsi-api-client-python"
DEFAULT_DOWNLOADER = DEFAULT_CLMS_DIR / "s3_hrwsi_downloader.py"
DEFAULT_ENV_YAML = DEFAULT_CLMS_DIR / "env.yaml"
DEFAULT_ROFENTAL_SETUP = DEFAULT_PHD_ROOT / "openamundsen_da" / "examples" / "rofental"
DEFAULT_ROFENTAL_PROJECT_YAML = (
    DEFAULT_ROFENTAL_SETUP / "projects" / "project_2024_2025" / "project_2024_2025.yml"
)
DEFAULT_ROI = DEFAULT_ROFENTAL_SETUP / "env" / "roi.gpkg"
DEFAULT_OUTPUT_ROOT = DEFAULT_PHD_ROOT / "02-Daten"

FSC_INCLUDE = re.compile(r"(FSC|FSCTOC|FSCOG)", re.IGNORECASE)
FSC_EXCLUDE = re.compile(r"(QC|QCOG|QUALITY|NDSI|FLAG)", re.IGNORECASE)
WSM_INCLUDE = re.compile(r"(WSM|SWS)", re.IGNORECASE)
WSM_EXCLUDE = re.compile(r"(QC|QUALITY|FLAG)", re.IGNORECASE)


@dataclass(frozen=True)
class DateWindow:
    start_date: str
    end_date: str


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _err(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)


def _run(cmd: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    _info(f"Run: {printable}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _find_conda(user_value: str | None = None, *, dry_run: bool = False) -> Path:
    if user_value:
        p = Path(user_value)
        if p.exists():
            return p
        if shutil.which(user_value):
            return Path(shutil.which(user_value))  # type: ignore[arg-type]
        raise FileNotFoundError(f"Provided conda executable not found: {user_value}")

    candidates: list[str | None] = [
        os.environ.get("CONDA_EXE"),
        shutil.which("conda"),
        shutil.which("mamba"),
        str(Path.home() / "miniconda3" / "condabin" / "conda.bat"),
        str(Path.home() / "anaconda3" / "condabin" / "conda.bat"),
    ]
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        if p.exists():
            return p
    if dry_run:
        _warn("Could not find conda in dry-run mode; using placeholder 'conda'")
        return Path("conda")
    raise FileNotFoundError(
        "Could not find 'conda'. Ensure Conda is installed and available in PATH/CONDA_EXE or pass --conda-exe."
    )


def _read_env_name_from_yaml(env_yaml: Path) -> str:
    name_pat = re.compile(r"^\s*name\s*:\s*([A-Za-z0-9_.-]+)\s*$")
    for line in env_yaml.read_text(encoding="utf-8").splitlines():
        m = name_pat.match(line)
        if m:
            return m.group(1)
    raise ValueError(f"Could not read env name from {env_yaml}")


def _parse_project_dates(project_yaml: Path) -> DateWindow:
    start_pat = re.compile(r"^\s*start_date\s*:\s*['\"]?(\d{4}-\d{2}-\d{2})")
    end_pat = re.compile(r"^\s*end_date\s*:\s*['\"]?(\d{4}-\d{2}-\d{2})")

    start: str | None = None
    end: str | None = None
    for line in project_yaml.read_text(encoding="utf-8").splitlines():
        if start is None:
            m = start_pat.match(line)
            if m:
                start = m.group(1)
                continue
        if end is None:
            m = end_pat.match(line)
            if m:
                end = m.group(1)
                continue

    if not start or not end:
        raise ValueError(f"Could not parse start_date/end_date from {project_yaml}")

    return DateWindow(start_date=start, end_date=end)


def _ensure_conda_env(conda_exe: Path, env_yaml: Path, env_name: str, *, dry_run: bool = False) -> None:
    if dry_run:
        _info(f"Dry-run: would ensure conda env '{env_name}' from {env_yaml}")
        return

    result = subprocess.run(
        [str(conda_exe), "env", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    env_paths = [Path(p) for p in data.get("envs", [])]
    env_names = {p.name for p in env_paths}

    if env_name in env_names:
        _info(f"Conda env '{env_name}' already exists")
        return

    _info(f"Creating conda env '{env_name}' from {env_yaml}")
    _run([str(conda_exe), "env", "create", "-f", str(env_yaml)], dry_run=dry_run)


def _copy_if_needed(src: Path, dst: Path, *, overwrite: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return "skipped_exists"
    shutil.copy2(src, dst)
    return "copied"


def _classify_tif(file_name: str) -> str | None:
    upper = file_name.upper()

    if FSC_INCLUDE.search(upper) and not FSC_EXCLUDE.search(upper):
        return "snowcover"

    if WSM_INCLUDE.search(upper) and not WSM_EXCLUDE.search(upper):
        return "wetsnow"

    return None


def _iter_tifs(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".tif", ".tiff"}:
            yield p


def _write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "file_name",
        "selected_kind",
        "target_path",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_mapping_notes(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = """Class mapping notes for CLMS HR-WSI -> openamundsen_da workflows

FSC (Fractional Snow Cover)
- Typical compatible coding: 0..100 valid FSC (%), 205 cloud, 210 water, 255 nodata.
- This aligns with current openamundsen_da snowcover defaults.

SWS/WSM (Sentinel-1 wet snow)
- Common coding in existing rofental examples: 110 wet, 125 dry/no-snow, 200 radar shadow, 210 water.
- Some documents/config examples still show older 1..6 class schemes.
- For rofental-style wet-snow assimilation, prefer 110/125/200/210 conventions.

Practical recommendation
- Validate one sample raster per product before full DA runs.
- If class codes differ, update your observation class mapping in project/setup config accordingly.
"""
    path.write_text(txt, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Standalone HR-WSI downloader for rofental-style openamundsen_da inputs")
    p.add_argument("--phd-root", type=Path, default=DEFAULT_PHD_ROOT)
    p.add_argument("--clms-client-dir", type=Path, default=DEFAULT_CLMS_DIR)
    p.add_argument("--downloader-script", type=Path, default=DEFAULT_DOWNLOADER)
    p.add_argument("--env-yaml", type=Path, default=DEFAULT_ENV_YAML)
    p.add_argument("--rofental-project-yaml", type=Path, default=DEFAULT_ROFENTAL_PROJECT_YAML)
    p.add_argument("--roi-vector", type=Path, default=DEFAULT_ROI)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--conda-exe", type=str, help="Optional path/command for conda executable")
    p.add_argument("--start-date", type=str, help="Override start date YYYY-MM-DD")
    p.add_argument("--end-date", type=str, help="Override end date YYYY-MM-DD")
    p.add_argument("--skip-env-create", action="store_true", help="Skip conda env creation check")
    p.add_argument("--overwrite", action="store_true", help="Overwrite copied output files if already present")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    try:
        if not args.downloader_script.exists():
            raise FileNotFoundError(f"Downloader script not found: {args.downloader_script}")
        if not args.env_yaml.exists():
            raise FileNotFoundError(f"env.yaml not found: {args.env_yaml}")
        if not args.rofental_project_yaml.exists():
            raise FileNotFoundError(f"Project YAML not found: {args.rofental_project_yaml}")
        if not args.roi_vector.exists():
            raise FileNotFoundError(f"ROI vector not found: {args.roi_vector}")

        date_window = _parse_project_dates(args.rofental_project_yaml)
        start_date = args.start_date or date_window.start_date
        end_date = args.end_date or date_window.end_date

        _info(f"Date window: {start_date} -> {end_date}")
        _info(f"ROI: {args.roi_vector}")

        run_tag = f"{start_date.replace('-', '')}_{end_date.replace('-', '')}"
        base_out = args.output_root / "openamundsen_da_examples" / "rofental" / "hrwsi" / run_tag
        raw_out = base_out / "raw_download"
        selected_snowcover = base_out / "obs" / "snowcover"
        selected_wetsnow = base_out / "obs" / "wetsnow"
        manifest_csv = base_out / "manifests" / "download_manifest.csv"
        notes_txt = base_out / "manifests" / "class_mapping_notes.txt"

        conda_exe = _find_conda(args.conda_exe, dry_run=args.dry_run)
        env_name = _read_env_name_from_yaml(args.env_yaml)

        if not args.skip_env_create:
            _ensure_conda_env(conda_exe, args.env_yaml, env_name, dry_run=args.dry_run)
        else:
            _warn("Skipping conda env existence/creation check by user request")

        raw_out.mkdir(parents=True, exist_ok=True)

        download_cmd = [
            str(conda_exe),
            "run",
            "-n",
            env_name,
            "python",
            str(args.downloader_script),
            str(raw_out),
            "-query_and_download",
            "-productType",
            "FSC",
            "SWS",
            "-vector",
            str(args.roi_vector),
            "-dateStart",
            start_date,
            "-dateEnd",
            end_date,
        ]
        _run(download_cmd, cwd=args.clms_client_dir, dry_run=args.dry_run)

        result_dir = raw_out / "result"
        if not args.dry_run and not result_dir.exists():
            raise FileNotFoundError(
                f"Expected downloader output at {result_dir}, but it does not exist."
            )

        rows: list[dict[str, str]] = []
        copied_snow = 0
        copied_wet = 0

        if not args.dry_run:
            for tif in _iter_tifs(result_dir):
                kind = _classify_tif(tif.name)
                if kind is None:
                    continue

                if kind == "snowcover":
                    dst = selected_snowcover / tif.name
                    status = _copy_if_needed(tif, dst, overwrite=args.overwrite)
                    if status == "copied":
                        copied_snow += 1
                else:
                    dst = selected_wetsnow / tif.name
                    status = _copy_if_needed(tif, dst, overwrite=args.overwrite)
                    if status == "copied":
                        copied_wet += 1

                rows.append(
                    {
                        "source_path": str(tif),
                        "file_name": tif.name,
                        "selected_kind": kind,
                        "target_path": str(dst),
                        "status": status,
                    }
                )

            _write_manifest(rows, manifest_csv)
            _write_mapping_notes(notes_txt)

            _info(f"Selected snowcover files copied: {copied_snow}")
            _info(f"Selected wetsnow files copied: {copied_wet}")
            _info(f"Manifest written: {manifest_csv}")
            _info(f"Class notes written: {notes_txt}")
        else:
            _info("Dry-run complete. No files copied.")

        _info(f"Done. Base output: {base_out}")
        return 0

    except subprocess.CalledProcessError as exc:
        _err(f"Command failed with exit code {exc.returncode}")
        return exc.returncode or 1
    except Exception as exc:
        _err(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
