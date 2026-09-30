"""Reuse verified terrain outputs while rebuilding original-region ROI products."""

import json
import os
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

from fram3s_grids.common import ROOTS, RESOLUTIONS, domain, raster_path, save_json, sha256, variant_dir
from fram3s_grids.delivery import prepare_roi_rasters, validate_resolution, finish_manifest, plot_overview, read_window
from fram3s_grids.geometry import build_geometry
from fram3s_grids.reporting import write_collection_readmes


def expected_files(specs: list) -> set:
    """Exact final delivery whitelist, including only necessary CRS companions."""
    paths = {
        f"01-aoi/{name}"
        for name in ("README.txt", "aoi.gpkg", "aoi_overview.qgz", "aoi_overview.png", "aoi_overview.pdf")
    }
    for row in specs:
        for kind in ROOTS:
            path = raster_path(Path("."), kind, row["region"], row["buffer_m"], row["resolution_m"])
            paths.update(str(path.with_suffix(suffix)) for suffix in (".tif", ".asc", ".prj"))
        paths.add(
            str(
                variant_dir(Path("01-aoi"), row["region"], row["buffer_m"])
                / f"{domain(row['region'], row['buffer_m'])}.zip"
            )
        )
    if len(paths) != 1145:
        raise ValueError(f"Expected 1145 unique delivery paths, got {len(paths)}")
    return paths


def finalize(work: Path) -> None:
    """Require the strict whitelist and refresh hashes after native QGIS generation."""
    specs = json.loads((work / "grid_specifications.json").read_text())
    expected = expected_files(specs)
    actual = {str(p.relative_to(work / "output")) for p in (work / "output").rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError(f"Delivery whitelist mismatch: missing={expected - actual}, extra={actual - expected}")
    qgis = json.loads((work / "qgis_validation.json").read_text())
    if qgis["invalid_layers"] or qgis["roi_rasters"] != 75 or qgis["subregions"] != 90:
        raise ValueError("QGIS validation failed")
    finish_manifest(work)
    save_json(
        work / "delivery_validation.json",
        {"files": len(actual), "failures": [], "roi_rule": "original region cell center"},
    )


def cleanup_existing(previous: Path, work: Path) -> None:
    """Build a fresh stage from a completed local build, preserving terrain bytes."""
    if any(work.iterdir()):
        raise FileExistsError("Cleanup requires an empty work directory")
    baseline = {row["path"]: row for row in json.loads((previous / "artifact_manifest.json").read_text())}
    shutil.copytree(previous / "sources/vectors", work / "sources/vectors")
    # Parent rasters remain read-only inputs; this workflow never recalculates them.
    (work / "parents").symlink_to(previous.resolve() / "parents", target_is_directory=True)
    shutil.copy2(previous / "source_manifest.json", work / "source_manifest.json")
    specs = build_geometry(work / "sources/vectors", work / "output", work)
    specs_path = previous / "grid_specifications.json"
    if not specs_path.exists():
        specs_path = previous / "output/01-aoi/grid_specifications.json"
    old_specs = json.loads(specs_path.read_text())
    if specs != old_specs:
        raise ValueError("Cleanup changed existing grid specifications")
    preserved = []
    for spec in specs:
        for kind in ("dem", "lc", "srf", "svf"):
            path = raster_path(Path("."), kind, spec["region"], spec["buffer_m"], spec["resolution_m"])
            for suffix in (".tif", ".asc", ".prj"):
                relative = str(path.with_suffix(suffix))
                source = previous / "output" / relative
                entry = baseline[relative]
                if source.stat().st_size != entry["bytes"] or sha256(source) != entry["sha256"]:
                    raise ValueError(f"Previous output changed: {relative}")
                target = work / "output" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, target)
                preserved.append(entry)
    prepare_roi_rasters(work)
    shapes = gpd.read_file(work / "output/01-aoi/aoi.gpkg", layer="boundaries")
    for res in RESOLUTIONS:
        rows = json.loads((previous / f"delivery_{res}.json").read_text())
        # Refresh ROI-dependent records; retain authoritative terrain paths and geometry.
        for row in rows:
            with rasterio.open(work / "output" / row["paths"]["roi"]) as src:
                roi = src.read(1) == 1
            row["roi_cells"] = int(roi.sum())
            row["roi_raster_area_m2"] = int(roi.sum()) * res**2
            row["missing_roi_cells"] = {}
            for kind in ("dem", "lc", "srf", "svf"):
                values = read_window(work / "parents" / f"{res}m" / f"{kind}.tif", row["bounds"])
                row["missing_roi_cells"][kind] = int(((~np.isfinite(values) | (values == -9999)) & roi).sum())
            quality = read_window(work / "parents" / f"{res}m/terrain_quality.tif", row["bounds"])
            for field, bit in [
                ("roi_srf_edge_cells", 4),
                ("roi_svf_proximity_cells", 8),
                ("roi_partial_source_cells", 2),
            ]:
                row[field] = int(((quality & bit != 0) & roi).sum())
            row["polygon_area_m2"] = float(shapes.loc[shapes.region == row["region"]].geometry.iloc[0].area)
        save_json(work / f"delivery_{res}.json", rows)
        failures = validate_resolution(work, res)
        if failures:
            raise ValueError(failures)
        print(f"Validated {res} m: 15 original-region masks and unchanged terrain", flush=True)
    save_json(
        work / "unchanged_rasters.json", {"files": preserved, "count": len(preserved), "grid_geometry_unchanged": True}
    )
    plot_overview(work / "output")
    write_collection_readmes(work)
    finish_manifest(work)
