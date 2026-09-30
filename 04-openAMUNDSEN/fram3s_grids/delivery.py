"""Regional crops, independent validation, cartography and artifact manifests."""

import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds

from fram3s_grids.common import (
    CRS,
    NODATA,
    ROOTS,
    domain,
    export_ascii,
    grid,
    raster_path,
    save_json,
    sha256,
    variant_dir,
    write_tif,
)


def read_window(path: Path, bounds: tuple) -> np.ndarray:
    """Read a crop on the shared integer grid; reject fractional window offsets."""
    with rasterio.open(path) as src:
        window = from_bounds(*bounds, transform=src.transform)
        if any(abs(value - round(value)) > 1e-8 for value in window.flatten()):
            raise ValueError(f"Nonaligned crop {path}: {window}")
        return src.read(1, window=window.round_offsets().round_lengths())


def source_coverage_preflight(work: Path) -> list:
    """Check missing DEM and class values at every target ROI before terrain processing."""
    root = work / "output/01-aoi"
    specs = json.loads((work / "grid_specifications.json").read_text())
    boundaries = gpd.read_file(root / "aoi.gpkg", layer="boundaries")
    rows = []
    for spec in specs:
        res = spec["resolution_m"]
        shape, transform = grid(spec["bounds"], res)
        geom = boundaries.loc[boundaries.region == spec["region"]].geometry.iloc[0]
        roi = rasterize([(geom, 1)], out_shape=shape, transform=transform, dtype="uint8").astype(bool)
        row = {"region": spec["region"], "buffer_m": spec["buffer_m"], "resolution_m": res}
        for kind in ("dem", "lc"):
            values = read_window(work / "parents" / f"{res}m" / f"{kind}.tif", spec["bounds"])
            row[f"missing_{kind}_roi_cells"] = int(((~np.isfinite(values) | (values == NODATA)) & roi).sum())
        rows.append(row)
    save_json(work / "input_coverage_preflight.json", rows)
    return [row for row in rows if row["missing_dem_roi_cells"] or row["missing_lc_roi_cells"]]


def prepare_roi_rasters(work: Path) -> None:
    """Make the geometry-only masks available for QGIS while terrain is processing."""
    output = work / "output"
    root = output / "01-aoi"
    specs = json.loads((work / "grid_specifications.json").read_text())
    boundaries = gpd.read_file(root / "aoi.gpkg", layer="boundaries")
    for spec in specs:
        path = raster_path(output, "roi", spec["region"], spec["buffer_m"], spec["resolution_m"])
        if path.exists():
            continue
        shape, transform = grid(spec["bounds"], spec["resolution_m"])
        geom = boundaries.loc[boundaries.region == spec["region"]].geometry.iloc[0]
        roi = rasterize([(geom, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8")
        write_tif(path, roi, transform, nodata=255)
        export_ascii(path)


def deliver_resolution(work: Path, resolution: int) -> list:
    """Export all 15 variants from a completed parent, with model ASCII companions."""
    output = work / "output"
    aoi = output / "01-aoi"
    specs = json.loads((work / "grid_specifications.json").read_text())
    boundaries = gpd.read_file(aoi / "aoi.gpkg", layer="boundaries")
    parent = work / "parents" / f"{resolution}m"
    results = []
    for spec in specs:
        if spec["resolution_m"] != resolution:
            continue
        name, buffer_m, bounds = spec["region"], spec["buffer_m"], spec["bounds"]
        shape, transform = grid(bounds, resolution)
        polygon = boundaries.loc[boundaries.region == name].geometry.iloc[0]
        roi = rasterize([(polygon, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8", all_touched=False)
        row = {
            **spec,
            "roi_cells": int(roi.sum()),
            "roi_raster_area_m2": int(roi.sum()) * resolution**2,
            "polygon_area_m2": polygon.area,
            "paths": {},
            "missing_roi_cells": {},
        }
        for kind in ROOTS:
            values = roi if kind == "roi" else read_window(parent / f"{kind}.tif", bounds)
            if values.shape != shape:
                raise ValueError(f"Parent does not cover {name}, {buffer_m}, {resolution}")
            if kind != "roi":
                missing = (~np.isfinite(values) | (values == NODATA)) & (roi == 1)
                row["missing_roi_cells"][kind] = int(missing.sum())
            path = raster_path(output, kind, name, buffer_m, resolution)
            write_tif(path, values, transform, nodata=255 if kind == "roi" else NODATA)
            export_ascii(path)
            row["paths"][kind] = str(path.relative_to(output))
        quality = read_window(parent / "terrain_quality.tif", bounds)
        row["roi_srf_edge_cells"] = int(((quality & 4 != 0) & (roi == 1)).sum())
        row["roi_svf_proximity_cells"] = int(((quality & 8 != 0) & (roi == 1)).sum())
        row["roi_partial_source_cells"] = int(((quality & 2 != 0) & (roi == 1)).sum())
        results.append(row)
    save_json(work / f"delivery_{resolution}.json", results)
    return results


def validate_resolution(work: Path, resolution: int) -> list:
    """Read exports back, check ROI coverage, compare source parent crops and value ranges."""
    output = work / "output"
    rows = json.loads((work / f"delivery_{resolution}.json").read_text())
    boundaries = gpd.read_file(output / "01-aoi/aoi.gpkg", layer="boundaries")
    failures = []
    for row in rows:
        shape, transform = grid(row["bounds"], resolution)
        with rasterio.open(output / row["paths"]["roi"]) as src:
            roi = src.read(1) == 1
        geom = boundaries.loc[boundaries.region == row["region"]].geometry.iloc[0]
        expected_roi = rasterize([(geom, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)
        if not np.array_equal(roi, expected_roi):
            failures.append(f"ROI geometry mismatch: {row['domain']}")
        for kind, relative in row["paths"].items():
            path = output / relative
            with rasterio.open(path) as src:
                values = src.read(1)
                if (
                    src.shape != shape
                    or src.transform != transform
                    or src.crs.to_epsg() != 25832
                    or src.nodata != (255 if kind == "roi" else NODATA)
                ):
                    failures.append(f"Geometry metadata: {relative}")
            with rasterio.open(path.with_suffix(".asc")) as src:
                ascii_values = src.read(1)
                if (
                    src.shape != shape
                    or src.transform != transform
                    or src.crs.to_epsg() != 25832
                    or src.nodata != (255 if kind == "roi" else NODATA)
                ):
                    failures.append(f"ASCII geometry metadata: {relative}")
                if not np.allclose(values, ascii_values, rtol=1e-7, atol=1e-6):
                    failures.append(f"ASCII roundtrip: {relative}")
            if kind == "roi" and not np.isin(values, [0, 1]).all():
                failures.append(f"Invalid ROI values: {relative}")
            valid = np.isfinite(values) & (values != (255 if kind == "roi" else NODATA))
            if not valid[roi].all():
                failures.append(f"Missing {int((~valid & roi).sum())} ROI cells: {relative}")
            if kind == "lc" and not np.isin(values[valid], np.arange(1, 14)).all():
                failures.append(f"Invalid land-cover class: {relative}")
            if kind == "srf" and not (values[roi] > 0).all():
                failures.append(f"Nonpositive SRF: {relative}")
            if kind == "svf" and not ((values[roi] >= 0) & (values[roi] <= 1)).all():
                failures.append(f"SVF out of range: {relative}")
            if kind != "roi":
                expected = read_window(work / "parents" / f"{resolution}m" / f"{kind}.tif", row["bounds"])
                if not np.array_equal(values, expected):
                    failures.append(f"Parent crop mismatch: {relative}")
    save_json(work / f"validation_{resolution}.json", {"stacks": len(rows), "failures": failures})
    return failures


def plot_overview(output: Path) -> None:
    """Show original region ROIs and their three terrain context rectangles."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    root = output / "01-aoi"
    shapes = gpd.read_file(root / "aoi.gpkg", layer="boundaries")
    extents = gpd.read_file(root / "aoi.gpkg", layer="grid_extents")
    subregions = gpd.read_file(root / "aoi.gpkg", layer="subregions")
    fig = plt.figure(figsize=(17, 12), layout="constrained")
    gs = fig.add_gridspec(2, 6)
    axes = [fig.add_subplot(gs[0, i : i + 2]) for i in (0, 2, 4)]
    axes += [fig.add_subplot(gs[1, i : i + 2]) for i in (1, 3)]
    labels = {
        "euregio": "Euregio",
        "tyrol": "Tyrol (including East Tyrol)",
        "north_tyrol": "North Tyrol",
        "south_tyrol": "South Tyrol",
        "trentino": "Trentino",
    }
    colors = {0: "#177e89", 5000: "#e79532", 10000: "#b04a82"}
    for ax, (name, title) in zip(axes, labels.items()):
        shapes[shapes.region == name].plot(ax=ax, color="#c5dee1", edgecolor="#216b80", linewidth=0.7)
        core = shapes.loc[shapes.region == name].geometry.iloc[0]
        clipped = subregions.geometry.intersection(core)
        clipped[~clipped.is_empty & (clipped.area > 0)].boundary.plot(
            ax=ax, color="#4e5559", linewidth=0.35, alpha=0.75
        )
        for buffer_m in (10000, 5000, 0):
            extents[
                (extents.region == name) & (extents.buffer_m == buffer_m) & (extents.resolution_m == 100)
            ].boundary.plot(ax=ax, color=colors[buffer_m], linewidth=0.9, linestyle="--")
        ax.set_title(title, fontweight="bold")
        ax.set_aspect("equal")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(axis="both", labelsize=8)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
    handles = [Patch(facecolor="#c5dee1", edgecolor="#216b80", label="Original region / ROI")]
    handles += [
        Line2D([0], [0], ls="--", color=color, label=f"{b // 1000} km terrain context") for b, color in colors.items()
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=10)
    fig.suptitle(
        "Fram3S regions and terrain context extents\nETRS89 / UTM 32N · 50, 100, 250, 500 and 1,000 m", fontsize=17
    )
    fig.savefig(root / "aoi_overview.png", dpi=170)
    fig.savefig(root / "aoi_overview.pdf")
    plt.close(fig)


def validate_vectors(work: Path) -> dict:
    """Verify exported geometry, preserved subregion attributes and region membership."""
    from fram3s_grids.geometry import read_regions
    from fram3s_grids.geometry import variants
    from shapely.geometry import box

    output = work / "output/01-aoi"
    source = work / "sources/vectors"
    # prepare_sources stages raster inputs; the runner may use any read-only source root.
    if not source.exists():
        raise FileNotFoundError("Vector validation requires the source snapshot at work/sources/vectors")
    regions, _ = read_regions(source)
    boundaries = gpd.read_file(output / "aoi.gpkg", layer="boundaries")
    partitions = gpd.read_file(output / "aoi.gpkg", layer="roi_partitions")
    for name, buffer_m, expected, bounds in variants(regions):
        rectangle = box(*bounds)
        boundary = boundaries.loc[boundaries.region == name].geometry.iloc[0]
        if expected.symmetric_difference(boundary).area > 0.001:
            raise ValueError("Exported boundary changed")
        partition = partitions[(partitions.region == name) & (partitions.buffer_m == buffer_m)]
        shp = gpd.read_file("zip://" + str(variant_dir(output, name, buffer_m) / f"{domain(name, buffer_m)}.zip"))
        for item in (partition, shp):
            if (
                set(item.ROI) != {0, 1}
                or len(item) != 2
                or item.crs.to_epsg() != 25832
                or not item.geometry.is_valid.all()
            ):
                raise ValueError("Invalid vector partition export")
            if item.loc[item.ROI == 1].geometry.iloc[0].symmetric_difference(expected).area > 0.001:
                raise ValueError("Shapefile/GPKG ROI differs")
            if item.geometry.unary_union.symmetric_difference(rectangle).area > 0.001:
                raise ValueError("Exported partition does not cover its grid rectangle")
            if item.geometry.iloc[0].intersection(item.geometry.iloc[1]).area > 0.001:
                raise ValueError("Exported partition overlaps")
    original = (
        gpd.read_file(source / "01-aoi/SUBREGIONS/raw/subregions_avalanche_report_4326_raw.gpkg")
        .to_crs(CRS)
        .set_index("id")
        .sort_index()
    )
    exported = gpd.read_file(output / "aoi.gpkg", layer="subregions").set_index("id").sort_index()
    if len(exported) != 90 or not original.index.equals(exported.index):
        raise ValueError("Subregion identifiers changed")
    import pandas as pd

    pd.testing.assert_frame_equal(
        original.drop(columns="geometry"), exported.drop(columns="geometry"), check_dtype=False
    )
    delta = max(a.symmetric_difference(b).area for a, b in zip(original.geometry, exported.geometry))
    if delta > 0.001:
        raise ValueError("Subregion geometry changed")
    selected = json.loads((work / "geometry_validation.json").read_text())["north_tyrol_subregion_ids"]
    if not (exported.loc[selected].province == "tyrol").all():
        raise ValueError("North Tyrol subregion selection contains another province")
    expected_ids = original.loc[
        (original.province == "tyrol") & (original.geometry.intersection(regions["north_tyrol"]).area > 0)
    ].index
    if set(selected) != set(expected_ids):
        raise ValueError("North Tyrol subregion selection mismatch")
    result = {
        "polygon_variants": 15,
        "subregions": 90,
        "north_tyrol_subregions": len(selected),
        "maximum_subregion_symmetric_difference_m2": delta,
        "attributes_preserved": True,
    }
    save_json(work / "vector_validation.json", result)
    return result


def finish_manifest(work: Path) -> None:
    """Require complete validated delivery and collect auditable per-stack summaries."""
    from fram3s_grids.common import RESOLUTIONS

    rows = []
    for resolution in RESOLUTIONS:
        validation = json.loads((work / f"validation_{resolution}.json").read_text())
        if validation["failures"]:
            raise ValueError(f"Validation failed at {resolution} m: {validation['failures']}")
        rows.extend(json.loads((work / f"delivery_{resolution}.json").read_text()))
    if len(rows) != 75:
        raise ValueError("Expected exactly 75 stacks")
    validate_vectors(work)
    validate_context_rois(work)
    root = work
    save_json(root / "collection_manifest.json", rows)
    flat = [
        {
            **{k: v for k, v in row.items() if k not in ("paths", "missing_roi_cells", "bounds")},
            **dict(zip(("xmin", "ymin", "xmax", "ymax"), row["bounds"])),
            **{kind + "_path": path for kind, path in row["paths"].items()},
        }
        for row in rows
    ]
    with (root / "grid_inventory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    save_json(root / "source_manifest.json", json.loads((work / "source_manifest.json").read_text()))
    artifacts = [
        {"path": str(p.relative_to(work / "output")), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted((work / "output").rglob("*"))
        if p.is_file()
    ]
    save_json(work / "artifact_manifest.json", artifacts)


def validate_context_rois(work: Path) -> None:
    """Compare geographic core cells across context variants at every resolution."""
    from fram3s_grids.common import REGIONS, RESOLUTIONS

    rows = []
    for name in REGIONS:
        for res in RESOLUTIONS:
            with rasterio.open(raster_path(work / "output", "roi", name, 0, res)) as src:
                core, bounds = src.read(1), src.bounds
            for buffer_m in (5000, 10000):
                path = raster_path(work / "output", "roi", name, buffer_m, res)
                if not np.array_equal(core, read_window(path, bounds)):
                    raise ValueError(f"Context changes core cells: {name}, {res}, {buffer_m}")
                with rasterio.open(path) as src:
                    if int(src.read(1).sum()) != int(core.sum()):
                        raise ValueError(f"Context adds ROI cells: {name}, {res}, {buffer_m}")
            rows.append({"region": name, "resolution_m": res, "core_cells": int(core.sum())})
    save_json(work / "roi_context_validation.json", {"failures": [], "comparisons": 50, "cores": rows})
