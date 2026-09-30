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
    specs = json.loads((root / "grid_specifications.json").read_text())
    boundaries = gpd.read_file(root / "aoi_overview.gpkg", layer="boundaries")
    rows = []
    for spec in specs:
        res = spec["resolution_m"]
        shape, transform = grid(spec["bounds"], res)
        geom = boundaries.loc[
            (boundaries.region == spec["region"]) & (boundaries.buffer_m == spec["buffer_m"])
        ].geometry.iloc[0]
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
    specs = json.loads((root / "grid_specifications.json").read_text())
    boundaries = gpd.read_file(root / "aoi_overview.gpkg", layer="boundaries")
    for spec in specs:
        path = raster_path(output, "roi", spec["region"], spec["buffer_m"], spec["resolution_m"])
        if path.exists():
            continue
        shape, transform = grid(spec["bounds"], spec["resolution_m"])
        geom = boundaries.loc[
            (boundaries.region == spec["region"]) & (boundaries.buffer_m == spec["buffer_m"])
        ].geometry.iloc[0]
        roi = rasterize([(geom, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8")
        write_tif(path, roi, transform, nodata=255)
        export_ascii(path)
    core_path = root / "kathi_north_tyrol_100m/roi_north_tyrol_core_100.tif"
    if not core_path.exists():
        shape, transform = grid((578000, 5175000, 784000, 5298000), 100)
        geom = boundaries.loc[(boundaries.region == "north_tyrol") & (boundaries.buffer_m == 0)].geometry.iloc[0]
        values = rasterize([(geom, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8")
        write_tif(core_path, values, transform, nodata=255)
        export_ascii(core_path)
        gpd.GeoDataFrame({"ROI": [1]}, geometry=[geom], crs=CRS).to_file(
            core_path.parent / "north_tyrol_core.gpkg", driver="GPKG"
        )


def deliver_resolution(work: Path, resolution: int) -> list:
    """Export all 15 variants from a completed parent, with model ASCII companions."""
    output = work / "output"
    aoi = output / "01-aoi"
    specs = json.loads((aoi / "grid_specifications.json").read_text())
    boundaries = gpd.read_file(aoi / "aoi_overview.gpkg", layer="boundaries")
    parent = work / "parents" / f"{resolution}m"
    results = []
    for spec in specs:
        if spec["resolution_m"] != resolution:
            continue
        name, buffer_m, bounds = spec["region"], spec["buffer_m"], spec["bounds"]
        shape, transform = grid(bounds, resolution)
        polygon = boundaries.loc[(boundaries.region == name) & (boundaries.buffer_m == buffer_m)].geometry.iloc[0]
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
        quality_dir = variant_dir(aoi, name, buffer_m) / f"{resolution}m" / "quality"
        for kind in ("source_coverage", "terrain_context_distance", "terrain_quality"):
            values = read_window(parent / f"{kind}.tif", bounds)
            write_tif(quality_dir / f"{kind}.tif", values, transform, nodata=255 if kind == "terrain_quality" else -1)
            if kind == "terrain_quality":
                row["roi_srf_edge_cells"] = int(((values & 4 != 0) & (roi == 1)).sum())
                row["roi_svf_proximity_cells"] = int(((values & 8 != 0) & (roi == 1)).sum())
                row["roi_partial_source_cells"] = int(((values & 2 != 0) & (roi == 1)).sum())
        if name == "north_tyrol" and buffer_m == 5000 and resolution == 100:
            core = boundaries.loc[(boundaries.region == name) & (boundaries.buffer_m == 0)].geometry.iloc[0]
            core_roi = rasterize([(core, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8")
            path = aoi / "kathi_north_tyrol_100m" / "roi_north_tyrol_core_100.tif"
            write_tif(path, core_roi, transform, nodata=255)
            export_ascii(path)
            gpd.GeoDataFrame({"ROI": [1]}, geometry=[core], crs=CRS).to_file(
                path.parent / "north_tyrol_core.gpkg", driver="GPKG"
            )
        results.append(row)
    save_json(work / f"delivery_{resolution}.json", results)
    return results


def validate_resolution(work: Path, resolution: int) -> list:
    """Read exports back, check ROI coverage, compare source parent crops and value ranges."""
    output = work / "output"
    rows = json.loads((work / f"delivery_{resolution}.json").read_text())
    boundaries = gpd.read_file(output / "01-aoi/aoi_overview.gpkg", layer="boundaries")
    failures = []
    for row in rows:
        shape, transform = grid(row["bounds"], resolution)
        with rasterio.open(output / row["paths"]["roi"]) as src:
            roi = src.read(1) == 1
        geom = boundaries.loc[
            (boundaries.region == row["region"]) & (boundaries.buffer_m == row["buffer_m"])
        ].geometry.iloc[0]
        expected_roi = rasterize([(geom, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)
        if not np.array_equal(roi, expected_roi):
            failures.append(f"ROI geometry mismatch: {row['domain']}")
        for kind, relative in row["paths"].items():
            path = output / relative
            with rasterio.open(path) as src:
                values = src.read(1)
                if src.shape != shape or src.transform != transform or src.crs.to_epsg() != 25832:
                    failures.append(f"Geometry metadata: {relative}")
            with rasterio.open(path.with_suffix(".asc")) as src:
                ascii_values = src.read(1)
                if src.transform != transform or src.crs.to_epsg() != 25832:
                    failures.append(f"ASCII geometry metadata: {relative}")
                if not np.allclose(values, ascii_values, rtol=1e-7, atol=1e-6):
                    failures.append(f"ASCII roundtrip: {relative}")
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
    """Plot actual region polygons, buffers, extents and all subregion boundaries."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    root = output / "01-aoi"
    shapes = gpd.read_file(root / "aoi_overview.gpkg", layer="boundaries")
    extents = gpd.read_file(root / "aoi_overview.gpkg", layer="grid_extents")
    sub = gpd.read_file(root / "subregions/subregions_25832.gpkg")
    fig, axes = plt.subplots(2, 3, figsize=(17, 12), layout="constrained")
    labels = {
        "euregio": "Euregio",
        "tyrol": "Tyrol (including East Tyrol)",
        "north_tyrol": "North Tyrol",
        "south_tyrol": "South Tyrol",
        "trentino": "Trentino",
    }
    colors = {0: "#177e89", 5000: "#e79532", 10000: "#b04a82"}
    for ax, (name, title) in zip(axes.flat, labels.items()):
        selected = shapes[shapes.region == name]
        for buffer_m in (10000, 5000, 0):
            selected[selected.buffer_m == buffer_m].plot(
                ax=ax, color=colors[buffer_m], alpha=0.14, edgecolor=colors[buffer_m]
            )
            extents[
                (extents.region == name) & (extents.buffer_m == buffer_m) & (extents.resolution_m == 100)
            ].boundary.plot(ax=ax, color=colors[buffer_m], linewidth=0.8, linestyle="--")
        limits = ax.get_xlim(), ax.get_ylim()
        sub.boundary.plot(ax=ax, color="#4e5559", linewidth=0.25, alpha=0.6)
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_title(title, fontweight="bold")
        ax.set_aspect("equal")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(axis="both", labelsize=8)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
    ax = axes.flat[-1]
    ax.set_title("Common origin and 1 km envelope", fontweight="bold")
    for spacing, color, lw in [(1000, "black", 2), (250, "#b04a82", 1.2), (100, "#177e89", 0.6)]:
        for x in range(0, 1001, spacing):
            ax.plot([x, x], [0, 1000], color=color, lw=lw, alpha=0.7)
            ax.plot([0, 1000], [x, x], color=color, lw=lw, alpha=0.7)
    ax.set_aspect("equal")
    ax.set_xlim(-30, 1030)
    ax.set_ylim(-30, 1030)
    ax.set_xlabel("Distance from a shared 1 km grid corner (m)")
    ax.legend(
        handles=[
            Line2D([0], [0], color=color, label=f"{res:,} m", lw=1.5)
            for res, color in [(100, "#177e89"), (250, "#b04a82"), (1000, "black")]
        ],
        loc="upper right",
        framealpha=0.95,
        fontsize=9,
    )
    handles = [
        Patch(
            facecolor=colors[b],
            alpha=0.4,
            label=f"{'Original boundary' if b == 0 else str(b // 1000) + ' km true buffer'}",
        )
        for b in colors
    ]
    handles += [
        Line2D([0], [0], ls="--", color="gray", label="Shared outer rectangle at all resolutions"),
        Line2D([0], [0], lw=0.5, color="#4e5559", label="Avalanche-report subregions"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=10)
    fig.suptitle(
        "Fram3S regions, true polygon buffers and aligned model grids\nETRS89 / UTM 32N · 50, 100, 250, 500 and 1,000 m",
        fontsize=17,
    )
    fig.savefig(root / "aoi_overview.png", dpi=170)
    fig.savefig(root / "aoi_overview.pdf")
    plt.close(fig)


def validate_vectors(work: Path) -> dict:
    """Verify exported geometry, preserved subregion attributes and region membership."""
    from fram3s_grids.geometry import read_regions
    from fram3s_grids.common import envelope
    from shapely.geometry import box

    output = work / "output/01-aoi"
    source = work / "sources/vectors"
    # prepare_sources stages raster inputs; the runner may use any read-only source root.
    if not source.exists():
        raise FileNotFoundError("Vector validation requires the source snapshot at work/sources/vectors")
    regions, _ = read_regions(source)
    boundaries = gpd.read_file(output / "aoi_overview.gpkg", layer="boundaries")
    for _, row in boundaries.iterrows():
        expected = regions[row.region].buffer(int(row.buffer_m), quad_segs=64) if row.buffer_m else regions[row.region]
        rectangle = box(*envelope(expected.bounds))
        if regions[row.region].difference(expected).area > 0.001:
            raise ValueError("Buffer does not contain the original region")
        if expected.symmetric_difference(row.geometry).area > 0.001:
            raise ValueError("Exported boundary changed")
        folder = variant_dir(output, row.region, int(row.buffer_m)) / "vectors"
        prefix = domain(row.region, int(row.buffer_m))
        partition = gpd.read_file(folder / f"{prefix}.gpkg", layer="roi_partition")
        shp = gpd.read_file("zip://" + str(folder / f"{prefix}.zip"))
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
        if partition.geometry.iloc[0].intersection(partition.geometry.iloc[1]).area > 0.001:
            raise ValueError("Exported partition overlaps")
    original = (
        gpd.read_file(source / "01-aoi/SUBREGIONS/raw/subregions_avalanche_report_4326_raw.gpkg")
        .to_crs(CRS)
        .set_index("id")
        .sort_index()
    )
    exported = gpd.read_file(output / "subregions/subregions_25832.gpkg").set_index("id").sort_index()
    if len(exported) != 90 or not original.index.equals(exported.index):
        raise ValueError("Subregion identifiers changed")
    import pandas as pd

    pd.testing.assert_frame_equal(
        original.drop(columns="geometry"), exported.drop(columns="geometry"), check_dtype=False
    )
    delta = max(a.symmetric_difference(b).area for a, b in zip(original.geometry, exported.geometry))
    if delta > 0.001:
        raise ValueError("Subregion geometry changed")
    selected = json.loads((output / "geometry_validation.json").read_text())["north_tyrol_subregion_ids"]
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
    save_json(output / "vector_validation.json", result)
    return result


def kathi_package(output: Path) -> None:
    """Supply core ROI geometry and explicit paths to the complete context-grid stack."""
    import shutil
    import zipfile
    from shapely.geometry import box

    aoi = output / "01-aoi"
    folder = aoi / "kathi_north_tyrol_100m"
    shapes = gpd.read_file(aoi / "aoi_overview.gpkg", layer="boundaries")
    core = shapes.loc[(shapes.region == "north_tyrol") & (shapes.buffer_m == 0)].geometry.iloc[0]
    rectangle = box(578000, 5175000, 784000, 5298000)
    gpkg = folder / "north_tyrol_core.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    gpd.GeoDataFrame({"ROI": [1]}, geometry=[core], crs=CRS).to_file(gpkg, layer="north_tyrol_core", driver="GPKG")
    gpd.GeoDataFrame({"context_m": [5000]}, geometry=[rectangle], crs=CRS).to_file(gpkg, layer="extent", driver="GPKG")
    partition = gpd.GeoDataFrame({"ROI": [1, 0]}, geometry=[core, rectangle.difference(core)], crs=CRS)
    partition.to_file(gpkg, layer="roi_partition", driver="GPKG")
    shpdir = folder / "shapefile"
    shpdir.mkdir(exist_ok=True)
    partition.to_file(shpdir / "north_tyrol_core.shp", driver="ESRI Shapefile", encoding="UTF-8")
    with zipfile.ZipFile(folder / "north_tyrol_core.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(shpdir.iterdir()):
            archive.write(path, path.name)
    # A self-contained model-grid handoff uses one domain identifier and the core ROI.
    grids = folder / "grids"
    grids.mkdir(exist_ok=True)
    for kind in ROOTS:
        src = (
            folder / "roi_north_tyrol_core_100.tif"
            if kind == "roi"
            else raster_path(output, kind, "north_tyrol", 5000, 100)
        )
        for suffix in (".tif", ".asc", ".prj"):
            original = src.with_suffix(suffix)
            shutil.copy2(original, grids / f"{kind}_north_tyrol_core_100{suffix}")
    (folder / "README.txt").write_text(
        "North Tyrol core ROI with 5 km surrounding context\n"
        "CRS: EPSG:25832; resolution: 100 m; domain: north_tyrol_core.\n"
        "Bounds: 578000, 5175000 : 784000, 5298000. Columns: 2060; rows: 1230.\n"
        "The model ROI is the original North Tyrol boundary, excluding East Tyrol.\n"
        "The 5 km buffered polygon defines the surrounding grid envelope only for this handoff.\n"
        "GeoPackage includes the original boundary, extent and ROI=1/ROI=0 partition.\n"
        "Shapefile ZIP contains that partition. The grids directory is self-contained.\n"
        "The general collection's north_tyrol/buffer_05000m ROI instead includes the true buffer.\n"
    )


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
    kathi_package(work / "output")
    validate_kathi_package(work)
    root = work / "output/01-aoi"
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


def validate_kathi_package(work: Path) -> None:
    """Check the self-contained handoff against the already validated regional stack."""
    folder = work / "output/01-aoi/kathi_north_tyrol_100m"
    shape, transform = grid((578000, 5175000, 784000, 5298000), 100)
    core = gpd.read_file(folder / "north_tyrol_core.gpkg", layer="north_tyrol_core").geometry.iloc[0]
    expected_roi = rasterize([(core, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8")
    for kind in ROOTS:
        expected = (
            expected_roi
            if kind == "roi"
            else read_window(work / "parents/100m" / f"{kind}.tif", (578000, 5175000, 784000, 5298000))
        )
        for suffix in ("tif", "asc"):
            with rasterio.open(folder / "grids" / f"{kind}_north_tyrol_core_100.{suffix}") as src:
                values = src.read(1)
                if src.shape != shape or src.transform != transform or src.crs.to_epsg() != 25832:
                    raise ValueError(f"Kathi geometry mismatch: {kind}.{suffix}")
                if not np.allclose(values, expected, rtol=1e-7, atol=1e-6):
                    raise ValueError(f"Kathi grid values differ: {kind}.{suffix}")
                if kind != "roi" and ((values == NODATA) | ~np.isfinite(values))[expected_roi == 1].any():
                    raise ValueError(f"Kathi ROI has missing {kind} values")
    partition = gpd.read_file("zip://" + str(folder / "north_tyrol_core.zip"))
    if set(partition.ROI) != {0, 1} or not partition.geometry.is_valid.all():
        raise ValueError("Invalid Kathi Shapefile partition")
    if partition.loc[partition.ROI == 1].geometry.iloc[0].symmetric_difference(core).area > 0.001:
        raise ValueError("Kathi Shapefile differs from original North Tyrol")
    save_json(
        folder / "validation.json",
        {"layers": 5, "columns": shape[1], "rows": shape[0], "core_roi_cells": int(expected_roi.sum()), "failures": []},
    )
