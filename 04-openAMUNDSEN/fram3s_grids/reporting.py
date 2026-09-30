"""Diagnostic summaries and user-facing collection documentation."""

import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

from fram3s_grids.common import RESOLUTIONS, ROOTS, raster_path, save_json


def raster_diagnostics(work: Path, report: Path) -> dict:
    """Summarize source versus target classes and inspect the 100 m Euregio layers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report.mkdir(parents=True, exist_ok=True)
    rows = []
    for res in RESOLUTIONS:
        parent = work / "parents" / f"{res}m"
        row = {"resolution_m": res}
        for kind in ("dem", "lc", "srf", "svf"):
            path = parent / f"{kind}.tif"
            if not path.exists():
                continue
            with rasterio.open(path) as src:
                values = src.read(1, masked=True).compressed()
            row[kind] = {
                "minimum": float(values.min()),
                "mean": float(values.mean()),
                "maximum": float(values.max()),
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
            }
            if kind == "lc":
                codes, counts = np.unique(values, return_counts=True)
                row[kind]["area_km2_by_class"] = {
                    str(int(code)): float(count * res**2 / 1e6) for code, count in zip(codes, counts)
                }
        rows.append(row)
    save_json(report / "parent_statistics.json", rows)
    output = work / "output"
    root = output / "01-aoi"
    core = gpd.read_file(root / "aoi_overview.gpkg", layer="boundaries")
    core = core[(core.region == "euregio") & (core.buffer_m == 0)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 12), layout="constrained")
    kinds = [
        ("dem", "Elevation (m)", "terrain"),
        ("lc", "Land-cover class", "tab20"),
        ("srf", "Snow redistribution factor", "RdBu_r"),
        ("svf", "Sky-view factor", "viridis"),
        ("source_coverage", "Fraction covered by source DEM", "viridis"),
        ("terrain_quality", "Terrain-quality bit flags", "magma"),
    ]
    for ax, (kind, title, cmap) in zip(axes.flat, kinds):
        path = (
            raster_path(output, kind, "euregio", 10000, 100)
            if kind in ROOTS
            else (root / "euregio/buffer_10000m/100m/quality" / f"{kind}.tif")
        )
        with rasterio.open(path) as src:
            values = src.read(1, masked=True, out_shape=(src.height // 3, src.width // 3))
            bounds = src.bounds
        kwargs = {}
        if kind == "srf":
            kwargs = {"vmin": 0.3, "vmax": 1.7}
        elif kind in ("svf", "source_coverage"):
            kwargs = {"vmin": 0, "vmax": 1}
        image = ax.imshow(values, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), cmap=cmap, **kwargs)
        core.boundary.plot(ax=ax, color="black", linewidth=0.4)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(labelsize=7)
        fig.colorbar(image, ax=ax, shrink=0.7)
    fig.suptitle(
        "Fram3S processing diagnostics: Euregio, 10 km buffer, 100 m\nBlack outline: original Euregio boundary",
        fontsize=14,
    )
    fig.savefig(report / "processing_diagnostics.png", dpi=150)
    fig.savefig(report / "processing_diagnostics.pdf")
    plt.close(fig)
    land_cover_changes(work, report)
    return {"resolutions": rows}


def land_cover_changes(work: Path, report: Path) -> None:
    """Compare class areas over the source footprint and map aggregation changes."""
    import matplotlib.pyplot as plt
    from rasterio.warp import Resampling, reproject

    source = json.loads((work / "source_manifest.json").read_text())["lc"]
    xmin, ymin, xmax, ymax = source["bounds"]
    area_rows, change_rows = [], []
    fig, axes = plt.subplots(2, 3, figsize=(13, 11), layout="constrained")
    for ax, res in zip(axes.flat, RESOLUTIONS):
        with rasterio.open(work / "parents" / f"{res}m/lc.tif") as target:
            classes = target.read(1)
            transform = target.transform
            # Clip edge-cell areas to the original source rectangle, so both area
            # totals cover exactly the same footprint rather than the padded grid.
            left = transform.c + np.arange(target.width) * res
            top = transform.f - np.arange(target.height) * res
            widths = np.maximum(0, np.minimum(left + res, xmax) - np.maximum(left, xmin))
            heights = np.maximum(0, np.minimum(top, ymax) - np.maximum(top - res, ymin))
            weights = heights[:, None] * widths[None, :] / 1e6
            for code in range(1, 14):
                before = source["class_counts"][str(code)] * 20**2 / 1e6
                after = float(weights[classes == code].sum())
                area_rows.append(
                    {
                        "resolution_m": res,
                        "class": code,
                        "source_area_km2": before,
                        "target_area_km2": after,
                        "change_km2": after - before,
                        "change_percent": 100 * (after - before) / before,
                    }
                )
            nearest = np.full(classes.shape, -9999, dtype="int16")
            with rasterio.open(work / "sources/lc.tif") as original:
                reproject(
                    rasterio.band(original, 1),
                    nearest,
                    dst_transform=transform,
                    dst_crs=target.crs,
                    dst_nodata=-9999,
                    resampling=Resampling.nearest,
                )
            valid = np.isin(classes, np.arange(1, 14)) & np.isin(nearest, np.arange(1, 14))
            changed = (classes != nearest) & valid
            change_rows.append(
                {
                    "resolution_m": res,
                    "comparable_cells": int(valid.sum()),
                    "different_cells": int(changed.sum()),
                    "different_percent": 100 * float(changed.sum()) / int(valid.sum()),
                }
            )
            # Use the same 1 km display cells and fraction scale at every resolution.
            stride = 1000 // res
            padded_shape = tuple(int(np.ceil(size / stride)) * stride for size in valid.shape)
            counts = []
            for mask in (changed, valid):
                displayed = np.zeros(padded_shape, dtype="uint8")
                displayed[: target.height, : target.width] = mask
                counts.append(
                    displayed.reshape(padded_shape[0] // stride, stride, padded_shape[1] // stride, stride).sum(
                        axis=(1, 3)
                    )
                )
            fraction = np.divide(counts[0], counts[1], out=np.full(counts[0].shape, np.nan), where=counts[1] > 0)
            image = ax.imshow(fraction, cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"{res} m: {change_rows[-1]['different_percent']:.1f}% of comparable cells")
            ax.set_axis_off()
    axes.flat[-1].set_axis_off()
    fig.colorbar(
        image,
        cax=axes.flat[-1].inset_axes([0.12, 0.5, 0.76, 0.07]),
        orientation="horizontal",
        label="Fraction of target cells differing\nwithin each 1 km display cell",
    )
    fig.suptitle(
        "Dominant-area land cover versus nearest source class\nCommon 1 km display grid and fraction scale"
    )
    fig.savefig(report / "land_cover_changes.png", dpi=150)
    fig.savefig(report / "land_cover_changes.pdf")
    plt.close(fig)
    for name, rows in (("land_cover_class_areas", area_rows), ("land_cover_changed_cells", change_rows)):
        with (report / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_collection_readmes(work: Path) -> None:
    """Explain active paths, source preservation and model-grid conventions."""
    output = work / "output"
    for kind, root in ROOTS.items():
        text = (
            "Fram3S aligned spatial inputs\n\n"
            "Regions: euregio, tyrol (including East Tyrol), north_tyrol, south_tyrol, trentino.\n"
            "Structure: <region>/buffer_00000m|buffer_05000m|buffer_10000m/<resolution>m/.\n"
            "Resolutions: 50, 100, 250, 500, 1000 m. CRS: EPSG:25832.\n"
            "Each region/buffer has a common rectangle snapped outward to 1 km boundaries.\n"
            "Buffers expand the actual region polygon; ROI=1 uses the cell-center rule.\n"
            "GeoTIFF and model ASCII have the same grid and values. Coordinates are aligned;\n"
            "continuous raster values retain their fractional precision.\n\n"
            "The full inventory, QGIS project, overview and source manifest are in 01-aoi.\n"
            "Legacy generated datasets are retained in timestamped archive folders.\n"
            "Raw land-cover, forest and glacier source collections remain available.\n"
            "Original source inputs and archive hashes are documented in the publication record\n"
            "under workspace/context/plans/open/fram3s_grid_alignment_processing.\n\n"
            "DEM uses valid-source area averages; land cover uses greatest covered class area.\n"
            "SRF follows calculateSRF.py at each resolution, normalized once on its parent.\n"
            "SVF is calculated on the shared parent, then cropped.\n"
            "The 10 km variants include terrain-edge limitations. Consult AOI quality rasters:\n"
            "flags 1=missing DEM, 2=partial source coverage, 4=SRF context proximity,\n"
            "8=SVF proximity indicator (not a quantitative error bound).\n"
        )
        if kind == "roi":
            text += (
                "\nOpen aoi_overview.qgz in QGIS. All paths are relative.\n"
                "Subregions: subregions/subregions_25832.gpkg, 90 polygons with original attributes.\n"
                "Province boundaries are derived from the archived avalanche-report regions.\n"
                "North Tyrol is separate from Tyrol including East Tyrol.\n"
                "Vector GeoPackages contain boundary, extent and ROI partition layers.\n"
                "Shapefile ZIPs contain the two-feature ROI partition.\n"
                "For Kathi use kathi_north_tyrol_100m: original North Tyrol as the core ROI,\n"
                "with the larger 5 km context grid. Its ROI differs from the true buffered ROI.\n"
            )
        (output / root / "README.txt").write_text(text)
