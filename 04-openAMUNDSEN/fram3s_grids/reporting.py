"""Diagnostic summaries and user-facing collection documentation."""

import csv
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

from fram3s_grids.common import RESOLUTIONS, save_json


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
    core = gpd.read_file(root / "aoi.gpkg", layer="boundaries")
    core = core[core.region == "euregio"]
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
        path = work / "parents/100m" / f"{kind}.tif"
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
        "Fram3S processing diagnostics: shared 100 m parent\nBlack outline: original Euregio boundary",
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
    fig.suptitle("Dominant-area land cover versus nearest source class\nCommon 1 km display grid and fraction scale")
    fig.savefig(report / "land_cover_changes.png", dpi=150)
    fig.savefig(report / "land_cover_changes.pdf")
    plt.close(fig)
    for name, rows in (("land_cover_class_areas", area_rows), ("land_cover_changed_cells", change_rows)):
        with (report / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_collection_readmes(work: Path) -> None:
    """Write the single delivery guide; provenance and diagnostics stay outside delivery."""
    (work / "output/01-aoi/README.txt").write_text(
        "Fram3S aligned spatial inputs\n\n"
        "CRS: ETRS89 / UTM zone 32N (EPSG:25832). Resolutions: 50, 100, 250, 500, 1000 m.\n"
        "Regions: Euregio, Tyrol (North + East Tyrol), North Tyrol, South Tyrol, Trentino.\n"
        "Paths: <layer>/<region>/buffer_00000m|buffer_05000m|buffer_10000m/<resolution>m/.\n"
        "Layers: 01-aoi (roi), 03-landcover (lc), 05-dem, 06-srf, 07-svf.\n"
        "ROI=1 denotes cell centers in the ORIGINAL region; outside cells are valid 0.\n"
        "The 0/5/10 km terrain buffers change only the surrounding terrain rectangle, snapped to 1 km.\n"
        "All resolutions share aligned coordinates; 100 and 250 m cells do not nest exactly.\n"
        "GeoTIFF, ASCII and required .prj files are supplied. ROI NoData=255; other layers=-9999.\n"
        "DEM, SRF and SVF retain continuous values. LC retains classes 1..13.\n\n"
        "aoi.gpkg: boundaries (5), grid_extents (75), roi_partitions (30), subregions (90).\n"
        "Each terrain buffer directory has one Shapefile ZIP partitioning its rectangle into ROI=1/0.\n"
        "Open aoi_overview.qgz in QGIS: relative paths, Arial labels, Euregio 100 m / 5 km view.\n"
        "The overview PNG/PDF shows original regions and terrain buffer rectangles.\n\n"
        "Tyrol 100 m / 5 km: 578000, 5167000 : 808000, 5298000; 2300 columns x 1310 rows.\n"
        "The station-availability audit concerns North Tyrol only, not East Tyrol.\n"
        "Terrain edge/source limitations remain documented in the external processing records.\n"
        "Records: workspace/context/plans/open/fram3s_grid_alignment_processing.\n"
        "Superseded data and raw sources: fram3s/90-archive/grid_collection_cleanup/<date>/.\n"
    )
