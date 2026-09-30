"""Source staging and per-resolution parent processing without source mutation."""

import importlib.util
import logging
import math
import shutil
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_cdt

from fram3s_grids.common import CRS, NODATA, envelope, grid, save_json, sha256, write_tif

SOURCES = {
    "dem": "05-dem/euregio/dem_euregio_20.asc",
    "lc": "03-landcover/lc_eusalp/openAMUNDSEN-euregio/lc_euregio_20_eusalp.asc",
    "provinces": "01-aoi/PROVINCE_BOUNDARY/province_boundary_4326.gpkg",
    "subregions": "01-aoi/SUBREGIONS/raw/subregions_avalanche_report_4326_raw.gpkg",
}
LOG = logging.getLogger(__name__)


def prepare_sources(source: Path, work: Path) -> dict:
    """Pixel-scan and stage source rasters, recording hashes and explicit CRS evidence."""
    from importlib.metadata import version

    folder = work / "sources"
    folder.mkdir(parents=True, exist_ok=True)
    records = {}
    for kind, relative in SOURCES.items():
        path = source / relative
        LOG.info("Source scan: %s", path)
        record = {
            "path": relative,
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256(path),
        }
        if kind in ("dem", "lc"):
            with rasterio.open(path) as src:
                values = src.read(1, masked=True)
                valid = ~np.ma.getmaskarray(values) & np.isfinite(values.data)
                record.update(
                    bounds=list(src.bounds),
                    transform=list(src.transform),
                    shape=list(src.shape),
                    valid_cells=int(valid.sum()),
                    missing_cells=int((~valid).sum()),
                    minimum=float(values.data[valid].min()),
                    maximum=float(values.data[valid].max()),
                    declared_crs=str(src.crs),
                    assigned_crs=CRS,
                )
                if kind == "lc":
                    codes, counts = np.unique(values.data[valid], return_counts=True)
                    record["class_counts"] = {str(int(code)): int(count) for code, count in zip(codes, counts)}
                    if not np.isin(codes, np.arange(1, 14)).all():
                        raise ValueError(f"Unexpected land cover classes: {codes}")
                write_tif(folder / f"{kind}.tif", np.where(valid, values.data, NODATA).astype("float32"), src.transform)
        if kind not in ("dem", "lc"):
            target = folder / "vectors" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        records[kind] = record
        save_json(work / "source_manifest.json", records)
    records["environment"] = {
        name: version(name) for name in ("numpy", "rasterio", "geopandas", "shapely", "openamundsen", "numba", "scipy")
    }
    records["crs_evidence"] = (
        "The existing lc_berchtesgaden_eusalp.R declares the DEM and derived land-cover grids "
        "as ETRS89 / UTM 32N; the North Tyrol snapshot builder declares EPSG:25832. "
        "Source sidecars alone are incomplete. Coordinates are retained during assignment."
    )
    save_json(work / "source_manifest.json", records)
    return records


def average(source_values, source_transform, shape, transform, nodata=NODATA):
    """Area-weighted mean of valid source pixels on a target grid."""
    result = np.full(shape, NODATA, dtype="float32")
    reproject(
        source_values,
        result,
        src_transform=source_transform,
        src_crs=CRS,
        src_nodata=nodata,
        dst_transform=transform,
        dst_crs=CRS,
        dst_nodata=NODATA,
        resampling=Resampling.average,
        num_threads=2,
    )
    return result


def dominant_class(source_values, source_transform, shape, transform):
    """Select the greatest covered area; ascending iteration breaks ties by lowest code."""
    valid = (source_values >= 1) & (source_values <= 13)
    best = np.zeros(shape, dtype="float32")
    result = np.full(shape, NODATA, dtype="int16")
    for code in range(1, 14):
        LOG.info("Land-cover area fraction: class %s", code)
        binary = np.where(valid, (source_values == code).astype("float32"), NODATA)
        fraction = average(binary, source_transform, shape, transform)
        update = fraction > best + 1e-7
        result[update] = code
        best[update] = fraction[update]
    return result


def source_fraction(source_valid, source_transform, shape, transform):
    """Fraction of each destination cell covered by valid source pixels, including borders."""
    fraction = average(source_valid.astype("float32"), source_transform, shape, transform, nodata=None)
    fraction[fraction == NODATA] = 0
    left, top = source_transform.c, source_transform.f
    right = left + source_valid.shape[1] * source_transform.a
    bottom = top + source_valid.shape[0] * source_transform.e
    res = transform.a
    x = transform.c + np.arange(shape[1]) * res
    y = transform.f - np.arange(shape[0]) * res
    x_fraction = np.maximum(0, np.minimum(x + res, right) - np.maximum(x, left)) / res
    y_fraction = np.maximum(0, np.minimum(y, top) - np.maximum(y - res, bottom)) / res
    return (fraction * y_fraction[:, None] * x_fraction[None, :]).astype("float32")


def prepare_parent(work: Path, resolution: int) -> Path:
    """Resample each source directly, never cascading between output resolutions."""
    target = work / "parents" / f"{resolution}m"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "prepared.json").exists():
        return target
    with rasterio.open(work / "sources/dem.tif") as src:
        bounds = envelope(src.bounds)
        shape, transform = grid(bounds, resolution)
        values = src.read(1)
        dem = average(values, src.transform, shape, transform)
        fraction = source_fraction((values != NODATA) & np.isfinite(values), src.transform, shape, transform)
        write_tif(target / "dem.tif", dem, transform)
        write_tif(target / "source_coverage.tif", fraction, transform, nodata=-1)
    del values, dem, fraction
    with rasterio.open(work / "sources/lc.tif") as src:
        values = src.read(1)
        classes = dominant_class(values, src.transform, shape, transform)
        write_tif(target / "lc.tif", classes, transform)
    save_json(target / "prepared.json", {"resolution_m": resolution, "bounds": bounds, "shape": shape})
    return target


def load_srf():
    """Import the existing calculation, preserving its formula and settings."""
    path = Path(__file__).resolve().parents[1] / "calculateSRF.py"
    spec = importlib.util.spec_from_file_location("fram3s_srf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def terrain_parent(work: Path, resolution: int) -> None:
    """Calculate terrain products once on each shared parent grid."""
    from openamundsen import terrain

    target = work / "parents" / f"{resolution}m"
    if (target / "terrain_complete.json").exists():
        return
    started = time.time()
    with rasterio.open(target / "dem.tif") as src:
        dem = src.read(1)
        transform = src.transform
    valid = (dem != NODATA) & np.isfinite(dem)
    dem_nan = np.where(valid, dem, np.nan)
    if not (target / "srf.tif").exists():
        LOG.info("SRF parent at %s m", resolution)
        srf = load_srf().compute_srf(np.ma.masked_invalid(dem_nan), resolution)
        write_tif(target / "srf.tif", srf.filled(NODATA).astype("float32"), transform)
        del srf
    if not (target / "svf.tif").exists():
        LOG.info("SVF parent at %s m", resolution)
        svf = terrain.sky_view_factor(dem_nan, resolution, azim_step=10, elev_step=1, num_sweeps=1)
        svf[~valid] = NODATA
        write_tif(target / "svf.tif", svf.astype("float32"), transform)
        del svf
    # Chebyshev distance conservatively covers all eight openness directions.
    padded = np.pad(valid, 1, constant_values=False)
    distance = distance_transform_cdt(padded, metric="chessboard")[1:-1, 1:-1].astype("float32") * resolution
    write_tif(target / "terrain_context_distance.tif", distance, transform, nodata=-1)
    flags = np.zeros(dem.shape, dtype="uint8")
    flags[~valid] |= 1
    with rasterio.open(target / "source_coverage.tif") as src:
        coverage = src.read(1)
    flags[coverage < 1 - 1e-6] |= 2
    flags[distance <= math.ceil(5000 / resolution) * resolution] |= 4
    flags[distance <= 5000] |= 8
    write_tif(target / "terrain_quality.tif", flags, transform, nodata=255)
    save_json(
        target / "terrain_complete.json",
        {
            "resolution_m": resolution,
            "seconds": time.time() - started,
            "srf_small_cardinal_distance_m": math.ceil(100 / resolution) * resolution,
            "srf_large_cardinal_distance_m": math.ceil(5000 / resolution) * resolution,
            "diagonal_distance_multiplier": math.sqrt(2),
            "normalization": "valid parent DEM cells",
            "quality_bits": {
                "1": "missing DEM",
                "2": "partial source DEM coverage",
                "4": "potentially incomplete SRF neighborhood",
                "8": "within 5 km of missing terrain; SVF proximity indicator only",
            },
        },
    )
