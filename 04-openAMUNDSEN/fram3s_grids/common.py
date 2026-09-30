"""File contracts and deterministic spatial helpers."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

CRS = "EPSG:25832"
NODATA = -9999
RESOLUTIONS = (50, 100, 250, 500, 1000)
REGIONS = ("euregio", "tyrol", "north_tyrol", "south_tyrol", "trentino")
BUFFERS = (0, 5000, 10000)
ROOTS = {"roi": "01-aoi", "lc": "03-landcover", "dem": "05-dem", "srf": "06-srf", "svf": "07-svf"}


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: object) -> None:
    """Atomically replace a JSON progress or metadata file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def envelope(bounds: tuple, spacing: int = 1000) -> tuple:
    """Snap a rectangle outward without rounding polygon coordinates."""
    return (
        math.floor(bounds[0] / spacing) * spacing,
        math.floor(bounds[1] / spacing) * spacing,
        math.ceil(bounds[2] / spacing) * spacing,
        math.ceil(bounds[3] / spacing) * spacing,
    )


def grid(bounds: tuple, resolution: int) -> tuple:
    """Return shape and affine transform for an exactly divisible rectangle."""
    xmin, ymin, xmax, ymax = bounds
    assert (xmax - xmin) % resolution == 0 and (ymax - ymin) % resolution == 0
    return (
        (int((ymax - ymin) / resolution), int((xmax - xmin) / resolution)),
        from_origin(xmin, ymax, resolution, resolution),
    )


def domain(region: str, buffer_m: int) -> str:
    return f"{region}_b{buffer_m:05d}"


def variant_dir(root: Path, region: str, buffer_m: int) -> Path:
    return root / region / f"buffer_{buffer_m:05d}m"


def raster_path(output: Path, kind: str, region: str, buffer_m: int, resolution: int) -> Path:
    return (
        variant_dir(output / ROOTS[kind], region, buffer_m)
        / f"{resolution}m"
        / f"{kind}_{domain(region, buffer_m)}_{resolution}.tif"
    )


def write_tif(path: Path, values: np.ndarray, transform, nodata=NODATA) -> None:
    """Write lossless, tiled, compressed GeoTIFF with embedded CRS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=values.dtype,
        crs=CRS,
        transform=transform,
        nodata=nodata,
        tiled=True,
        compress="deflate",
        predictor=3 if values.dtype.kind == "f" else 2,
        BIGTIFF="IF_SAFER",
    ) as dst:
        dst.write(values, 1)


def export_ascii(path: Path) -> Path:
    """Export a model ASCII grid and an unambiguous WKT projection sidecar."""
    from rasterio.shutil import copy

    target = path.with_suffix(".asc")
    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        copy(path, target, driver="AAIGrid", SIGNIFICANT_DIGITS=9)
    target.with_suffix(".prj").write_text(rasterio.crs.CRS.from_string(CRS).to_wkt() + "\n")
    return target
