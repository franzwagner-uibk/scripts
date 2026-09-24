#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
geoConverter.py

Simple batch and single-file converter for:
  - Shapefile (.shp) <-> GeoPackage (.gpkg)
  - Shapefile (.shp) / GeoPackage (.gpkg) -> GeoJSON (.geojson)
  - North Tyrol / Oetztal ROI -> validated two-feature zipped shapefile

The original conversions use the CONFIG section below. Run both saved ROI
jobs with ``python geoConverter.py --mode roi2shp``; see GEO_CONVERTER.md.
For the original conversions, edit the variables and run:

    python geoConverter.py

Author: Franz Wagner
Date: 2025-12-02
"""

import argparse
import hashlib
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional
from zipfile import ZIP_DEFLATED, ZipFile

import geopandas as gpd


# =========================
# ===== CONFIG (GLOBAL) ===
# =========================

# Mode of operation:
#   "shp2gpkg"  - convert Shapefile  -> GeoPackage
#   "gpkg2shp"  - convert GeoPackage -> Shapefile
#   "shp2json"  - convert Shapefile  -> GeoJSON
#   "gpkg2json" - convert GeoPackage -> GeoJSON
#   "roi2shp"   - build saved North Tyrol / Oetztal ROI ZIPs
MODE: str = "shp2gpkg"

# Input can be a single file or a directory.
INPUT_PATH: str = r"F:\fram3s\01-data\01-aoi\TESTSITES\testsites_rofental_extended.shp"

# Output directory. If None, outputs are written next to the inputs.
OUTPUT_DIR: Optional[str] = None

# If INPUT_PATH is a directory, decide whether to recurse into subfolders.
RECURSIVE: bool = False

# Overwrite existing output files?
OVERWRITE: bool = False

# Logging level: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL: str = "INFO"


# Saved ROI jobs. EPSG:25832 is ETRS89 / UTM zone 32N.
# The ASCII raster has no embedded CRS; its model YAML declares EPSG:25832.
ROI_BASE = Path(r"F:\fram3s\01-data\01-aoi")
ROI_PROVINCES = ROI_BASE / "PROVINCE_BOUNDARY/province_boundary_4326.gpkg"
ROI_RASTER = Path("M:/Ötztal/openamundsen/oetztal/grids/roi_oetztal_50.asc")
ROI_EPSG = 25832
ROI_NORTH_TYROL_MARGIN_M = 5000
ROI_OUTPUTS = {
    "north_tyrol": ROI_BASE / "EUREGIO_BOUNDARY/north_tyrol_roi_25832.zip",
    "oetztal": ROI_BASE / "OETZTAL/oetztal_roi_25832.zip",
}
# Polygon overlays may accumulate sub-millimeter floating-point error.
ROI_AREA_TOLERANCE_M2 = 0.001


# =================
# LOGGING SETUP
# =================
def configure_logging(level: str = LOG_LEVEL) -> None:
    """Configure simple console logging."""
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
    )
    logging.info("Logging initialized (level=%s).", level.upper())


# =================
# HELPER FUNCTIONS
# =================
def _expected_suffix(mode: str) -> str:
    if mode in ("shp2gpkg", "shp2json"):
        return ".shp"
    if mode in ("gpkg2shp", "gpkg2json"):
        return ".gpkg"
    raise ValueError(f"Unsupported MODE: {mode}")


def discover_input_files(mode: str, input_path: Path, recursive: bool) -> List[Path]:
    """Return list of input files based on mode and INPUT_PATH."""
    expected = _expected_suffix(mode)

    if input_path.is_file():
        if input_path.suffix.lower() != expected:
            logging.error("INPUT_PATH has wrong extension for MODE %s: %s", mode, input_path)
            return []
        return [input_path]

    if not input_path.is_dir():
        logging.error("INPUT_PATH does not exist or is not a directory: %s", input_path)
        return []

    pattern = f"*{expected}"
    if recursive:
        files = sorted(p for p in input_path.rglob(pattern) if p.is_file())
    else:
        files = sorted(p for p in input_path.glob(pattern) if p.is_file())

    logging.info("Found %d input file(s) in %s (recursive=%s).", len(files), input_path, recursive)
    return files


def ensure_output_dir(path: Path) -> None:
    """Create output directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def output_directory_for(src: Path, configured_output: Optional[Path]) -> Path:
    """Decide where to place the converted file."""
    if configured_output is not None:
        return configured_output
    # Default: same folder as source
    return src.parent


def convert_shp_to_gpkg(src: Path, dst_dir: Path, overwrite: bool) -> bool:
    """Convert a single shapefile to GeoPackage."""
    try:
        ensure_output_dir(dst_dir)
        dst = dst_dir / f"{src.stem}.gpkg"

        if dst.exists():
            if not overwrite:
                logging.info("Skipping existing GPKG (overwrite=False): %s", dst)
                return True
            logging.warning("Overwriting existing GPKG: %s", dst)
            dst.unlink()

        logging.debug("Reading SHP: %s", src)
        gdf = gpd.read_file(src)

        logging.debug("Writing GPKG: %s", dst)
        gdf.to_file(dst, layer=src.stem, driver="GPKG")

        logging.info("Converted SHP -> GPKG: %s -> %s", src, dst)
        return True
    except Exception as exc:
        logging.error("Failed SHP -> GPKG: %s | %s", src, exc)
        return False


def convert_gpkg_to_shp(src: Path, dst_dir: Path, overwrite: bool) -> bool:
    """Convert a single GeoPackage (first layer) to Shapefile."""
    try:
        ensure_output_dir(dst_dir)
        dst = dst_dir / f"{src.stem}.shp"

        if dst.exists():
            if not overwrite:
                logging.info("Skipping existing SHP (overwrite=False): %s", dst)
                return True
            logging.warning("Overwriting existing SHP: %s", dst)
            # Remove all shapefile sidecar files
            for side in dst_dir.glob(f"{src.stem}.*"):
                side.unlink()

        logging.debug("Reading GPKG: %s", src)
        # By default, GeoPandas reads the first layer.
        gdf = gpd.read_file(src)

        logging.debug("Writing SHP: %s", dst)
        gdf.to_file(dst, driver="ESRI Shapefile")

        logging.info("Converted GPKG -> SHP: %s -> %s", src, dst)
        return True
    except Exception as exc:
        logging.error("Failed GPKG -> SHP: %s | %s", src, exc)
        return False


def convert_shp_to_geojson(src: Path, dst_dir: Path, overwrite: bool) -> bool:
    """Convert a single shapefile to GeoJSON."""
    try:
        ensure_output_dir(dst_dir)
        dst = dst_dir / f"{src.stem}.geojson"

        if dst.exists():
            if not overwrite:
                logging.info("Skipping existing GeoJSON (overwrite=False): %s", dst)
                return True
            logging.warning("Overwriting existing GeoJSON: %s", dst)
            dst.unlink()

        logging.debug("Reading SHP: %s", src)
        gdf = gpd.read_file(src)

        logging.debug("Writing GeoJSON: %s", dst)
        gdf.to_file(dst, driver="GeoJSON")

        logging.info("Converted SHP -> GeoJSON: %s -> %s", src, dst)
        return True
    except Exception as exc:
        logging.error("Failed SHP -> GeoJSON: %s | %s", src, exc)
        return False


def convert_gpkg_to_geojson(src: Path, dst_dir: Path, overwrite: bool) -> bool:
    """Convert a single GeoPackage (first layer) to GeoJSON."""
    try:
        ensure_output_dir(dst_dir)
        dst = dst_dir / f"{src.stem}.geojson"

        if dst.exists():
            if not overwrite:
                logging.info("Skipping existing GeoJSON (overwrite=False): %s", dst)
                return True
            logging.warning("Overwriting existing GeoJSON: %s", dst)
            dst.unlink()

        logging.debug("Reading GPKG: %s", src)
        # By default, GeoPandas reads the first layer.
        gdf = gpd.read_file(src)

        logging.debug("Writing GeoJSON: %s", dst)
        gdf.to_file(dst, driver="GeoJSON")

        logging.info("Converted GPKG -> GeoJSON: %s -> %s", src, dst)
        return True
    except Exception as exc:
        logging.error("Failed GPKG -> GeoJSON: %s | %s", src, exc)
        return False


# =================
# REPEATABLE ROI EXPORTS
# =================
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def north_tyrol_roi(source: Path) -> tuple:
    """Extract North Tyrol, excluding East Tyrol, and add the configured margin."""
    from shapely import force_2d
    from shapely.geometry import box

    provinces = gpd.read_file(source, layer="province_boundary_4326")
    _require(provinces.crs is not None, "Province source CRS is missing")
    tyrol = provinces.loc[provinces.province == "tyrol"]
    _require(len(tyrol) == 1, "Expected exactly one province=tyrol feature")
    geometry = force_2d(tyrol.geometry.iloc[0])
    _require(geometry.geom_type == "MultiPolygon" and len(geometry.geoms) == 2,
             "Expected separate North Tyrol and East Tyrol polygons")
    parts = gpd.GeoSeries(list(geometry.geoms), crs=provinces.crs).to_crs(ROI_EPSG)
    north = max(parts, key=lambda part: part.area)
    east = min(parts, key=lambda part: part.area)
    _require(north.is_valid, "Invalid North Tyrol geometry")
    # East Tyrol is discarded; its source ring has a known self-intersection.
    # Do not repair it or let its topology alter the selected northern component.
    _require(north.centroid.x < east.centroid.x,
             "Could not identify the larger western North Tyrol component")
    xmin, ymin, xmax, ymax = north.bounds
    margin = ROI_NORTH_TYROL_MARGIN_M
    _require(margin >= 0, "North Tyrol margin must be nonnegative")
    return north, box(xmin - margin, ymin - margin, xmax + margin, ymax + margin)


def binary_raster_roi(source: Path) -> tuple:
    """Polygonize a binary UTM raster without dropping zero-valued cells or holes."""
    import numpy as np
    from osgeo import gdal, ogr, osr
    from shapely import from_wkb
    from shapely.geometry import box
    from shapely.ops import unary_union

    gdal.UseExceptions()
    raster = gdal.Open(str(source), gdal.GA_ReadOnly)
    _require(raster.RasterCount == 1, "ROI raster must have exactly one band")
    original = raster.ReadAsArray()
    _require(set(np.unique(original).tolist()) == {0, 1},
             "ROI raster must contain both 0 and 1, and no NoData or other values")
    transform = raster.GetGeoTransform()
    x, dx, rx, y, ry, dy = transform
    _require(dx > 0 and dy < 0 and rx == ry == 0, "ROI raster must have a north-up rectangular grid")
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(ROI_EPSG)
    if raster.GetProjection():
        embedded = raster.GetSpatialRef()
        embedded.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        _require(bool(embedded.IsSame(srs)), "Raster CRS conflicts with configured ROI_EPSG")
    rectangle = box(x, y + raster.RasterYSize * dy, x + raster.RasterXSize * dx, y)
    # OGR's Memory name works with both older GDAL and the bundled QGIS GDAL.
    vector = ogr.GetDriverByName("Memory").CreateDataSource("")
    layer = vector.CreateLayer("roi", srs, ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("ROI", ogr.OFTInteger))
    gdal.Polygonize(raster.GetRasterBand(1), None, layer, 0, [])
    pieces = [from_wkb(bytes(feature.GetGeometryRef().ExportToWkb()))
              for feature in layer if feature.GetField("ROI") == 1]
    inside = unary_union(pieces)
    expected_area = int(np.count_nonzero(original == 1)) * dx * -dy
    _require(abs(inside.area - expected_area) <= ROI_AREA_TOLERANCE_M2, "Polygonization changed ROI area")
    return inside, rectangle, (original, transform)


def validate_roi_zip(archive: Path, inside: object, rectangle: object, raster_grid: tuple = None) -> dict:
    """Reopen an archive and verify its schema, partition and optional raster round trip."""
    import numpy as np

    name = archive.stem
    with ZipFile(archive) as zipped:
        expected = {name + ext for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg")}
        _require(set(zipped.namelist()) == expected and zipped.testzip() is None, "Invalid ZIP contents")
    vsi = "/vsizip/" + archive.resolve().as_posix() + "/" + name + ".shp"
    frame = gpd.read_file(vsi)
    _require(len(frame) == 2 and set(frame.columns) == {"ROI", "geometry"}, "Expected two ROI features")
    _require(frame.crs is not None and frame.crs.to_epsg() == ROI_EPSG, "Incorrect output CRS")
    _require(np.issubdtype(frame.ROI.dtype, np.integer) and set(frame.ROI) == {0, 1}, "ROI must be integer 0/1")
    _require(frame.geometry.is_valid.all() and not frame.geometry.is_empty.any(), "Invalid ROI geometry")
    _require(not frame.geometry.has_z.any(), "ROI output must be two-dimensional")
    saved_inside = frame.loc[frame.ROI == 1, "geometry"].iloc[0]
    outside = frame.loc[frame.ROI == 0, "geometry"].iloc[0]
    errors = {
        "overlap_m2": saved_inside.intersection(outside).area,
        "rectangle_error_m2": saved_inside.union(outside).symmetric_difference(rectangle).area,
        "roi_error_m2": saved_inside.symmetric_difference(inside).area,
    }
    _require(max(errors.values()) <= ROI_AREA_TOLERANCE_M2, f"ROI partition mismatch: {errors}")
    report = {"features": 2, "epsg": ROI_EPSG, "roi_values": [0, 1],
              "bounds": list(rectangle.bounds), "inside_area_m2": saved_inside.area, **errors}
    if raster_grid is not None:
        from osgeo import gdal, ogr, osr

        gdal.UseExceptions()
        original, transform = raster_grid
        target = gdal.GetDriverByName("MEM").Create("", original.shape[1], original.shape[0], 1, gdal.GDT_Byte)
        target.SetGeoTransform(transform)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(ROI_EPSG)
        target.SetProjection(srs.ExportToWkt())
        target.GetRasterBand(1).Fill(255)
        vector = ogr.Open(vsi, 0)
        gdal.RasterizeLayer(target, [1], vector.GetLayer(0), options=["ATTRIBUTE=ROI"])
        mismatches = int(np.count_nonzero(target.ReadAsArray() != original))
        _require(mismatches == 0, f"Raster round trip changed {mismatches} cells")
        report.update(raster_mismatches=mismatches, inside_cells=int(np.count_nonzero(original == 1)),
                      outside_cells=int(np.count_nonzero(original == 0)))
    return report


def create_roi_zip(inside: object, rectangle: object, destination: Path, raster_grid: tuple = None) -> dict:
    """Export one rectangle partition using the existing GPKG-to-SHP converter."""
    import numpy as np
    from shapely.geometry import MultiPolygon

    _require(not destination.exists(), f"Output already exists: {destination}")
    _require(destination.parent.is_dir(), f"Destination directory missing: {destination.parent}")
    _require(inside.geom_type in {"Polygon", "MultiPolygon"} and inside.is_valid and not inside.is_empty,
             "ROI must be a valid, nonempty polygon or multipolygon")
    _require(rectangle.equals(rectangle.envelope) and rectangle.covers(inside), "Rectangle must enclose the ROI")
    outside = rectangle.difference(inside)
    _require(not outside.is_empty and outside.is_valid, "Outside ROI must have valid nonempty area")
    geometries = [MultiPolygon([part]) if part.geom_type == "Polygon" else part for part in (inside, outside)]
    with tempfile.TemporaryDirectory(prefix="geoconverter_roi_") as directory:
        work = Path(directory)
        gpkg = work / (destination.stem + ".gpkg")
        frame = gpd.GeoDataFrame({"ROI": np.array([1, 0], dtype=np.int32)}, geometry=geometries, crs=ROI_EPSG)
        frame.to_file(gpkg, layer=destination.stem, driver="GPKG", index=False)
        _require(convert_gpkg_to_shp(gpkg, work / "shapefile", overwrite=False), "Shapefile conversion failed")
        staged = work / destination.name
        with ZipFile(staged, "x", ZIP_DEFLATED) as archive:
            for extension in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
                component = work / "shapefile" / (destination.stem + extension)
                archive.write(component, component.name)
        validate_roi_zip(staged, inside, rectangle, raster_grid)
        with staged.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
        _require(_sha256(staged) == _sha256(destination), "Delivered ZIP checksum mismatch")
    report = validate_roi_zip(destination, inside, rectangle, raster_grid)
    report.update(destination=str(destination), sha256=_sha256(destination), bytes=destination.stat().st_size)
    return report


def run_roi_jobs(job: str = "all", output_dir: Optional[Path] = None) -> list:
    """Build saved jobs; preflight all targets and stage both before delivery."""
    _require(job in {"all", *ROI_OUTPUTS}, f"Unknown ROI job: {job}")
    names = list(ROI_OUTPUTS) if job == "all" else [job]
    destinations = {name: output_dir / ROI_OUTPUTS[name].name if output_dir is not None else ROI_OUTPUTS[name]
                    for name in names}
    for destination in destinations.values():
        _require(not destination.exists(), f"Output already exists: {destination}")
        _require(destination.parent.is_dir(), f"Destination directory missing: {destination.parent}")
    sources = {name: ROI_PROVINCES if name == "north_tyrol" else ROI_RASTER for name in names}
    hashes = {name: _sha256(source) for name, source in sources.items()}
    reports = []
    with tempfile.TemporaryDirectory(prefix="geoconverter_jobs_") as directory:
        staged_jobs = []
        for name in names:
            if name == "north_tyrol":
                inside, rectangle = north_tyrol_roi(sources[name])
                raster_grid = None
            else:
                inside, rectangle, raster_grid = binary_raster_roi(sources[name])
            staged = Path(directory) / destinations[name].name
            create_roi_zip(inside, rectangle, staged, raster_grid)
            staged_jobs.append((name, staged, inside, rectangle, raster_grid))
        for name, source in sources.items():
            _require(_sha256(source) == hashes[name], f"Source changed during conversion: {source}")
        for name, staged, inside, rectangle, raster_grid in staged_jobs:
            destination = destinations[name]
            with staged.open("rb") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
            _require(_sha256(staged) == _sha256(destination), "Delivered ZIP checksum mismatch")
            report = validate_roi_zip(destination, inside, rectangle, raster_grid)
            report.update(destination=str(destination), sha256=_sha256(destination),
                          bytes=destination.stat().st_size, source_sha256=hashes[name])
            reports.append(report)
    return reports


# =================
# MAIN ORCHESTRATION
# =================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["shp2gpkg", "gpkg2shp", "shp2json", "gpkg2json", "roi2shp"],
                        default=MODE.lower())
    parser.add_argument("--roi-job", choices=["all", "north_tyrol", "oetztal"], default="all")
    parser.add_argument("--roi-output-dir", type=Path, help="Existing directory for fresh ROI ZIPs")
    args = parser.parse_args()
    configure_logging(LOG_LEVEL)
    mode = args.mode
    if mode == "roi2shp":
        try:
            print(json.dumps(run_roi_jobs(args.roi_job, args.roi_output_dir), indent=2, ensure_ascii=False))
        except Exception as exc:
            logging.error("ROI export failed: %s", exc)
            raise SystemExit(1) from exc
        return
    if mode not in {"shp2gpkg", "gpkg2shp", "shp2json", "gpkg2json"}:
        logging.error("Unsupported MODE: %s", MODE)
        return

    input_path = Path(INPUT_PATH)
    output_root = Path(OUTPUT_DIR) if OUTPUT_DIR is not None else None

    logging.info("Mode        : %s", mode)
    logging.info("INPUT_PATH  : %s", input_path)
    logging.info("OUTPUT_DIR  : %s", output_root if output_root is not None else "<same as input>")
    logging.info("RECURSIVE   : %s", RECURSIVE)
    logging.info("OVERWRITE   : %s", OVERWRITE)

    files = discover_input_files(mode, input_path, RECURSIVE)
    if not files:
        logging.warning("No input files found. Nothing to do.")
        return

    success = 0
    total = len(files)

    for idx, src in enumerate(files, start=1):
        logging.info("[%d/%d] Processing: %s", idx, total, src)
        out_dir = output_directory_for(src, output_root)

        if mode == "shp2gpkg":
            ok = convert_shp_to_gpkg(src, out_dir, OVERWRITE)
        elif mode == "gpkg2shp":
            ok = convert_gpkg_to_shp(src, out_dir, OVERWRITE)
        elif mode == "shp2json":
            ok = convert_shp_to_geojson(src, out_dir, OVERWRITE)
        elif mode == "gpkg2json":
            ok = convert_gpkg_to_geojson(src, out_dir, OVERWRITE)
        else:
            ok = False  # Should not reach here due to earlier check.

        if ok:
            success += 1

    logging.info("Finished. %d/%d file(s) converted successfully.", success, total)


if __name__ == "__main__":
    main()

