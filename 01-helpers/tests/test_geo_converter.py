"""Exercise ROI geometry and file contracts using synthetic data."""
import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from osgeo import gdal, osr
from shapely.geometry import MultiPolygon, Polygon, box

SPEC = importlib.util.spec_from_file_location("geo_converter", Path(__file__).parents[1] / "geoConverter.py")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def write_raster(path, data, epsg=25832):
    gdal.UseExceptions()
    dataset = gdal.GetDriverByName("GTiff").Create(str(path), data.shape[1], data.shape[0], 1, gdal.GDT_Int16)
    dataset.SetGeoTransform((620000, 50, 0, 5200000, 0, -50))
    if epsg is not None:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(data)
    dataset = None


def test_partition_preserves_hole_and_disconnected_parts(tmp_path):
    outer = box(600000, 5200000, 600300, 5200300)
    hole = box(600100, 5200100, 600200, 5200200)
    inside = MultiPolygon([Polygon(outer.exterior.coords, [hole.exterior.coords]),
                           box(600400, 5200000, 600450, 5200050)])
    rectangle = box(599950, 5199950, 600500, 5200350)
    destination = tmp_path / "partition.zip"
    report = converter.create_roi_zip(inside, rectangle, destination)
    assert report["features"] == 2
    assert report["overlap_m2"] == report["rectangle_error_m2"] == report["roi_error_m2"] == 0
    saved = gpd.read_file(f"/vsizip/{destination}/partition.shp").set_index("ROI")
    assert saved.loc[0].geometry.covers(hole)
    assert saved.loc[1].geometry.equals(inside)
    digest = converter._sha256(destination)
    with pytest.raises(ValueError, match="already exists"):
        converter.create_roi_zip(inside, rectangle, destination)
    assert converter._sha256(destination) == digest


@pytest.mark.parametrize("epsg", [25832, None])
def test_binary_raster_round_trip_with_hole_and_island(tmp_path, epsg):
    data = np.array([[0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 0, 1, 0],
                     [0, 1, 0, 1, 0, 0, 0],
                     [0, 1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0]], dtype=np.int16)
    source = tmp_path / "roi.tif"
    write_raster(source, data, epsg)
    inside, rectangle, grid = converter.binary_raster_roi(source)
    report = converter.create_roi_zip(inside, rectangle, tmp_path / "raster.zip", grid)
    assert report["raster_mismatches"] == 0
    assert report["inside_cells"] == 9
    assert report["outside_cells"] == 26
    assert report["inside_area_m2"] == 9 * 2500


@pytest.mark.parametrize("bad_value", [-9999, 2])
def test_raster_rejects_nodata_and_unexpected_classes(tmp_path, bad_value):
    source = tmp_path / "invalid.tif"
    write_raster(source, np.array([[0, 1, bad_value]], dtype=np.int16))
    with pytest.raises(ValueError, match="no NoData or other values"):
        converter.binary_raster_roi(source)


def test_raster_rejects_conflicting_crs(tmp_path):
    source = tmp_path / "wrong_crs.tif"
    write_raster(source, np.array([[0, 1]], dtype=np.int16), 32632)
    with pytest.raises(ValueError, match="CRS conflicts"):
        converter.binary_raster_roi(source)


@pytest.mark.parametrize("invalid_east", [False, True])
def test_north_tyrol_selection_margin_and_2d(tmp_path, invalid_east):
    north = Polygon([(10, 47, 3), (11, 47, 3), (11, 48, 3), (10, 48, 3), (10, 47, 3)])
    east = Polygon([(12, 46.8, 3), (12.1, 46.8, 3), (12.1, 47, 3), (12, 47, 3), (12, 46.8, 3)])
    if invalid_east:
        east = Polygon([(12, 46.8, 3), (12.1, 46.8, 3), (12.05, 46.9, 3),
                        (12.1, 47, 3), (12, 47, 3), (12.05, 46.9, 3), (12, 46.8, 3)])
        assert not east.is_valid
    source = tmp_path / "provinces.gpkg"
    gpd.GeoDataFrame({"province": ["tyrol"]}, geometry=[MultiPolygon([east, north])], crs=4326).to_file(
        source, layer="province_boundary_4326", driver="GPKG")
    inside, rectangle = converter.north_tyrol_roi(source)
    assert not inside.has_z
    expected = gpd.GeoSeries([north], crs=4326).to_crs(25832).iloc[0]
    assert inside.equals(expected)
    xmin, ymin, xmax, ymax = inside.bounds
    assert rectangle.bounds == (xmin - 5000, ymin - 5000, xmax + 5000, ymax + 5000)


def test_all_jobs_preflight_before_writing_first_output(tmp_path, monkeypatch):
    first, second = tmp_path / "north.zip", tmp_path / "oetztal.zip"
    second.write_bytes(b"keep this")
    monkeypatch.setattr(converter, "ROI_OUTPUTS", {"north_tyrol": first, "oetztal": second})
    with pytest.raises(ValueError, match="already exists"):
        converter.run_roi_jobs()
    assert not first.exists()
    assert second.read_bytes() == b"keep this"


def test_original_gpkg_to_shp_conversion(tmp_path):
    source = tmp_path / "original.gpkg"
    gpd.GeoDataFrame({"value": [7]}, geometry=[box(0, 0, 10, 10)], crs=25832).to_file(source, driver="GPKG")
    assert converter.convert_gpkg_to_shp(source, tmp_path / "shapes", overwrite=False)
    result = gpd.read_file(tmp_path / "shapes/original.shp")
    assert result.value.tolist() == [7]
    assert result.crs.to_epsg() == 25832
