"""Numerical contracts for aligned model inputs, independent of the private data."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fram3s_grids.common import envelope, grid
from fram3s_grids.processing import average, dominant_class, load_srf, source_fraction


def test_outward_envelope_dimensions_and_non_nesting():
    bounds = envelope((578580.18, 5175569.98, 783711.01, 5297900.63))
    assert bounds == (578000, 5175000, 784000, 5298000)
    assert grid(bounds, 100)[0] == (1230, 2060)
    for res in (50, 100, 250, 500, 1000):
        shape, transform = grid(bounds, res)
        assert shape[1] * res == bounds[2] - bounds[0]
        assert transform.c % res == 0 and transform.f % res == 0
    assert 250 % 100 != 0


def test_area_weighted_average_with_offset_and_nodata():
    source = np.array([[0, 10, 20], [0, 10, 20]], dtype="float32")
    values = average(source, from_origin(0, 2, 1, 1), (1, 1), from_origin(0.5, 2, 2, 2))
    assert values[0, 0] == 10  # 0.5*0 + 1*10 + 0.5*20, divided by 2
    source[:, 1] = -9999
    assert average(source, from_origin(0, 2, 1, 1), (1, 1), from_origin(0.5, 2, 2, 2))[0, 0] == 10


def test_dominant_class_weights_not_pixel_counts_and_ties():
    source = np.array([[2, 1, 2], [2, 1, 2]], dtype="float32")
    # One complete class-1 column outweighs two quarter-width class-2 columns.
    assert dominant_class(source, from_origin(0, 2, 1, 1), (1, 1), from_origin(0.75, 2, 1.5, 2))[0, 0] == 1
    assert dominant_class(source, from_origin(0, 2, 1, 1), (1, 1), from_origin(0.5, 2, 2, 2))[0, 0] == 1
    source[:] = -9999
    assert dominant_class(source, from_origin(0, 2, 1, 1), (1, 1), from_origin(0.5, 2, 2, 2))[0, 0] == -9999


def test_partial_source_coverage_includes_uncovered_rectangle():
    fraction = source_fraction(np.ones((2, 2), bool), from_origin(1, 3, 1, 1), (2, 2), from_origin(0, 4, 2, 2))
    np.testing.assert_allclose(fraction, 0.25)


def test_roi_zero_is_valid_and_center_rule_preserves_partition():
    transform = from_origin(0, 2, 1, 1)
    a = rasterize([(box(0, 0, 1.1, 2), 1)], out_shape=(2, 2), transform=transform, dtype="uint8")
    b = rasterize([(box(1.1, 0, 2, 2), 1)], out_shape=(2, 2), transform=transform, dtype="uint8")
    assert np.array_equal(a + b, np.ones((2, 2)))


def test_srf_formula_matches_frozen_legacy_reference():
    # Direct independent reproduction of the existing formula, including normalization.
    from openamundsen import terrain
    import numba

    dem = np.array([[100, 120, 140], [90, 500, 160], [80, 70, 50]], dtype="float32")
    for res in (50, 250, 1000):
        numba.set_num_threads(1)
        large = terrain.openness(dem.astype(float), res, 5000, negative=True).astype("float32")
        small = terrain.openness(dem.astype(float), res, 100, negative=True).astype("float32")
        raw = (np.clip(3 * (large - 1), 0.1, 1.6) + np.clip(3 * (small - 1.2), 0.1, 1.6)) / 2
        adjusted = 1 + (raw - 1) * dem / dem.max()
        expected = adjusted / adjusted.mean()
        parallel_threads = min(2, numba.config.NUMBA_NUM_THREADS)
        numba.set_num_threads(parallel_threads)
        for _ in range(2):
            np.testing.assert_allclose(load_srf().compute_srf(np.ma.array(dem), res), expected, rtol=2e-6)
            assert numba.get_num_threads() == parallel_threads


def test_srf_import_has_no_log_file_side_effect(tmp_path):
    original = Path(__file__).resolve().parents[1] / "calculateSRF.py"
    isolated = tmp_path / "calculateSRF.py"
    isolated.write_bytes(original.read_bytes())
    spec = importlib.util.spec_from_file_location("isolated_srf", isolated)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not list(tmp_path.glob("*.log"))


def test_context_changes_extent_but_not_original_roi():
    from fram3s_grids.geometry import variants
    from fram3s_grids.common import REGIONS

    original = box(120, 250, 1720, 1880)
    rows = list(variants(dict.fromkeys(REGIONS, original)))[:3]
    assert [row[1] for row in rows] == [0, 5000, 10000]
    assert all(row[2].equals(original) for row in rows)
    assert rows[0][3] == (0, 0, 2000, 2000)
    assert rows[1][3] == (-5000, -5000, 7000, 7000)
    cores = []
    for _, _, geom, bounds in rows:
        shape, transform = grid(bounds, 100)
        values = rasterize([(geom, 1)], out_shape=shape, transform=transform, dtype="uint8")
        yy, xx = np.nonzero(values)
        cores.append(set(zip(transform.c + (xx + 0.5) * 100, transform.f - (yy + 0.5) * 100)))
    assert cores[0] == cores[1] == cores[2]


def test_ascii_is_self_contained_without_optional_xml(tmp_path):
    import rasterio
    from fram3s_grids.common import write_tif, export_ascii

    for name, values, nodata in [
        ("roi", np.array([[0, 1], [1, 0]], dtype="uint8"), 255),
        ("dem", np.array([[12.3456, -9999], [27.891, 0]], dtype="float32"), -9999),
    ]:
        path = tmp_path / f"{name}.tif"
        transform = from_origin(578000, 5298000, 100, 100)
        write_tif(path, values, transform, nodata=nodata)
        asc = export_ascii(path)
        assert not list(tmp_path.glob("*.aux.xml"))
        with rasterio.open(asc) as src:
            assert src.crs.to_epsg() == 25832 and src.nodata == nodata and src.transform == transform
            np.testing.assert_allclose(src.read(1), values, rtol=1e-7, atol=1e-6)


def test_delivery_whitelist_excludes_diagnostics_and_person_packages():
    from fram3s_grids.common import REGIONS, BUFFERS, RESOLUTIONS
    from fram3s_grids.cleanup import expected_files

    specs = [
        {"region": region, "buffer_m": buffer_m, "resolution_m": res}
        for region in REGIONS
        for buffer_m in BUFFERS
        for res in RESOLUTIONS
    ]
    paths = expected_files(specs)
    assert len(paths) == 1145
    assert sum(path.endswith(".zip") for path in paths) == 15
    assert sum(path.endswith(".tif") for path in paths) == 375
    assert not any("quality" in path or "kathi" in path or path.endswith((".json", ".aux.xml")) for path in paths)


def test_central_vectors_preserve_attributes_and_partitions(tmp_path):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import MultiPolygon
    from fram3s_grids.geometry import build_geometry
    from fram3s_grids.delivery import validate_vectors

    source = tmp_path / "sources/vectors"
    provinces = source / "01-aoi/PROVINCE_BOUNDARY/province_boundary_4326.gpkg"
    provinces.parent.mkdir(parents=True)
    geometries = [
        MultiPolygon([box(600000, 5230000, 620000, 5250000), box(650000, 5200000, 660000, 5210000)]),
        box(620000, 5180000, 640000, 5200000),
        box(620000, 5160000, 640000, 5180000),
    ]
    gpd.GeoDataFrame({"province": ["tyrol", "south_tyrol", "trentino"]}, geometry=geometries, crs="EPSG:25832").to_crs(
        4326
    ).to_file(provinces, driver="GPKG")
    original = gpd.GeoDataFrame(
        {
            "id": [f"AT-{i}" for i in range(90)],
            "province": ["tyrol"] * 90,
            "country": ["AT"] * 90,
            "original_note": [None if i % 2 else "retained" for i in range(90)],
        },
        geometry=[box(601000 + i * 100, 5231000, 601080 + i * 100, 5231080) for i in range(90)],
        crs="EPSG:25832",
    ).to_crs(4326)
    subpath = source / "01-aoi/SUBREGIONS/raw/subregions_avalanche_report_4326_raw.gpkg"
    subpath.parent.mkdir(parents=True)
    original.to_file(subpath, driver="GPKG")
    specs = build_geometry(source, tmp_path / "output", tmp_path)
    assert len(specs) == 75
    assert len(list((tmp_path / "output").rglob("*.gpkg"))) == 1
    assert len(list((tmp_path / "output").rglob("*.zip"))) == 15
    assert not list((tmp_path / "output").rglob("*.shp")) and not list((tmp_path / "output").rglob("*.json"))
    exported = gpd.read_file(tmp_path / "output/01-aoi/aoi.gpkg", layer="subregions")
    pd.testing.assert_frame_equal(original.drop(columns="geometry"), exported.drop(columns="geometry"))
    assert validate_vectors(tmp_path)["attributes_preserved"]
