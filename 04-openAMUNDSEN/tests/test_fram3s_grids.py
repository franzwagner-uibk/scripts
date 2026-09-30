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
