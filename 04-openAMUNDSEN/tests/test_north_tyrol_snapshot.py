"""Tests for the dedicated North Tyrol snapshot builder."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from ruamel.yaml import YAML


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from north_tyrol_snapshot import (  # noqa: E402
    MAPS_YAML,
    PINNED_IMAGE,
    AsciiGridHeader,
    SnapshotOptions,
    _builder_commit,
    _cropped_geotransform,
    _date_first,
    _forcing_source_extent,
    _trim_incomplete_forcing_tail,
    _validated_geotransform,
    ascii_crop_window,
    classify_fsc,
    copy_verified,
    crop_ascii_grid,
    hydrological_seasons,
    longest_missing_run,
    preflight,
    project_configuration,
    setup_configuration,
    validate_options,
    write_json,
    write_pending_projects,
)


def test_north_tyrol_map_template_enables_classified_station_overview() -> None:
    config = YAML(typ="safe").load(MAPS_YAML)
    panels = config["maps"]["subdomain_example_setup_overview"]["panels"]

    assert panels[0]["show_subdomain_labels"] is True
    assert "roi_label" not in panels[0]
    assert panels[1]["station_marker_mode"] == "sources_and_roles"
    assert panels[1]["station_match_tolerance_m"] == 10
    assert panels[1]["below_items"] == [{"kind": "station_categories"}]
    assert panels[4]["station_marker_mode"] == "sources_and_roles"


def test_hydrological_seasons_include_complete_leap_year_window() -> None:
    seasons = hydrological_seasons(2019, 2020)

    assert [season.name for season in seasons] == ["project_2019_2020", "project_2020_2021"]
    assert seasons[0].start == datetime(2019, 10, 1, 0, 0)
    assert seasons[0].end == datetime(2020, 9, 30, 21, 0)


def test_hydrological_seasons_reject_reverse_range() -> None:
    with pytest.raises(ValueError, match="end_year"):
        hydrological_seasons(2022, 2021)


def test_fsc_classification_is_mutually_exclusive_and_complete() -> None:
    values = np.array([0.0, 50.0, 100.0, 205.0, 255.0, 210.0, 215.0, np.nan])

    classes = classify_fsc(values)

    assert classes["valid"].tolist() == [True, True, True, False, False, False, False, False]
    assert classes["cloud"].tolist() == [False, False, False, True, True, False, False, False]
    assert classes["water"].tolist() == [False, False, False, False, False, True, False, False]
    assert classes["nodata"].tolist() == [False, False, False, False, False, False, True, True]
    assert not classes["unknown"].any()


def test_fsc_classification_reports_unknown_values() -> None:
    assert classify_fsc(np.array([150.0]))["unknown"].tolist() == [True]


def test_ascii_crop_window_aligns_to_original_cells() -> None:
    header = AsciiGridHeader(5, 4, 100.0, 200.0, 100.0, "-9999")

    assert ascii_crop_window(header, (199.0, 299.0, 401.0, 501.0)) == (0, 4, 0, 4)


def test_ascii_crop_preserves_original_numeric_tokens(tmp_path: Path) -> None:
    source = tmp_path / "source.asc"
    source.write_text(
        "ncols 4\n"
        "nrows 3\n"
        "xllcorner 0\n"
        "yllcorner 0\n"
        "cellsize 100\n"
        "NODATA_value -9999\n"
        "1.000 2.00 3.0 4\n"
        "5 6 7 8\n"
        "9 10 11 12\n",
        encoding="utf-8",
    )
    destination = tmp_path / "crop.asc"

    record = crop_ascii_grid(source, destination, (100.0, 100.0, 300.0, 300.0))

    assert destination.read_text(encoding="utf-8").splitlines()[6:] == ["2.00 3.0", "6 7"]
    assert record["window"] == [0, 2, 1, 3]


def test_dataset_level_geotransform_is_validated_and_updated_for_crop() -> None:
    import xarray as xr

    dataset = xr.Dataset(
        {"spatial_ref": xr.DataArray(0)},
        coords={"x": [573025.0, 573075.0, 573125.0], "y": [5308625.0, 5308575.0]},
        attrs={"GeoTransform": "50 0 573000 0 -50 5308650 0 0 1"},
    )

    style = _validated_geotransform(dataset, Path("scene.nc"))
    cropped = dataset.isel(x=slice(1, None))

    assert style == 9
    assert _cropped_geotransform(cropped, style) == (
        "50 0 573050 0 -50 5308650 0 0 1"
    )


def test_geotransform_coordinate_mismatch_is_rejected() -> None:
    import xarray as xr

    dataset = xr.Dataset(
        {"spatial_ref": xr.DataArray(0)},
        coords={"x": [573025.0, 573075.0], "y": [5308625.0, 5308575.0]},
        attrs={"GeoTransform": "50 0 0 0 -50 0 0 0 1"},
    )

    with pytest.raises(ValueError, match="does not match"):
        _validated_geotransform(dataset, Path("scene.nc"))


def test_raw_copy_hash_linkage(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"original source bytes")

    record = copy_verified(source, tmp_path / "raw" / "source.bin")

    assert record["sha256"]
    assert Path(record["raw_copy"]).read_bytes() == source.read_bytes()


@pytest.mark.parametrize(
    ("values", "expected"),
    [([False, True, True, False, True], 2), ([False, False], 0), ([True, True, True], 3)],
)
def test_longest_missing_run(values: list[bool], expected: int) -> None:
    assert longest_missing_run(values) == expected


def test_forcing_source_extent_accepts_supported_timestamp_column(tmp_path: Path) -> None:
    source = tmp_path / "station.csv"
    source.write_text(
        "datetime,temp\n2017-09-30 00:00:00,1.0\n2023-09-30 21:00:00,2.0\n",
        encoding="utf-8",
    )

    assert _forcing_source_extent(source) == (
        datetime(2017, 9, 30, 0, 0),
        datetime(2023, 9, 30, 21, 0),
    )


@pytest.mark.parametrize("contents", ["temp\n1.0\n", "date,temp\n"])
def test_forcing_source_extent_rejects_missing_or_empty_timestamps(
    tmp_path: Path, contents: str
) -> None:
    source = tmp_path / "station.csv"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="column|timestamps"):
        _forcing_source_extent(source)


def test_forcing_working_frame_places_normalized_date_first() -> None:
    import pandas as pd

    frame = pd.DataFrame({"temp": [273.15], "precip": [0.0], "date": ["2017-10-01"]})

    ordered = _date_first(frame, Path("station.csv"))

    assert ordered.columns.tolist() == ["date", "temp", "precip"]


def test_forcing_tail_is_trimmed_to_last_native_model_grid_timestamp() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "date": pd.date_range("2021-10-18 07:00:00", periods=4, freq="h"),
            "temp": [1.0, 2.0, 3.0, 4.0],
        }
    )

    trimmed, dropped = _trim_incomplete_forcing_tail(
        frame, datetime(2017, 10, 1), Path("station.csv")
    )

    assert trimmed["date"].max() == pd.Timestamp("2021-10-18 09:00:00")
    assert dropped == 1


def test_project_configuration_is_pending_and_uses_correct_fsc_classes() -> None:
    project = project_configuration(hydrological_seasons(2022, 2022)[0])
    snowcover = project["obs"]["snowcover"]
    data_assimilation = project["data_assimilation"]

    assert data_assimilation["prior_forcing"]["ensemble_size"] == 30
    assert data_assimilation["assimilation_events"] == []
    assert data_assimilation["likelihood"]["scf"]["min_support_coverage_ratio"] == 0.1
    assert "rebase_open_loop" not in data_assimilation["rejuvenation"]
    assert list(snowcover["classes"]["cloud"]) == [205, 255]
    assert list(snowcover["classes"]["water"]) == [210]
    assert list(snowcover["classes"]["nodata"]) == [215]


def test_setup_configuration_parses_with_pinned_openamundsen() -> None:
    import openamundsen
    import pandas as pd

    stations = pd.DataFrame([{"id": "P.S000", "x": 100.0, "y": 200.0}])
    config = setup_configuration(stations, hydrological_seasons(2017, 2022))
    grid_variables = config["output_data"]["grids"]["variables"]

    assert {"var": "snow.depth", "name": "snowdepth_instantaneous"} in grid_variables

    parsed = openamundsen.parse_config(config)

    assert parsed.start_date == datetime(2017, 10, 1, 0, 0)
    assert parsed.end_date == datetime(2023, 9, 30, 21, 0)


def test_pending_projects_have_no_active_project_tree(tmp_path: Path) -> None:
    seasons = hydrological_seasons(2017, 2022)

    write_pending_projects(tmp_path, seasons)

    assert not (tmp_path / "projects").exists()
    project_dirs = sorted((tmp_path / "projects_pending_events").iterdir())
    assert len(project_dirs) == 6
    for project_dir in project_dirs:
        project_path = project_dir / f"{project_dir.name}.yml"
        project = YAML(typ="safe").load(project_path)
        assert project["data_assimilation"]["assimilation_events"] == []
        assert (project_dir / "PENDING_EVENTS").is_file()


def test_validate_options_refuses_existing_final_target(tmp_path: Path) -> None:
    options = SnapshotOptions(tmp_path / "source", tmp_path / "target", 2017, 2022, 100, PINNED_IMAGE)
    options.final_path.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        validate_options(options)


def test_validate_options_refuses_resampling_resolution(tmp_path: Path) -> None:
    options = SnapshotOptions(tmp_path / "source", tmp_path / "target", 2017, 2022, 250, PINNED_IMAGE)

    with pytest.raises(ValueError, match="native 100 m"):
        validate_options(options)


def test_write_json_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"

    write_json(output, {"z": 1, "a": [2, 3]})

    assert output.read_text(encoding="utf-8") == json.dumps(
        {"z": 1, "a": [2, 3]}, indent=2, sort_keys=True
    ) + "\n"


def test_builder_commit_uses_validated_environment_value(monkeypatch: pytest.MonkeyPatch) -> None:
    commit = "2" * 40
    monkeypatch.setenv("NORTH_TYROL_SNAPSHOT_BUILDER_COMMIT", commit)

    assert _builder_commit() == commit


def test_builder_commit_rejects_invalid_environment_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NORTH_TYROL_SNAPSHOT_BUILDER_COMMIT", "unknown")

    with pytest.raises(ValueError, match="40-character SHA"):
        _builder_commit()


def test_preflight_discovers_complete_contract_fixture(tmp_path: Path) -> None:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point, box

    source = tmp_path / "fram3s"
    roi_path = source / "01-data/01-aoi/TESTSITES/Testsite_North_Tyrol.gpkg"
    raw_regions_path = source / "01-data/01-aoi/SUBREGIONS/raw/subregions_avalanche_report_raw_25832.gpkg"
    polygons = [box(index * 1000.0, 0.0, (index + 1) * 1000.0, 1000.0) for index in range(8)]
    regions = gpd.GeoDataFrame(
        {"id": [f"AT-07-{index:02d}" for index in range(8)]}, geometry=polygons, crs="EPSG:25832"
    )
    roi_path.parent.mkdir(parents=True)
    raw_regions_path.parent.mkdir(parents=True)
    regions.to_file(roi_path, driver="GPKG")
    regions.to_file(raw_regions_path, driver="GPKG")

    forcing_dir = source / "01-data/02-meteo/01-data/01-initial/openamundsen-v2"
    forcing_meta_path = source / "01-data/02-meteo/02-meta/gpkg/meta-all.gpkg"
    forcing_dir.mkdir(parents=True)
    forcing_rows = []
    for index in range(163):
        station_name = f"S{index:03d}"
        forcing_rows.append(
            {
                "provider": "P",
                "stn_name": station_name,
                "stn_name_orig": station_name,
                "elev": 1000.0 + index,
                "geometry": Point(100.0 + (index % 8) * 1000.0, 500.0),
            }
        )
        if index < 159:
            timestamps = "2017-09-30 00:00:00\n2023-09-30 21:00:00\n"
        elif index == 159:
            timestamps = "2017-09-30 00:00:00\n"
        elif index == 160:
            timestamps = "2023-09-30 21:00:00\n"
        elif index == 161:
            timestamps = "2004-09-06 00:00:00\n2005-08-02 00:00:00\n"
        else:
            timestamps = "2023-10-09 00:00:00\n2024-12-02 00:00:00\n"
        rows = "".join(f"{timestamp},1,1,1,1,1\n" for timestamp in timestamps.splitlines())
        (forcing_dir / f"P.{station_name}.csv").write_text(
            "date,temp,precip,rel_hum,sw_in,wind_speed\n" + rows,
            encoding="utf-8",
        )
    forcing_meta_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(forcing_rows, crs="EPSG:25832").to_file(forcing_meta_path, driver="GPKG")

    snow_dir = source / "01-data/02-meteo/01-data/02-snow_obs/Tirol_snow_depth"
    snow_dir.mkdir(parents=True)
    snow_rows = []
    for index in range(35):
        station_id = f"SNOW{index:03d}"
        x = 100.0 + (index % 8) * 1000.0
        snow_rows.append(
            {"id": station_id, "name": station_id, "lat": 47.0, "lon": 11.0, "alt": 1200.0, "x": x, "y": 500.0}
        )
        (snow_dir / f"{station_id}.csv").write_text("time,snow_depth\n", encoding="utf-8")
    pd.DataFrame(snow_rows).to_csv(snow_dir / "stations_snow_depth.csv", index=False)

    fsc_dir = source / "50-eurac/SCF_Eurac_v3/SCF_Eurac_v3"
    fsc_dir.mkdir(parents=True)
    expected_counts = [114, 121, 116, 119, 145, 123]
    for start_year, count in zip(range(2017, 2023), expected_counts, strict=True):
        start = datetime(start_year, 10, 1)
        for offset in range(count):
            scene_date = start + timedelta(days=offset)
            (fsc_dir / f"SnowFLAKES_{scene_date:%Y%m%d}_v3_eurac.nc").touch()

    for relative in (
        "01-data/05-dem/euregio/dem_euregio_100.asc",
        "01-data/03-landcover/lc_eusalp/openAMUNDSEN-euregio/lc_euregio_100_eusalp.asc",
        "01-data/06-srf/euregio/srf_euregio_100.asc",
    ):
        path = source / relative
        path.parent.mkdir(parents=True)
        path.touch()

    options = SnapshotOptions(source, tmp_path / "target", 2017, 2022, 100, PINNED_IMAGE)

    result = preflight(options)

    assert result["status"] == "PREFLIGHT_OK"
    assert result["subdomain_count"] == 8
    assert result["forcing_spatial_file_count"] == 163
    assert result["forcing_station_count"] == 161
    assert result["forcing_outside_window_count"] == 2
    assert result["snow_station_count"] == 35
    assert result["fsc_scene_count"] == 738
