# GeoConverter

`geoConverter.py` retains the original config-driven SHP/GPKG/GeoJSON conversions
and adds repeatable ROI exports. Use Python with GeoPandas, Shapely 2, NumPy and
GDAL Python bindings (`osgeo`). QGIS 3.44.8's Python environment on the workstation
already provides these dependencies. Tests additionally require pytest.

## Saved ROI jobs

Run from the repository root:

```console
python 01-helpers/geoConverter.py --mode roi2shp
python 01-helpers/geoConverter.py --mode roi2shp --roi-job north_tyrol
python 01-helpers/geoConverter.py --mode roi2shp --roi-job oetztal
```

In Windows PowerShell, use QGIS's environment instead of a plain Python install:

```powershell
& 'C:\Program Files\QGIS 3.44.8\bin\python-qgis-ltr.bat' '\\wsl.localhost\Ubuntu\home\franz\workspace\repos\scripts\01-helpers\geoConverter.py' --mode roi2shp
```

Both outputs use **ETRS89 / UTM zone 32N (EPSG:25832)**. Each ZIP contains a single
2D polygon shapefile with exactly two records and one integer attribute:
`ROI=1` inside, `ROI=0` in the surrounding rectangle. Holes and disconnected
parts are preserved in their respective records. No simplification is applied.

| Job | Source under `F:\fram3s\01-data\01-aoi` unless stated | Output under the same base |
| --- | --- | --- |
| North Tyrol | `PROVINCE_BOUNDARY\province_boundary_4326.gpkg` | `EUREGIO_BOUNDARY\north_tyrol_roi_25832.zip` |
| Ötztal | `M:\Ötztal\openamundsen\oetztal\grids\roi_oetztal_50.asc` | `OETZTAL\oetztal_roi_25832.zip` |

North Tyrol is the larger western component of `province=tyrol`; East Tyrol is
excluded. After projection, its axis-aligned bounding rectangle is extended by
5,000 m on each side. The merged Euregio boundary lacks internal province borders
and is intentionally not used.

Ötztal retains its complete raster extent and exact pixel edges. The ASCII source
has no embedded CRS; EPSG:25832 is assigned explicitly from its model YAML.
Rasters must be north-up, single-band and contain both 0 and 1, without NoData or
other classes. An embedded CRS that conflicts with EPSG:25832 is rejected.

## Repeating exports

The saved paths, target CRS and North Tyrol margin live in the `ROI_*` configuration
block. `--roi-output-dir` redirects the selected jobs to an **existing** directory:

```console
python 01-helpers/geoConverter.py --mode roi2shp --roi-output-dir C:/temp/fresh-roi
```

An existing ZIP causes a nonzero exit before either saved job is started. ROI
exports never overwrite files, even if the legacy `OVERWRITE` option is enabled.
Choose a fresh output directory or move previous ZIPs before repeating a job.

Both jobs are prepared and verified in temporary storage before delivery. Inputs
are hashed before and after preprocessing. The existing GPKG-to-SHP converter
performs the final vector conversion; no intermediate files are left alongside
inputs or final ZIPs. Each archive contains `.shp`, `.shx`, `.dbf`, `.prj` and `.cpg`
files at its root.

Every staged and delivered ZIP is reopened to verify its schema, CRS, valid
geometry, boundary preservation, nonoverlapping areas and complete rectangle
coverage. Floating-point overlay residuals must be below 0.001 square meters.
Ötztal must also reproduce every source raster cell when rasterized back onto the
original grid. JSON validation results and archive SHA-256 digests are printed.
A filesystem failure during delivery can leave a partial output; inspect it and
move it aside before retrying. The tool does not delete or replace existing files.

## Tests

```console
python -m pytest -q 01-helpers/tests/test_geo_converter.py
```

Synthetic tests cover holes, disconnected pieces, exact raster round trips,
missing or conflicting raster CRS, invalid raster classes, province selection,
5 km padding, overwrite protection and the original GPKG-to-SHP conversion.
