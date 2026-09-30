# Fram3S aligned grids

The builder produces 75 stacks: five regions, three terrain contexts and five resolutions. The publisher keeps matching terrain files in place and replaces the AOI delivery after validation. Superseded files and raw sources move into a dated central archive.

## Delivery contract

- CRS: ETRS89 / UTM zone 32N, EPSG:25832, aligned from `(0, 0)`.
- Regions: Euregio, Tyrol including North and East Tyrol, North Tyrol, South Tyrol and Trentino.
- Resolutions: 50, 100, 250, 500 and 1000 m.
- Contexts: 0, 5000 and 10000 m. The buffered polygon determines a rectangle snapped outward to 1000 m. All resolutions share that rectangle; 100 and 250 m cells do not nest directly.
- **ROI always represents the original region.** Cell centers inside are 1, outside are valid 0, NoData is 255. Context variants have identical geographic core cells at each resolution.
- DEM, land-cover, SRF and SVF NoData is −9999. Continuous values keep their fractional precision.

There are exactly **1145 files** under the five active layer roots:

- 1125 GeoTIFF, ASCII and `.prj` files: five layers × 75 stacks × three files.
- 15 Shapefile ZIPs under `01-aoi/<region>/buffer_<context>m/`.
- Five AOI root files: `README.txt`, `aoi.gpkg`, `aoi_overview.qgz`, `aoi_overview.png` and `aoi_overview.pdf`.

The central GeoPackage contains `boundaries` (5 original regions), `grid_extents` (75 specifications), `roi_partitions` (30 features: inside/outside for 15 rectangles) and `subregions` (90 features with original attributes). ZIPs contain the same two-part partitions. Detailed boundaries are not rounded or simplified. Invalid Tyrol geometry is repaired with `make_valid`, as recorded outside delivery.

The Tyrol 100 m / 5 km grid has bounds `(578000, 5167000, 808000, 5298000)` and 2300 columns × 1310 rows. Use the standard `tyrol/buffer_05000m/100m` paths. There is no person-specific package. The existing station-availability audit covers North Tyrol only; it does not establish East Tyrol availability.

## Numerical processing

DEM uses valid-source area-weighted averages from the 20 m source. Land cover selects the class with greatest source coverage among 13 classes; ties use the lowest class code. Source glacier treatment is retained. Uncovered rectangle cells remain NoData; partial cells average only their valid source area.

SRF runs `calculateSRF.py` on each parent with the existing nominal 100 m and 5 km openness radii, elevation scaling and parent-wide mean normalization. The wrapper serializes openness because its shared minimum array is unsafe in parallel, then restores the caller's thread setting. Coarse cells retain the formula's rounded-up search radius. SVF runs the openAMUNDSEN terrain routine on each parent with 10° azimuth spacing and one sweep. Regional grids are crops of these shared parents.

Every ROI cell must have valid data. Source coverage fractions, potential SRF neighborhood truncation and SVF proximity flags remain in the work directory's parent rasters and diagnostics. The SVF flag is a proximity indicator, not an error bound. Cleanup does not recompute or alter terrain values.

## Execution

Use image `ghcr.io/openamundsen/openamundsen-da@sha256:f3834a701e116b9ab11c50677d94236bffcd5d9adb045ae6b871b3ccf2c98723`. Its Python is `/opt/conda/envs/openamundsen_da/bin/python`. Mount code and sources read-only, and a fresh work directory read-write. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, a writable `NUMBA_CACHE_DIR` and an appropriate `NUMBA_NUM_THREADS` limit.

```text
python buildFram3sGrids.py prepare --source /source --work /work
python buildFram3sGrids.py resolution --work /work --resolution 1000
python buildFram3sGrids.py resolution --work /work --resolution 500
python buildFram3sGrids.py resolution --work /work --resolution 250
python buildFram3sGrids.py resolution --work /work --resolution 100
python buildFram3sGrids.py resolution --work /work --resolution 50
python buildFram3sGrids.py finish --work /work
```

For cleanup of the previous buffered-ROI collection, reuse its completed local build:

```text
python buildFram3sGrids.py cleanup --source /previous-work --work /fresh-work
```

This verifies hashes of all 900 non-ROI files, hard-links them into staging, preserves geometry, regenerates 75 masks, validates all raster pairs against the parent grids and rebuilds vectors and overview. Both work directories must be on the same local filesystem and mounted at stable paths. Do not modify linked terrain files. Use a fresh work directory; completed checkpoints must not be reused across changed code or inputs.

Generate and reopen the project in separate native QGIS processes:

```text
python qgisFram3sProject.py --root <native-stage>/01-aoi --records <external-records>
python qgisFram3sProject.py --root <native-stage>/01-aoi --records <external-records> --validate-only
```

Windows QGIS requires native Windows staging for GeoPackages; SQLite locking through WSL UNC paths is unreliable. Copy the staged AOI to a native Windows folder, generate and validate there, then copy the `.qgz` and external validation records back. All project sources are relative. The headless Qt runtime explicitly loads installed Windows Arial if necessary. Validation requires Arial, 75 transparent-outside ROI renderers, all vector features, 90 subregions and the opening layer set: Euregio 100 m / 5 km ROI plus Tyrol, South Tyrol and Trentino outlines. The other masks and all subregion views start off. One extent layer per region/context avoids duplicate outlines.

```text
python buildFram3sGrids.py finalize --work /work
```

Finalization requires exactly 1145 whitelisted files and successful QGIS evidence, validates the vectors and core-cell invariance, then hashes the final artifacts. JSONs, CSV inventories, source manifests, diagnostics, previews and logs remain outside `output/`. The overview contains five region panels showing original ROIs and context rectangles.

## Publication and rollback

```text
powershell -File publishFram3sGrids.ps1 -WorkRoot <work> -DestinationRoot F:\fram3s\01-data -RecordRoot <records>
powershell -File publishFram3sGrids.ps1 -WorkRoot <work> -DestinationRoot F:\fram3s\01-data -RecordRoot <records> -Publish
```

The first call checks structural prerequisites without changing data. Publication verifies original input hashes, hashes every retained terrain file against staging and hashes replacements before moving anything. Existing terrain that differs or unexpected active files stop publication before archive moves. New terrain files may be published into an empty delivery; replacing existing terrain is a separate operation.

The publisher moves the old AOI, existing layer archives, raw land-cover/forest/glacier folders, redundant XMLs and layer READMEs into `fram3s/90-archive/grid_collection_cleanup/<UTC timestamp>/`, preserving paths relative to `01-data`. It writes a journal before moves. Same-volume moves are checked against a complete path, byte-size and modification-time inventory; raw archive files are not all rehashed. The four original processing inputs are rehashed before moving. Every retained delivery file and every copied file is checked with SHA-256; copied files are checked again at destination. Final file paths must match the whitelist exactly.

Reopen the published QGIS project in a fresh process with validation output outside delivery. Check the published whitelist again after QGIS closes. For rollback, move only the journal's copied files aside and reverse its archive moves. Keep retained terrain files and unlisted files untouched. An interrupted publication must be reconciled from its journal before another run.

To rebuild from originals after central archival, point `--source` at the dated central archive and use `--archive-stamp` for the earlier per-layer archive timestamp. This reconstructs the four original paths into a fresh local input cache and records their provenance.

## Tests

```text
python -m pytest -q tests/test_fram3s_grids.py
python tests/verify_fram3s_publication.py
```

The second command requires Windows. Numerical tests cover alignment, area aggregation, categorical ties, NoData, source edges, core-cell invariance, independent ASCII metadata, the delivery whitelist and SRF reproducibility. The Windows fixture covers dry runs, changed input/terrain rejection, unexpected-file rejection, raw-source archival, unchanged terrain and all 1145 final hashes, including Windows 8.3 paths. GitHub Actions runs both jobs. Production validation additionally checks every full raster export and all detailed vector partitions.
