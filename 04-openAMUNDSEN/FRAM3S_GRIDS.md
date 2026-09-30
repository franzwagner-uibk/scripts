# Fram3S aligned grids

`buildFram3sGrids.py` builds a staged collection of five regions, three true polygon-buffer variants and five resolutions. It does not overwrite or archive source data. `qgisFram3sProject.py` creates the portable QGIS project using an installed PyQGIS runtime. `publishFram3sGrids.ps1` verifies source hashes, archives replaced files and publishes the validated collection using native Windows network-drive access.

## Contract

- CRS: EPSG:25832; coordinate origin `(0, 0)`.
- Regions: Euregio, Tyrol including East Tyrol, North Tyrol, South Tyrol and Trentino.
- Buffers: 0, 5,000 and 10,000 m around the actual polygon.
- Resolutions: 50, 100, 250, 500 and 1,000 m. Each region/buffer uses one rectangle snapped outward to 1,000 m.
- ROI: cell center inside gives 1; outside gives valid 0. NoData is distinct.
- DEM: area-weighted average of valid 20 m source elevations.
- Land cover: greatest valid source area among the existing 13 classes; equal-area ties select the lowest code. Existing glacier classification is retained as part of that source.
- SRF: `calculateSRF.py` on each parent DEM, with its existing formula, nominal 100 m and 5 km radii, elevation scaling and mean normalization. Normalization is over valid parent cells, never individual regional crops.
- SVF: openAMUNDSEN sky-view calculation on each full parent, azimuth spacing 10 degrees and one sweep. Regional products are crops, not independently recalculated edge treatments.

The parent source extent is snapped outward to 1 km. Values outside available source coverage remain NoData. Target cells with partial source coverage average the valid source fraction; that fraction is provided explicitly. Complete coverage of every ROI cell is required, and edge limitations are reported separately.

The upstream openness implementation updates a shared minimum array inside a parallel loop. The SRF wrapper runs that calculation with one Numba thread and restores the caller's thread setting afterward. This avoids nondeterministic lost minima without changing the formula. Coarse grids still use the legacy rounded-up cell search radius; see each parent's metadata for effective distances.

## Reproducible execution

The validated execution environment is the existing image ID `sha256:f3834a701e116b9ab11c50677d94236bffcd5d9adb045ae6b871b3ccf2c98723` (local tag `ghcr.io/openamundsen/openamundsen-da:0.9.4`). The source manifest records package versions. Python is `/opt/conda/envs/openamundsen_da/bin/python` in that image.

Mount the source data root read-only at `/source`, this directory read-only at `/code`, and a new local work directory read-write at `/work`. Set the working directory to `/code`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `NUMBA_NUM_THREADS=24` and a writable `NUMBA_CACHE_DIR` outside the checkout.

```text
python buildFram3sGrids.py prepare --source /source --work /work
python buildFram3sGrids.py resolution --work /work --resolution 1000
python buildFram3sGrids.py resolution --work /work --resolution 500
python buildFram3sGrids.py resolution --work /work --resolution 250
python buildFram3sGrids.py resolution --work /work --resolution 100
python buildFram3sGrids.py resolution --work /work --resolution 50
python buildFram3sGrids.py finish --work /work
```

Run the QGIS script with the installed QGIS Python launcher:

```text
python qgisFram3sProject.py --root <work>/output/01-aoi
python qgisFram3sProject.py --root <published>/01-aoi --validate-only
```

After QGIS generation, refresh the artifact manifest with `finish`. Source and output directories must be different. Use a fresh work directory for changed sources or processing code; checkpoints support continuing interrupted work with the same inputs and implementation, not mixing revisions.

## Outputs and review

The work directory contains source hashes, source-pixel statistics, parent rasters, per-resolution validation, and a publishable `output/` subtree. Layer folders retain their established names. Each region/buffer/resolution contains GeoTIFF and ASCII grids. ASCII uses nine significant digits and explicit projection sidecars.

AOI output includes all boundary/buffer/extent geometries, ROI partition Shapefile ZIPs, a 90-feature subregion GeoPackage with original attributes, the map PNG/PDF, the QGIS project, and the machine-readable grid inventory. The QGIS project uses relative paths and references active subregion files, not archived paths.

The Kathi core mask uses the original North Tyrol boundary on the 100 m grid with a 5 km context envelope: `(578000, 5175000, 784000, 5298000)`, 2060 columns by 1230 rows. It differs intentionally from the buffered region's ROI mask.

Quality mask bits are additive: 1 = missing DEM, 2 = partial source coverage, 4 = potential SRF neighborhood truncation, 8 = proximity within 5 km to missing terrain for SVF review. The SRF flag uses a conservative square neighborhood. The SVF flag is not a quantitative error bound or proof that more distant cells are unaffected.

Archive and publish only after all 75 stacks pass validation, vector/subregion preservation is verified, and QGIS opens all layers. Preserve raw land-cover/glacier/forest inputs. Archive legacy generated grids and all old AOI contents with hashes and a reversible old-to-new path manifest. Keep historical model runs unchanged.

## Tests

```text
python -m pytest -q tests/test_fram3s_grids.py
```

Tests cover outward snapping, weighted aggregation, categorical ties and missing data, source-edge coverage, ROI partition behavior and SRF numerical reproducibility. Production validation additionally reads every export back and compares it with its parent crop.

## Rebuilding after archiving

The originals remain in the publication timestamp's archives. On a fresh work directory, use:

```text
python buildFram3sGrids.py prepare --source /source --archive-stamp <timestamp> --work /work
```

This reconstructs the four required source paths in a local snapshot and records their archived provenance. Use the archive timestamp from the publication journal. For slow mounted I/O, an existing local snapshot may be used only after its hashes have been verified against the authoritative source files.

On Windows, validate the publication prerequisites first, then pass `-Publish` for the authorized publication:

```text
powershell -File publishFram3sGrids.ps1 -WorkRoot <work> -DestinationRoot F:\fram3s\01-data -RecordRoot <report>
powershell -File publishFram3sGrids.ps1 -WorkRoot <work> -DestinationRoot F:\fram3s\01-data -RecordRoot <report> -Publish
```

The journal is written before archival moves. It records every source-to-archive move and each archived file's SHA-256. No publication collision is overwritten. If publication is interrupted, use the journal and artifact manifest to identify the generated files, move those aside, then reverse the recorded archive moves. Do not restore by deleting unlisted files.

Windows QGIS cannot reliably open GeoPackages through the WSL UNC filesystem because of SQLite locking. Generate and check the project against a native Windows staging copy of the referenced AOI files, then copy the project back. Relative paths are verified in the project XML. Reopen the published project directly on the Windows drive to validate the final location.
