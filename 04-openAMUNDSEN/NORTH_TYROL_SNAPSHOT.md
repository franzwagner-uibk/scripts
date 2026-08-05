# North Tyrol multi-season snapshot builder

`buildNorthTyrolSnapshot.py` creates a self-contained, event-neutral 100 m
North Tyrol setup. It preserves byte-identical selected Fram3S inputs under
`data_raw/`, writes native-resolution working subsets under `data_working/`
and creates complete forcing, station and FSC candidate inventories.

For the 2017-09-30 forcing lookback through 2023-09-30 snapshot window, the
fixed station contract contains 161 stations. Selection first requires a station
file within the eight-subdomain ROI plus 10 km, then retains every source whose
timestamp extent overlaps that window. Sources entirely before or after the
window are excluded; partial per-variable coverage is retained and reported by
the inventories rather than silently filled.

EURAC GeoTransform metadata is accepted from either the CF grid-mapping
variable or the NetCDF dataset attributes. It must match the native x/y
coordinates exactly; each lossless crop receives a correspondingly shifted
GeoTransform and is never resampled.

The builder deliberately writes projects below `projects_pending_events/` with
empty `assimilation_events` and a `PENDING_EVENTS` marker. Do not move a project
to `projects/` until its observation events and provisional DA settings have
been reviewed.

## P8 preflight

Run inside the pinned openAMUNDSEN-DA image with Fram3S mounted read-only:

```bash
python /scripts/04-openAMUNDSEN/buildNorthTyrolSnapshot.py \
  --source-root /mnt/fram3s/fram3s \
  --target-root /home/franz/workspace/openamundsen_da_runs/north_tyrol_subdomain_runs \
  --start-year 2017 \
  --end-year 2022 \
  --resolution 100 \
  --image ghcr.io/openamundsen/openamundsen-da:0.9.4@sha256:f3834a701e116b9ab11c50677d94236bffcd5d9adb045ae6b871b3ccf2c98723 \
  --preflight
```

If Fram3S exposes `01-data/` directly below `/mnt/fram3s`, use that directory as
`--source-root`. The builder never searches for or falls back to a different
source root.

Remove `--preflight` only after reviewing its counts and target. A normal build
refuses an existing final target, builds in a timestamped staging directory and
atomically promotes only a snapshot that passes all validations.

## Tests

```bash
docker run --rm \
  -v /home/franz/workspace/repos/scripts:/work:ro \
  -w /work \
  ghcr.io/openamundsen/openamundsen-da:0.9.4@sha256:f3834a701e116b9ab11c50677d94236bffcd5d9adb045ae6b871b3ccf2c98723 \
  python -m pytest -q 04-openAMUNDSEN/tests/test_north_tyrol_snapshot.py
```
