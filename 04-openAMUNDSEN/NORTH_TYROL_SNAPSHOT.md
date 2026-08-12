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
The shared openAMUNDSEN output configuration keeps those 161 forcing points and
adds the 35 selected snow stations under their independent station IDs. Duplicate
coordinates are valid, but duplicate point names fail snapshot generation.
Working station files end at the last native source timestamp aligned with the
3 h model grid. Any later incomplete off-grid tail rows are omitted and counted
explicitly in the forcing coverage inventory; they are not aggregated or filled.

EURAC GeoTransform metadata is accepted from either the CF grid-mapping
variable or the NetCDF dataset attributes. It must match the native x/y
coordinates exactly; each lossless crop receives a correspondingly shifted
GeoTransform and is never resampled.

The builder deliberately writes projects below `projects_pending_events/` with
empty `assimilation_events` and a `PENDING_EVENTS` marker. Do not move a project
to `projects/` until its observation events and provisional DA settings have
been reviewed.

## Event scheduling and finalization

`scheduleDAEvents.py` is the generic interface to the pure normalized-table
scheduler in `da_event_scheduler.py`. The current North Tyrol policy is
versioned at `policies/north_tyrol_alternating_6day_v3.yml`; the v2 policy remains
available unchanged for reproducibility. Version 3 uses alternating
FSC and station-HS targets every six days from October 7 through July 31. FSC
selection uses a stable reference footprint that excludes water and pixels that
are nodata throughout the selected archive. The water mask must be identical in
every retained scene; variability fails instead of being inferred. A domain
passes with at most 20% cloud, at most 20% non-cloud invalid pixels, at least
one valid pixel and finite uncertainty for every valid pixel. Candidate scenes
rank by valid support,
uncertainty p90 and mean, target offset and date after the project and leaf
fulfillment constraints are satisfied.

The fixed targets are not moved. FSC and station observations are matched
within four days so retained adjacent slots can satisfy the five-to-seven-day
gap contract despite irregular acquisitions. Station observations are matched
to the unique nearest value within half the model timestep of the configured
daily assimilation time; for this 3 h setup the inclusive tolerance is 1.5 h.
When exactly two equally near values symmetrically bracket the model time,
version 3 uses their arithmetic mean if both lie inside that window and their
timestamps are no more than 24 h apart. The real source offset still controls
candidate ranking, so exact observations are preferred. A missing slot
remains an explicit exception and does not alter later slot types. Each
observation type must fill at least 85% of its feasible annual targets in the
top-level project and in every leaf with feasible support.

The scheduler treats those fulfillment limits as hard constraints. It
maximizes the temporally compatible retained-event count, then resolves the
maximum-cardinality feasible schedule in deterministic per-target candidate
rank order with memoized backtracking. This preserves leaf-specific support
without materializing the combinatorial cross-product of all leaf miss counts.

To inspect a normalized inventory without creating output:

```bash
python 04-openAMUNDSEN/scheduleDAEvents.py \
  --policy 04-openAMUNDSEN/policies/north_tyrol_alternating_6day_v3.yml \
  --fsc-inventory SNAPSHOT/inventories/fsc_scene_subdomain_quality.csv \
  --snow-inventory SNAPSHOT/inventories/snow_station_timestep_support.csv \
  --station-metadata SNAPSHOT/data_working/obs/stations/stations_snow_depth.csv \
  --start 2022-10-01 \
  --end 2023-09-30 \
  --output-dir /unused/in/preflight \
  --preflight
```

`finalizeNorthTyrolProjects.py` is the North Tyrol adapter. It accepts both the
original event-neutral snapshot and the current documentation-shaped canonical
setup, provided a legacy snapshot contains the exact-timestep station inventory;
daily station summaries are intentionally not accepted. A legacy FSC inventory
must also contain stable reference/water-mask evidence and an explicit valid-FSC
uncertainty count. Older inventories must be rebuilt from the retained NetCDF
scenes and cannot fall back to a generic uncertainty count. Preflight is read-only.
A canonical refresh builds a same-filesystem
sibling while excluding old derived leaves and runtime artifacts, writes only
the reviewed final event lists to each project YAML, removes the legacy event
filter, regenerates all 48 leaves and validates their steps without propagation.
It swaps the sibling into the canonical path only after acceptance and restores
the original on any post-swap failure.

The canonical station table remains documentation-shaped and does not require a
scheduler-only `subdomain_id` column. The adapter derives membership in memory
from each station's EPSG:25832 `x`/`y` coordinates and the accepted
`env/subdomains.gpkg`. Every station must be covered by exactly one valid,
uniquely identified subdomain polygon; missing or ambiguous membership fails
without rewriting station metadata.

Canonical refresh refuses runtime locks, live model processes and containers
mounted on the setup. Host checks inspect both process arguments and
`/proc/<pid>/cwd`, including relative-path commands started inside the setup.
It also refuses completed results, restart data or model state by default.
After the listed runtime artifacts have been reviewed, the
explicit `--discard-runtime-artifacts` option authorizes their replacement in
the staged transaction; it never overrides a live lock or process.

The refresh writes deterministic target, event, quality, leaf-selection,
exception and shared-role audits under `raw/metadata/`. It also inventories
exact forcing flatlines and pairs co-temporal station snow depth with the native
EURAC pixel and 3x3 neighborhood. It also writes areal FSC context in fixed
250 m elevation bands and by land-cover class, assigning native FSC pixels to
the containing native 100 m DEM and land-cover cell without resampling. These tables support review only; they never
replace the final `data_assimilation.assimilation_events` lists in project YAML.

The reviewed digest-pinned core image validates all 48 regenerated leaves
before promotion. Active DA and benchmark stations must have same-ID
observation CSVs and configured model points, and each station-HS event must
have exact half-timestep support. Daily openAMUNDSEN snow-depth and SWE outputs
must use `freq: D`; both compact variables must request at least one metric.
The canonical transaction manifest records the scheduler commit, policy and
image digests, parent provenance, discarded runtime paths and final promotion
result.

The accepted source polygons contain small overlaps that v0.9.4 correctly
refuses. The adapter leaves the hashed working GeoPackage unchanged and derives
a preparation-only partition in sorted subdomain-ID order. It removes overlap
from later polygons while preserving the exact union, then records source and
output hashes and removed areas under `provenance/`.

Recorded files under `data_working/` remain byte-identical. Final station roles
live in `data_finalized/obs/stations/`, which links to the immutable station
series and contains only the finalized role metadata as a new file. Project
configs reference this finalized layer; preparation copies its spatial subset.

openAMUNDSEN-DA v0.9.4 separates subdomain-tree preparation from the
observation/step preparation normally entered by `subdomains run`. The adapter
runs the official `subdomains prepare` CLI and then invokes that exact pinned
v0.9.4 preparation routine for each subdomain without entering propagation.
Any `results`, restart state or model artifact fails final acceptance.

```bash
python 04-openAMUNDSEN/finalizeNorthTyrolProjects.py \
  --setup-root /home/franz/workspace/openamundsen_da_runs/north_tyrol_subdomain_runs/north_tyrol_subdomains_100m \
  --policy 04-openAMUNDSEN/policies/north_tyrol_alternating_6day_v3.yml \
  --image ghcr.io/openamundsen/openamundsen-da@sha256:REVIEWED_CORE_IMAGE_DIGEST \
  --preflight
```

Remove `--preflight` only from a clean reviewed scripts commit. When Git is not
available to the finalizer runtime, set `NORTH_TYROL_FINALIZER_COMMIT` to the
exact reviewed 40-character Git commit. The final state is `READY_TO_RUN`; ES50
model execution is deliberately a separate approval gate.

If and only if reviewed runtime artifacts are intentionally being replaced,
add `--discard-runtime-artifacts` to the non-preflight command.

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

When the runtime image does not contain Git, set
`NORTH_TYROL_SNAPSHOT_BUILDER_COMMIT` to the exact 40-character commit checked
out on the host. The build fails rather than writing unknown commit provenance.

## Tests

```bash
docker run --rm \
  -v /home/franz/workspace/repos/scripts:/work:ro \
  -w /work \
  ghcr.io/openamundsen/openamundsen-da:0.9.4@sha256:f3834a701e116b9ab11c50677d94236bffcd5d9adb045ae6b871b3ccf2c98723 \
  python -m pytest -q -p no:cacheprovider \
    04-openAMUNDSEN/tests/test_north_tyrol_snapshot.py \
    04-openAMUNDSEN/tests/test_da_event_scheduler.py
```
