---
published: false
---

# Canonical North Tyrol Station Membership Repair

## Problem

The canonical North Tyrol setup stores station identity, role, uncertainty and
coordinates in `obs/stations/stations_da_metadata.csv`, but intentionally does
not persist a scheduler-only `subdomain_id` column. The canonical scheduler
adapter currently requires that column and therefore fails preflight before it
can rebuild the six project schedules.

The accepted `env/subdomains.gpkg` is the authoritative, non-overlapping
subdomain geometry. A read-only P8 audit confirmed that every one of the 35
station coordinates is covered by exactly one of its eight polygons.

## Design

The canonical adapter will derive station membership in memory before invoking
the source-independent scheduler:

1. Read the accepted subdomain polygons from `env/subdomains.gpkg` and require
   unique, non-empty `id` values in EPSG:25832.
2. Parse finite `x` and `y` coordinates from every station metadata row.
3. Use polygon `covers` semantics so a station exactly on a boundary is not
   silently discarded.
4. Require exactly one covering polygon for every station. Zero or multiple
   matches fail with the station ID and matching polygon IDs.
5. Add `subdomain_id` only to normalized in-memory copies passed to the
   scheduler. Do not rewrite `stations_da_metadata.csv` and do not consult the
   prior `raw/metadata/da_selection_audit.csv`, which is an output of an older
   schedule rather than an authoritative spatial input.

Both `--preflight` and the transactional canonical refresh already enter the
same schedule-building path, so one adapter change covers both workflows. The
generic scheduler, project YAML ownership, observation timing and scientific
roles remain unchanged.

## Failure and compatibility behavior

- Missing or invalid station coordinates fail before schedule selection.
- Missing, invalid, duplicate-ID or wrong-CRS subdomain geometry fails before
  schedule selection.
- Ambiguous boundary/overlap membership fails rather than choosing by file
  order or nearest geometry.
- The legacy snapshot adapter remains unchanged because its normalized station
  inventory already contains `subdomain_id`.
- No openAMUNDSEN-DA core, CLI or scientific configuration changes are needed.

## Focused validation

Add tests proving:

- deterministic one-to-one membership for stations in multiple subdomains;
- boundary inclusion through `covers`;
- clear failures for zero and multiple matches, invalid coordinates, duplicate
  polygon IDs and a CRS other than EPSG:25832;
- canonical schedule construction succeeds without a persisted
  `subdomain_id` and leaves the source metadata byte-identical;
- legacy scheduling behavior is unchanged.

Run only the North Tyrol scheduler/finalizer test module and the existing
snapshot configuration checks in the pinned immutable core image. After green
CI, use the reviewed scripts commit on P8 and rerun the read-only preflight
before any transactional refresh.
