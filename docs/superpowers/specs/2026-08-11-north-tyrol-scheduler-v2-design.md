# North Tyrol Scheduler v2 and Transactional Refresh Design

Date: 2026-08-11

Status: approved for implementation

## Objective

Replace the first North Tyrol event selection with a deterministic, reusable
selection contract and refresh all six project configurations without running
the model. The scheduler remains independent of openAMUNDSEN-DA. A thin adapter
owns North Tyrol paths, observation normalization, configuration updates,
subdomain preparation, validation and atomic promotion.

The implementation supports both the original event-neutral snapshot and the
current documentation-shaped setup. Layout detection is strict and
unambiguous. An unknown or mixed layout is refused.

## Fixed scientific policy

The policy remains YAML-driven and versioned. Each hydrological-year project
uses fixed targets every six days from October 7 through July 31, alternating
`scf` and `station_hs` and starting with FSC. Candidates may be selected within
four days of a target. Consecutive filled targets must remain five to seven days
apart. A skipped target does not change later target types.

FSC quality is evaluated separately for every scene and subdomain:

- the reference footprint excludes water pixels;
- the archive-wide water mask must be identical in every retained scene;
  scene-to-scene water-mask variability is rejected rather than guessed;
- pixels that are nodata throughout the complete selected archive are excluded
  from the stable reference footprint;
- cloud fraction is cloud pixels divided by the reference footprint and must
  not exceed 20%;
- non-cloud invalid fraction is nodata pixels divided by the reference
  footprint and must not exceed 20%;
- cloud is not counted again as invalid;
- at least one valid FSC pixel is required and every valid FSC pixel must have a
  finite uncertainty value;
- a scene is a top-level candidate when at least one subdomain passes;
- observed snow fraction is not a selection criterion.

Fulfillment is measured against feasible targets: a target is feasible when at
least one candidate of its required type exists inside its search window. The
scheduler must fill at least 85% of feasible FSC targets and 85% of feasible
station-HS targets in the top-level project and in every leaf with feasible
support. It reports total, feasible, selected and unsupported target counts
separately so unavailable source data never appear as a selection failure.

Station-HS availability uses the unique nearest finite observation within half
one model timestep of the configured daily assimilation timestamp. For the
three-hour North Tyrol setup this is an inclusive 1.5-hour tolerance. Equal
nearest matches fail as ambiguous. A daily count or an observation farther away
on the same date is insufficient. Scheduling and audits retain the model
timestamp, matched observation timestamp, delta and active station IDs.

Selected observation dates are unique across both observation types and all six
projects. FSC source scenes are also unique within a project. Subject to the
project and leaf fulfillment constraints, FSC choices rank by total valid
support, uncertainty p90, uncertainty mean, target offset, date and source
identity. Station-HS choices rank by active DA support and then use the same
deterministic target-offset and identity fallbacks.

Fulfillment is a hard scheduling constraint, not a score that may be traded
between leaves. A memoized constraint solver tracks the remaining allowed
misses for every project/type and leaf/type constraint. It first computes the
maximum temporally compatible event count, then explores candidates in the
documented deterministic rank order and backtracks only when a later hard
constraint requires it. FSC source identities are part of that search state,
so a duplicate scene is skipped or causes backtracking rather than invalidating
an otherwise recoverable path after selection. Failed constraint states are
memoized. Safe branch-and-bound checks compare every constraint's remaining
required support with its remaining-slot maximum and compare total required
leaf support with the optimistic sum of the strongest remaining candidate
supports. These relax temporal and source conflicts, so they can reject only
mathematically impossible branches. A path that overfills one leaf therefore
cannot hide a shortfall in another, while feasible and infeasible
50-slot/eight-leaf problems remain bounded. Candidate ranking is applied only
within the maximum-cardinality schedules that satisfy every hard constraint.

## Shared station roles

One role split is shared by all six projects. Roles are mutually exclusive.

- Domains with one or two stations use every station for DA and have no
  holdout.
- Domains with three stations target one holdout.
- Domains with four or more stations target two holdouts.

Holdout selection first maximizes the exact-timestep temporal support retained
by the remaining DA stations across the candidate slot windows, then prioritizes
elevation spread among holdouts with station ID as the stable final tie breaker.
After scheduling, every feasible
station-HS event is checked against half-timestep-matched active IDs. If all active
stations in a domain are holdouts, the minimum deterministic set is promoted to
DA. Every reduction is recorded. No role may be both DA and benchmark.

## Scheduler architecture and outputs

`da_event_scheduler.py` contains only normalized-table logic and standard
Python data structures. `scheduleDAEvents.py` is its generic CLI. Policy schema
v2 adds explicit FSC reference-footprint fields, half-timestep station matching
and feasible-target fulfillment while retaining strict validation.

The scheduler produces deterministic records for:

- every target and whether it is feasible;
- every retained event and its target delta;
- FSC reference, cloud, invalid, valid and uncertainty metrics by supporting
  subdomain;
- station active IDs and DA/holdout counts at the selected timestamp;
- shared station roles and every adaptive reduction;
- unavailable, unfilled and quality-rejected targets;
- per-project fulfillment and global date-uniqueness validation.

The North Tyrol adapter also writes two scientific-QC layers outside the
runtime schedule. First, it pairs station snow depth with selected EURAC scenes
using the same half-timestep contract and reports the native FSC pixel and
native 3x3 neighborhood. Separate areal FSC summaries by elevation and land
cover provide spatial-support context; a point station is never compared as if
it were a subdomain mean. These semi-independent comparisons are consistency
evidence, not causal proof. Second, an exact-value forcing-flatline inventory
records long plateaus for review, including the Eissee/AT-07-22 investigation,
without silently modifying or rejecting forcing data.

The elevation-band width is an explicit scientific choice and is intentionally
not fixed by this implementation until separately approved. The areal helper
therefore requires a positive width instead of applying a hidden default.

Only `data_assimilation.assimilation_events` in each project YAML is the
authoritative final selection. The adapter removes legacy
`subdomain_event_filter` selection config. CSV and JSON outputs are audits,
never an alternate runtime schedule.

## North Tyrol adapter and layouts

Legacy snapshot mode retains the existing sibling-staging workflow and recorded
hash checks. It requires the builder's exact-timestep station inventory and
refuses legacy daily-only summaries because they cannot prove half-timestep
support. It also refuses FSC inventories that predate the stable-reference,
stable-water-mask and uncertainty-valid-count schema. Such inventories must be
rebuilt from retained NetCDF scenes; `uncertainty_count` is not accepted as a
proxy for complete uncertainty on valid FSC pixels.

Canonical refresh mode recognizes the documentation-shaped setup by its single
setup YAML and required `env/`, `grids/`, `meteo/`, `obs/`, `projects/` and
`raw/` directories. It normalizes FSC scene/subdomain quality from active
NetCDFs and polygons and station support from active station CSV timestamps and
the root station metadata. Preflight is read-only.

Normal refresh creates a complete same-filesystem sibling staging copy while
excluding heavy derived subdomain trees, results, restart states and model
logs. It never edits the accepted canonical root. Within staging it:

1. writes the shared finalized station roles;
2. updates all six top-level project schedules and observation-filter policy;
3. preserves ES50, compact retention and all unrelated DA parameters;
4. ensures openAMUNDSEN grid outputs include daily snow depth and daily SWE;
5. requires `freq: D` for both openAMUNDSEN outputs and nonempty compact metrics
   for both `snowdepth_daily` and `swe_daily`;
6. removes all derived leaf projects and regenerates them with the reviewed,
   digest-pinned image;
7. recreates deterministic observation files and steps without propagation;
8. validates the entire staged setup and automatically promotes it only after
   acceptance.

Before promotion, the digest-pinned core image runs its strengthened pre-run
assimilation-requirement validator over all 48 leaves. This proves same-ID
station CSV and model-point identity for active DA and benchmark roles and
exact half-timestep support for every authored station-HS event. The adapter
does not reimplement or weaken that runtime contract.

The image argument must contain an immutable `@sha256:` digest. It is recorded
but is not tied to the obsolete v0.9.4 digest.

## Configuration contract

The shared openAMUNDSEN setup must request at least:

```yaml
output_data:
  grids:
    variables:
      - {var: snow.depth, name: snowdepth_daily, freq: D}
      - {var: snow.swe, name: swe_daily, freq: D}
```

Existing instantaneous snow-depth output is preserved. Every project keeps
`snowdepth_daily` and `swe_daily` in the compact DA grid variables. The refresh
does not alter PF seeds, perturbations, likelihoods, uncertainty settings,
plots, maps or scientific inputs.

## Transaction and failure behavior

The adapter validates source layout, project identities, observation inputs,
schedule feasibility, station roles and image digest before copying. All
destructive cleanup happens only in staging. Staging failures are marked
`INCOMPLETE`; the canonical tree remains untouched.

A canonical refresh refuses runtime lock markers, a host model process or a
container mounted on the setup. Host discovery checks both command arguments
and `/proc/<pid>/cwd`, so a relative-path run started inside the setup is also
detected. Completed results, restart data or model state also cause refusal
unless the operator explicitly supplies
`--discard-runtime-artifacts` after reviewing the listed paths. That
acknowledgement never overrides a live process or lock.

After staged acceptance, the canonical root is renamed to a temporary sibling
backup and staging is renamed to the canonical name. The promoted path is
validated again. A post-swap failure restores the original root. The backup is
removed only after the promoted setup passes. Promotion is automatic after
validation; model propagation remains a separate explicit approval.

The final transaction manifest records the scheduler commit, policy digest,
immutable image, canonical parent identity, parent transaction/config digests,
discarded runtime paths and whether the transaction is only staged or fully
promoted. It hashes configuration provenance, not the scientific data tree.

## Acceptance

Acceptance requires:

- six top-level projects covering 2017/18 through 2022/23;
- globally unique selected timestamps and no duplicate FSC source scene;
- at least 85% selection of feasible targets for both types in every project
  and every leaf with feasible support;
- half-timestep station match evidence for every HS event;
- reference-footprint FSC evidence for every selected scene/subdomain;
- mutually exclusive shared roles with every domain of at most two stations
  entirely assigned to DA;
- both daily snow depth and daily SWE in the openAMUNDSEN and compact DA output
  contracts;
- 48 regenerated leaf projects with steps equal to retained leaf events plus
  initialization;
- successful strengthened core requirement validation for all 48 leaves;
- no results, restart state, model state, container output or propagation;
- only relative, contained preparation links;
- deterministic audits and a final transaction manifest.

## Focused verification

Minimal synthetic tests cover reference-footprint FSC quality, feasible-target
fulfillment, cross-type/global uniqueness, half-timestep station matching, adaptive
sparse-domain roles, deterministic audits, both layout contracts, required snow
outputs and rollback-safe promotion. Existing snapshot-builder tests cover the
maintained generated configurations. No model propagation is part of this
workstream.
