# DA Event Scheduler and North Tyrol Finalization Design

Date: 2026-08-05

Status: approved for implementation

## Objective

Add a reusable, source-independent observation-event scheduler and a thin North Tyrol adapter that finalizes the accepted six-year snapshot without running model propagation. The final snapshot contains six ensemble-size-50 projects, prepared subdomain projects and deterministic steps, and is marked `READY_TO_RUN`.

## Scope and invariants

- Cover hydrological years 2017/18 through 2022/23.
- Preserve all accepted raw and working data bytes and their existing provenance.
- Keep scheduler logic independent of snapshot paths and openAMUNDSEN-DA imports.
- Keep source-path discovery, YAML mutation, container execution and promotion in the North Tyrol adapter.
- Use the reviewed, digest-pinned openAMUNDSEN-DA v0.9.4 image.
- Run only `openamundsen-da subdomains prepare`; never propagate the model.
- Build and validate a complete sibling staging tree before replacing the canonical snapshot.
- Restore the accepted snapshot if post-swap validation fails.

## Interfaces

```text
scheduleDAEvents.py
  --policy PATH
  --fsc-inventory PATH
  --snow-inventory PATH
  --station-metadata PATH
  --start YYYY-MM-DD
  --end YYYY-MM-DD
  --output-dir PATH
  [--preflight]
```

```text
finalizeNorthTyrolProjects.py
  --setup-root PATH
  --policy PATH
  --image IMAGE@DIGEST
  [--preflight]
```

Preflight reads, normalizes, schedules and validates without writing output or changing the snapshot. Normal scheduler execution refuses a non-empty output directory. Normal finalization refuses an unexpected snapshot state, builds in a timestamped sibling, and promotes only a fully accepted staging tree.

## Scheduling policy

The versioned YAML policy defines a fixed six-day target cadence from October 7 through July 31. Slots alternate `scf` and `station_hs`, starting with FSC. Unavailable slots are skipped without changing the type of later slots, and each skip is retained as an explicit exception.

FSC candidates are searched within four days of a target. A scene is admitted when at least one subdomain has cloud fraction at or below 20%. Observed snow fraction is not a filter. Candidate choice is deterministic and prioritizes filled slots, supported subdomains, lower cloud and uncertainty, target proximity and earlier dates. A scene may be selected once, and duplicate event dates are forbidden.

Station-HS candidates prioritize the number of subdomains with DA support, active holdouts, total active DA stations, target proximity and earlier dates. Adjacent retained slots normally have actual dates five to seven days apart. A longer gap is permitted only across an explicitly skipped slot. Every project must fulfill at least 85% of the FSC targets and 85% of the station-HS targets.

## Station roles

One station split is shared by all six projects, and DA and holdout roles are mutually exclusive. Per subdomain, target two holdouts when at least four stations exist, one holdout when two or three stations exist, and no holdout for a single-station domain. Selection prioritizes temporal coverage and then elevation spread with stable station-ID tie breaking.

The split must preserve DA support for scheduled station-HS events. A domain's holdout target may be reduced only when needed to retain DA support, and every reduction is recorded. FSC has no holdouts.

## Outputs

The scheduler writes deterministic CSV detail tables and JSON summaries for target slots, retained events, station roles, quality metrics and exceptions. Records include target and selected dates, selection deltas, supporting subdomains or stations, source identity and deterministic ranking fields.

The finalizer copies the six pending project skeletons to an authoritative `projects/` tree, changes only the event schedule, station-role references, `ensemble_size: 50` and `output.retention: compact`, and preserves other accepted DA parameters. It records the parent manifest, hashes the former pending tree, and writes a finalization manifest containing the reviewed commit, policy hash, pinned image digest, parent-manifest hash, final tree hash, event counts and per-domain dropped events.

## Preparation and promotion

For each staged top-level project, the adapter invokes official v0.9.4 `openamundsen-da subdomains prepare`. Preparation must produce eight filtered subdomain projects and their deterministic steps. Expected step count is the retained subdomain event count plus the initialization step.

Before promotion, validation requires:

- six top-level ES50 projects with compact retention;
- at least 85% fulfillment for both observation types in every project;
- exact, exclusive station roles and complete exception inventories;
- real FSC scene and uncertainty linkage for every selected FSC event;
- 48 prepared project/subdomain combinations with correct step counts;
- unchanged raw and working data hashes;
- no pending tree, results, restart state, model process, container or tmux job.

After validation, the adapter renames the canonical directory to a sibling backup and the staging directory to the canonical name. It validates the promoted path and restores the backup on failure. On success it removes `READY_FOR_EVENT_SELECTION`, creates `READY_TO_RUN`, retains only `projects/`, and deletes the temporary backup.

## Failure behavior

Input, policy, schedule, coverage, image, preparation or validation failures are fatal. A failed pre-swap staging tree is marked `INCOMPLETE`, and the canonical snapshot remains untouched. A failed post-swap validation restores the original directory. Existing non-empty generic outputs and unexpected canonical snapshot states are refused.

## Minimal verification

One focused synthetic pytest module covers slot boundaries and alternation, deterministic FSC selection, skipped slots, duplicate and gap constraints, cloud gating, fulfillment, adaptive station roles, preflight refusal and transaction rollback. The existing North Tyrol snapshot builder tests remain unchanged. CI runs only these explicit focused files in the pinned v0.9.4 image.
