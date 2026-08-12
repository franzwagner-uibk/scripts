# North Tyrol station-tie policy v3 design

## Purpose

The canonical North Tyrol preflight contains hourly station series in which a
missing midnight value can leave observations at 23:00 and 01:00 equally near
the 00:00 model state. Policy v2 correctly rejects ambiguous matches, but it
cannot express the approved midpoint interpolation rule now implemented in
openAMUNDSEN-DA.

Policy v3 mirrors the core scientific contract while preserving v2 unchanged.
Final DA event selection remains outside openAMUNDSEN-DA and is authored only
through each project YAML `assimilation_events` list.

## Matching and scheduling

The unique nearest finite, nonnegative observation inside half one model
timestep remains preferred. Exactly two equal-distance values may be averaged
only when one is before and one after the model time, both are inside the
half-timestep window and their timestamps are separated by no more than 24
hours, inclusively. Duplicate timestamps, same-side ties, more than two ties
and wider pairs remain errors.

An accepted mean counts as station support. Its effective time is the model
timestamp, while candidate ranking retains the real source offset so a direct
observation is preferred. Station uncertainty and roles are unchanged.
Selected interpolations are logged at INFO with both timestamps, values and the
mean. Existing scheduler CSV and JSON schemas remain unchanged.

## Policy and audit

The v2 policy file remains byte-identical. A new v3 policy records the fixed
matching name and 24-hour span. These are validated scientific constants, not
user-selectable openAMUNDSEN-DA settings.

The canonical refresh also enables the existing areal FSC diagnostic with 250
m elevation bands. The DEM and land-cover inputs remain on their native 100 m
grid, while native FSC pixels are assigned to their containing terrain cell;
none of these inputs is resampled.

## Rollout

The scripts PR follows merged core commit
`0a9c87753d1e59bd13552f0882353301ed2b8991` and pins its immutable image digest
`sha256:8fa2bda758be2b98e88a3a2fdb616f5ef5504d9b4fe9a2d7dde4d799a47e327d`.
Lenovo P8 first runs a read-only preflight. A successful
preflight is followed by the existing same-filesystem transactional refresh of
all six projects and 48 leaves. The accepted canonical setup is not changed on
failure, and no model propagation is launched.

## Verification

Focused tests cover v2 preservation, v3 parsing, valid midpoint support,
malformed ties, exact-over-interpolated ranking, selected-event INFO logs,
station-role support and the 250 m native-grid audit. P8 acceptance requires
the core pre-run validator to accept every final HS event, at least 85% of
feasible support per type and leaf, complete SWE and snow-depth outputs and no
runtime process or model state before promotion.
