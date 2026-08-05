# Minimal North Tyrol snapshot CI design

## Goal

Provide one reproducible green check for pull request 5 before merge. The check
must exercise only the dedicated North Tyrol snapshot-builder tests and must not
access Fram3S, build a snapshot or run openAMUNDSEN-DA projects.

## Workflow

Add `.github/workflows/north-tyrol-snapshot.yml` with one job named
`north-tyrol-snapshot-tests` on `ubuntu-latest`.

The workflow runs for pull requests and pushes to `main` only when the dedicated
builder, its wrapper, tests, documentation or workflow file changes. It grants
only `contents: read`, checks out the repository and runs exactly this test file:

```text
04-openAMUNDSEN/tests/test_north_tyrol_snapshot.py
```

Tests execute inside the immutable image already used for local and P8
acceptance:

```text
ghcr.io/openamundsen/openamundsen-da:0.9.4@sha256:f3834a701e116b9ab11c50677d94236bffcd5d9adb045ae6b871b3ccf2c98723
```

The repository mount is read-only, container networking is disabled after the
image is available, pytest caching is disabled and the job timeout is 15
minutes.

## Explicit exclusions

The workflow does not run linting, unrelated tests, a version matrix, source
discovery, Fram3S preflight, fixture generation outside pytest, snapshot builds,
model runs, cache restoration or artifact upload.

## Acceptance

The workflow is accepted when GitHub Actions reports all 26 targeted tests
passing on the exact head commit of pull request 5. The PR may then be merged
without changing its merge method or deleting unrelated branches.
