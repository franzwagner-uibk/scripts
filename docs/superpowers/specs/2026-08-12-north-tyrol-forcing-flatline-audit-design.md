# North Tyrol Forcing Flatline Audit Design

Date: 2026-08-12

Status: approved for implementation

## Objective

Repair the scripts-only North Tyrol forcing-quality audit without changing any
forcing value or openAMUNDSEN-DA runtime behavior. The audit must inspect the
native hourly `openamundsen-v2` station series even though the model consumes
them at a three-hour timestep.

## Source contract

Each forcing CSV must have strictly increasing timestamps and an inferred
native cadence of exactly one hour. Missing hourly rows are permitted but end
an otherwise constant plateau. The audit considers only the model-consumed
columns `temp`, `precip`, `rel_hum`, `sw_in`, `wind_speed` and `wind_dir` when
present. The unused `hs` column and unrelated metadata columns are excluded.

An exact finite-value plateau is retained when its first and last observation
are at least 24 hours apart. A plateau lasting at least 168 hours is classified
as severe. Zero-valued precipitation plateaus are retained as
`dry_zero_precip`; every other retained plateau is a
`candidate_stuck_sensor`. These classifications are review evidence only. No
forcing record is filled, masked, excluded or rewritten.

## Project-window outputs

The audit detects plateaus once from the shared forcing source and intersects
them with each of the six project windows. The clipped overlap must still last
at least 24 hours to be retained for that project. The detailed CSV records the
project, station, variable, value, clipped and source interval, sample count,
duration, inferred source cadence, classification and severity.

A second CSV records overlapping candidate stuck-sensor plateaus for two or
more variables at the same station and project. Adjacent segments with the same
active variable set are combined deterministically. The JSON summary records
source cadence, thresholds, project windows, per-project counts, per-variable
counts, station counts and multivariable overlap counts.

The 2017/18 output must contain the known Eissee temperature, relative-humidity,
wind-speed and wind-direction plateau from 2018-04-05 15:00 through the project
end at 2018-09-30 21:00.

## Failure behavior and verification

Missing time columns, non-increasing timestamps, non-hourly native cadence,
missing project windows and an unexpected forcing-file count fail before
canonical promotion. Timestamp gaps are valid and explicitly terminate a
plateau. Focused tests cover hourly data under a three-hour model setup, gaps,
inclusive duration thresholds, severe classification, dry precipitation,
multivariable overlaps and project-window clipping.

The existing transactional finalizer writes the corrected audit into staging.
Scientific inputs remain byte-identical and the canonical P8 setup is promoted
only after its existing complete validation succeeds.
