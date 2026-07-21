# CMU eVTOL Mission Segmentation Protocol v0.1 — Draft and Disabled

## Status

```text
status: draft
production_enabled: false
emit_mission_records: false
emit_phase_labels: false
emit_rpt_labels: false
implementation_authorized: false
```

This document defines acceptance, rejection, and uncertainty rules for a future
implementation. It does not claim production segmentation is complete and does
not authorize data generation. Enabling production requires a newly reviewed
protocol version and a validated implementation; changing one Boolean is not
sufficient.

## Provenance and references

- CMU KiltHub article `14226830`, eVTOL Battery Dataset Version 3
- raw schema: `cmu_evtol_kilthub_v3_raw_v0.1`
- [raw ingestion rules](cmu_evtol_raw_ingestion_v0_1.md)
- `data/manifests/cmu_evtol/cmu_evtol_vtc6_cell_protocols.csv`
- `data/manifests/cmu_evtol/cmu_evtol_vtc6_asset_inventory_v3.csv`
- `data/manifests/cmu_evtol/cmu_evtol_vtc6_raw_qc_v3_120s.csv`
- `data/manifests/cmu_evtol/cmu_evtol_mission_phase_evidence_v0_1.csv`

The evidence CSV SHA-256 is
`3245f58d90f3f0bea0761652c22e93fd2128d3de84be97238f2899e2ec0e0613`.
Its clean-main generation revision is
`7fd9fbe380dac390dabe6404960d690fb228e8dc`.

Power evidence uses
`electrical_power_W = Ecell_V * I_mA / 1000`. Candidate thresholds come only
from the descriptive evidence audit and are not frozen production constants.

## Evidence coverage

Supported only as audited evidence:

- baseline: VAH01, VAH17, VAH27;
- condition: VAH02, VAH12, VAH05.

The remaining 16 cells—VAH06, VAH07, VAH09, VAH10, VAH11, VAH13, VAH15,
VAH16, VAH20, VAH22, VAH23, VAH24, VAH25, VAH26, VAH28, and VAH30—have not
undergone phase-evidence validation and remain deferred.

## Permitted inputs and evidence

The future implementation may use canonical `cell_id`, `source_file`,
`source_row`, `time_s`, `Ecell_V`, `I_mA`, `cycleNumber`, `Ns`,
`continuous_block_id`, and `continuous_block_boundary_reason`.

Boundary decisions may combine the canonical block, source-row order, time,
Ns transitions, signed current/power, real duration, completeness checks, and
cell condition metadata. `cycleNumber` remains an observation only: it cannot
be a mission key, boundary, split key, or globally unique cycle identifier.

The ordinary candidate shape is high-discharge run → lower-discharge run →
high-discharge run. Ns 4→5→6 is auxiliary transition evidence only and cannot
determine a phase by itself.

## Candidate enumeration

Within one `continuous_block_id`, form contiguous same-Ns runs in ascending
`source_row` order. A run with high-discharge-power evidence is an anchor; only
its next two adjacent runs are examined for lower→high evidence. Enumeration
never crosses a block. An anchor missing either following run still emits an
uncertain QC candidate instead of disappearing.

Runs cannot be shared between candidates. If candidates overlap, the earliest
anchor by `source_row` wins; the later candidate is retained as uncertain with
reason `overlaps_earlier_candidate`. Ns 4 may support an anchor but cannot alone
create an accepted result.

## Primary classification

Every enumerated candidate has exactly one `primary_status`, selected with this
strict precedence:

```text
out_of_scope > hard_reject > uncertain > accept
```

Multiple triggers are retained in `all_reason_codes`. `primary_reason` is the
first triggered code in the listed reason-code order for the winning status.
Reason codes are unique across status groups.

`out_of_scope` applies before waveform classification to the 16 deferred cells,
unsupported schema/dataset versions, and unknown cell conditions.

## Accept

All conditions are necessary:

1. All three runs lie in one `continuous_block_id`.
2. Source rows and timestamps are strictly increasing.
3. Every duration is finite and positive.
4. Every run is discharge-dominated.
5. Power follows the high→lower→high order.
6. Power and duration agree with evidence for that cell condition.
7. No canonical discontinuity occurs inside the candidate.
8. The waveform does not match RPT-like low-rate evidence.

## Hard reject

Hard-reject a candidate that crosses a block, has non-finite/duplicate/reversed
timestamps, has zero or negative duration, has definitively incorrect segment
order, is severely contaminated by charge/rest, spans a known gap, or clearly
matches a long low-rate diagnostic sequence. RPT-like candidates are excluded
from ordinary mission candidates; this draft does not emit RPT labels.

## Uncertain

Retain an uncertain outcome for a missing segment, interrupted or partial
waveform, early end-of-life cutoff, proximity to a block/file boundary without
crossing it, power/duration close to but outside screening ranges, overlap with
an earlier candidate, or inability to distinguish mission from diagnostic
sequence. A missing segment is not a hard reject unless another hard-reject
reason also fires.

Rejected and uncertain outcomes must retain QC counts and reasons. Silent drops
are forbidden.

## Diagnostic screening thresholds

The v0.1 evidence audit used these screening ranges:

| Quantity | Range/value |
|---|---:|
| high discharge power | 35–70 W |
| lower discharge power | 5–30 W |
| first high duration | 45–120 s |
| lower duration | 200–1500 s |
| last high duration | 60–150 s |
| minimum discharge fraction | 0.9 |
| RPT-like discharge power | 0.5–4 W |
| RPT-like minimum duration | 3600 s |
| RPT-like minimum voltage drop | 0.8 V |
| RPT-like maximum end voltage | 2.7 V |

These values are authorized only for v0.1 diagnostic candidate screening.
Condition-specific final acceptance tolerances remain unfrozen. The next
implementation may be a diagnostic classifier at most; it cannot emit formal
mission records.

## Proposed identifiers and output only

The deterministic identifier draft is:

```text
cmu_v3_mission_v0_1:{cell_id}:b{continuous_block_id}:s{start_row}:e{end_row}
```

It contains no `cycleNumber` and is not generated by this draft.

The proposed output fields are `segmentation_schema_version`, `cell_id`,
`mission_id`, `continuous_block_id`, mission source-row/time bounds,
`primary_status`, `primary_reason`, `all_reason_codes`, `completeness_status`,
`exclusion_reason`, `protocol_condition`, takeoff/cruise/landing source-row
bounds, durations and median powers, and `qc_flags`.

`primary_status` is the single classification outcome. `completeness_status`
describes structural observation only (for example complete, partial, or
interrupted) and cannot override classification precedence.
No SOC, SOH, capacity, split, scaler, or model field is permitted.

## Deferred

Production segmentation, complete RPT identification, capacity lookup, SOC,
dataset splits, and sequence windows remain undefined. No mission records or
phase labels are produced in v0.1.
