# CMU eVTOL KiltHub Version 3 Raw Ingestion v0.1

## Scope

This layer inventories and loads the original CMU KiltHub article `14226830`,
eVTOL Battery Dataset, Version 3 assets. It is intentionally limited to raw
schema validation, row provenance, and continuous time-block QC.

It does **not** define missions, flight phases, RPTs, capacity references, SOC,
SOH, splits, resampling, model windows, or training inputs.

## Assets

- Main time series: `VAH##.csv`
- Optional impedance summary: `VAH##_impedance.csv`
- Expected main cells: the 22 cells registered in the v0.1 cell protocol manifest
- Missing impedance never invalidates a main time series; VAH26 is expected to
  have no impedance file in Version 3.

The asset inventory records dataset name, article ID, dataset version, cell ID,
file presence, size, SHA-256, data-row count, and observed columns. Missing files
remain explicit inventory rows.

## Canonical raw schema

The loader preserves the ten original numeric fields and their meanings:

```text
time_s
Ecell_V
I_mA
EnergyCharge_W_h
QCharge_mA_h
EnergyDischarge_W_h
QDischarge_mA_h
Temperature__C
cycleNumber
Ns
```

It adds only provenance and continuity fields:

```text
cell_id
source_file
source_row
continuous_block_id
continuous_block_boundary_reason
```

`source_row` is the 1-based data-record ordinal after the CSV header. Input row
order is preserved. The loader does not sort or filter rows.

Current follows the dataset convention: `I_mA > 0` is charge and `I_mA < 0` is
discharge. Values remain in mA; no sign or unit conversion is performed.

`cycleNumber` is tester-native and unreliable as a unique physical-cycle count.
It is preserved but never used for a block ID, mission ID, or split. `Ns` is
also preserved without assigning phase semantics.

## Continuous-block boundaries

The first record starts block zero. A later record starts a new block when one
or more of these rules fires:

1. current `time_s` is non-finite;
2. previous `time_s` is non-finite;
3. current `time_s` equals previous `time_s`;
4. current `time_s` is less than previous `time_s`;
5. the positive time gap exceeds `gap_threshold_s`;
6. the caller supplies that source row as a known concatenation boundary.

Boundary rows are retained as the first rows of their new blocks. Multiple
reasons may be recorded on one row, separated by `|`. Changes or reversals in
`cycleNumber`, and changes in `Ns`, never create blocks by themselves.

## Gap threshold

`gap_threshold_s` is an explicit loader parameter and is recorded in metadata.
The engineering default is 120 s, selected for raw QC because Version 3 asset
reconnaissance observed 10 s and 30 s nominal cadences. It is four times the
slower observed cadence.

This default is **not** a frozen benchmark research constant. Formal downstream
protocols must review sampling statistics and explicitly pass the selected
threshold. If a future asset version does not support the 10/30 s observation,
the caller must provide a threshold explicitly.

## Metadata and QC

Every ingestion result records:

- article ID and dataset version;
- source SHA-256;
- loader/schema version;
- gap threshold, unit, and whether it was defaulted;
- all block rules and per-rule trigger counts;
- input and output rows;
- block count;
- zero filtered rows and empty filter-reason counts;
- original-unit mapping;
- UTC generation time;
- installed software version and Git revision/dirty state when available.

Non-finite, duplicate, and reversed timestamps are retained and surfaced in the
shared `QCResult`. No raw asset is modified.
