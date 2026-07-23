# CMU eVTOL Transfer Benchmark Protocol v0.1

## Status and scope

This document records the development comparison protocol for the first CMU eVTOL transfer benchmark implementation. It uses the existing development candidate identities and dynamic-capacity SOC reference; these remain development artifacts rather than formal mission labels or ground truth.

The authoritative cell-level protocol assignments are stored in:

```text
data/manifests/cmu_evtol/cmu_evtol_vtc6_cell_protocols.csv
```

Assignments are based on the dataset-supplied `README.txt`. Raw tester `cycleNumber` values are explicitly not treated as globally reliable cycle identifiers.

## Domains

- Source domain: Panasonic NCR18650PF.
- Main source temperature: 25 °C.
- Target domain: CMU eVTOL Battery Dataset using Sony-Murata 18650 VTC6 cells.
- Common model inputs: voltage, current, and cell-surface temperature.
- A future CMU SOC label must be called an SOC reference rather than measured ground truth.

## Development target population

The supervised development comparison treats each of VAH01, VAH17, VAH27,
VAH30, VAH15, VAH12, and VAH11 as an independent target. Models are never
adapted on one target cell and evaluated as though it were another target.
Baseline and condition results remain separately identified by the manifest's
`condition` column.

## Split integrity

- Row-level random splitting is prohibited.
- Window-level random splitting before mission assignment is prohibited.
- A complete mission is the smallest splittable target unit.
- Every row and every future sequence window from one mission must remain in the same split.
- Each cell is ordered by `mission_start_source_row`, with
  `mission_cycle_id` as a deterministic tie-breaker.
- Per cell, `floor(60%)` missions form Train, `floor(20%)` form Validation,
  and the remainder form Test.
- Target-test missions must never participate in normalization, adaptation,
  validation, tuning, early stopping, threshold selection, or model selection.
- Zero-shot and every supervised setting use the exact same Test missions.

## Target-label budgets

Supervised target budgets are chronological prefixes of each target cell's
Train split:

- shared comparison budgets: 1%, 5%, and 10% of Train;
- Vidal-style Partial: 40% of Train (approximately 24% of all missions);
- Vidal-style Full: 100% of Train (approximately 60% of all missions).

- Budgets are constructed from complete missions, never rows, samples, or windows.
- For each target cell, the denominator is that cell's Train mission count.
- Train missions are selected as chronological prefixes.
- The percentage and ceiling operation are applied independently to each cell.
- Budget subsets are nested: 1% ⊂ 5% ⊂ 10% ⊂ Partial ⊂ Full.
- The realized mission count, sample count, and physical duration must later be reported for every budget.
- Every model evaluated at a target-label budget must use the exact same fold-specific budget manifest.
- Transfer and same-budget scratch runs must reuse that same manifest without resampling or replacement.

Formal mission IDs are not assigned in v0.1. The deterministic development
budget manifest at
`data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1.csv`
stores every core7 development-candidate `mission_cycle_id`, its Train,
Validation, or Test assignment, and its budget membership.
It is explicitly `development_only`, uses `soc_dynamic` as a reference rather
than ground truth, and does not make those candidates formal benchmark labels.

## Reporting tracks

Results are reported separately for each supervision track:

1. **Zero-shot** — Panasonic labelled source only; no CMU adaptation inputs or labels.
2. **Unsupervised target adaptation** — Panasonic labelled source plus unlabelled CMU adaptation inputs; no CMU SOC-reference labels.
3. **Supervised few-shot transfer** — source-pretrained model plus the fixed 1%, 5%, or 10% labelled adaptation-mission subset.
4. **Same-budget scratch** — random initialization using exactly the same labelled mission subset as the corresponding few-shot transfer run.
5. **Full-target upper bound** — all eligible labelled missions from the adaptation cells; the held-out test cell remains excluded from training and selection.

Tracks with different target supervision must not be merged into one leaderboard.

## Transfer gain and negative transfer

For a fixed model, fold, seed, metric, and target-label budget:

```text
Transfer Gain = Error(same-budget scratch) - Error(transfer few-shot)
```

- Positive gain indicates beneficial transfer.
- Gain near zero indicates limited pretraining value.
- Negative gain indicates negative transfer.

Transfer and scratch comparisons must share the same target missions, test cell, preprocessing permissions, evaluation records, and random-seed schedule.

## Deferred decisions

Version 0.1 does not decide:

- mission reconstruction rules;
- the formal CMU SOC-reference version;
- RPT detection and dynamic-capacity rules;
- resampling interval;
- physical context length, sequence length, or stride;
- mission identifiers;
- early/middle/late-life boundaries;
- label-quality inclusion thresholds.

These decisions require separate evidence and review before data preparation. Implementations must not infer them from this protocol.

## Prohibited operations in v0.1

The v0.1 protocol task must not:

- split raw files into missions;
- compute SOC or SOH;
- generate model sequences;
- train or evaluate models;
- modify raw data;
- use condition cells in primary baseline folds.
