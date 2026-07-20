# CMU eVTOL Transfer Benchmark Protocol v0.1

## Status and scope

This document freezes the cell-level comparison protocol for the first CMU eVTOL transfer benchmark implementation. Version 0.1 is protocol-only: it does not define mission identifiers, generate SOC references, transform raw data, construct sequences, or train models.

The authoritative cell-level protocol assignments are stored in:

```text
data/manifests/cmu_evtol_vtc6_cell_protocols.csv
```

Assignments are based on the dataset-supplied `README.txt`. Raw tester `cycleNumber` values are explicitly not treated as globally reliable cycle identifiers.

## Domains

- Source domain: Panasonic NCR18650PF.
- Main source temperature: 25 °C.
- Target domain: CMU eVTOL Battery Dataset using Sony-Murata 18650 VTC6 cells.
- Common model inputs: voltage, current, and cell-surface temperature.
- A future CMU SOC label must be called an SOC reference rather than measured ground truth.

## Primary benchmark population

The primary benchmark uses only cells explicitly labelled `Baseline` by the source README. The v0.1 manifest identifies VAH01, VAH17, and VAH27 as baseline cells.

All non-baseline cells are excluded from primary-fold construction. They are reserved for later independent stress-test groups:

- discharge-power shift;
- cruise-duration shift;
- charge-current shift;
- thermal or cooling shift;
- end-of-charge-voltage shift;
- other or uncertain conditions, if later evidence requires that classification.

Stress-test groups must not be silently pooled into the primary baseline benchmark.

## Leave-one-cell-out evaluation

The primary evaluation uses three fixed leave-one-cell-out folds. In each fold:

1. one baseline cell is the held-out target-test cell;
2. the other two baseline cells form the target adaptation/validation population;
3. the held-out test cell contributes no input, label, scaler statistics, hyperparameter evidence, early-stopping evidence, or model-selection evidence before final evaluation;
4. every model uses the same folds and held-out test cells.

The explicit cell lists are stored in the protocol YAML. Mission identifiers are intentionally absent from v0.1 and will be introduced only after mission reconstruction is independently reviewed.

## Split integrity

- Row-level random splitting is prohibited.
- Window-level random splitting before mission assignment is prohibited.
- A complete mission is the smallest splittable target unit.
- Every row and every future sequence window from one mission must remain in the same split.
- Target-test missions and cells must never participate in normalization, adaptation, validation, tuning, early stopping, threshold selection, or model selection.
- Condition stress-test cells must not enter primary baseline adaptation folds.

## Target-label budgets

Supervised target budgets are 1%, 5%, and 10% of eligible adaptation missions, calculated independently within each adaptation cell.

- Budgets are constructed from complete missions, never rows, samples, or windows.
- For each adaptation cell, the denominator is that cell's own eligible complete-mission count.
- Eligible missions are ordered chronologically within each adaptation cell and selected as a chronological prefix.
- The percentage and ceiling operation are applied independently to each adaptation cell, so every non-empty cell contributes at least one mission at the 1% budget.
- The per-cell selections are merged only after selection to form one fold-specific budget manifest.
- Budget subsets must be nested: 1% is a subset of 5%, which is a subset of 10%.
- The realized mission count, sample count, and physical duration must later be reported for every budget.
- Every model evaluated at a target-label budget must use the exact same fold-specific budget manifest.
- Transfer and same-budget scratch runs must reuse that same manifest without resampling or replacement.

No mission IDs are assigned in v0.1.

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
