# Liu 2025 DARM: Panasonic to CMU eVTOL extension plan

## Goal

Complete the paper reproduction first, then reuse the frozen Panasonic 25 °C
source artifact while adapting DARM independently to selected CMU eVTOL target
cells. The downstream benchmark must not change the source records, source SOC
labels, source normalizer, source encoder, SOC predictor, or paper hyperparameters.

The CMU extension is a development benchmark. Its dynamic-capacity SOC is a
reference label, not independent ground truth, and remains
`formal_benchmark_eligible=false`.

## Frozen source contract

The completed `25c_to_0c` reproduction must emit one reusable source artifact
before any target adaptation:

- source encoder and SOC predictor state;
- source-fitted min-max normalizer state;
- source manifest path and SHA-256;
- source record and window inventory;
- feature order `[voltage, current, temperature]`;
- sampling rate 1 Hz, window length 40, and stride 1;
- Git commit, dirty state, seed, and resolved source-training settings.

Every CMU target run must load this same source artifact, initialize a fresh
target encoder from it, and leave the source encoder and SOC predictor frozen.
Source training must not be repeated independently for each CMU target.

## Existing components to reuse

| Existing component | Reuse decision |
|---|---|
| `DARM` model and paper losses | Reuse unchanged |
| Liu source manifest and source label policy | Reuse unchanged |
| `MinMaxNormalizer` | Reuse the source-fitted state; never fit on CMU |
| Shared checkpoint helpers | Reuse for source and per-target artifacts |
| CMU dynamic-capacity SOC artifacts | Reuse as development evaluation references |
| CMU mission and budget manifest | Reuse mission identities and Train/Validation/Test isolation |
| CMU Vidal adapter validation and 1 Hz record resampling | Reuse the feature/provenance path |
| CMU per-cell condition manifest | Reuse condition names and independent-target assignments |
| `compute_metrics` and per-cell aggregation | Reuse |
| Git/hash/run metadata helpers | Reuse |

The tracked CMU inputs are:

```text
outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1/
outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1_extension4/
data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1.csv
data/manifests/cmu_evtol/cmu_evtol_vtc6_cell_protocols.csv
```

No CMU raw-data reconstruction is required for this extension.

## Components that must not be reused unchanged

- Vidal's 50-step, stride-50 windows: DARM requires 40 steps and stride 1.
- Vidal's target-fitted scaler path: unsupervised DARM must use only the frozen
  Panasonic source normalizer.
- Vidal's supervised fine-tuning loop: DARM adaptation uses target features
  without target SOC labels.
- `PanasonicAdapter` protocol assignments: they encode Vidal 2019's 25 °C
  split and must not redefine the Liu source protocol.
- The current Liu raw-MAT loader as the CMU interface: CMU already has validated
  mission records and provenance.
- The generic experiment runner: its current adapter and transfer abstractions
  do not enforce the paper-specific DARM source/target state contract.

## Minimal required extensions

1. Split the Liu runner into a source stage and a target stage without changing
   the existing end-to-end reproduction result.
2. Save the source normalizer inside the reusable source checkpoint.
3. Add a CMU feature-only adapter path that reads only identity, time, voltage,
   current, and temperature for target adaptation. It must not require or load
   `soc_dynamic`, capacity values, or label metadata.
4. Add record-safe Liu windows with length 40 and stride 1. Adaptation windows
   must return no target tensor; evaluation windows may expose the development
   SOC reference.
5. Select CMU adaptation and test missions from the existing manifest before
   window construction. Validation and Test missions must not enter adaptation,
   normalization, early stopping, or model selection.
6. Start every target cell from the same frozen source artifact and a fresh
   target encoder. Never carry an adapted encoder from one target cell to the
   next.

## Target domains

The first campaign reuses the established core-seven independent targets:

| Cell | Condition |
|---|---|
| VAH01 | baseline protocol |
| VAH17 | baseline protocol |
| VAH27 | baseline protocol |
| VAH30 | 35 °C chamber |
| VAH15 | extended 1000 s cruise |
| VAH12 | short 400 s cruise |
| VAH11 | 20% lower discharge power |

Cells remain independent targets. Baseline cells are not pooled, and condition
cells are not pooled with baseline cells.

## Unlabelled adaptation budget

The paper exposes one target Cycle 1 record, approximately 8,778 seconds, to
unsupervised adaptation. The primary CMU comparison should therefore use the
chronological prefix of complete Train missions whose cumulative duration first
reaches that physical-time budget. This avoids giving a target hundreds of
hours of unlabelled adaptation data merely because more missions are available.

| Cell | Complete Train missions | Full Train hours | Matched missions | Matched hours |
|---|---:|---:|---:|---:|
| VAH01 | 496 | 135.0 | 9 | 2.45 |
| VAH17 | 588 | 160.1 | 9 | 2.45 |
| VAH27 | 343 | 93.4 | 9 | 2.45 |
| VAH30 | 540 | 147.0 | 9 | 2.45 |
| VAH15 | 325 | 106.5 | 8 | 2.62 |
| VAH12 | 1381 | 222.5 | 16 | 2.58 |
| VAH11 | 1321 | 359.6 | 9 | 2.45 |

An all-Train-missions run may be added later as a sensitivity analysis. It must
not replace or be mixed with the duration-matched primary result.

## Evaluation

For every target cell, report both:

- source-only prediction using the frozen source-initialized target encoder;
- DARM prediction after unlabelled adaptation.

Evaluation uses only the existing Test missions and their development SOC
reference. Report overall and mission-macro MAE/RMSE, per-condition identity,
adaptation runtime, source-artifact hash, adaptation mission count and duration,
and the signed gain over source-only. Negative transfer must remain visible.

## Output layout

```text
outputs/benchmarks/panasonic_to_cmu_evtol/liu_2025_darm/
└── duration_matched/
    └── seed_<seed>/
        └── <run_id>/
            ├── source_artifact.json
            ├── run_metadata.json
            ├── cell_metrics.csv
            └── <cell_id>/
                ├── adapt_last.pt
                ├── adapt_best.pt
                ├── training_history.csv
                └── test_predictions.parquet
```

## Required checks

- Paper reproduction results remain unchanged after source/target stage split.
- Every CMU adaptation loader yields features only.
- No CMU SOC or capacity column is read by the adaptation path.
- Source normalizer state and source checkpoint hash are identical for all cells.
- Adaptation, Validation, and Test mission sets are disjoint.
- No window crosses a mission boundary.
- Each target starts from the same source state, not another target's adapted state.
- Outputs record `label_is_ground_truth=false` and
  `formal_benchmark_eligible=false`.

## Execution order

1. Finish and freeze the Panasonic 25 °C to 0 °C paper reproduction.
2. Refactor source-artifact persistence with a regression smoke test.
3. Implement and test the CMU feature-only adaptation path.
4. Run VAH30 first as the closest existing thermal condition stress test.
5. Run all core-seven cells independently with the duration-matched budget.
6. Add the all-Train sensitivity analysis only after the primary campaign.
