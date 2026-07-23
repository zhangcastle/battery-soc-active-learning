# Vidal Panasonic → CMU eVTOL development results v0.1

## Scope

This note records the first complete Vidal-style Panasonic-to-CMU campaign
executed from `notebooks/benchmarks/01_vidal_lstm_tl.ipynb` and saved in the
notebook at Git commit `06c7156`.

The comparison uses the same chronological, complete-mission split for every
setting:

- Train: first 60% of each cell's missions;
- Validation: next 20%;
- Test: final approximately 20%;
- no mission crosses a split;
- every setting uses the same Test missions.

The source is the canonical Panasonic Case 1 checkpoint and its matching
source scaler. Supervised runs restart from that same checkpoint independently
for each cell and fit a target scaler using only the selected Train missions.

These are development results. The CMU SOC is a reference label rather than
ground truth, the phase roles are diagnostic `high_1/lower/high_2` roles, only
seed 42 was run, and `formal_benchmark_eligible=false`.

## Settings

| Setting | Target missions used for training | Approximate share of all missions |
|---|---:|---:|
| Zero-shot | 0 | 0% |
| 1% budget | first 1% of Train | 0.6% |
| 5% budget | first 5% of Train | 3% |
| 10% budget | first 10% of Train | 6% |
| Vidal-style Partial | first 40% of Train | 24% |
| Vidal-style Full | all Train | 60% |

“Vidal-style Partial” is the standardized CMU setting, not an exact
reproduction of the paper's dataset-specific partial protocol.

## Test results

Overall MAE and RMSE are reported in SOC percentage points below. Values in
the generated CSV files are fractions on `[0, 1]`.

### Overall MAE

| Cell | Condition | Zero-shot | 1% | 5% | 10% | Partial | Full |
|---|---|---:|---:|---:|---:|---:|---:|
| VAH01 | baseline | 45.86 | 5.77 | 5.55 | 4.44 | 5.21 | 2.66 |
| VAH17 | baseline | 40.55 | 5.77 | 4.61 | 3.87 | 3.68 | 0.98 |
| VAH27 | baseline | 20.46 | 6.12 | 6.12 | 5.30 | 4.44 | 2.00 |
| VAH30 | 35°C chamber | 66.67 | 6.06 | 5.00 | 4.45 | 2.78 | 1.14 |
| VAH15 | 1000 s cruise | 15.43 | 3.61 | 3.26 | 2.87 | 2.26 | 0.85 |
| VAH12 | 400 s cruise | 52.61 | 14.88 | 10.90 | 10.10 | 12.57 | 4.47 |
| VAH11 | 20% lower discharge power | 7.31 | 15.78 | 11.58 | 9.25 | 8.73 | 5.31 |

### Overall RMSE

| Cell | Zero-shot | 1% | 5% | 10% | Partial | Full |
|---|---:|---:|---:|---:|---:|---:|
| VAH01 | 50.27 | 9.50 | 9.03 | 7.25 | 6.27 | 3.05 |
| VAH17 | 45.82 | 9.30 | 7.62 | 6.92 | 5.91 | 1.16 |
| VAH27 | 28.06 | 9.32 | 9.03 | 8.05 | 6.79 | 2.34 |
| VAH30 | 68.24 | 9.48 | 8.41 | 7.75 | 6.18 | 1.41 |
| VAH15 | 22.76 | 6.25 | 5.70 | 4.99 | 4.11 | 0.98 |
| VAH12 | 56.65 | 22.16 | 18.36 | 17.33 | 14.04 | 7.27 |
| VAH11 | 12.60 | 21.55 | 16.17 | 13.54 | 11.30 | 6.01 |

The unweighted mean cell MAE falls from 35.56 percentage points at zero-shot
to 8.29, 6.72, 5.77, 5.67, and 2.49 for 1%, 5%, 10%, Partial, and Full,
respectively. Full adaptation reduces mean cell MAE by about 93% relative to
zero-shot.

## Findings by condition

### Baseline cells

VAH01, VAH17, and VAH27 show substantial zero-shot variation despite sharing
the baseline protocol. Their zero-shot MAEs range from 20.46% to 45.86%.
Therefore, protocol name alone does not explain transfer difficulty; cell
history, aging state, and feature distribution remain important.

All three baseline cells benefit strongly from target labels. VAH17 reaches
0.98% MAE with Full adaptation, while VAH01 and VAH27 reach 2.66% and 2.00%.
The result supports cell-specific adaptation rather than treating the three
baseline cells as interchangeable samples from one target.

### VAH30: 35°C chamber

VAH30 has the worst zero-shot result: 66.67% MAE. This is consistent with the
large temperature-domain mismatch observed during feature-range auditing.
It also has the largest absolute adaptation gain: Full adaptation lowers MAE
to 1.14%. Temperature shift is therefore severe for direct transfer but highly
learnable when target labels are available.

### VAH15: extended 1000 s cruise

VAH15 transfers comparatively well before adaptation (15.43% MAE) and is the
best supervised target after Full adaptation (0.85% MAE). Increasing label
availability improves it consistently. Extended cruise duration alone does
not appear to create the same transfer barrier as the thermal or short-cruise
conditions.

### VAH12: short 400 s cruise

VAH12 is difficult in zero-shot (52.61% MAE) and remains the hardest target
after Full adaptation (4.47% MAE). Its 10% setting outperforms Partial in MAE
(10.10% versus 12.57%), although Partial has lower RMSE (14.04% versus
17.33%). More training missions therefore did not produce a uniformly better
error distribution under the fixed 515-epoch schedule.

This may reflect optimization variance, lifecycle drift between chronological
Train and Test, or a condition-specific mismatch. Training and validation
histories should be checked before attributing it to a scientific mechanism.

### VAH11: 20% lower discharge power

VAH11 is the strongest zero-shot target (7.31% MAE), but 1%, 5%, 10%, and
Partial adaptation worsen MAE to 15.78%, 11.58%, 9.25%, and 8.73%.
This is clear negative transfer at small and medium label budgets. Only Full
adaptation improves on zero-shot, reaching 5.31%.

The likely lesson is not that adaptation is generally harmful, but that a
small chronological prefix is not representative enough for this condition,
or that full-network fine-tuning with 515 epochs is too aggressive at small
sample sizes. This must be tested rather than assumed.

## Label efficiency and monotonicity

- A very small target budget gives large gains on six of seven cells.
- Performance is not strictly monotonic with mission count.
- Partial is better than 10% on five cells, worse on VAH01 and VAH12.
- Full is the best setting on every cell.
- VAH11 demonstrates why negative-transfer reporting must remain part of the
  benchmark rather than reporting only successful adaptation cases.

Because each budget uses a chronological prefix, these results measure both
label quantity and which lifecycle period is labeled. They should not be
interpreted as a pure sample-size ablation.

## Runtime

| Setting | Total runtime |
|---|---:|
| Zero-shot | 141.3 s |
| 1% | 165.1 s |
| 5% | 190.4 s |
| 10% | 230.3 s |
| Partial | 470.1 s |
| Full | 980.7 s |

These are the observed Colab runtimes for this run and are not hardware-neutral
benchmarks.

## Artifacts

The runs were written under:

```text
/content/drive/MyDrive/battery_soc_benchmark/outputs/benchmarks/
  panasonic_to_cmu_evtol/vidal/<setting>/seed_42/<run_id>/
```

Run IDs recorded by the notebook:

| Setting | Run ID |
|---|---|
| Zero-shot | `20260723_135854` |
| 1% | `20260723_140115` |
| 5% | `20260723_140400` |
| 10% | `20260723_140710` |
| Partial | `20260723_141101` |
| Full | `20260723_141851` |

Each run contains its per-cell predictions, histories and metrics plus
`cell_metrics.csv` and `run_metadata.json`. The notebook output is committed
for review, but the Drive artifacts remain the authoritative detailed output.

## Interpretation limits and next checks

1. Results use one seed; uncertainty across seeds is unknown.
2. The reference SOC is generated by the development dynamic-capacity
   pipeline and is not independently measured ground truth.
3. `high_1/lower/high_2` are development waveform roles, not frozen
   take-off/cruise/landing labels.
4. The fixed 515-epoch schedule has no model selection or early stopping;
   VAH11 and VAH12 require training-history review.
5. The seven target cells represent different protocols and lifecycle
   histories; condition effects are not isolated causal effects.
6. The next comparison should preserve these exact splits, budgets, test
   missions, source artifact, metrics, and negative-transfer reporting for
   every reproduced method.
