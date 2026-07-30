# Liu 2025 DARM Panasonic → CMU eVTOL development results v0.1

## Scope

This note records the first complete Liu 2025 DARM Panasonic-to-CMU
core-seven campaign executed from
`notebooks/reproductions/liu_2025/20_liu_2025_darm_25c_to_0c.ipynb` and saved
in the notebook at Git commit `792fb71`.

The source is the frozen Panasonic 18650PF 25 °C estimator and its matching
source normalizer. Each CMU cell is treated as an independent target and
starts from the same source artifact. Adaptation uses a chronological prefix
of complete Train missions whose duration first reaches the paper Cycle 1
budget of 8,778 seconds:

| Cell | Condition | Adaptation missions | Approximate duration |
|---|---|---:|---:|
| VAH01 | baseline | 9 | 2.45 h |
| VAH17 | baseline | 9 | 2.45 h |
| VAH27 | baseline | 9 | 2.45 h |
| VAH30 | 35 °C chamber | 9 | 2.45 h |
| VAH15 | 1000 s cruise | 8 | 2.62 h |
| VAH12 | 400 s cruise | 16 | 2.58 h |
| VAH11 | 20% lower discharge power | 9 | 2.45 h |

DARM reads only target voltage, current, temperature, time, and mission
identity during adaptation. CMU SOC references are used only for post-hoc
evaluation on held-out Test missions and never for normalization, gradient
updates, early stopping, or checkpoint selection.

The run uses seed 42, 1 Hz inputs, 40-step stride-one windows, batch size 256,
up to 100 adaptation epochs, Adam learning rate `1e-4`, loss weights
`L_adv + 10 L_rec + L_mmd`, joint GRL strength 0.1, frozen target BatchNorm
running statistics, and patience 20 on unlabeled total loss. The reported
DARM model is the last checkpoint.

These are development results. CMU dynamic-capacity SOC is a reference rather
than independently measured ground truth, and
`formal_benchmark_eligible=false`.

## Test results

Overall MAE and RMSE are reported in SOC percentage points. Source only is
the zero-shot result before any CMU adaptation.

### Overall RMSE

| Cell | Condition | Source only | DARM | Relative change |
|---|---|---:|---:|---:|
| VAH01 | baseline | 48.25 | 24.05 | −50.2% |
| VAH17 | baseline | 43.65 | 22.90 | −47.5% |
| VAH27 | baseline | 28.31 | 25.87 | −8.6% |
| VAH30 | 35 °C chamber | 54.18 | 36.91 | −31.9% |
| VAH15 | 1000 s cruise | 24.09 | 19.42 | −19.4% |
| VAH12 | 400 s cruise | 54.01 | 36.69 | −32.1% |
| VAH11 | 20% lower discharge power | 20.97 | 38.30 | **+82.7%** |
| **Unweighted mean** |  | **39.07** | **29.16** | **−25.4%** |

### Overall MAE

| Cell | Condition | Source only | DARM | Relative change |
|---|---|---:|---:|---:|
| VAH01 | baseline | 46.09 | 17.61 | −61.8% |
| VAH17 | baseline | 41.08 | 17.79 | −56.7% |
| VAH27 | baseline | 23.50 | 21.62 | −8.0% |
| VAH30 | 35 °C chamber | 52.48 | 31.28 | −40.4% |
| VAH15 | 1000 s cruise | 19.72 | 16.43 | −16.7% |
| VAH12 | 400 s cruise | 51.36 | 29.43 | −42.7% |
| VAH11 | 20% lower discharge power | 13.25 | 36.57 | **+176.0%** |
| **Unweighted mean** |  | **35.36** | **24.39** | **−31.0%** |

Mission-macro and window-weighted metrics are nearly identical, so these
results are not driven by a small number of longer Test missions.

## Findings

- DARM improves both RMSE and MAE on six of seven independent targets.
- The three baseline cells obtain a mean RMSE reduction of 39.4%.
- VAH30 and VAH12 have the largest zero-shot errors among the condition cells
  and both gain about 32% RMSE from unsupervised adaptation.
- VAH15 is the easiest successful condition target after adaptation.
- VAH11 is a severe negative-transfer case: its source model is already
  comparatively strong, and adaptation damages the learned representation.
- Absolute DARM errors remain too high for a practical SOC estimator; the
  result demonstrates transfer benefit rather than deployment-level accuracy.

## Framework-level comparison

For context, the completed Vidal campaign uses supervised Full target
fine-tuning, while Liu DARM is unsupervised and duration matched. This is a
comparison of complete framework outcomes, not an equal-supervision ablation.

| Framework | Initial mean RMSE | Final mean RMSE | Initial mean MAE | Final mean MAE |
|---|---:|---:|---:|---:|
| Vidal: zero-shot → Full fine-tuning | 40.63 | 3.17 | 35.56 | 2.49 |
| Liu: source only → DARM | 39.07 | 29.16 | 35.36 | 24.39 |

The two frameworks have closely matched zero-shot difficulty across cells,
but supervised Vidal Full realizes much larger final gains. Detailed Vidal
results are recorded in
`docs/benchmarks/vidal_panasonic_to_cmu_evtol_results_v0_1.md`.

## Artifacts

The run was written to:

```text
/content/drive/MyDrive/battery_soc_benchmark/outputs/liu_2025/
  cmu_core7/duration_matched_grl_0p1/
```

The root contains `condition_metrics.csv`. Each cell directory contains
`results.json`, adaptation diagnostics, resumable checkpoints, and the final
adaptation state. The committed notebook output provides a reviewable summary;
Drive remains the authoritative detailed artifact location.

## Interpretation limits

1. Only seed 42 was run; uncertainty across seeds is unknown.
2. CMU SOC is a development reference and not measured ground truth.
3. Cells differ in condition and lifecycle history, so condition effects are
   not isolated causal effects.
4. The last DARM checkpoint is selected without target labels and may not
   coincide with the post-hoc lowest Test RMSE epoch.
5. VAH11 must remain visible as negative transfer rather than being replaced
   by its source-only result after looking at target labels.
