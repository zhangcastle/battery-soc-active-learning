# Shen 2023 TATN Panasonic → CMU eVTOL development results v0.1

## Scope

This note records the first complete Shen et al. (2023) TATN Panasonic-to-CMU
campaign executed from `notebooks/benchmarks/32_tatn_panasonic25_to_cmu.ipynb`
using `scripts/benchmarks/run_tatn_cmu.py`. The notebook runs the same
CMU core-seven protocol under four settings, crossing two source checkpoints
with two adaptation scenarios.

These are development results. CMU SOC is a dynamic-capacity reference rather
than measured ground truth, only seed behaviour implied by the script default
was used, and `formal_benchmark_eligible=false`.

## Source checkpoints

| Checkpoint | Origin |
|---|---|
| `trained` | Source trained in this repository following the Panasonic 25 °C paper protocol: all nine 25 °C profiles, 1 Hz, 90-step non-overlapping windows, Adam lr 1e-3, batch 64, up to 300 epochs with label-safe early stopping. |
| `released` | The authors' published `pre-25.pt` checkpoint, loaded into the required released-repository tensor layout unchanged. |

CMU adaptation reuses the same chronological complete-mission prefix budget
used by the Liu 2025 DARM extension (at least 8,778 seconds), and CMU target
normalisation uses only the Panasonic 25 °C source ranges. Each CMU cell
starts from the unchanged source checkpoint; no adapted state carries across
cells. Transfer uses up to 150 epochs, batch size 64.

## Adaptation scenarios

- **Unsupervised** — the adaptation loader exposes no CMU SOC-reference
  labels; TATN's own domain-adaptation mechanism is the only adaptation
  signal.
- **Semisupervised** — for each CMU cell, the earliest adaptation mission
  contributes the first 50% of its SOC reference labels; all remaining
  adaptation missions stay unlabelled and use TATN-selected pseudo-labels.
  Test missions remain held out under both scenarios.

## Test results

Values are overall MAE and RMSE in SOC percentage points, "Source only"
being the frozen source checkpoint evaluated directly on the CMU test
missions with no adaptation.

### Result 1 — trained source, unsupervised

| Cell | Condition | Source only MAE | TATN MAE | Source only RMSE | TATN RMSE |
|---|---|---:|---:|---:|---:|
| VAH01 | baseline | 6.677 | 22.006 | 8.047 | 23.971 |
| VAH17 | baseline | 6.854 | 6.854 | 7.754 | 7.754 |
| VAH27 | baseline | 6.074 | 12.970 | 6.564 | 15.731 |
| VAH30 | 35°C chamber | 9.961 | 9.961 | 13.058 | 13.058 |
| VAH15 | 1000 s cruise | 7.262 | 36.182 | 8.054 | 37.376 |
| VAH12 | 400 s cruise | 8.659 | 8.659 | 12.120 | 12.120 |
| VAH11 | 20% lower discharge power | 4.391 | 30.264 | 9.486 | 30.264 |
| **Mean** |  | **7.13** | **18.13** |  |  |

Unsupervised adaptation degrades six of seven cells and leaves three
unchanged; only VAH17, VAH30, and VAH12 are unaffected because the run
did not alter their predictions relative to source-only.

### Result 2 — released source (`pre-25.pt`), unsupervised

| Cell | Condition | Source only MAE | TATN MAE | Source only RMSE | TATN RMSE |
|---|---|---:|---:|---:|---:|
| VAH01 | baseline | 26.694 | 26.694 | 30.213 | 30.213 |
| VAH17 | baseline | 27.466 | 26.394 | 30.938 | 30.210 |
| VAH27 | baseline | 25.009 | 22.106 | 29.058 | 26.265 |
| VAH30 | 35°C chamber | 26.992 | 26.907 | 30.237 | 30.000 |
| VAH15 | 1000 s cruise | 28.866 | 27.213 | 32.977 | 31.511 |
| VAH12 | 400 s cruise | 20.296 | 20.296 | 22.830 | 22.830 |
| VAH11 | 20% lower discharge power | 24.223 | 23.341 | 34.929 | 33.635 |
| **Mean** |  | **25.65** | **24.71** |  |  |

Against the much weaker released checkpoint, unsupervised adaptation gives
small, mostly positive gains instead of the degradation seen with the
self-trained source.

### Result 3 — trained source, semisupervised

| Cell | Condition | Source only MAE | TATN MAE | Source only RMSE | TATN RMSE |
|---|---|---:|---:|---:|---:|
| VAH01 | baseline | 6.677 | 17.076 | 8.048 | 19.408 |
| VAH17 | baseline | 6.854 | 15.222 | 7.754 | 17.523 |
| VAH27 | baseline | 6.074 | 15.746 | 6.564 | 18.329 |
| VAH30 | 35°C chamber | 9.963 | 10.324 | 13.059 | 11.938 |
| VAH15 | 1000 s cruise | 7.262 | 14.695 | 8.053 | 18.762 |
| VAH12 | 400 s cruise | 8.660 | 10.988 | 12.120 | 12.214 |
| VAH11 | 20% lower discharge power | 4.392 | 22.081 | 9.487 | 23.778 |
| **Mean** |  | **7.13** | **15.16** |  |  |

Adding 50%-of-first-mission labels plus pseudo-labels reduces the damage
seen in Result 1 (mean MAE 18.13 → 15.16) but adaptation is still net
negative on every one of the seven cells relative to the strong source-only
baseline.

### Result 4 — released source, semisupervised

| Cell | Condition | Source only MAE | TATN MAE | Source only RMSE | TATN RMSE |
|---|---|---:|---:|---:|---:|
| VAH01 | baseline | 26.695 | 19.019 | 30.213 | 22.472 |
| VAH17 | baseline | 27.466 | 26.390 | 30.938 | 26.390 |
| VAH27 | baseline | 25.008 | 17.278 | 29.058 | 20.810 |
| VAH30 | 35°C chamber | 26.992 | 25.837 | 30.237 | 29.107 |
| VAH15 | 1000 s cruise | 28.866 | 25.088 | 32.977 | 30.504 |
| VAH12 | 400 s cruise | 20.297 | 17.994 | 22.831 | 20.799 |
| VAH11 | 20% lower discharge power | 24.218 | 14.385 | 34.926 | 19.215 |
| **Mean** |  | **25.65** | **20.28** |  |  |

This is the only one of the four settings with a consistently favourable
mean gain (25.65 → 20.28 MAE), although VAH12, VAH15, and VAH30 still
worsen slightly.

## Findings

- TATN's adaptation mechanism, as reproduced here, behaves in opposite
  directions depending on which source checkpoint it starts from. Against
  the strong self-trained source it is net harmful on every cell in both
  scenarios (Results 1 and 3). Against the weaker released checkpoint it
  is net helpful or roughly neutral (Results 2 and 4).
- This is the opposite qualitative behaviour from Vidal fine-tuning, Liu
  DARM, and Bian MDA, which all improve on their own strong sources for
  most cells (see the cross-method comparison note). It should be treated
  as a signal to review the TATN adaptation implementation and
  hyperparameters against the paper before drawing a scientific conclusion
  about the method itself.
- VAH11 shows the most severe degradation of any cell under the trained
  source in both scenarios (+25.9 and +19.4 MAE points), consistent with
  its role as the easiest zero-shot target across every method tested on
  this benchmark.
- Semisupervised labels (50% of one mission plus pseudo-labels) partially
  offset the damage relative to fully unsupervised adaptation under both
  source checkpoints, but do not reverse it under the trained source.

## Interpretation limits

1. Results reflect the run as executed in the notebook; seed control and
   repeat count are not documented in the notebook beyond the script
   defaults, so run-to-run variance is unknown.
2. CMU SOC is a development dynamic-capacity reference, not measured
   ground truth.
3. The self-trained vs. released source difference could reflect
   differences in feature scaling, training data coverage, or an
   implementation detail of TATN's adaptation loss rather than a property
   of the method itself; this needs code-level review before being reported
   as a finding about TATN.
4. Only the core-seven CMU cells were evaluated; conclusions do not
   generalise to the full CMU population.

## Related files

| Purpose | Path |
|---|---|
| Notebook | `notebooks/benchmarks/32_tatn_panasonic25_to_cmu.ipynb` |
| Runner script | `scripts/benchmarks/run_tatn_cmu.py` |
| Model | `src/battery_benchmark/reproductions/shen_2023/model.py` |
| Data loader | `src/battery_benchmark/reproductions/shen_2023/data.py` |
