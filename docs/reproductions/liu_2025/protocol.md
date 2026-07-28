# Liu et al. 2025 DARM reproduction

## Current scope

This phase implements one fixed-temperature task: Panasonic 18650PF
`25 C -> 0 C`.

| Role | Files | Labels |
|---|---|---|
| `source_train` | 25 C Cycle 1-4 | Used to train the source encoder and SOC predictor |
| `source_test` | 25 C LA92, UDDS, US06, HWFET, NN | Evaluation only |
| `target_adapt` | 0 C Cycle 1 | Forbidden; the adaptation loader contains no SOC tensor |
| `target_test` | 0 C LA92, UDDS, US06, HWFET, NN | Evaluation only |

The authoritative file assignment and hashes are in
`data/manifests/liu_2025/panasonic_25c_to_0c.csv`.

The public 25 C data contains two separate HWFET sessions. The paper reports
and plots one approximately 7000-second HWFET sequence, so this protocol uses
the first session (`HWFTa`) and excludes the later repeat (`HWFTb`). The paper
does not publish the exact filename, so selection of the first session remains
an explicit reconstruction assumption.

## Related files

| Purpose | Path |
|---|---|
| Dataset split and hashes | `data/manifests/liu_2025/panasonic_25c_to_0c.csv` |
| Model | `src/battery_benchmark/reproductions/liu_2025/darm.py` |
| Manifest validation | `src/battery_benchmark/reproductions/liu_2025/protocol.py` |
| Training entry point | `scripts/reproductions/liu_2025/run_reproduction.py` |
| Colab entry point | `notebooks/reproductions/liu_2025/20_liu_2025_darm_25c_to_0c.ipynb` |
| Protocol and assumptions | `docs/reproductions/liu_2025/protocol.md` |

## Paper settings

- Input: current, voltage, cell temperature (`paper_ivt`)
- Sampling: 1 Hz
- Window: 40 time steps, reconstructed stride 1
- Source training: Adam, learning rate 0.001, up to 500 epochs
- Target adaptation: Adam, learning rate 0.0001, up to 100 epochs
- Batch size: 256
- Loss: `L_adv + 10 * L_rec + L_mmd`
- Early stopping patience: 20

## Reconstruction assumptions

The paper does not publish code or fully specify the following details:

- SOC reference formula: this implementation uses per-file cycler-Ah normalisation.
- Window stride: 1.
- DWS convolution kernel: 3.
- Decoder upsampling: nearest-neighbour.
- Attention scoring: learned scalar softmax over the five latent steps.
- MK-MMD bandwidths: median distance multiplied by `[0.5, 1, 2, 4]`.
- Adversarial sign: gradient reversal.
- Target BatchNorm defaults to `legacy` (running statistics and affine
  parameters update). `frozen_stats` freezes running statistics only, and
  `fully_frozen` also freezes affine parameters.
- Adversarial update defaults to `grl_joint`. `alternating_flipped` separates
  the discriminator and target-encoder steps; `eq12_joint` is a diagnostic
  literal reading of Eq. 12 and is never presented as the recommended method.
- Adaptation outputs use the last checkpoint. Lowest total loss is retained
  only as a proxy checkpoint because GRL is not a joint minimization objective.
- Target-label checkpoint comparisons are post-hoc diagnostic oracles and
  never participate in optimization, early stopping, or formal selection.
- Source milestone comparisons likewise use source-test labels only after
  training; they diagnose generalization and never select the formal source
  artifact.
- Early-stopping monitor: current-stage training-loss proxy.
- Min-max normalisation scope: the baseline fits source Cycle 1-4 only. The
  paper states that measurements are transformed to `[0, 1]` but does not
  identify the fitting scope. `--source-normalization source_domain` is a
  transductive diagnostic that fits the source scaler on all 25 C input
  covariates without reading test SOC labels; the historical default remains
  `source_train`. `--target-normalization target_adapt` diagnoses
  a separate scaler fitted only on unlabeled target Cycle 1, while
  `target_domain` diagnoses normalization over all target input covariates
  without using target SOC labels.
- Feature order: the original reconstruction used
  `[voltage, current, temperature]`; `--feature-order paper_ivt` follows the
  paper's explicit `[current, voltage, temperature]` definition and requires a
  separately trained source artifact.

These assumptions must remain visible when comparing against the paper.

## Colab

Use `notebooks/reproductions/liu_2025/20_liu_2025_darm_25c_to_0c.ipynb`.
The notebook defaults to a smoke run. Change `RUN_MODE` to `"formal"` for the
paper settings. Formal checkpoints and results should be written to Google
Drive so `--resume` survives a Colab disconnect.

An L4 GPU has ample memory for this approximately 198k-parameter model and
batch size 256. Runtime, not memory, is the constraint: 500 source epochs may
take hours, and Colab session limits can interrupt the run.

## Downstream CMU extension

The paper reproduction remains the source-of-truth source protocol. The
subsequent Panasonic-to-CMU development benchmark is specified separately in
`docs/benchmarks/liu_2025_panasonic_to_cmu_evtol_plan.md`.
