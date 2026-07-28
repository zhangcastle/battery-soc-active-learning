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

- Input: voltage, current, cell temperature
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
- Early-stopping monitor: current-stage training loss.
- Min-max normalisation scope: the baseline fits source Cycle 1-4 only. The
  paper states that measurements are transformed to `[0, 1]` but does not
  identify the fitting scope; `--target-normalization target_adapt` diagnoses
  a separate scaler fitted only on unlabeled target Cycle 1.

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
