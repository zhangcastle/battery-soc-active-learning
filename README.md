# Battery SOC Active-Learning Confirmation

A minimal, Colab-friendly confirmation of whether active target-window selection
can reduce the labelled data needed by Vidal Case 2 full-model fine-tuning.

The confirmation runs exactly three lines with the same model and environment:
full target data, a random subset, and a Regression-CLUE subset at the same
budget. Regression-CLUE combines query-by-committee regression uncertainty
([Wu, 2019](https://arxiv.org/abs/1805.00334)) with uncertainty-weighted hidden
embedding clustering adapted from
[CLUE](https://github.com/virajprabhu/CLUE). It is an adaptation for sequence
regression, not an exact reproduction of the classification method.

## Minimal confirmation

Open `notebooks/active_learning_confirmation.ipynb` in Colab. It mounts the
same `/content/drive/MyDrive/battery_soc_benchmark` layout used by the original
repo, reuses the existing LG HG2 seed-42 checkpoint, trains seed 43 once when
needed, and runs one source-to-target pair at a 10% label budget. The result is one table:
`confirmation_summary.csv`.

Because the repository is private, enable a Colab secret named `GITHUB_TOKEN`
or `GITHUB_1` with read access to this repository before running the first cell.

The notebook creates the second formal Case 1 checkpoint automatically because
two independently trained source models are needed to estimate regression
disagreement without reading target SOC labels.

> **Status:** Vidal 2019 Case 1–4 reproduction pipelines and the CMU Version 3
> ingestion/block/RPT pipeline are implemented; unified CMU transfer remains in
> development.
> See [Known Uncertainties](docs/known_uncertainties.md).

---

## Why this repository exists

The long-term goal is to:

1. Reproduce Vidal et al.'s cross-chemistry transfer learning results.
2. Compare them fairly against other SOC estimation + transfer methods
   (MMD, CORAL, DANN, pseudo-labels, source-free adaptation).
3. Transfer the best methods to the CMU eVTOL battery dataset,
   where target-domain SOC labels may be unavailable.

**The existing `battery-state-of-charge-estimation` repository is a
reference only.** This repository is independent and does not modify it.

---

## Quick start

### Google Colab

1. Open [`notebooks/00_colab_setup.ipynb`](notebooks/00_colab_setup.ipynb).
2. Follow the cells to mount Drive, clone the repo, install, and run the smoke test.
3. See [`docs/google_drive_layout.md`](docs/google_drive_layout.md) for persistent paths.

### Local machine

```bash
git clone https://github.com/your-org/battery-soc-transfer-benchmark
cd battery-soc-transfer-benchmark
pip install -e .

# Run the synthetic smoke test (no data required)
python scripts/train.py \
    --config configs/experiments/smoke_test.yaml \
    --paths  configs/paths/local.yaml
```

Or with Make:

```bash
make install
make smoke
make test
```

---

## Repository structure

```
battery-soc-transfer-benchmark/
├── notebooks/          Grouped by datasets, reproductions, and benchmarks
├── configs/            Matching dataset/reproduction/benchmark namespaces
├── src/battery_benchmark/
│   ├── datasets/       Dataset-owned pipelines (for example CMU eVTOL)
│   ├── reproductions/  Paper-owned implementations (Vidal 2019)
│   ├── data/           Shared adapters, records, schema, splits
│   ├── preprocessing/  Normalisation, sequence building
│   ├── labels/         Shared label primitives
│   ├── models/         Reusable model definitions
│   ├── transfer/       Transfer learning methods
│   ├── training/       Trainer, checkpointing, callbacks
│   ├── evaluation/     Metrics, evaluator, plots
│   ├── experiments/    ExperimentRunner, result schema
│   └── utils/          Config, paths, device, logging
├── scripts/            Matching dataset/reproduction/benchmark CLIs
├── tests/              Mirrors implementation ownership
├── docs/               Protocols and audit conclusions by owner
└── outputs/            Ignored generated artifacts in three namespaces
```

---

## Paper-faithful vs. unified benchmark mode

| Mode | Purpose | Use when |
|------|---------|----------|
| **Paper-faithful** | Reproduce original method exactly | Verifying published numbers |
| **Unified benchmark** | Same conditions for all methods | Fair cross-method comparison |

See [`docs/fairness_protocol.md`](docs/fairness_protocol.md).

---

## Labelled vs. unlabelled target-domain experiments

The benchmark supports all label availability scenarios:

| Scenario | `label_availability.target_labels` |
|----------|-------------------------------------|
| Source-only baseline | `false` |
| Fine-tuning (supervised) | `true` |
| Unsupervised DA (MMD, CORAL, DANN) | `false` |
| Self-training | `false` (pseudo-labels generated internally) |
| Source-free | `false` |

CMU eVTOL target SOC labels are unavailable during adaptation.
Evaluation uses a coulomb-counted proxy reference.
See [`docs/datasets/cmu_evtol/overview.md`](docs/datasets/cmu_evtol/overview.md).

---

## Implementation status

| Component | Status |
|-----------|--------|
| Synthetic dataset | ✅ Complete |
| VidalLSTM + Case 1–4 pipelines | ✅ Implemented |
| Source-only transfer | ✅ Complete |
| Fine-tuning transfer | ✅ Complete |
| Scratch baseline | ✅ Complete |
| Trainer + checkpointing | ✅ Complete |
| Metrics (MAE/RMSE/MAX) | ✅ Complete |
| Colab setup notebook | ✅ Complete |
| Panasonic / LG HG2 / Samsung 30T adapters | ✅ Implemented |
| CMU Version 3 raw/block/RPT pipeline | ✅ Implemented |
| CMU mission/SOC benchmark pipeline | ⚠️ Development-only |
| MMD / CORAL / DANN | ⏳ Placeholder |
| Pseudo-label / source-free | ⏳ Placeholder |
| Vidal reproduction | ⚠️ Implemented reconstruction; documented assumptions remain |
| Liu 2025 DARM (25°C → 0°C) | ⚠️ Manifest-driven reconstruction; formal run pending |
| Shen 2023 TATN Tables II/IV/V/VI/VII | ✅ Colab-ready paper reproduction |
| CMU transfer | ⚠️ Protocol established; formal benchmark not complete |

---

## Running the smoke test

```bash
python scripts/train.py \
    --config configs/experiments/smoke_test.yaml \
    --paths  configs/paths/local.yaml
```

Expected output:
```
Experiment: smoke_test
Source → Target: synthetic_A → synthetic_B
Transfer method: source_only
Source test  MAE:       <value>
Source test  RMSE:      <value>
Outputs saved to:       outputs/benchmarks/smoke_test
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Known warnings

- Paper reproductions are **not complete**. See `docs/known_uncertainties.md`.
- Samsung dataset availability is unconfirmed.
- CMU eVTOL SOC reference is physics-derived, not directly measured.

---

## License

MIT. See [LICENSE](LICENSE).


test git
