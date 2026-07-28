# Google Drive storage layout

This is the project-wide reference for persistent files used from Google
Colab. Runtime code and caches stay under `/content`; datasets, checkpoints,
and experiment results that must survive a Colab disconnect stay in Drive.

## Root paths

```text
Repository in Colab:  /content/battery-soc-transfer-benchmark
Temporary cache:      /content/battery_soc_cache
Google Drive root:    /content/drive/MyDrive/battery_soc_benchmark
```

The canonical configuration is
`configs/paths/colab_drive.yaml`. A different Drive location should be set
there or passed as a runtime root; Python source files should not be edited to
change personal storage paths.

## Persistent layout

```text
/content/drive/MyDrive/battery_soc_benchmark/
├── data/
│   ├── raw/                         Optional immutable raw datasets
│   ├── processed/                   Generated dataset artifacts
│   ├── labelled/
│   │   └── vidal_sequence_v2/       Vidal sequence-v2 training input
│   └── manifests/                   Dataset manifests copied for Colab runs
└── outputs/
    ├── logs/
    ├── reproductions/
    │   └── vidal_2019/
    │       ├── case1/
    │       ├── case2/
    │       ├── case3/
    │       └── case4/
    ├── benchmarks/
    │   └── panasonic_to_cmu_evtol/
    │       ├── liu_2025_darm/
    │       │   └── duration_matched/
    │       └── vidal/
    └── vidal_case1/ ... vidal_case4/  Existing legacy reproduction artifacts
```

Raw data should not be copied into the Git repository. The repository may
contain compact, reviewable generated CMU outputs where explicitly allowed by
its `.gitignore`; Drive remains the storage location for large training and
benchmark artifacts.

## Vidal reproduction outputs

The preferred layout for new reproduction runs is:

```text
outputs/reproductions/vidal_2019/
├── case1/<dataset>/seed_<seed>/<run_id>/
├── case2/source_<source>__target_<target>/seed_<seed>/<run_id>/
├── case3/source_<source>__target_<target>/seed_<seed>/<run_id>/
└── case4/
    ├── pretraining/seed_<seed>/<run_id>/
    └── target_<target>/seed_<seed>/<run_id>/
```

Existing formal runs were created before this namespace was introduced and
remain under:

```text
outputs/vidal_case1/
outputs/vidal_case2/
outputs/vidal_case3/
outputs/vidal_case4/
```

Do not move or rename these legacy runs: notebooks and provenance records may
refer to their exact paths. Readers may discover both layouts, but a
checkpoint, scaler, and resolved config must always come from the same run
directory.

## Liu 2025 reproduction

The current Panasonic `25 C -> 0 C` notebook writes to:

```text
outputs/liu_2025/25c_to_0c/
├── smoke/
├── formal/
├── formal_paper_ivt/
├── formal_paper_protocol_v2_source_only/
├── diagnostic_source_checkpoints/
├── diagnostic_preprocessing_audit/
├── diagnostic_update_semantics_20260728/
│   ├── bn_forward_audit/
│   ├── adversarial_grl_legacy/
│   ├── adversarial_grl_frozen_stats/
│   ├── adversarial_alternating_frozen_stats/
│   ├── adversarial_eq12_frozen_stats/
│   ├── mmd_legacy/
│   └── mmd_frozen_stats/
├── diagnostic_frozen_stats_grl_full_loss_10ep/
├── diagnostic_frozen_stats_reconstruction_mmd_10ep/
├── diagnostic_frozen_stats_alternating_full_loss_10ep/
├── diagnostic_fully_frozen_alternating_full_loss_10ep/
├── diagnostic_frozen_stats_alternating_adversarial_only_10ep/
├── diagnostic_bn_frozen/
├── diagnostic_clip/
├── diagnostic_no_reconstruction/
├── diagnostic_adversarial_only/
├── diagnostic_mmd_only/
├── diagnostic_target_normalized/
└── diagnostic_target_domain_normalized/
```

All `diagnostic_*` directories reuse `formal/source_artifact.pt`; they
do not retrain or overwrite the formal source model. `diagnostic_clip/` clips
source-normalized inputs to `[0, 1]`, while `diagnostic_no_reconstruction/`
sets the adaptation reconstruction weight to zero.
`diagnostic_adversarial_only/` and `diagnostic_mmd_only/` isolate the two
remaining alignment losses.
`diagnostic_target_normalized/` fits a target min-max scaler using only the
unlabelled target Cycle 1.
`diagnostic_target_domain_normalized/` is a transductive diagnostic whose
scaler sees all 0 C input covariates, but no target SOC labels.
`formal_paper_ivt/` is a separate full run using the paper's stated
`[current, voltage, temperature]` channel order.
`formal_paper_protocol_v2_source_only/` is the corrected source-only check:
one 25 C HWFET session and a diagnostic scaler fitted to all 25 C input
covariates without reading source-test SOC labels. It does not run adaptation
or overwrite `formal_paper_ivt/`.
`diagnostic_source_checkpoints/` compares the paper-order source artifact,
best checkpoint, and last checkpoint without retraining.
`diagnostic_preprocessing_audit/` compares SOC label formulas and reports
source-scaler range violations without retraining.
`diagnostic_update_semantics_20260728/` isolates target-BN forward drift and
the three adversarial update semantics using the same paper-order source
artifact, seed, data order, and one adaptation epoch. It does not replace any
formal run.
`diagnostic_frozen_stats_grl_full_loss_10ep/` reuses that paper-order source
artifact for a ten-epoch full-loss run with frozen target BN statistics and
joint GRL updates.
`diagnostic_frozen_stats_reconstruction_mmd_10ep/` removes the adversarial
term, while `diagnostic_frozen_stats_alternating_full_loss_10ep/` replaces
joint GRL with alternating flipped-label updates; all other settings match
the ten-epoch full-loss diagnostic.
`diagnostic_fully_frozen_alternating_full_loss_10ep/` repeats the alternating
run with target-BN affine parameters frozen to isolate their contribution to
late feature drift.
`diagnostic_frozen_stats_alternating_adversarial_only_10ep/` removes
reconstruction and MMD to test the alternating adversarial game without
competing target-encoder losses.

### Current canonical Panasonic source bundle

```text
outputs/vidal_case1/panasonic_18650pf/seed_42/
└── 20260716_090443_c4728bba/
    ├── checkpoints/epoch_515.pt
    ├── scaler.json
    └── resolved_config.yaml
```

CMU zero-shot must use this Panasonic Case 1 scaler. It must not use a
target-fitted Case 2 scaler or fit a scaler on CMU data.

## Panasonic → CMU Vidal benchmark

```text
outputs/benchmarks/panasonic_to_cmu_evtol/vidal/
├── zero_shot/seed_42/<run_id>/
├── budget_1pct/seed_42/<run_id>/
├── budget_5pct/seed_42/<run_id>/
├── budget_10pct/seed_42/<run_id>/
├── vidal_partial/seed_42/<run_id>/
├── vidal_full/seed_42/<run_id>/
└── summary/
```

Each setting run owns its per-cell predictions, histories, metrics,
checkpoints/scalers where applicable, `cell_metrics.csv`, and
`run_metadata.json`. The current run IDs and analysis are recorded in
`docs/benchmarks/vidal_panasonic_to_cmu_evtol_results_v0_1.md`.

## Environment variables

| Variable | Meaning | Typical Colab value |
|---|---|---|
| `BATTERY_BENCHMARK_ROOT` | Repository checkout | `/content/battery-soc-transfer-benchmark` |
| `VIDAL_ARTIFACT_ROOT` | Root searched for existing Vidal artifacts | `/content/drive/MyDrive/battery_soc_benchmark/outputs` |

`GITHUB_TOKEN` is a Colab Secret used only to clone or update the private
repository. It is not a storage path and must never be printed or written to
Drive, notebook output, configuration, or Git.

## Run-directory requirements

Persistent experiment directories should be immutable after completion and
contain enough provenance to resolve:

- dataset and protocol/config;
- generator or experiment entry point;
- input artifact paths and hashes;
- Git revision and dirty state;
- seed, device, and generation time;
- checkpoint, scaler, metrics, predictions, and QC belonging to that run.

Smoke/development outputs must remain visibly separate from formal outputs,
usually under `_smoke/` or a metadata status field. A file's presence alone
does not establish that a run is formal or scientifically validated.

## Colab usage

```python
from google.colab import drive
drive.mount("/content/drive")
```

Use `notebooks/00_colab_setup.ipynb` for repository setup and Drive checks.
Experiment notebooks should consume the roots above rather than embed private
Drive URLs or user-machine absolute paths.
