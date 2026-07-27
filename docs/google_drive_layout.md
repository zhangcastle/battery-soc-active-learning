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
