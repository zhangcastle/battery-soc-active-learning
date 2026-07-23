# Project structure

The repository separates shared infrastructure, dataset pipelines, paper
reproductions, and unified benchmarks. A file belongs in the narrowest scope
that owns its scientific assumptions.

```text
battery-soc-transfer-benchmark/
├── configs/
│   ├── datasets/cmu_evtol/       CMU dataset and protocol configuration
│   ├── reproductions/vidal_2019/ Vidal paper cases and model configuration
│   ├── benchmarks/               Cross-paper / cross-dataset comparisons
│   ├── models/                   Shared model configurations
│   ├── transfer/                 Shared transfer-method configurations
│   └── paths/                    Runtime path configuration
├── data/
│   ├── raw/                      Immutable local source data; ignored by Git
│   ├── processed/                Generated datasets; ignored by Git
│   └── manifests/
│       ├── cmu_evtol/            CMU assets, protocols, and compact QC tables
│       └── vidal_2019/           Vidal source-file split manifests
├── docs/
│   ├── benchmarks/               Unified benchmark results and analyses
│   ├── datasets/cmu_evtol/       CMU protocols and evidence audits
│   └── reproductions/vidal_2019/ Vidal reproduction protocol and Colab notes
├── notebooks/
│   ├── datasets/cmu_evtol/       CMU preparation notebooks
│   ├── reproductions/vidal_2019/ Vidal paper-case notebooks
│   └── benchmarks/               Unified comparison notebooks
├── scripts/
│   ├── datasets/cmu_evtol/       CMU ingestion, QC, audit, and RPT CLIs
│   ├── reproductions/vidal_2019/ Vidal labeling and case CLIs
│   └── benchmarks/               Unified benchmark CLIs
├── src/battery_benchmark/
│   ├── datasets/cmu_evtol/       CMU raw/block/RPT/SOC dataset logic
│   ├── reproductions/vidal_2019/ Vidal model, labels, windows, and cases
│   ├── data/                      Shared schemas, records, adapters, and splits
│   ├── preprocessing/             Shared normalization and sequence tools
│   ├── models/                    Reusable non-paper-specific models
│   ├── training/                  Shared trainers and checkpoints
│   ├── evaluation/                Shared metrics and evaluators
│   ├── transfer/                  Shared transfer-learning methods
│   ├── experiments/               Unified runner and result schema
│   ├── labels/                    Shared label primitives
│   └── utils/                     Paths, environment, Git, logging, hashing
├── tests/
│   ├── datasets/cmu_evtol/       CMU pipeline and protocol tests
│   ├── reproductions/vidal_2019/ Vidal reproduction tests
│   ├── benchmarks/               Unified benchmark protocol and manifest tests
│   └── test_*.py                  Shared infrastructure tests
└── outputs/
    ├── datasets/<dataset>/<pipeline>/
    ├── reproductions/<paper>/<case>/
    └── benchmarks/<benchmark>/<method>/
```

## Ownership rules

- Dataset assumptions such as CMU `Ns`, continuous blocks, RPT detection, and
  the fixed 3.0 Ah development reference stay under `datasets/cmu_evtol`.
- Paper assumptions such as Vidal's 571-parameter LSTM, Case 1–4, sequence-v2
  labels, and paper split manifests stay under `reproductions/vidal_2019`.
- Panasonic, LG, and Samsung raw adapters remain in shared `data/`: they are
  inputs to both the Vidal reproduction and future unified benchmarks.
- Normalization, sequence construction, training, metrics, generic models, and
  transfer methods remain shared because they have no dataset- or paper-owned
  scientific rule.
- Reviewable generated outputs, including CMU QC and lookup artifacts, may be
  committed for verification. Producers record dataset, protocol/config,
  generator, input hashes, generation time, and Git revision plus dirty state
  in metadata.

Historical top-level output directories are legacy artifacts. New commands use
the namespaced layout documented in `outputs/README.md`.
