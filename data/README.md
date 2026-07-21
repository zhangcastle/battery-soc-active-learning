# Data Directory

This directory is intentionally empty in version control.

Raw and processed battery datasets are **never committed to Git**.

## Directory layout

```
data/
├── raw/         # Original unmodified files from dataset providers
├── interim/     # Intermediate processing artefacts
├── processed/   # Standardised per-cycle DataFrames ready for training
└── manifests/   # Small CSV/YAML index files (can be committed)
```

## Obtaining datasets

| Dataset | Source | Status | Notes |
|---------|--------|--------|-------|
| Panasonic NCR18650PF | [McMaster / Mendeley](https://data.mendeley.com/datasets/wykht8y7tg/1) | ✅ In `raw/Panasonic_18650PF/` | See `notebooks/reproductions/vidal_2019/02_prepare_panasonic.ipynb` |
| LG HG2 | [McMaster / Mendeley](https://data.mendeley.com/datasets/wykht8y7tg/1) | ✅ In `raw/LG_HG2/` | See `notebooks/reproductions/vidal_2019/03_prepare_lg_hg2.ipynb` |
| Samsung 30T | [McMaster / Mendeley](https://data.mendeley.com/datasets/9xyvy2njj3/1) | ✅ In `raw/Samsung_30T/` | See naming note below |
| Samsung 48G | Unknown | ❌ Not found | See `raw/Samsung_48G/README.md` |
| CMU eVTOL | CMU Battery Data Lab | ⏳ Not yet obtained | See `docs/datasets/cmu_evtol/overview.md` |

### Samsung 30T naming note

Vidal et al. (2022) refers to this cell as **"Samsung T30"**, which appears to be a
typo or variant spelling of the actual product name **Samsung INR21700-30T** (3 Ah).
The dataset (Kollmeyer & Skells, McMaster, [Mendeley DOI 10.17632/9xyvy2njj3.1](https://data.mendeley.com/datasets/9xyvy2njj3/1),
published December 2020) was collected in 2019 and is consistent with the dataset
described in the paper. The benchmark uses `samsung_30t` as the canonical name.

## Google Colab usage

When running on Colab, place datasets in Google Drive and configure
`configs/paths/colab_drive.yaml` with your Drive paths.
The benchmark symlinks Drive paths so datasets are never duplicated.

See `notebooks/00_colab_setup.ipynb` for setup instructions.
