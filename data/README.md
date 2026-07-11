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

| Dataset | Source | Notes |
|---------|--------|-------|
| Panasonic NCR18650PF | McMaster University (public) | See `notebooks/02_prepare_panasonic.ipynb` |
| LG HG2 | McMaster University (public) | See `notebooks/03_prepare_lg_hg2.ipynb` |
| Samsung 48G | McMaster University (public) | See `notebooks/04_prepare_samsung.ipynb` |
| Samsung 30T | To be confirmed | See `docs/known_uncertainties.md` |
| CMU eVTOL | CMU Battery Data Lab | See `docs/cmu_evtol_protocol.md` |

## Google Colab usage

When running on Colab, place datasets in Google Drive and configure
`configs/paths/colab_drive.yaml` with your Drive paths.
The benchmark symlinks Drive paths so datasets are never duplicated.

See `notebooks/00_colab_setup.ipynb` for setup instructions.
