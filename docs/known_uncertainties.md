# Known Uncertainties

This document lists aspects of the benchmark that are not yet resolved.
**Do not claim paper-faithful results for any item listed here.**

---

## 1. Vidal et al. sequence construction

**Status:** Unconfirmed.

The exact method used in Vidal et al. (2019 ITEC) to segment continuous battery
recordings into LSTM input sequences has not yet been confirmed from the paper.
This includes:
- Sequence length
- Whether sequences span cycle boundaries
- Whether each step or only the last step is used as the prediction target

**Impact:** The reported architecture is `VidalLSTM` (`hidden_size=10`, 1 layer),
but training performance may differ from the paper until sequence construction
is verified.

---

## 2. Vidal et al. normalisation details

**Status:** Unconfirmed.

The Vidal et al. (2019 ITEC) paper's feature normalisation method (min-max
per-feature, z-score, etc.) and the data used to fit the scaler (per-cycle,
global, or per-profile) are not yet confirmed.

---

## 3. Samsung dataset availability and structure

### Samsung 30T — RESOLVED

**Status:** Confirmed. Data in `data/raw/Samsung_30T/`.

The dataset used is **Samsung INR21700-30T** (3 Ah), collected at McMaster University
by Dr. Phillip Kollmeyer, published on Mendeley December 2020
([DOI 10.17632/9xyvy2njj3.1](https://data.mendeley.com/datasets/9xyvy2njj3/1)).
Test data was collected in 2019 and is consistent with the Vidal et al. (2019 ITEC) paper.

Vidal et al. refers to this cell as **"Samsung T30"** — this is a typo/variant of the
actual name "Samsung 30T" (or "3.0T" as written in the Readme). The benchmark uses
the canonical name `samsung_30t`.

File structure: temperature subdirectories (`-20degC`, `-10degC`, `0degC`, `10degC`,
`25degC`, `40degC`), each containing `.mat` drive-cycle files (UDDS, HWFET, LA92, US06,
Mixed 1–8) and charge files.

### Samsung 48G — UNRESOLVED

**Status:** Dataset not found. `data/raw/Samsung_48G/` is a placeholder.

No public McMaster Samsung 48G (INR21700-48G, ~4.8 Ah) dataset has been located.
The McMaster Mendeley page (`wykht8y7tg/1`) does not appear to include a Samsung 48G.

**Impact:** `configs/experiments/vidal_case3.yaml` (Panasonic → Samsung 48G) cannot
be run until this dataset is obtained.

**Next steps:** Search IEEE DataPort and other repositories; contact Dr. Kollmeyer.

**Affected configs:** `configs/datasets/samsung_48g.yaml`

---

## 4. CMU eVTOL SOC reference

**Status:** Physics-derived only.

The CMU eVTOL dataset does not provide directly measured SOC as a ground-truth
label.  Any SOC reference used for evaluation is derived from coulomb counting
with dynamic capacity estimation and therefore:
- Contains accumulated integration error
- Depends on capacity estimation quality
- Should be reported as a proxy reference, not ground truth

**Mitigation:** Evaluation should clearly state `label_source = "coulomb_counted_dynamic_capacity"`.
Physical-consistency metrics (boundary violations, monotonicity) should be
reported separately from proxy accuracy metrics.

See `docs/cmu_evtol_protocol.md` for the recommended evaluation protocol.

---

## 5. Transfer method implementations

**Status:** Placeholders only.

The following transfer methods are interface stubs and raise `NotImplementedError`:
- MMD (Maximum Mean Discrepancy)
- CORAL (Correlation Alignment)
- DANN (Domain-Adversarial Neural Network)
- Pseudo-label self-training
- Source-free adaptation

These will be implemented in future iterations.

---

## 6. Dataset adapters

**Status:** Placeholders only.

The following dataset adapters raise `NotImplementedError`:
- `PanasonicDataset`
- `LGDataset`
- `Samsung48GDataset`
- `Samsung30TDataset`
- `CMUeVTOLDataset`

Only `SyntheticDataset` is fully functional for the current smoke test.
