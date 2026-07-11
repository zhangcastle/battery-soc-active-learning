# Known Uncertainties

This document lists aspects of the benchmark that are not yet resolved.
**Do not claim paper-faithful results for any item listed here.**

---

## 1. Vidal et al. sequence construction

**Status:** Unconfirmed.

The exact method used in Vidal et al. (2022) to segment continuous battery
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

The paper's feature normalisation method (min-max per-feature, z-score, etc.)
and the data used to fit the scaler (per-cycle, global, or per-profile) are
not yet confirmed.

---

## 3. Samsung dataset availability and structure

**Status:** Unconfirmed.

The availability and file structure of the Samsung 48G and Samsung 30T datasets
referenced in Vidal et al. require verification.  If these datasets are not
publicly available, alternative datasets must be identified.

**Affected configs:** `configs/datasets/samsung_48g.yaml`, `configs/datasets/samsung_30t.yaml`

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
