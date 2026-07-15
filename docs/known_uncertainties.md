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

### Samsung 30T — STRONGLY SUPPORTED (identity audit complete)

**Status:** STRONGLY SUPPORTED. Data in `data/raw/Samsung_30T/`.
Manifest: `data/manifests/samsung_30t_vidal2019_25c_manifest.csv`.
Report: `docs/samsung_30t_vidal2019_25c_split_report.md`.

The public dataset is **Samsung INR21700-30T** (3 Ah), collected at McMaster
University by Dr. Phillip Kollmeyer, published on Mendeley December 2020
([DOI 10.17632/9xyvy2njj3.1](https://data.mendeley.com/datasets/9xyvy2njj3/1)).
Data was collected March 2019, chronologically consistent with the 2019 paper.

Vidal et al. refers to this cell as **"Samsung T30"**.  The README names it "3.0T";
filenames use "IN21700_30T".  "T30" is most plausibly a transposition of "30T"
(digits and letter inverted), not a different cell.  All verifiable attributes
(capacity, format, profiles, institution, collection date) align.

**Not CONFIRMED** because no author-provided explicit statement cross-references
"Samsung T30" (paper) to "Samsung INR21700-30T" (public dataset).

See §5 below for Samsung-specific unresolved items.

### Samsung 48G — UNRESOLVED

**Status:** Dataset not found. `data/raw/Samsung_48G/` is a placeholder.

No public McMaster Samsung 48G (INR21700-48G, ~4.8 Ah) dataset has been located.
The McMaster Mendeley page (`wykht8y7tg/1`) does not appear to include a Samsung 48G.

**Impact:** `configs/experiments/vidal_case3.yaml` (Panasonic → Samsung 48G) cannot
be run until this dataset is obtained.

**Next steps:** Search IEEE DataPort and other repositories; contact Dr. Kollmeyer.

**Affected configs:** `configs/datasets/samsung_48g.yaml`

---

## 4. LG HG2 dataset — LG-specific unresolved items

### 4a. HWFET 25°C — public file contains no drive-cycle excitation — UNRESOLVED

**Status:** Data quality failure confirmed; cause unknown.

The file `10-29-18_01.36 551_HWFET_25degC_LGHG2.mat` contains **600 seconds of
zero current** at ~4.186 V (near full charge, at rest). The public HWFET-labelled
file contains no drive-cycle excitation. This file is excluded from the manifest
(`qc_status = FAIL`).

- **Impact:** No usable HWFET 25°C data exists in the public dataset. This matches
  the paper's protocol (HWFET is not listed as a test profile in Vidal 2019 Table I),
  so the Vidal reproduction is not affected.
- **Unresolved:** Why the public HWFET-labelled file contains no drive-cycle
  excitation is unknown. Possible explanations: test aborted at start, equipment
  issue, or file mislabelled.
- **Next step:** Check HWFET files at other temperatures (0°C, 10°C, 40°C,
  n10°C, n20°C) to determine whether the 25°C instance is uniquely affected.

### 4b. Mixed{N} = Mix{N} mapping — INFERRED

**Status:** Inferred, not confirmed from paper text.

The paper uses "Mix1–8" (Table I); the public dataset uses "Mixed1–8" (filenames).
The count-based and chronological mapping `Mixed{N} → Mix{N}` is the most
plausible correspondence, supported by the dataset README ("mix 1-8 consist of
random mix of UDDS, HWFET, LA92, US06"). However, neither source explicitly
states this equivalence.

- **Impact:** If the naming is non-sequential (e.g., Mixed3 ≠ Mix3 in some
  categorization), the split assignment is still correct at the file level but
  the paper-profile label in the manifest would be incorrect.
- **Next step:** If supplementary materials or original code from Vidal et al.
  become available, cross-check the mix profile order.

### 4c. Charge Ah counter reset behavior — OBSERVED

**Status:** Observed in 4 charge files; behavior documented.

Files `549_Charge`, `551_Charge2`, `551_Charge8`, and `552_Charge9` show
`Ah_end = 0` but `Ah_max > 0`. This appears to be a cycler behavior where the
Ah counter resets to 0 at the end of the recording step. The `Ah_max` value
(peak accumulated Ah) is used as the canonical "Ah charged" metric.

- **Impact:** None on split assignment. Relevant for future SOC computation:
  the adapter must use `Ah_max` (not `Ah_end`) for these files, or integrate
  the current-time trajectory directly.

---

## 5. Samsung INR21700-30T dataset — Samsung-specific unresolved items

### 5a. Battery identity "T30" vs. "30T" — STRONGLY SUPPORTED, not CONFIRMED

**Status:** Strongly supported.

The Vidal 2019 paper calls the cell "Samsung T30"; the public dataset README calls
it "Samsung 3.0T"; filenames use "IN21700_30T".  "T30" is most plausibly a
transposition of "30T".  No author-provided explicit cross-reference exists.

- **Impact:** The split is labelled PROVISIONAL.  Benchmark results using Samsung
  30T should acknowledge the identity uncertainty if claiming paper-faithful
  reproduction of Vidal Case 4.
- **Resolution path:** A direct statement from Dr. Kollmeyer or a supplementary
  from Vidal et al. would confirm or reject the equivalence.

### 5b. Missing second Cap_1C .mat at 25 °C — UNRESOLVED

**Status:** Data gap.

The README documents a second 1C discharge test ("performed before the Mix3
cycle"), but no corresponding `Cap_1C` `.mat` file exists in the 25 °C directory.
The Charge9/Charge10 two-charge sequence and the ~70-minute gap between them are
consistent with the missing test.

- **Impact:** Charge9 and Charge10 split assignments were inferred by
  reconstruction.  Charge9 (tiny top-off) is excluded; Charge10 (full recharge)
  is assigned to full_train/partial_train as the pre-Mixed3 charge.
- **Resolution path:** None identified; the file does not appear in the public
  Mendeley release.

### 5c. Partial-train profile selection for Samsung T30 — INFERRED

**Status:** Mix1–3 used; inferred from paper text ("cycles MIX 1-3 … for Cases
3 and 4").

The paper does not give a Samsung T30-specific partial-train table.  Samsung 48G
is the documented exception (uses Mix 1, 2, 6).

- **Impact:** If Samsung T30 also uses a non-sequential subset (e.g., Mix 1, 2, 6),
  the `partial_train` flags need updating.
- **Resolution path:** Author supplementary or code release.

### 5d. Samsung 30T 25 °C HWFET validity — CONFIRMED VALID, but excluded by protocol

**Status:** Valid drive cycle; excluded per paper protocol.

Unlike the LG HG2 HWFET 25 °C file (which was a rest file), the Samsung 30T
HWFET 25 °C file (`734_HWFET_25degC_IN21700_30T.mat`) is a real drive cycle
(2600 s, I_min = −18.4 A, I_max = +12.0 A regen).  It is excluded because
Vidal 2019 Table I does not list HWFET as a test profile for Samsung T30.

- **Impact:** None on the Vidal reproduction.  If future work includes HWFET as
  a test profile for Samsung, the manifest should be updated.

---

## 6. SequenceRecord — known implementation limitations (low-priority tech debt)

**Status:** Non-blocking. Normal usage is not affected.

The following limitations exist in the current `SequenceRecord` implementation
(`src/battery_benchmark/data/sequence_record.py`):

### 6a. `metadata` dict is not defensively copied

`SequenceRecord.__post_init__` copies the incoming `data` DataFrame to prevent
mutation of the caller's original.  However, the `metadata` dict is **not**
copied — `record.metadata` and the caller's original dict are the same object.
If the caller modifies the original dict after construction, `record.metadata`
will reflect those changes.

- **Impact under normal use:** None, as long as callers do not modify the
  metadata dict after passing it to the constructor.  The adapters being built
  for this project follow that convention.
- **Risk:** A careless caller could silently corrupt a record's metadata after
  construction, bypassing the validation that ran in `__post_init__`.
- **Recommended fix:** Add `object.__setattr__(self, "metadata", dict(self.metadata))`
  in `__post_init__` (a shallow copy is sufficient for all current metadata
  values, which are scalars or None).

### 6b. No type guards on `metadata` and `qc` arguments

`SequenceRecord` currently validates the *content* of the `metadata` dict and
the *values* inside `qc`, but does not check that:
- `metadata` is actually a `dict` (not a list, object, etc.),
- `qc` is actually a `QCResult` instance (not an arbitrary object).

- **Impact under normal use:** None; all callers use the correct types.
- **Risk:** Misleading `AttributeError` at validation time rather than a clear
  `TypeError` at construction time.
- **Recommended fix:** Add two explicit type checks in `__post_init__`:
  ```python
  if not isinstance(self.metadata, dict):
      raise TypeError(f"metadata must be a dict, got {type(self.metadata).__name__!r}")
  if not isinstance(self.qc, QCResult):
      raise TypeError(f"qc must be a QCResult, got {type(self.qc).__name__!r}")
  ```

### 6c. No regression tests for the above edge cases

Tests for `metadata`-is-not-dict and `qc`-is-not-QCResult are not yet in
`tests/test_sequence_record.py`.

**Resolution plan (all low priority, not blocking current work):**
1. Shallow-copy `metadata` in `__post_init__`.
2. Add `isinstance` guards for `metadata` and `qc`.
3. Add corresponding regression tests (3–4 test cases).

---

## 7. CMU eVTOL SOC reference

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

## 8. Transfer method implementations

**Status:** Placeholders only.

The following transfer methods are interface stubs and raise `NotImplementedError`:
- MMD (Maximum Mean Discrepancy)
- CORAL (Correlation Alignment)
- DANN (Domain-Adversarial Neural Network)
- Pseudo-label self-training
- Source-free adaptation

These will be implemented in future iterations.

---

## 9. Dataset adapters

**Status:** Placeholders only.

The following dataset adapters raise `NotImplementedError`:
- `PanasonicDataset`
- `LGDataset`
- `Samsung48GDataset`
- `Samsung30TDataset`
- `CMUeVTOLDataset`

Only `SyntheticDataset` is fully functional for the current smoke test.
# Vidal Case 1

- The paper does not disclose a reproducible fixed sequence length; 50 samples
  is a reconstruction assumption and must be recorded in every run.
- The exact normalisation formula and shuffle behaviour are not disclosed;
  per-battery train-only min-max scaling and configurable shuffle are used.
- PyTorch differs from MATLAB R2018b. The second PyTorch LSTM bias is fixed at
  zero to reproduce the reported 571 trainable parameters.
- Samsung 48G is missing, so the current reproduction covers three batteries.
- SOC v2 is a cycler-derived benchmark reference label, not independent truth.
