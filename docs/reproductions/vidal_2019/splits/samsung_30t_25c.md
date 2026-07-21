# Samsung INR21700-30T — 25 °C Split Report
## Vidal et al. (2019 ITEC) Reproduction

**Status:** PROVISIONAL  
**Date:** 2026-07-13  
**Manifest:** `data/manifests/vidal_2019/samsung_30t_vidal2019_25c_manifest.csv`  
**Script:** `scripts/inspect_samsung_30t_25c_split.py`

---

## 1. Scope

This report documents the paper-faithful file-level split reconstruction for the
Samsung INR21700-30T dataset at 25 °C, for use in reproducing Vidal et al. (2019
ITEC), "Li-ion Battery State of Charge Estimation Using Long Short-Term Memory
Recurrent Neural Network with Transfer Learning."

Temperature in scope: **25 °C only** (the Vidal 2019 experiments used a thermal
chamber set to 25 °C; multi-temperature data are present in the public dataset
but are not used for this reproduction).

This task covers:
- Identity verification of the public Samsung 30T dataset against the paper's
  "Samsung T30"
- Raw-file inspection (fields, QC, SHA-256, signal fingerprints)
- 25 °C timeline reconstruction
- File-level split assignment (full_train / partial_train / test / exclude)
- Charge chronology and assignment
- Manifest generation and validation

This task does **not** cover: SOC calculation, feature normalisation, LSTM sequence
construction, model training, or adapter implementation.

---

## 2. Samsung T30 versus Samsung 30T — Identity Audit

### 2.1 Public Dataset

| Attribute | Value |
|-----------|-------|
| Public name (filenames) | `IN21700_30T` (e.g., `734_UDDS_25degC_IN21700_30T.mat`) |
| README description | "Samsung 3.0T cell" |
| Format | 21700 cylindrical |
| Nominal capacity | 3 Ah |
| Test institution | McMaster University, Hamilton, Ontario, Canada |
| Experimenter | Dr. Phillip Kollmeyer (`phillip.kollmeyer@gmail.com`) |
| Cycler | Digatron Firing Circuits Universal Battery Tester, 75 A / 5 V channel |
| 25 °C session dates | 2019-03-14 (HPPC) through 2019-03-21 (PausCycl) |
| Mendeley publication | December 2020, DOI: 10.17632/9xyvy2njj3.1 |

### 2.2 Vidal 2019 Paper

The paper (Vidal et al., 2019 ITEC) refers to the battery as **"Samsung T30"**
with a nominal capacity of approximately 3 Ah.  The paper does not reproduce
the Mendeley dataset identifier or the cycler log details.

### 2.3 Evidence Assessment

| Evidence item | Status |
|---------------|--------|
| Cell format (21700) | **Confirmed** from filenames (`IN21700_30T`) |
| Nominal capacity (3 Ah) | **Confirmed** from README ("3Ah Samsung 3.0T cell") |
| Test profiles (Mixed1–8, UDDS, HWFET, LA92, US06) | **Confirmed** from raw directory |
| Temperature (25 °C) | **Confirmed** from subdirectory name and file metadata |
| Institution (McMaster, Dr. Kollmeyer) | **Confirmed** from README |
| Data collection date (2019) | **Confirmed** from file timestamps (March 2019) |
| Chronological compatibility with Vidal 2019 paper | **Confirmed** — data from March 2019 predates the 2019 ITEC conference |
| "T30" vs. "30T" name equivalence | **Inferred** — "T30" is most plausibly a transposition of "30T"; the README calls the cell "3.0T"; no author-provided cross-reference exists |

### 2.4 Naming Analysis

- **Samsung naming convention for high-drain cylindrical cells**: The INR21700-30T
  designation follows Samsung's convention: IN = INR chemistry,
  R = round/cylindrical, 21700 = cell size, 30 = 3000 mAh, T = high-drain
  variant.
- **"3.0T"** in the README represents the same cell written as "3.0 Ah T-series."
- **"T30"** in the Vidal paper is most likely a transposition of "30T" (digits
  and letter inverted).  No evidence suggests "T30" is a different cell.

### 2.5 Identity Verdict

**STRONGLY SUPPORTED**

The public Samsung INR21700-30T dataset corresponds to the "Samsung T30" battery
in Vidal et al. (2019) with high probability.  All verifiable attributes (format,
capacity, profiles, temperature, institution, collection date) align.  The name
discrepancy ("T30" vs. "30T") is a plausible transposition with no contradicting
evidence.

Not CONFIRMED because the authors do not provide an explicit written cross-
reference between "Samsung T30" (paper) and "Samsung INR21700-30T" (public
dataset).

---

## 3. Exact Vidal 2019 Protocol for the Samsung Battery

Source: Paper §IV, Table I (Vidal et al., 2019 ITEC).

Samsung T30 serves as the **target domain** in **Case 4**:
- Source domain: Panasonic 18650PF (25 °C)
- Target domain: Samsung T30 (25 °C)
- Transfer learning objective: train with limited Samsung T30 data, evaluate on
  Samsung T30 test profiles.

| Split | Profiles | Notes |
|-------|---------|-------|
| full_train | Mix1–Mix8 + associated CC charges | Upper-bound / full-target experiment |
| partial_train | Mix1–Mix3 + associated CC charges | Transfer-learning partial target data; paper states "cycles MIX 1-3 … for Cases 3 and 4" (Samsung 48G is the exception using Mix 1, 2, 6) |
| test | UDDS, LA92, US06 + associated CC charges | Held-out evaluation |
| exclude | HWFET, HPPC, C20DisCh, Dis_0p5C, Dis_2C, Cap_1C, PausCycl | Not listed as training or test profiles |

**Note on HWFET:** The Samsung 30T HWFET 25 °C file (`734_HWFET_25degC_IN21700_30T.mat`)
is a **valid drive cycle** (2600 s, I_min = −18.4 A, I_max = +12.0 A), unlike the
LG HG2 HWFET file which was a rest-only file.  HWFET is excluded here solely
because Table I does not list it as a selected test profile for Samsung T30.

---

## 4. Raw Dataset Metadata and Chronology

The Samsung INR21700-30T dataset at 25 °C was collected in a single continuous
testing session spanning 2019-03-14 to 2019-03-21.

The session sequence, as inferred from file timestamps:

```
2019-03-14  Characterisation: HPPC
2019-03-16  Characterisation: C20DisCh
2019-03-17  Characterisation: Dis_0p5C
2019-03-18  Characterisation: Dis_2C
2019-03-18  Charge1 (tiny top-off, 0.022 Ah) → Cap_1C (first, 1C discharge)
2019-03-18  Charge2 (full, 2.937 Ah) → UDDS → Charge3
            HWFET → Charge4 → LA92 → Charge5 → US06 → Charge6
2019-03-19  Mixed1 → Charge7 → Mixed2 → Charge8
            [~35-hour pause]
2019-03-20  Charge9 (tiny top-off, 0.050 Ah) → [Cap_1C second, no .mat saved]
            → Charge10 (full, 2.917 Ah) → Mixed3 → Charge11
            Mixed4 → Charge12
2019-03-21  Mixed5 → Charge13 → Mixed6 → Charge14
            Mixed7 → Charge15 → Mixed8 → Charge16
            PausCycl (last file; no trailing charge)
```

---

## 5. Raw Directory Structure

```
data/raw/Samsung_30T/
├── Readme IN21700-30T.txt
├── Technical Information and Experimental Test Results for Samsung IN21700-30T.pdf
├── IN21700-30T HPPC OCV CC_DIS Data.xlsx
├── 25degC/          ← CANONICAL 25 °C DIRECTORY
│   ├── 34 .mat files (selected for manifest)
│   └── 11 .csv files (CSV copies of same cycler output; not in manifest)
├── 40degC/
├── 10degC/
├── 0degC/
├── -10degC/
└── -20degC/
```

The CSV files present alongside .mat files in each temperature directory are CSV
copies of the same cycler output, noted in the README.  They are excluded from
the manifest (`.mat` files are canonical).

---

## 6. Raw File Inventory (25 °C .mat files)

| Filename | n_rows | Duration (s) | Profile | Type | Split |
|----------|--------|-------------|---------|------|-------|
| 03-14-19_17.34 729_HPPC_25degC | 55 457 | 123 890 | HPPC | characterization | exclude |
| 03-16-19_04.00 729_C20DisCh_25degC | 2 590 | 155 123 | C20DisCh | characterization | exclude |
| 03-17-19_23.07 729_Dis_0p5C_25degC | 227 | 13 123 | Dis_0p5C | characterization | exclude |
| 03-18-19_02.46 729_Dis_2C_25degC | 301 | 17 110 | Dis_2C | characterization | exclude |
| 03-18-19_13.03 734_Charge1_25degC | 41 | 2 039 | Charge1 | charge | exclude |
| 03-18-19_13.37 734_Cap_1C_25degC | 423 | 4 184 | Cap_1C | characterization | exclude |
| 03-18-19_14.47 734_Charge2_25degC | 107 | 5 963 | Charge2 | charge | test |
| 03-18-19_16.26 734_UDDS_25degC | 65 656 | 6 565 | UDDS | drive_cycle | test |
| 03-18-19_18.16 734_Charge3_25degC | 104 | 5 825 | Charge3 | charge | test |
| 03-18-19_19.53 734_HWFET_25degC | 26 000 | 2 600 | HWFET | drive_cycle | exclude |
| 03-18-19_20.36 734_Charge4_25degC | 104 | 5 822 | Charge4 | charge | test |
| 03-18-19_22.13 734_LA92_25degC | 43 292 | 4 333 | LA92 | drive_cycle | test |
| 03-18-19_23.25 734_Charge5_25degC | 104 | 5 839 | Charge5 | charge | test |
| 03-19-19_01.03 734_US06_25degC | 19 910 | 1 991 | US06 | drive_cycle | test |
| 03-19-19_01.36 734_Charge6_25degC | 104 | 5 824 | Charge6 | charge | test |
| 03-19-19_03.13 734_Mixed1_25degC | 32 402 | 3 240 | Mixed1 | drive_cycle | full+partial |
| 03-19-19_04.07 734_Charge7_25degC | 104 | 5 836 | Charge7 | charge | full+partial |
| 03-19-19_05.44 734_Mixed2_25degC | 36 365 | 3 636 | Mixed2 | drive_cycle | full+partial |
| 03-19-19_06.45 734_Charge8_25degC | 105 | 5 833 | Charge8 | charge | full+partial |
| 03-20-19_17.52 740_Charge9_25degC | 44 | 2 148 | Charge9 | charge | exclude |
| 03-20-19_19.37 740_Charge10_25degC | 107 | 5 968 | Charge10 | charge | full+partial |
| 03-20-19_21.16 740_Mixed3_25degC | 26 200 | 2 620 | Mixed3 | drive_cycle | full+partial |
| 03-20-19_22.00 740_Charge11_25degC | 104 | 5 804 | Charge11 | charge | full+partial |
| 03-20-19_23.37 740_Mixed4_25degC | 37 004 | 3 700 | Mixed4 | drive_cycle | full only |
| 03-21-19_00.39 740_Charge12_25degC | 103 | 5 820 | Charge12 | charge | full only |
| 03-21-19_02.16 740_Mixed5_25degC | 26 656 | 2 665 | Mixed5 | drive_cycle | full only |
| 03-21-19_03.00 740_Charge13_25degC | 103 | 5 818 | Charge13 | charge | full only |
| 03-21-19_04.37 740_Mixed6_25degC | 34 783 | 3 478 | Mixed6 | drive_cycle | full only |
| 03-21-19_05.35 740_Charge14_25degC | 104 | 5 830 | Charge14 | charge | full only |
| 03-21-19_07.12 740_Mixed7_25degC | 29 561 | 2 956 | Mixed7 | drive_cycle | full only |
| 03-21-19_08.01 740_Charge15_25degC | 104 | 5 830 | Charge15 | charge | full only |
| 03-21-19_09.39 740_Mixed8_25degC | 42 168 | 4 217 | Mixed8 | drive_cycle | full only |
| 03-21-19_10.49 740_Charge16_25degC | 104 | 5 837 | Charge16 | charge | full only |
| 03-21-19_12.26 740_PausCycl_25degC | 14 757 | 1 476 | PausCycl | other | exclude |

**Total: 34 .mat files**

---

## 7. Public Raw-Name to Paper-Profile Mapping

| Public filename pattern | `paper_profile` in manifest | Mapping confidence |
|------------------------|---------------------------|-------------------|
| `…_Mixed1_…` | `Mix1` | High |
| `…_Mixed2_…` | `Mix2` | High |
| `…_Mixed3_…` | `Mix3` | High |
| `…_Mixed4_…` | `Mix4` | High |
| `…_Mixed5_…` | `Mix5` | High |
| `…_Mixed6_…` | `Mix6` | High |
| `…_Mixed7_…` | `Mix7` | High |
| `…_Mixed8_…` | `Mix8` | High |
| `…_UDDS_…` | `UDDS` | High |
| `…_LA92_…` | `LA92` | High |
| `…_US06_…` | `US06` | High |
| `…_HWFET_…` | `` (blank; excluded by protocol) | High (excluded) |
| `…_Charge{2–8,10–16}_…` (selected) | `CC` | Inferred from chronology |
| `…_Charge{1,9}_…` (excluded) | `` (blank) | Excluded; not a training input |
| `…_HPPC_…`, `…_C20DisCh_…`, etc. | `` (blank) | Excluded characterisation |

The Samsung 30T public dataset uses the **"Mixed"** naming convention (not "Mix"
or "Cycle_"), identical to LG HG2.  No renaming is required.

---

## 8. Full-Training File List

17 files (8 drive cycles + 9 charge files):

**Drive cycles:**
- Mixed1, Mixed2, Mixed3, Mixed4, Mixed5, Mixed6, Mixed7, Mixed8

**Charge files:**
- Charge7 (post-Mixed1), Charge8 (post-Mixed2), Charge10 (pre-Mixed3 / post-missing Cap_1C),
  Charge11 (post-Mixed3), Charge12 (post-Mixed4), Charge13 (post-Mixed5),
  Charge14 (post-Mixed6), Charge15 (post-Mixed7), Charge16 (post-Mixed8)

---

## 9. Partial-Training File List

7 files (3 drive cycles + 4 charge files), strict subset of full_train:

**Drive cycles:**
- Mixed1, Mixed2, Mixed3

**Charge files:**
- Charge7 (post-Mixed1), Charge8 (post-Mixed2), Charge10 (pre-Mixed3),
  Charge11 (post-Mixed3)

**Repository semantics note:** In this repository, `partial_train = 1` flags
files belonging to the limited target-domain subset used in transfer learning
(Cases 3/4 in Vidal 2019).  All `partial_train = 1` files also have
`full_train = 1`, reflecting that the partial subset is a strict subset of the
complete available training set.

---

## 10. Test File List

8 files (3 drive cycles + 5 charge files):

**Drive cycles:**
- UDDS, LA92, US06

**Charge files:**
- Charge2 (pre-UDDS), Charge3 (post-UDDS), Charge4 (post-HWFET/pre-LA92),
  Charge5 (post-LA92), Charge6 (post-US06)

**Note on Charge6:** Charge6 falls chronologically between US06 (test) and
Mixed1 (train).  Under the preceding-cycle-wins rule, Charge6 is assigned to
test (US06 precedes).  This means Mixed1 has no preceding charge file in the
training split.  Mixed1's SOC initialisation uses `soc_anchor_source =
"full_charge_protocol"` — the starting voltage (~4.185 V after Charge6) confirms
a fully charged state regardless of which split Charge6 belongs to.

---

## 11. Excluded File List

9 files:

| File | Reason |
|------|--------|
| HPPC | Characterisation test |
| C20DisCh | Characterisation test |
| Dis_0p5C | Characterisation test |
| Dis_2C | Characterisation test |
| Cap_1C | First 1C discharge test; characterisation |
| Charge1 | Tiny top-off (0.022 Ah) before first Cap_1C; not a training input |
| Charge9 | Tiny top-off (0.050 Ah) before missing second Cap_1C; not a training input |
| HWFET | Valid drive cycle but not listed as test/train in Table I |
| PausCycl | Real drive excitation but not a Vidal paper profile |

---

## 12. Charge Chronology and Assignment

The charge assignment rule is: each charge file is assigned to the split of the
drive cycle that **immediately precedes the charge** (preceding-cycle-wins),
with a **following-cycle fallback** when no drive cycle directly precedes the
charge (e.g. the preceding segment is excluded or absent).

```
Charge1  → [tiny top-off] → exclude
  precedes: Cap_1C (characterisation)
  
Charge2  → test
  precedes: UDDS
  following-cycle fallback: Cap_1C excluded; UDDS is next

Charge3  → test
  precedes: UDDS (Charge3 immediately follows UDDS)

Charge4  → test
  precedes: HWFET (excluded); following-cycle fallback: LA92 is next

Charge5  → test
  precedes: LA92

Charge6  → test
  precedes: US06 [preceding-cycle-wins; next is Mixed1 (train)]
  SOC anchor for Mixed1: full_charge_protocol (no cross-split dependency)

Charge7  → full_train + partial_train
  precedes: Mixed1

Charge8  → full_train + partial_train
  precedes: Mixed2

Charge9  → [tiny top-off 0.050 Ah] → exclude
  precedes: missing Cap_1C (no .mat file saved at 25°C)

Charge10 → full_train + partial_train
  precedes: Mixed3 (following-cycle fallback; missing Cap_1C excluded)

Charge11 → full_train + partial_train
  precedes: Mixed3

Charge12 → full_train
  precedes: Mixed4

Charge13 → full_train
  precedes: Mixed5

Charge14 → full_train
  precedes: Mixed6

Charge15 → full_train
  precedes: Mixed7

Charge16 → full_train
  precedes: Mixed8 [next is PausCycl (excluded); no trailing charge after PausCycl]
```

All charge pairings are resolved or inferred under the documented
reconstruction rule.  Charge9 and Charge10 carry
`pairing_status = "inferred under README + chronology evidence"` because
their connection to the missing Cap_1C test relies on README protocol
statements and the observed gap structure, not direct file inspection.
No charge pairings remain unresolved.

---

## 13. Capacity-Test Interruptions and SOC-Anchor Handling

### 13.1 First Cap_1C (25 °C session)

The first capacity test (`734_Cap_1C_25degC_IN21700_30T.mat`, 2019-03-18 13:37)
follows a tiny top-off charge (Charge1, 0.022 Ah) and discharges the cell to
3.148 V (Ah_end = −2.945 Ah = ~2.945 Ah capacity at 25 °C).  Charge2 then
fully recharges the cell before UDDS.

### 13.2 Missing Second Cap_1C (25 °C session)

The README documents a second 1C discharge test performed before Mixed3.  At
25 °C no Cap_1C .mat file is present between Charge8 (2019-03-19 06:45) and
Charge9 (2019-03-20 17:52).  Evidence for the missing test:
- Charge9 starts ~35 hours after Charge8 (long pause, consistent with setup)
- Charge9 is a tiny top-off (0.050 Ah at 0.922 A, ~36 min) — typical pre-test
  preparation
- Charge10 (2019-03-20 19:37) is a full recharge (2.917 Ah at 2.999 A, ~100 min)
  consistent with restoring a fully discharged cell
- Time gap between Charge9 end (~18:27) and Charge10 start (19:37) = ~70 min,
  consistent with a 1C discharge of 3 Ah (expected ~60 min plus setup)
- The Charge9 → Charge10 two-charge pattern matches the observed Cap_1C pattern
  at 0 °C and other temperatures where a `Cap_1C.mat` file is present

**Conclusion:** A second Cap_1C test occurred at 25 °C between 2019-03-19 and
2019-03-20.  The corresponding .mat file was not included in the public dataset
release.  This is an unresolved data gap (see Section 20).

### 13.3 SOC Initialisation

For all selected drive cycles, `soc_anchor_source = "full_charge_protocol"` or
`"preceding_full_charge"`.  Starting voltages near 4.185 V confirm fully charged
states.  `soc_init = 1.0` is a documented protocol assumption for all selected
drive cycles, consistent with the paper's full-charge-before-each-cycle protocol.

---

## 14. Duplicate/Combined-File Findings

### SHA-256 exact duplicates
None detected.  All 34 `.mat` files in the 25 °C directory have unique SHA-256
hashes.

### Signal-fingerprint duplicates
None detected.  All signal fingerprints (n_rows|duration|V₀|V₋₁|I₀|I₋₁|Ah₀|Ah₋₁|∑I₈|∑Ah₈)
are unique across the 25 °C manifest.

### CSV copies
Each temperature directory contains CSV copies of the cycler output alongside
the `.mat` files.  At 25 °C there are 11 CSV files.  The README confirms these
are generated from the same cycler output as the `.mat` files.  CSV files are
not included in the manifest (`.mat` is canonical).

### Combined files
No evidence of a long combined multi-cycle file in the 25 °C directory.  Each
file corresponds to a single test segment.

---

## 15. Data-Quality Findings

| Finding | Severity | Files | Impact |
|---------|---------|-------|--------|
| Duplicate final timestamp (dt=0) | Low | 29/34 files | Cycler artifact; last sample repeated; no impact on training |
| Missing second Cap_1C .mat | Medium | — | Charge9/Charge10 reconstruction required; SOC anchor for Mixed3 inferred |
| HPPC has 317 duplicate timestamp pairs | Low | HPPC only | Excluded characterisation file; no split impact |
| Ah counter reset at end for Dis_2C | Low | Dis_2C only | Excluded file |

**Overall assessment:** All mandatory QC checks pass (no QC FAIL on any selected
file).  29 of 34 files have a `duplicate_timestamp(dt=0)` warning — a common
cycler artifact where the Digatron cycler logs the final sample twice.  This is
not a data-quality failure; the adapter should deduplicate on load (e.g.,
`keep='first'` on the Time column).

---

## 16. Leakage Checks

Checks confirmed by `scripts/inspect_samsung_30t_25c_split.py`:

| Check | Result |
|-------|--------|
| No full_train/test overlap | PASS |
| No partial_train/test overlap | PASS |
| No excluded file has a selected flag | PASS |
| partial_train is a strict subset of full_train | PASS |
| No SHA-256 exact duplicate crosses train/test | PASS |
| No signal-fingerprint duplicate crosses train/test | PASS |
| No charge file assigned to test is used as training input | PASS |

**Charge6 note:** Charge6 is assigned to test (US06 precedes).  Mixed1's SOC
anchor is `full_charge_protocol` (independent of Charge6's split membership),
so no cross-split dependency exists.

---

## 17. Paper-Confirmed Facts

The following are directly stated or clearly implied in Vidal et al. (2019 ITEC):

1. Samsung T30 is used as a target domain battery.
2. The target battery nominal capacity is approximately 3 Ah.
3. The full training set includes Mix1–Mix8 drive profiles.
4. The partial training set includes Mix1–Mix3 (for Cases 3/4; Samsung 48G uses
   Mix 1, 2, 6 instead).
5. Test profiles are UDDS, LA92, and US06.
6. CC charge segments accompany drive cycles.
7. The experiment was conducted at 25 °C.

---

## 18. Dataset-Confirmed Facts

The following are directly confirmed from the public dataset (README + raw files):

1. 34 `.mat` files at 25 °C.
2. Naming convention: `Mixed1`–`Mixed8` (not "Mix" or "Cycle_").
3. Test date range: 2019-03-14 to 2019-03-21.
4. Data columns: `Time`, `TimeStamp`, `Voltage`, `Current`, `Ah`, `Wh`, `Power`,
   `Battery_Temp_degC`.
5. Drive cycles sampled at 0.1 s; charges/pauses at lower rates (README).
6. Drive cycles show I_max ≈ +12 A (regen) and I_min down to −36 A (12C discharge).
7. HWFET at 25 °C is a real drive cycle (2600 s, active regen current).
8. Two tiny top-off charges (Charge1: 0.022 Ah; Charge9: 0.050 Ah) flank the
   two Cap_1C capacity tests.
9. First Cap_1C .mat file is present (723_Cap_1C); second is absent from the
   public release.
10. Ah counter typically reset after each segment (README: "Ah counter typically
    reset after each charge, test, or drive cycle").
11. All charge files show Ah_end = 0.0 (Ah counter reset at completion).
12. CSV copies of all cycler data are co-located with `.mat` files; content is
    identical (README).
13. Repeated-run manifest SHA-256:
    `45e419237f283e7f54cb551ae98acc1c05a1dceac2258a80e913f5760a01da1e`

---

## 19. Reconstruction Assumptions

| # | Assumption | Basis |
|---|-----------|-------|
| A1 | "Samsung T30" (paper) = Samsung INR21700-30T (public) | Capacity, format, profiles, institution, collection date all match; "T30" is inferred transposition of "30T" |
| A2 | Mixed1–8 (public) = Mix1–8 (paper) | Direct name correspondence; no alternative names exist in the directory |
| A3 | partial_train = Mix1–3 for Samsung T30 | Paper text: "cycles MIX 1-3 … for Cases 3 and 4"; Samsung 48G is the explicit exception |
| A4 | Charge-to-drive-cycle pairing by file chronology | README confirms filename timestamps sort by test start time |
| A5 | soc_init = 1.0 for all selected drive cycles | Protocol: full-charge before every cycle; starting voltages ≈ 4.185 V confirm full state |
| A6 | Charge6 assigned to test (US06 precedes) | Preceding-cycle-wins rule; Mixed1 SOC anchor independent |
| A7 | Second Cap_1C test occurred but no .mat file saved | Inferred from Charge9/Charge10 gap structure and README protocol |
| A8 | Charge9 is a pre-Cap_1C top-off (not a training input) | Tiny Ah (0.050 Ah at 0.922 A); analogous to Charge1 structure |

---

## 20. Unresolved Questions

### 20a. Battery Identity — "T30" vs. "30T" — INFERRED
**Status:** STRONGLY SUPPORTED but not CONFIRMED.  
**Reason:** No author-provided explicit statement in the public dataset documentation
cross-referencing "Samsung T30" (paper) to "Samsung INR21700-30T" (public).  
**Impact:** Does not prevent benchmark use; keep label `STRONGLY_SUPPORTED` in
manifest.

### 20b. Missing Second Cap_1C .mat at 25 °C — UNRESOLVED
**Status:** File absent from public dataset release.  
**Evidence:** Gap structure (Charge9 tiny top-off → ~70 min gap → Charge10 full
recharge) and README protocol are consistent with a Cap_1C test that was not
saved as a .mat file.  
**Impact:** Does not affect the split; Mixed3 uses Charge10 as its pre-cycle
charge.  SOC anchor for Mixed3 = `preceding_full_charge` (Charge10).

### 20c. Partial-Train Profile Selection for Samsung T30 — INFERRED
**Status:** Mix1–3 inferred from paper text ("cycles MIX 1-3 … for Cases 3 and 4").  
**Reason:** The paper does not provide a Samsung T30-specific table distinguishing
it from the general Case 4 description.  
**Impact:** If the correct partial subset differs (e.g., Mix 1, 2, 6 as for
Samsung 48G), the `partial_train` flag assignments need updating.

### 20d. Charge6 Boundary Assignment — PROVISIONAL RULE
**Status:** Preceding-cycle-wins rule assigns Charge6 to test.  
**Reason:** The paper does not specify exact charge-file boundaries.  An
alternative interpretation would assign Charge6 to train (following Mixed1).  
**Impact:** Under the current rule, no leakage; Mixed1 SOC initialised at 1.0
independently.

### 20e. PausCycl Role — PROVISIONAL EXCLUDE
**Status:** Excluded.  
**Reason:** PausCycl at Samsung 30T 25 °C is a genuine drive excitation (not a
rest file: I_min = −18.87 A), but it is not a named paper profile.  
**Impact:** Excluded from all splits; if future analysis identifies it as a
partial-cycle run of a known profile, the manifest should be updated.

---

## 21. Differences Between Paper and Public Dataset

| Difference | Details |
|-----------|---------|
| Battery name | Paper: "Samsung T30" / Public: "Samsung INR21700-30T" (README: "3.0T") |
| HWFET validity | LG HG2 HWFET was a rest file; Samsung 30T HWFET is a valid drive cycle — but still excluded per Table I |
| Missing Cap_1C .mat | Second Cap_1C test occurred but no .mat file was included in the public dataset at 25 °C |
| Ah counter convention | README confirms Ah reset after each segment; drive-cycle Ah values are negative (discharge) starting from 0 |
| Duplicate final timestamp | Cycler artifact in 29/34 files; not mentioned in paper |
| Dataset publication date | Data collected March 2019; publicly released December 2020 |

---

## 22. Final Identity Status

**STRONGLY SUPPORTED**

The public Samsung INR21700-30T dataset is, with high probability, the dataset
used as "Samsung T30" in Vidal et al. (2019 ITEC).  All verifiable attributes
align.  The name discrepancy is a plausible transposition.  No contradicting
evidence was found.

This dataset may be used as a public Samsung 30T benchmark.  If exact paper-
faithful reproduction is required, the identity uncertainty should be
acknowledged in results reporting.

---

## 23. Final Split Status

**PROVISIONAL**

Reasons:
1. Battery identity is STRONGLY SUPPORTED, not CONFIRMED (§22).
2. Partial-train profile selection (Mix1–3) is inferred from paper text, not
   from a Samsung-specific table (§20c).
3. Missing second Cap_1C .mat at 25 °C is a data gap that required chronological
   reconstruction for Charge9/Charge10 assignment (§20b).

The split will remain PROVISIONAL until:
- An explicit author confirmation of Samsung T30 = Samsung INR21700-30T is found, and
- The partial-train profile selection is confirmed by a Samsung-specific paper source.

---

*Generated by `scripts/inspect_samsung_30t_25c_split.py`.  
Manifest SHA-256: `45e419237f283e7f54cb551ae98acc1c05a1dceac2258a80e913f5760a01da1e`.*
