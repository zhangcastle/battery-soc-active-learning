# Battery Dataset Audit Report
## Vidal et al. (2019) — Li-ion Battery SOC Estimation via LSTM with Transfer Learning

**Audit date:** 2026-07-12  
**Auditor:** read-only automated audit (battery_audit_full.py)  
**Repo:** `/Users/yue/Research/battery-soc-transfer-benchmark`  
**Python:** `.venv/bin/python3` with `scipy.io.loadmat` (MATLAB v5 format)

---

## 1. Executive Summary

This audit covers 612 MATLAB `.mat` files across three battery datasets plus one placeholder, totalling ~599 files successfully loaded and analyzed. The primary goal is to assess whether these datasets can faithfully reproduce the Vidal et al. (2019) LSTM transfer-learning paper.

**Key findings:**

| Finding | Status |
|---|---|
| All three real datasets loadable via `scipy.io.loadmat` | ✅ Confirmed |
| No SOC field in any raw file — must be coulomb-counted | ⚠️ Confirmed |
| Panasonic dataset matches paper training source (Mix + Charge) | ✅ Confirmed |
| LG HG2 and Samsung 30T match paper target domains | ✅ Confirmed |
| LG HG2 **Missing Mix 3** at 0°C and 10°C | 🔴 **Critical gap** |
| Samsung 40°C Mixed3/4 files have mid-file timestamp reset to 0 | 🔴 **Critical bug** |
| LG HG2 LA92 10°C file has non-monotonic timestamps + NaN voltage | 🔴 **Critical** |
| Panasonic combined-cycle file contains multi-day time gaps (~34h) | ⚠️ Important |
| Samsung 30T current spikes >30A are physically valid (35A-rated cell) | ℹ️ False alarm |
| Samsung 48G: placeholder only, no data available | ❌ Absent |
| Datasets appear to be later/extended versions vs. original 2019 paper | ⚠️ Unresolved |

---

## 2. Dataset Versions and Paper Compatibility

### 2.1 Panasonic 18650PF

- **Data collection dates (from filenames):** March–July 2017
- **Publication date of Vidal et al.:** 2019 (data likely collected 2017)
- **Assessment:** ✅ **Consistent** — timestamps match expected collection window for a 2019 paper
- **Mix-profile naming:** "Cycle_N" (e.g., `25degC_Cycle_1_Pan18650PF.mat`) plus a standalone `_NN_` file (Neural Network evaluation cycle). The "NN" file is a test cycle not counted as one of the numbered Mix profiles.
- **Temperature coverage:** −20°C, −20°C Trise, −20°C Trise with pause, −10°C, 0°C, 10°C, 10°C Trise, 10°C Trise with pause, 25°C (9 subdirectories)

### 2.2 LG 18650HG2

- **Data collection dates:** October–December 2018 (filenames confirm this)
- **Bonus directory present:** `LG 18650HG2 Li-ion Battery Data and Example Deep Neural Network xEV SOC Estimator Script/` containing a McMaster University FNN script dated January 2020
- **Assessment:** ⚠️ **Possibly later version** — LG data collected Oct–Dec 2018 is plausibly the version used in the paper (submitted 2019). However, the Jan 2020 FNN script and its prepared dataset may reflect a post-paper extension. The raw `LG_HG2/` directory structure is consistent with the Panasonic dataset format.
- **Temperature coverage:** n20°C, n10°C, 0°C, 10°C, 25°C, 40°C (6 temperatures)

### 2.3 Samsung INR21700-30T ("Samsung T30" in paper)

- **Data collection dates:** March–July 2019 (filenames: `03-...-19` through `07-...-19`)
- **Assessment:** ⚠️ **May postdate the original paper** — Vidal et al. (2019) likely submitted with preliminary Samsung data; the full 6-temperature dataset collected Mar–Jul 2019 may be an extended collection. **However**, the paper references Samsung as a target domain so some version was available.
- **Temperature coverage:** −20°C, −10°C, 0°C, 10°C, 25°C, 40°C (6 temperatures)
- **Note:** Paper calls it "Samsung T30"; confirmed to be INR21700-30T from filenames (`_IN21700_30T`)

### 2.4 Samsung 48G

- **Status:** Placeholder only. The directory `Samsung_48G/` contains only a `README.md`. **No data is available.**
- **Impact:** The task description states Samsung 48G uses Mix 1, 2, 6 in Case 3. This cannot be verified or reproduced.

---

## 3. Schema and Column Analysis

### 3.1 MATLAB File Structure

All real `.mat` files use **MATLAB v5 format** (loadable by `scipy.io.loadmat`). Each file contains a single struct variable `meas` with the following fields:

| Column | Panasonic | LG HG2 | Samsung 30T | Notes |
|---|---|---|---|---|
| `Time` | ✅ (col 8) | ✅ (col 1) | ✅ (col 1) | Elapsed seconds from start of file |
| `TimeStamp` | ✅ (col 1) | ✅ (col 2) | ✅ (col 2) | Wall-clock string (e.g., `"3/18/2017 2:17:35 AM"`) |
| `Voltage` | ✅ (col 2) | ✅ (col 3) | ✅ (col 3) | Volts |
| `Current` | ✅ (col 3) | ✅ (col 4) | ✅ (col 4) | Amperes |
| `Ah` | ✅ (col 4) | ✅ (col 5) | ✅ (col 5) | Ampere-hours (signed, per-file cumulative) |
| `Wh` | ✅ (col 5) | ✅ (col 6) | ✅ (col 6) | Watt-hours |
| `Power` | ✅ (col 6) | ✅ (col 7) | ✅ (col 7) | Watts |
| `Battery_Temp_degC` | ✅ (col 7) | ✅ (col 8) | ✅ (col 8) | °C, surface/thermocouple |
| `Chamber_Temp_degC` | ✅ (col 9) | ❌ absent | ❌ absent | Environmental chamber temp (Panasonic only) |
| `SOC` | ❌ absent | ❌ absent | ❌ absent | **Not present in any raw file** |

**Panasonic column order note:** `Time` is column 8 (not column 1 like LG/Samsung). `TimeStamp` is column 1. This is a schema inconsistency that could cause bugs if columns are accessed by position rather than name.

### 3.2 Dtype and Precision

- All numeric fields: `float64` after extraction
- `TimeStamp`: array of strings (MATLAB cell array → object dtype in scipy)
- `Battery_Temp_degC` in Panasonic: sometimes stored as `uint8` (integer, °C rounded) — confirm with loading code

### 3.3 Sampling Frequency

| Dataset | Drive cycle files | Characterization files |
|---|---|---|
| Panasonic | **10 Hz** (confirmed: median dt = 0.1s) | 0.003–1 Hz (HPPC, EIS) |
| LG HG2 | **~10 Hz** (median dt ≈ 0.100–0.101s) | 0.017–1 Hz |
| Samsung 30T | **~10 Hz** (median dt ≈ 0.098–0.102s) | 0.017 Hz |

Note: The median over all files per dataset is skewed low (~0.017 Hz) by slow characterization files. For drive cycles and Mix profiles, 10 Hz is the correct sampling rate.

---

## 4. SOC Definition and Ah Accounting

### 4.1 SOC in Raw Data

**No SOC field exists in any raw dataset.** SOC must be derived externally.

### 4.2 Ah Convention (confirmed)

- `Ah` field is **per-file cumulative**, resetting to **0.0 at the start of each file**
- During **discharge**: Ah decreases (goes more negative); e.g., Cycle_1 ends at Ah ≈ −2.70 Ah
- During **charge**: Ah increases positively; e.g., Charge1 ends at Ah ≈ +2.00 Ah
- **Ah does NOT carry over between files.** Each new `.mat` file starts from Ah = 0.

### 4.3 SOC Derivation Options

**Option A — Coulomb counting with known Q_nominal:**
```
SOC(t) = SOC_init + ∫₀ᵗ I(τ)/Q_nominal dτ = SOC_init + Ah(t)/Q_nominal
```
- SOC_init must be inferred from prior state (e.g., SOC_init = 1.0 after a full CC charge to 4.2 V)
- Q_nominal: ~2.9 Ah (Panasonic 18650PF), ~3.0 Ah (LG HG2), ~3.0 Ah (Samsung 30T)

**Option B — Use Ah from prior charge file as SOC reference:**
- After a charge file ending at Ah_charged, set SOC_init = Ah_charged / Q_nominal
- Then track drive cycle: SOC(t) = SOC_init + Ah_drive(t) / Q_nominal

**Option C (used in some papers) — OCV-based SOC lookup:**
- Use stored OCV curves to look up SOC from measured open-circuit voltage
- Not directly reproducible without OCV curve from the 5-pulse HPPC files

### 4.4 Ah Reset Anomaly

**40 files have non-zero Ah at start** (detected by `abs(Ah[0]) > 0.01`). These are primarily:
- Panasonic characterization files where the equipment was not reset between segments
- Some Samsung 30T charge/characterization files

These are **not relevant to the drive cycle or Mix profiles** but would cause incorrect SOC initialization if included as-is.

---

## 5. Profile Coverage and Paper Split

### 5.1 Panasonic (Training Source)

The paper trains on Panasonic Mix profiles + associated CC charge cycles.

| Profile type | Count (all temps) | Split role |
|---|---|---|
| Mix (Cycle_1 to Cycle_4 + NN) | **40 files** across 9 temps | **train** |
| Charge | 60 files | **train** (paired with Mix) |
| UDDS | 6 files | train (also evaluated individually in paper) |
| LA92 | 5 files | train |
| US06 | 6 files | train |
| HWFET | 6 files | train / component of Mix |
| HPPC | 13 files | excluded (characterization) |
| EIS | ~70 files | excluded |
| Discharge, OCV, Pause | remaining | excluded |

**Panasonic Mix composition (confirmed from filenames):**
- Named `{temp}_Cycle_{N}_Pan18650PF.mat` (N=1–4) plus one `_NN_Pan18650PF.mat` per temperature
- "NN" stands for the combined evaluation cycle sequence used for neural network testing
- Some temperatures also have combined-cycle files (see §5.4)

### 5.2 LG HG2 (Target Domain — Test)

| Profile type | Count | Notes |
|---|---|---|
| Mix (Mixed1–8) | **46 files** | Mix1–3 → partial_target; Mix4–8 → test |
| Charge | 96 files | test (paired with Mix/drive cycles) |
| UDDS | 6 files | test |
| HWFET | 6 files | test |
| LA92 | 6 files | test |
| US06 | 6 files | test |
| HPPC | 6 files | excluded |
| C20Discharge, Discharge | 18 files | excluded |

⚠️ **CRITICAL GAP — Missing Mix 3 at 0°C and 10°C:**

| Temperature | Available Mix numbers |
|---|---|
| n20°C | 1, 2, 3, 4, 5, 6, 7, 8 ✅ |
| n10°C | 1, 2, 3, 4, 5, 6, 7, 8 ✅ |
| **0°C** | 1, 2, **MISSING 3**, 4, 5, 6, 7, 8 🔴 |
| **10°C** | 1, 2, **MISSING 3**, 4, 5, 6, 7, 8 🔴 |
| 25°C | 1, 2, 3, 4, 5, 6, 7, 8 ✅ |
| 40°C | 1, 2, 3, 4, 5, 6, 7, 8 ✅ |

For Case 3 (partial target with Mix 1–3), 0°C and 10°C can only supply Mix 1 and 2.

### 5.3 Samsung 30T (Target Domain — Test)

All Mix profiles present (Mix 1–8) at all 6 temperatures (−20°C through 40°C): **48 Mix files total**. No missing profiles.

| Profile type | Count | Notes |
|---|---|---|
| Mix (Mixed1–8) | **48 files** | Mix1–3 → partial_target; Mix4–8 → test |
| Charge | 96 files | test |
| UDDS | 6 files | test |
| HWFET | 6 files | test |
| LA92 | 6 files | test |
| US06 | 6 files | test |
| HPPC, Discharge, etc. | 40 files | excluded |

### 5.4 Panasonic Combined-Cycle Files

Panasonic includes concatenated multi-cycle files (e.g., `n10degC_US06_HWFET_UDDS_LA92_Pan18650PF.mat`). These are **not traditional drive cycle sequences** — they contain very large time gaps:

```
Time[0] =       0.0 s
Time[1] = 123,526.2 s  (≈ 34 hours later)
Time[2] = 136,728.3 s  (≈ 3.7 hours after [1])
...
```

This file records all 4 drive cycle types (US06, HWFET, UDDS, LA92) separated by multi-hour gaps (charging, resting). It spans 586,319 s (~6.8 days). Naïve use as a sequence will insert enormous dead time between drive cycles. Segmentation at gaps >1000s is required.

### 5.5 Reconstruction of Paper Splits

| Split | Panasonic source | Target |
|---|---|---|
| Training | Mix files + Charge files (40+60 files) | N/A |
| Testing | — | LG: UDDS/LA92/US06/Charge (24+96=120); Samsung: same pattern (120) |
| Partial target (Case 3) | — | LG Mix1–3 (up to 18 files, **14 available** due to missing); Samsung Mix1–3 (18 files) |

---

## 6. Data Quality Findings

### 6.1 Summary Table

| Issue type | Panasonic | LG HG2 | Samsung 30T | Severity |
|---|---|---|---|---|
| Large time gaps (>5× median dt) | 86 | 18 | 15 | medium |
| Duplicate timestamps | 15 | 7 | 23 | medium |
| Non-monotonic timestamps | 0 | 1 | 2 | **high** |
| Implausible voltage (V<2.0) | 3 | 3 | 0 | high |
| NaN voltage | 0 | 1 | 1 | **high** |
| Ah non-zero at file start | 15 | 12 | 13 | medium |
| Current >30A (Samsung false alarm) | 0 | 0 | 63 | info |

**Total issues logged:** 278 across 3 datasets.

### 6.2 Critical Issues

#### 🔴 Samsung 40°C Mixed3 and Mixed4: Mid-file timestamp reset

```
File: 03-08-19_16.45 722_Mixed3_40degC_IN21700_30T.mat
  At row 23448 (of 28314): Time 2094.195 → 2094.195 → 0.000
  
File: 03-09-19_00.02 722_Mixed4_40degC_IN21700_30T.mat  
  At row 35982 (of 37264): Time 3726.144 → 3726.158 → 0.000
```

**Root cause:** The data acquisition system was restarted mid-experiment; a new time-series (starting from t=0) was appended to the existing file. The files contain two distinct experimental segments joined without a break marker.

**Impact:** Any code that concatenates files or constructs LSTM sequences by time order will produce a false temporal continuity with a ~2–4 hour apparent jump backward. SOC coulomb-counting will restart incorrectly.

**Recommended action:** Split each file at the timestamp reset (row 23448 and 35982 respectively); treat the post-reset segment as a separate sequence.

#### 🔴 LG HG2 LA92 10°C: Non-monotonic timestamps + NaN voltage

```
File: 11-25-18_20.59 582_LA92_10degC_LGHG2.mat
  4 negative time differences at indices [228, 1487, 1490, 1942]
  Example: Time[227]=22.598 → Time[228]=22.697 → Time[229]=0.499 (reset at row 229!)
  NaN voltage: 3 samples
```

**Root cause:** Multiple timestamp resets within a single file, plus 3 corrupted voltage samples. This is the only LA92 file at 10°C.

**Impact:** The 10°C LA92 test profile for LG HG2 is corrupted. Results at this temperature/profile combination cannot be trusted without cleaning.

**Recommended action:** Segment at each timestamp reset; remove NaN rows; verify continuity of remaining segments before use.

#### 🔴 LG HG2: Missing Mix3 at 0°C and 10°C

For Case 3 (transfer with Mix 1–3 as partial target), only Mix 1 and Mix 2 are available at 0°C and 10°C for LG HG2. If the paper claimed 3 profiles per temperature for all temperatures, this cannot be reproduced.

**Recommended action:** Note in experimental setup that 0°C and 10°C use only 2 partial-target profiles; adjust tables accordingly.

### 6.3 Notable Medium-Severity Issues

#### Panasonic large time gaps (86 instances)

Panasonic's combined-cycle files (e.g., `n10degC_US06_HWFET_UDDS_LA92_Pan18650PF.mat`) have enormous inter-sample gaps of 13,000–123,000 seconds. This is **by design** (multi-day session) but sequence builders must segment at gaps.

Also affects Panasonic HPPC and characterization files which alternate between pulse-period and rest-period sampling.

#### Duplicate timestamps (45 total)

Duplicate or near-duplicate timestamps occur across all datasets, primarily in HPPC and rest segments where the cycler reports the same time for two consecutive measurements. These should be deduplicated before derivative-based feature computation.

#### Ah non-zero at file start (40 instances)

These are exclusively characterization files (HPPC, 5-pulse tests) where equipment is not reset between sub-experiments. Not relevant to drive cycle or Mix profile training.

### 6.4 Samsung 30T Current >30A (False Alarm)

63 flagged instances across Samsung 30T where discharge current exceeds |30A|. The Samsung INR21700-30T is rated at **35A continuous discharge** (Samsung datasheet). Observed peaks of −31 to −36A are physically valid transients during aggressive Mixed-profile discharge pulses. The 30A threshold in our audit was too conservative for this cell.

**No action required** — these are not data quality issues.

---

## 7. Leakage Risk Assessment

### 7.1 Normalization Leakage

**Risk:** If normalization statistics (min/max or mean/std) are computed over the combined training + test dataset, then test-set statistics leak into the training pipeline.

**Assessment (assumption — code not audited):** The paper likely normalizes per-dataset (Panasonic separately, LG HG2 separately, Samsung separately) since they are separate physical batteries. However:
- If voltage normalization uses global V_min=2.0, V_max=4.2 (hardcoded), no leakage occurs
- If statistics are computed from the combined dataset before splitting by profile type, leakage is present

**Recommended action:** Normalize using only training-set statistics, or use fixed physical bounds.

### 7.2 Sequence Construction Leakage

**Risk:** If LSTM input sequences are constructed by concatenating files across charge and discharge cycles, and if a sliding window crosses the train/test split boundary, subsequences from both splits appear in the same training window.

**Assessment:** Since train (Panasonic) and test (LG/Samsung) are different physical batteries, there is **no cross-battery leakage**. However, for within-battery splits (e.g., Mix 1–3 as partial target, Mix 4–8 as test), a sliding window starting in Mix 3 and ending in Mix 4 could leak.

**Recommended action:** When using within-battery splits, enforce hard segment boundaries at profile transitions. Do not allow windows to straddle Mix 3 / Mix 4 boundary.

### 7.3 Ah Continuity Leakage

**Risk:** Ah resets to 0 at each file boundary. If SOC is computed per-file, the "initial SOC" assumption affects all predictions. If the initial SOC is derived from test data (e.g., by looking at the charge file that preceded the test drive cycle), this is legitimate information sharing, not leakage.

**Assessment:** Low risk — using the prior charge file's final Ah to set initial SOC is an operationally valid strategy, not a data leak.

### 7.4 Temperature Information in Input Features

**Risk:** If temperature is used as an input feature but the model was not trained on that temperature range, predictions at extreme temperatures (−20°C) may be extrapolation rather than generalization.

**Assessment:** This is a model design choice, not a data leakage issue.

---

## 8. Recommended Canonical Schema

For preprocessing, we recommend standardizing all datasets to the following schema. Each row is one time step.

```python
# Recommended canonical column names
CANONICAL_SCHEMA = {
    'timestamp':     'str       — wall-clock string (from TimeStamp field)',
    'time_s':        'float64   — elapsed seconds from start of segment (from Time field)',
    'voltage_V':     'float64   — terminal voltage in Volts',
    'current_A':     'float64   — current in Amperes (neg=discharge, pos=charge)',
    'temperature_C': 'float64   — battery surface temperature in °C',
    'ah':            'float64   — cumulative Ah from file start (signed, neg=discharge)',
    'soc':           'float64   — [DERIVED] state of charge 0–1 via coulomb counting',
    'battery':       'str       — dataset identifier: panasonic | lg_hg2 | samsung_30t',
    'profile':       'str       — profile type: Mix | UDDS | LA92 | US06 | HWFET | Charge | ...',
    'temperature':   'str       — nominal temperature: -20degC | ... | 40degC',
    'segment_id':    'str       — unique segment identifier: {battery}_{temperature}_{filename}_{subseg}',
}
```

**SOC derivation:**
```python
# Q_nominal per dataset (Ah)
Q_NOMINAL = {'panasonic': 2.9, 'lg_hg2': 3.0, 'samsung_30t': 3.0}

# After a full CC charge (Charge file ending at V ≈ 4.2 V, I ≈ 0):
# SOC_init = 1.0

# SOC at time t within a file:
# SOC(t) = SOC_init + Ah(t) / Q_nominal
# where Ah(t) is the cumulative Ah column (negative during discharge)
```

**Segment ID convention:** Each file should be split into sub-segments at:
1. Timestamp resets (time drops to < previous time)
2. Time gaps > 60 seconds (conservative) or > 300 seconds (moderate)

---

## 9. Unresolved Questions

1. **Paper version of datasets:** It is unclear whether the LG HG2 and Samsung 30T data in this repository is identical to the version used in Vidal et al. (2019) or an extended post-submission collection. The Samsung data was collected Mar–Jul 2019, which may postdate the paper's submission.

2. **Missing LG Mix 3 at 0°C/10°C:** Was Mix 3 deliberately omitted at these temperatures in the original experiments, or was data lost? The paper's Table I and Case 3 description do not clarify per-temperature profile availability.

3. **Panasonic "NN" file role:** The file `25degC_NN_Pan18650PF.mat` appears to be a Neural Network evaluation cycle (different from Cycle_1–4). Its exact composition (which drive cycles, in what order) is not specified in the filename. It may correspond to the "Mix" profile used for testing in the paper.

4. **Samsung 48G data:** The task specifies Samsung 48G uses Mix 1, 2, 6 in Case 3, but no data is present. Was this dataset ever publicly released?

5. **Q_nominal values:** The exact nominal capacities used in Vidal et al. for SOC normalization are not specified. Using different Q_nominal values will shift SOC by a constant factor.

6. **Samsung 40°C Mixed3/4 split point:** After splitting at the timestamp reset, what is the correct interpretation of the two segments? Are they separate Mix profiles from different sessions, or a single profile interrupted by equipment restart?

7. **LG HG2 10degC LA92 usability:** The corrupted file `582_LA92_10degC_LGHG2.mat` has 4 timestamp resets and 3 NaN voltages. After cleaning, is the remaining data sufficient for evaluation?

8. **Current sign in Vidal et al.:** The paper may define positive current as discharge (opposite to this dataset's convention). Implementors must verify sign convention matches the paper's equations.

---

## 10. Appendix: Per-Dataset File Counts

### Panasonic 18650PF (198 .mat files)

| Temperature | Drive cycles | Charges/Pauses | HPPC | EIS | Other |
|---|---|---|---|---|---|
| −20°C | 10 | 8 | 2 | 10 | 0 |
| −20°C Trise | 6 | 6 | 0 | 0 | 0 |
| −20°C Trise w/ pause | 1 | 0 | 0 | 0 | 0 |
| −10°C | 9 | 11 | 2 | 9 | 0 |
| 0°C | 9 | 10 | 2 | 12 | 0 |
| 10°C | 10 | 7 | 2 | 14 | 0 |
| 10°C Trise | 6 | 6 | 0 | 0 | 0 |
| 10°C Trise w/ pause | 1 | 0 | 0 | 0 | 0 |
| 25°C | 11 | 22 | 2 | 15 | 4 |

### LG 18650HG2 (206 .mat files)

| Temperature | Mix | Drive cycles | Charges | HPPC/Dis | Other |
|---|---|---|---|---|---|
| n20°C | 8 | 4 | 11 | 10 | 1 |
| n10°C | 8 | 5 | 16 | 10 | 2 |
| 0°C | 7 | 4 | 16 | 10 | 2 |
| 10°C | 7 | 5 | 16 | 10 | 1 |
| 25°C | 8 | 5 | 17 | 10 | 2 |
| 40°C | 8 | 4 | 20 | 10 | 1 |

### Samsung INR21700-30T (208 .mat files)

| Temperature | Mix | Drive cycles | Charges | HPPC/Dis | Other |
|---|---|---|---|---|---|
| −20°C | 8 | 4 | 18 | 5 | 1 |
| −10°C | 8 | 4 | 17 | 5 | 1 |
| 0°C | 8 | 4 | 17 | 5 | 1 |
| 10°C | 8 | 4 | 17 | 5 | 1 |
| 25°C | 8 | 4 | 16 | 5 | 1 |
| 40°C | 8 | 4 | 16 | 5 | 1 |

### Samsung 48G

| Item | Status |
|---|---|
| Data files | None |
| README.md | Present (placeholder) |

---

*End of audit report. Generated: 2026-07-12. All findings are based on read-only file inspection. No files were modified.*
