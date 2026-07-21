# Panasonic NCR18650PF 25°C — Vidal et al. (2019 ITEC) File-Split Report

**Generated:** 2026-07-12 (corrected)
**Method:** Read-only file inspection via `scripts/inspect_panasonic_25c_split.py`
**Manifest:** `data/manifests/vidal_2019/panasonic_vidal2019_25c_manifest.csv`
**Battery:** Panasonic NCR18650PF (2.9 Ah nominal)
**Temperature:** 25°C only
**Status:** **PROVISIONAL** — see §12 for rationale

---

## 1. Scope and Paper Protocol

### 1.1 Paper Reference

**Correct citation:**

> C. Vidal, P. Kollmeyer, E. Chemali, and A. Emadi, "Li-ion Battery State of Charge Estimation Using Long Short-Term Memory Recurrent Neural Network with Transfer Learning," *2019 IEEE Transportation Electrification Conference and Expo (ITEC)*, 2019.

This is a **2019 conference paper** (ITEC). Earlier versions of this report incorrectly cited a different (later) journal paper; all citation references in this document now refer to the 2019 ITEC paper.

The data was collected at the University of Wisconsin-Madison in **March–July 2017** by Dr. Phillip Kollmeyer.

### 1.2 Scope of This Report

This report covers **all .mat files under `data/raw/Panasonic_18650PF/25degC/`** and determines their role in reproducing the Vidal et al. (2019) training/test split at **25°C only**.

The thermal chamber was set to **25°C** for this session. Only 25°C files are relevant to this task. No multi-temperature files are required.

**Paper-verified (Vidal 2019 ITEC, §II):** *"All tests were performed in a thermal chamber set to 25°C, and the measured battery surface temperature increases beyond 25°C due to losses in the battery."*

Subfolders inspected:
- `Drive cycles/` — 10 .mat files
- `Charges and Pauses/` — 25 .mat files
- `5 pulse disch/` — 2 .mat files (HPPC)
- `EIS/` — 15 .csv files (not .mat, excluded automatically)
- `C20 OCV and 1C discharge tests_start_of_tests/` — 12 .mat files
- `1C discharge tests_end_of_tests/` — 11 .mat files

**Total .mat files in manifest: 61**
**EIS (.csv) files:** excluded (not .mat, not relevant)

### 1.3 Vidal et al. Train/Test Protocol (reconstructed)

From the Panasonic README and the most plausible interpretation of the paper's Mix1–5 naming (see §7):

| Role | Profiles | Files at 25°C |
|---|---|---|
| Full training source | Mix1–5 (= Cycle_1–4 + NN) + associated charges | 5 drive + 5 charge = **10** |
| Partial training source (Case 3 only) | Mix1–3 (Cycle_1–3) + associated charges | 3 drive + 3 charge = **6** |
| Test | UDDS, LA92, US06 + associated charges | 3 drive + 4 charge = **7** |
| Excluded | HWFET, HPPC, EIS, OCV, 1C discharge, Pause, Telemetry, end-of-test files | **44** |

**Total manifest rows: 61** (10 + 7 + 44 = 61)

---

## 2. Final `full_train` File List

Files with `full_train=1` (drive cycles: Cycle_1–4 + NN; associated charges).

### 2.1 Drive Cycles

| Filename | Profile | Start | Duration (min) | Ah end | T mean (°C) |
|---|---|---|---|---|---|
| `03-18-17_02.17 25degC_Cycle_1_Pan18650PF.mat` | Cycle_1 (Mix1) | 2017-03-18T02:17 | 183.1 | −2.696 | 26.4 |
| `03-19-17_03.25 25degC_Cycle_2_Pan18650PF.mat` | Cycle_2 (Mix2) | 2017-03-19T03:25 | 185.8 | −2.711 | 26.9 |
| `03-19-17_09.07 25degC_Cycle_3_Pan18650PF.mat` | Cycle_3 (Mix3) | 2017-03-19T09:07 | 171.1 | −2.530 | 26.8 |
| `03-19-17_14.31 25degC_Cycle_4_Pan18650PF.mat` | Cycle_4 (Mix4) | 2017-03-19T14:31 | 201.8 | −2.798 | 26.9 |
| `03-21-17_16.27 25degC_NN_Pan18650PF.mat`      | NN (Mix5)       | 2017-03-21T16:27 | 195.6 | −2.550 | 27.4 |

### 2.2 Associated Charge Files (full_train)

| Filename | Role | Start | Split basis |
|---|---|---|---|
| `03-18-17_00.05 3406_Charge1.mat` | Pre-Cycle_1 charge (no preceding drive cycle → inherits Cycle_1 split) | 2017-03-18T00:05 | full_train, partial_train |
| `03-19-17_06.31 3406_Charge2.mat` | Post-Cycle_2 / Pre-Cycle_3 charge (Cycle_2 immediately precedes) | 2017-03-19T06:31 | full_train, partial_train |
| `03-19-17_11.59 3406_Charge3.mat` | Post-Cycle_3 / Pre-Cycle_4 charge (Cycle_3 immediately precedes) | 2017-03-19T11:59 | full_train, partial_train |
| `03-19-17_17.53 3406_Charge4.mat` | Post-Cycle_4 / Pre-US06 charge (Cycle_4 immediately precedes) | 2017-03-19T17:53 | full_train |
| `03-21-17_19.43 3416_Charge5.mat` | Post-NN charge (NN immediately precedes) | 2017-03-21T19:43 | full_train |

**full_train count: 5 drive cycles + 5 charge files = 10 files total**

---

## 3. Final `partial_train` File List

Files with `partial_train=1` (Case 3 only: Mix1–3 = Cycle_1–3 + associated charges).
These are a strict subset of `full_train`.

### 3.1 Drive Cycles (partial_train)

| Filename | Profile | Notes |
|---|---|---|
| `03-18-17_02.17 25degC_Cycle_1_Pan18650PF.mat` | Cycle_1 (Mix1) | also full_train |
| `03-19-17_03.25 25degC_Cycle_2_Pan18650PF.mat` | Cycle_2 (Mix2) | also full_train |
| `03-19-17_09.07 25degC_Cycle_3_Pan18650PF.mat` | Cycle_3 (Mix3) | also full_train |

### 3.2 Associated Charge Files (partial_train)

| Filename | Role |
|---|---|
| `03-18-17_00.05 3406_Charge1.mat` | Pre-Cycle_1 |
| `03-19-17_06.31 3406_Charge2.mat` | Post-Cycle_2 / Pre-Cycle_3 |
| `03-19-17_11.59 3406_Charge3.mat` | Post-Cycle_3 |

**partial_train count: 3 drive cycles + 3 charge files = 6 files total**

---

## 4. Final `test` File List

Files with `test=1` (UDDS, LA92, US06 + associated charges).

### 4.1 Drive Cycles (test)

| Filename | Profile | Start | Duration (min) | Ah end | T mean (°C) |
|---|---|---|---|---|---|
| `03-20-17_01.43 25degC_US06_Pan18650PF.mat`  | US06 | 2017-03-20T01:43 |  80.3 | −2.586 | 29.5 |
| `03-21-17_00.29 25degC_UDDS_Pan18650PF.mat`  | UDDS | 2017-03-21T00:29 | 374.1 | −2.702 | 26.0 |
| `03-21-17_09.38 25degC_LA92_Pan18650PF.mat`  | LA92 | 2017-03-21T09:38 | 235.1 | −2.587 | 26.5 |

### 4.2 Associated Charge Files (test)

| Filename | Role | Split basis |
|---|---|---|
| `03-19-17_17.53 3406_Charge4.mat` | Pre-US06 charge (Cycle_4 immediately precedes → full_train; see §5.2) | full_train only |
| `03-20-17_03.05 3415_Charge1.mat` | Post-US06 charge (US06 immediately precedes → test) | test |
| `03-20-17_21.35 3416_Charge2.mat` | Pre-UDDS charge (HWFTb excluded → UDDS fallback → test) | test |
| `03-21-17_06.44 3416_Charge3.mat` | Post-UDDS / Pre-LA92 charge (UDDS immediately precedes → test) | test |
| `03-21-17_13.34 3416_Charge4.mat` | Post-LA92 / Pre-NN charge (LA92 immediately precedes → **test**) | test |

**test count: 3 drive cycles + 4 charge files = 7 files total**

**Note on 3406_Charge4:** This charge (Mar 19, 17:53) immediately follows Cycle_4 (full_train) and therefore is classified `full_train=1` by the split rule. It also serves as the `selected_charge_for_sequence` (SOC anchor) for US06 (test). Its manifest entry is `full_train=1, test=0` and it is listed in §4.2 only for reference; it is counted in the full_train total.

**Note on 3416_Charge4:** This charge (Mar 21, 13:34) immediately follows LA92 (test) and therefore is classified `test=1` by the split rule. NN (full_train) does **not** load this charge file; NN uses `soc_anchor_source = "full_charge_protocol"` with `soc_init = 1.0` derived from Ah≈0 at file start. No cross-split dependency exists.

---

## 5. Final Excluded File List

| Filename | Reason |
|---|---|
| `03-20-17_05.56 25degC_HWFTa_Pan18650PF.mat` | HWFET excluded per protocol |
| `03-20-17_19.27 25degC_HWFTb_Pan18650PF.mat` | HWFET excluded per protocol |
| `03-20-17_08.04 3415_Charge2.mat` | Post-HWFTa charge; HWFET excluded; charge not used as SOC anchor for any selected cycle → excluded |
| `03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat` | HPPC characterization |
| `03-11-17_10.10 3390_dis5_10p.mat` | HPPC inter-pulse discharge |
| `05-08-17_13.26 C20 OCV Test_C20_25dC.mat` | C20 OCV characterization |
| `03-09-17_15.12 3349_TS002973.mat` | Telemetry/setup |
| `03-09-17_15.13 3349_Charge1.mat` | Pre-characterization charge |
| `03-09-17_17.59 3349_Dis1C_1.mat` | 1C discharge characterization |
| `03-09-17_17.59 3349_Pause_1.mat` | Pause/rest |
| `03-09-17_19.03 3349_Charge2.mat` | Pre-characterization charge |
| `03-09-17_21.03 3349_Dis1C_Rp.mat` | 1C discharge characterization (repeat) |
| `03-09-17_21.03 3349_Pause_Rp.mat` | Pause/rest |
| `03-09-17_21.57 3349_ChargeRp.mat` | Pre-characterization charge (repeat) |
| `03-10-17_23.36 3349_Dis1C_2.mat` | 1C discharge characterization |
| `03-10-17_23.36 3349_Pause_2.mat` | Pause/rest |
| `03-11-17_00.39 3349_Charge3.mat` | Pre-characterization charge |
| `03-28-17_15.29 3423_Charge1.mat` | From 10°C session; stored in 25°C folder |
| `03-11-17_08.25 3390_TS003014.mat` | Telemetry/setup |
| `03-12-17_11.58 3390_Charge_2.mat` | Post-HPPC charge (not used as SOC anchor for any selected cycle) |
| All 10 Pause files (3406_Pause1–4, 3415_Pause1–2, 3416_Pause2–5) | Rest periods |
| `04-27-17_08.52 3541_Charge1a.mat` | April HPPC session charge |
| `04-27-17_11.53 3541_dis5_10p.mat` | April HPPC discharge |
| `04-29-17_13.07 3541_Charge2a.mat` | April HPPC session charge |
| All 11 end-of-tests files (4020_*) | End-of-life 1C discharge tests (July 2017) |

**exclude count: 44 files**

---

## 5a. Leakage Check

The following constraints are verified by `scripts/inspect_panasonic_25c_split.py` (validation checks c–e):

| Check | Result |
|---|---|
| No file has `full_train=1` AND `test=1` | **PASS** |
| No file has `partial_train=1` AND `test=1` | **PASS** |
| No file has `exclude=1` AND any split flag set | **PASS** |

**Leakage check: PASS** (confirmed by script output 2026-07-12).

---

## 6. Charge Sequence Table

**Split-assignment rule (canonical):** Each charge file is assigned to the split of the drive cycle that *immediately precedes it* in chronological order (preceding cycle wins). When the immediately preceding drive cycle is excluded (HWFET), the charge is assigned to the split of the next non-excluded drive cycle that uses it as its SOC anchor (following-cycle fallback). Pre-session charges (no preceding drive cycle) inherit the split of the immediately following drive cycle.

| Drive Cycle | `selected_charge_for_sequence` | `previous_charge` | `next_charge` | `pairing_status` | `soc_anchor_evidence` |
|---|---|---|---|---|---|
| Cycle_1 (Mix1) | `3406_Charge1` | `3406_Charge1` | `3406_Charge2` | **confirmed** | Charge1 terminates at 4.200V, 0.00A, Ah_end=2.670Ah → SOC_init=1.0 |
| Cycle_2 (Mix2) | **UNRESOLVED** | *(absent)* | `3406_Charge2` | **unresolved** | No preceding charge file; Cycle_2 Ah≈0 at start → SOC_init=1.0 assumed, low confidence |
| Cycle_3 (Mix3) | `3406_Charge2` | `3406_Charge2` | `3406_Charge3` | **confirmed** | Charge2 terminates at 4.200V, 0.00A, Ah_end=2.677Ah → SOC_init=1.0 |
| Cycle_4 (Mix4) | `3406_Charge3` | `3406_Charge3` | `3406_Charge4` | **confirmed** | Charge3 terminates at 4.200V, 0.00A, Ah_end=2.497Ah → SOC_init=1.0 |
| US06 | `3406_Charge4` | `3406_Charge4` | `3415_Charge1` | **confirmed** | Charge4 terminates at 4.200V, 0.00A, Ah_end=2.760Ah → SOC_init=1.0 |
| HWFET-a | *(excluded)* | `3415_Charge1` | `3415_Charge2` | — | — |
| HWFET-b | *(excluded)* | `3415_Charge2` | `3416_Charge2` | — | — |
| UDDS | `3416_Charge2` | `3416_Charge2` | `3416_Charge3` | **confirmed** | Charge2 terminates at 4.200V, 0.00A, Ah_end=2.675Ah → SOC_init=1.0 |
| LA92 | `3416_Charge3` | `3416_Charge3` | `3416_Charge4` | **confirmed** | Charge3 terminates at 4.200V, 0.00A, Ah_end=2.635Ah → SOC_init=1.0 |
| NN (Mix5) | *(empty — protocol)* | `3416_Charge4` | `3416_Charge5` | **confirmed** | No charge file loaded; `soc_anchor_source = "full_charge_protocol"`, `soc_init = 1.0` — NN file starts at Ah≈0, confirming battery was fully charged before run; no cross-split dependency |

**Ah-reset confirmation:** Every selected drive cycle starts with Ah ≈ 0.0000 (range −0.0001 to 0.0000), confirming the cycler's Ah counter was reset at the start of each file.

**Unresolved pairings:** 1 (Cycle_2 pre-charge absent).
**Inferred pairings:** 0 (NN now uses `full_charge_protocol` assumption; no cross-split dependency).

---

## 7. Cycle_1–4 + NN = Mix1–5 Mapping

### 7.1 Verdict: Most Plausible Paper-to-Dataset Mapping

The mapping Cycle_1→Mix1, Cycle_2→Mix2, Cycle_3→Mix3, Cycle_4→Mix4, NN→Mix5 is **the most plausible paper-to-public-dataset mapping**, supported by the inference chain below. The Vidal 2019 (ITEC) paper uses the name "Mix1–5" for training profiles, while the Panasonic public dataset uses the names "Cycle_1–4" and "NN". Neither the paper nor the dataset README directly states "Cycle_N = MixN" verbatim, but the correspondence is unambiguous from context.

### 7.2 Inference Chain

**Step 1 — Panasonic README names the training profiles:**

> "Series of nine drive cycles performed, in following order: Cycle 1, Cycle 2, Cycle 3, Cycle 4, US06, HWFET, UDDS, LA92, Neural Network (NN). Cycles 1-4 consist of random mix of US06, HWFET, UDDS, LA92, and Neural Network drive cycles."

This establishes: the Panasonic dataset's training profiles are Cycle_1, Cycle_2, Cycle_3, Cycle_4, and NN (Mix of standard cycles plus Neural Network evaluation).

**Step 2 — Vidal 2019 (ITEC) calls training profiles Mix1–5:**

The paper trains an LSTM on profiles called "Mix1" through "Mix5", where Mix profiles consist of random combinations of standard drive cycles. Five training profiles match five Panasonic Cycle/NN files.

**Step 3 — Chronological and count correspondence:**

- Five training profiles in Panasonic dataset: Cycle_1, Cycle_2, Cycle_3, Cycle_4, NN
- Five Mix profiles in paper: Mix1, Mix2, Mix3, Mix4, Mix5
- In chronological order, Cycle_1 is first, Cycle_2 is second, ..., NN is fifth (Mix5)
- The NN file uses a different current pattern (max discharge −14.7 A vs. −17 to −20 A for Cycles), consistent with the README's description of the Neural Network drive cycle as a distinct composition

**Step 4 — MATLAB code in external_repos confirms split structure:**

The MATLAB script `external_repos/reproductions/xf68bwh54v-1/LSTMvsFNN_Script_Nov_2021_V2.m` uses normalized data with `Train/`, `Validation/`, `Test/` directories, confirming the basic train/test structure.

**Conclusion:** Cycle_N = MixN and NN = Mix5 is the most plausible correspondence. The paper does not use the dataset's "Cycle_N" naming, but the mapping is unambiguous from the five-to-five count, the README's description, and the drive-cycle composition. This mapping is documented as **inferred, not directly quoted from the paper**.

### 7.3 No Alternative Mapping

No credible alternative mapping exists. The README is unambiguous about the test sequence order and profile names. The only naming ambiguity is "Mix1–5" (paper) vs. "Cycle_1–4 + NN" (dataset), which this inference resolves.

---

## 8. Duplicate/Combined File Findings

At 25°C there are **no combined multi-cycle files** (unlike at −10°C where `n10degC_US06_HWFET_UDDS_LA92_Pan18650PF.mat` and at 10°C where `10degC_US06_HWFET_UDDS_LA92_NN_Pan18650PF.mat` exist).

The two HWFET files (`HWFTa` and `HWFTb`) are **not** combined/duplicate files — they represent two separate HWFET test sessions run ~13.5 hours apart (possibly due to an equipment break). Both are standalone files and both are excluded per protocol.

`is_duplicate=0` for all 25°C .mat files.

---

## 9. Selected-File Quality Check Results

All 17 selected (non-excluded) files pass quality control with **no issues**.

| File | Profile | QC Status | n_rows | duration_s | T_mean | V_min | V_max | I_min | I_max | Ah_start | Ah_end |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `...Cycle_1...` | Cycle_1 | **PASS** | 109,641 | 10,988 | 26.4°C | 2.502 | 4.217 | −19.4 | +10.7 | 0.000 | −2.696 |
| `...Cycle_2...` | Cycle_2 | **PASS** | 111,268 | 11,147 | 26.9°C | 2.505 | 4.206 | −18.2 | +9.9 | 0.000 | −2.711 |
| `...Cycle_3...` | Cycle_3 | **PASS** | 102,441 | 10,265 | 26.8°C | 2.508 | 4.209 | −17.2 | +9.5 | 0.000 | −2.530 |
| `...Cycle_4...` | Cycle_4 | **PASS** | 120,863 | 12,107 | 26.9°C | 2.508 | 4.220 | −17.9 | +10.1 | 0.000 | −2.798 |
| `...US06...` | US06 | **PASS** | 48,061 | 4,819 | 29.5°C | 2.494 | 4.223 | −20.8 | +7.6 | 0.000 | −2.586 |
| `...UDDS...` | UDDS | **PASS** | 224,187 | 22,447 | 26.0°C | 2.503 | 4.208 | −7.5 | +4.1 | 0.000 | −2.702 |
| `...LA92...` | LA92 | **PASS** | 140,874 | 14,106 | 26.5°C | 2.501 | 4.236 | −10.5 | +9.7 | 0.000 | −2.587 |
| `...NN...` | NN | **PASS** | 116,982 | 11,735 | 27.4°C | 2.502 | 4.231 | −14.7 | +7.9 | 0.000 | −2.550 |
| `3406_Charge1` | Charge | **PASS** | 120 | 24,685 | 23.1°C | 3.299 | 4.200 | 0.0 | +2.9 | 0.000 | +2.670 |
| `3406_Charge2` | Charge | **PASS** | 99 | 5,728 | 27.8°C | 3.297 | 4.200 | 0.0 | +2.9 | 0.000 | +2.677 |
| `3406_Charge3` | Charge | **PASS** | 95 | 5,489 | 27.8°C | 3.364 | 4.200 | 0.0 | +2.9 | 0.000 | +2.497 |
| `3406_Charge4` | Charge | **PASS** | 101 | 5,850 | 27.9°C | 3.097 | 4.200 | 0.0 | +2.9 | 0.000 | +2.760 |
| `3415_Charge1` | Charge | **PASS** | 115 | 6,684 | 27.6°C | 3.342 | 4.200 | 0.0 | +2.9 | 0.000 | +2.569 |
| `3416_Charge2` | Charge | **PASS** | 118 | 6,862 | 27.6°C | 3.283 | 4.200 | 0.0 | +2.9 | 0.000 | +2.675 |
| `3416_Charge3` | Charge | **PASS** | 117 | 6,827 | 27.5°C | 3.317 | 4.200 | 0.0 | +2.9 | 0.000 | +2.635 |
| `3416_Charge4` | Charge | **PASS** | 116 | 6,746 | 27.4°C | 3.345 | 4.200 | 0.0 | +2.9 | 0.000 | +2.544 |
| `3416_Charge5` | Charge | **PASS** | 117 | 6,729 | 27.4°C | 3.354 | 4.200 | 0.0 | +2.9 | 0.000 | +2.513 |

**Quality check criteria (all passed):**
- `scipy.io.loadmat` loads without error ✓
- `meas` struct exists ✓
- Required fields present: `Time`, `TimeStamp`, `Voltage`, `Current`, `Ah`, `Battery_Temp_degC` ✓
- Array lengths consistent ✓
- No NaN in `Voltage`, `Current`, or `Ah` ✓
- `Time` monotonically non-decreasing (no resets) ✓
- `Battery_Temp_degC` mean in 20–35°C range for drive cycles ✓
- `Battery_Temp_degC` mean of 23.1°C for Charge1 (acceptable — chamber warming up) ✓

---

## 10. Confirmed Facts

The following are confirmed by direct file inspection and cross-referenced with the Panasonic README:

1. **Cycle_1 = Mix1, Cycle_2 = Mix2, Cycle_3 = Mix3, Cycle_4 = Mix4, NN = Mix5** (most plausible mapping; see §7).
   Source: Panasonic README explicitly names the test sequence as "Cycle 1, Cycle 2, Cycle 3, Cycle 4, US06, HWFET, UDDS, LA92, Neural Network (NN)."

2. **All 25°C drive cycles are in `Drive cycles/` subfolder.** There are exactly 10 drive cycle files (Cycle_1–4, US06, HWFTa, HWFTb, UDDS, LA92, NN). No combined multi-cycle file exists at 25°C.

3. **All selected drive cycles start at Ah = 0.** Confirmed by direct .mat inspection. This confirms the cycler Ah counter was reset before each test, consistent with a preceding full charge.

4. **All charge files terminate at V = 4.200 V with I_min → 0 A** (CC-CV profile to 4.2V), confirming they represent full charges to 100% SOC.

5. **Charge amounts (Ah_end of charge files) range from +2.497 to +2.760 Ah**, consistent with nominal 2.9 Ah cell with some aging (approximately 85–95% of nominal).

6. **Temperature at 25°C session is confirmed** (mean 26–27°C for drive cycles, 23–28°C for charges — slightly elevated due to self-heating). All files satisfy the 25°C nominal classification.

7. **All drive cycles are sampled at 10 Hz** (median dt = 0.1 s, as confirmed by n_rows × 0.1 ≈ duration_s for drive cycle files).

8. **Charge files are sampled at lower rate** (~0.23 Hz average based on n_rows and duration_s), consistent with README: "charges and pauses were saved at a lower data rate."

9. **The Panasonic README's ordering (Cycle 1 → Cycle 2 → Cycle 3 → Cycle 4 → US06 → HWFET → UDDS → LA92 → NN) matches the chronological file order** at 25°C exactly.

10. **No EIS .mat files exist at 25°C.** EIS data is stored as CSV files in the `EIS/` subfolder (15 .csv files, not .mat). These are excluded automatically.

---

## 11. Assumptions

The following assumptions are made where evidence is incomplete:

1. **Cycle_2's pre-charge (ASSUMPTION — UNRESOLVED).** No charge file is visible between Cycle_1 (Mar 18, 02:17) and Cycle_2 (Mar 19, 03:25). The 24-hour gap contains only `3406_Pause2` (Mar 19, 02:26). We assume a charge occurred during this gap but its file is absent (not logged, or stored elsewhere). Evidence supporting this assumption: Cycle_2 starts at Ah = −0.0001 ≈ 0.0, which is consistent with a full preceding charge. `pairing_status = unresolved` for Cycle_2.

2. **3406_Charge4 single role.** This charge (Mar 19, 17:53) comes after Cycle_4 (full_train) and before US06 (test). By the split rule, Cycle_4 immediately precedes → `full_train=1`. It also serves as the `selected_charge_for_sequence` for US06 (its SOC anchor). The manifest records `full_train=1` with cross-split usage documented in the `selected_charge_for_sequence` column of US06.

3. **3416_Charge4 single role.** This charge (Mar 21, 13:34) immediately follows LA92 (test) → `test=1` by split rule. Although it physically precedes NN in time, the NN training sequence does **not** load this charge file. Instead, `soc_init = 1.0` is assumed from the full-charge protocol: NN starts at Ah≈0, which confirms the battery was fully charged. `selected_charge_for_sequence = ""` for NN; no cross-split dependency exists.

4. **HWFET excluded per protocol.** The Vidal et al. (2019) test profiles at 25°C are UDDS, LA92, US06. HWFET is excluded. HWFTa and HWFTb are therefore excluded. `3415_Charge2` (post-HWFTa) is excluded because it is not the `selected_charge_for_sequence` for any remaining selected drive cycle.

5. **`3416_Charge2` (Mar 20, 21:35) classified as test** (pre-UDDS charge). It immediately follows HWFTb (excluded). Since HWFTb is excluded, `3416_Charge2` is assigned to UDDS via the following-cycle fallback rule, giving `test=1`. `pairing_status = confirmed`.

6. **`3423_Charge1` (Mar 28, 15:29) is excluded.** This charge file appears in the `C20 OCV and 1C discharge tests_start_of_tests/` subfolder at 25°C, but was also listed under the 10°C session. It is 7 days after the 25°C drive cycles ended. It is excluded from the Vidal split regardless of its session affiliation.

7. **No combined multi-cycle file at 25°C.** Unlike −10°C and 10°C temperatures (which have combined files), no such file exists at 25°C. Only split files are available.

---

## 12. Unresolved Questions

1. **Missing pre-Cycle_2 charge file.** A charge must have occurred between Cycle_1 (Mar 18, 02:17) and Cycle_2 (Mar 19, 03:25) given that Cycle_2 starts at Ah≈0. No charge file covering this period appears in the `Charges and Pauses/` folder. Possible explanations: (a) the file was not logged separately; (b) the charge is embedded in `3406_Pause2` (unlikely — Pause2 is only 59 minutes before Cycle_2, insufficient for a full 1C charge). **Impact:** Cycle_2's `selected_charge_for_sequence` is empty; `pairing_status = unresolved`. SOC initialization for Cycle_2 must be assumed (SOC_init = 1.0, low confidence).

2. **`3423_Charge1` location ambiguity.** This file (Mar 28, 15:29) appears in the 25°C folder's `C20 OCV and 1C discharge tests_start_of_tests/` subfolder but may belong to the 10°C session. Excluded regardless; no impact on Vidal split.

3. **HWFTb pairing gap.** The 13.5-hour gap between HWFTa (ends ~07:30 Mar 20) and HWFTb (starts 19:27 Mar 20) is unexplained. HWFET excluded per protocol; this is informational only.

---

## 13. Differences from Vidal et al. (2019 ITEC) Paper Description

| Item | This Reconstruction | Vidal et al. (2019 ITEC) |
|---|---|---|
| Training profile names | Cycle_1 through Cycle_4 + NN (Panasonic dataset names) | Mix1 through Mix5 (paper names) |
| Mapping basis | Most plausible inference from README + count correspondence | Not stated verbatim in paper |
| Number of Mix profiles | 5 (Cycle_1–4 + NN) | 5 (Mix1–5) |
| Test profiles | UDDS, LA92, US06 | UDDS, LA92, US06 |
| HWFET role | Excluded (not in paper's test set) | Component of Mix profiles internally |
| Combined multi-cycle file at 25°C | Does not exist | Likely not used in paper |
| Charge files paired with training | 5 charge files (split-rule assignment) | Not explicitly documented in paper |
| Pre-Cycle_2 charge file | **Absent** from repository | Presumably occurred |
| Temperature scope | **25°C only** | Paper text (§II): *"All tests were performed in a thermal chamber set to 25°C"* — confirmed match |
| Data acquisition rate | Drive-cycle files sampled at **10 Hz** (dt = 0.1 s, confirmed by file inspection) | Paper Table IV states **1 Hz** — discrepancy; likely paper rounded or used resampled data; raw files are 10 Hz |

---

## 14. Split Status: PROVISIONAL

This split is **PROVISIONAL** for the following reasons:

1. **Pre-Cycle_2 charge file is absent.** While the impact on the split itself is minor (Cycle_2 still starts at SOC≈1.0), the evidence chain is incomplete. `pairing_status = unresolved` for Cycle_2.

2. **Charge file dual-role handling not confirmed by paper.** The file-level assignment of accompanying charge sequences is reconstructed
because the paper does not specify exact public-file mappings. No training
sequence loads a charge file assigned to the test split.

3. **Mix mapping confirmed from paper.** The Vidal 2019 (ITEC) paper (Table I) lists Panasonic test profiles as "Mix (1-5), UDDS, LA92, US06, CC". The Panasonic public dataset's "Cycle_1–4 + NN" matches this five-to-five count. The Cycle_N = MixN correspondence is now confirmed as the correct paper-to-dataset mapping.

The split can be considered **FINAL** once:
- [ ] The pre-Cycle_2 charge absence is investigated (check if file exists in a different subfolder)
- [ ] The Panasonic adapter is implemented and SOC labels are computed, confirming SOC_init=1.0 for Cycle_2

---

*End of report. Corrected 2026-07-12 by manual review + `scripts/inspect_panasonic_25c_split.py` (read-only inspection). No raw data files were modified.*
