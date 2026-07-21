# LG HG2 25°C — Vidal et al. (2019 ITEC) File-Split Report

**Generated:** 2026-07-12
**Method:** Read-only file inspection via `scripts/inspect_lg_hg2_25c_split.py`
**Manifest:** `data/manifests/vidal_2019/lg_hg2_vidal2019_25c_manifest.csv`
**Battery:** LG INR18650HG2 (3 Ah nominal)
**Temperature:** 25°C only
**Status:** **PROVISIONAL** — see §19 for rationale

---

## 1. Scope

This report covers all `.mat` files under `data/raw/LG_HG2/25degC/` and determines their role in reproducing the Vidal et al. (2019 ITEC) training/test split at **25°C only**.

**Paper-verified (Vidal 2019 ITEC, §II):** *"All tests were performed in a thermal chamber set to 25°C, and the measured battery surface temperature increases beyond 25°C due to losses in the battery."*

**Total .mat files in 25degC directory: 36**

CSV files (also present in the directory) are exact duplicates of the `.mat` files saved by the cycler and are excluded; only `.mat` files are used for the manifest.

---

## 2. Exact Vidal 2019 (ITEC) Paper Protocol for LG HG2

**Citation:**

> C. Vidal, P. Kollmeyer, E. Chemali, and A. Emadi, "Li-ion Battery State of Charge Estimation Using Long Short-Term Memory Recurrent Neural Network with Transfer Learning," *2019 IEEE Transportation Electrification Conference and Expo (ITEC)*, 2019.

**From Table I of the paper:**

| Parameter | Value |
|---|---|
| Battery | LG HG2 |
| Capacity | 3 Ah |
| Format | Cylindrical 18650 |
| Test profiles | Mix (1-8), UDDS, LA92, US06, CC |

**From §IV.B of the paper:**

> "The Training datasets consist of all of the available Mixed drive cycles for each battery, as listed in Table I, and the accompanying charges (CC) for each mixed drive cycle. The Test dataset is the UDDS, LA92, and US06 profiles and the corresponding charges (CC)."

> "For cases 3 and 4, partial Training datasets consisting of cycles MIX 1-3 ... were used."

**Inferred split from paper:**

| Role | Profiles | Notes |
|---|---|---|
| Full training | Mix1–8 + accompanying CC charges | All eight mix profiles |
| Partial training (Case 3 only) | Mix1–3 + accompanying CC charges | Strict subset of full training |
| Test | UDDS, LA92, US06 + corresponding CC charges | Standard drive cycles |
| Excluded | HWFET, HPPC, OCV, capacity, discharge, pause | Not in paper protocol |

---

## 3. Raw Directory Structure

```
data/raw/LG_HG2/
├── 25degC/                    ← THIS REPORT (36 .mat files)
│   ├── 10-25-18_*.mat         ← characterization (HPPC, C20DisCh, initial charge)
│   ├── 10-28-18_*.mat         ← characterization + test drive cycles (UDDS, HWFET)
│   ├── 10-29-18_*.mat         ← test drive cycles (LA92, US06) + Mixed1–2
│   ├── 10-30-18_*.mat         ← Mixed3–6 + characterization (Cap_1C)
│   ├── 10-31-18_*.mat         ← Mixed7–8 + PausCycl
│   └── *.csv                  ← CSV duplicates of .mat (excluded)
├── 0degC/
├── 10degC/
├── 40degC/
├── n10degC/
├── n20degC/
├── Readme file - Description of Experimental Tests.txt
└── Technical Information and Experimental Test Results for LG 18650HG2.pdf
```

---

## 4. Raw File Inventory (25°C)

| Filename | n_rows | duration_s | Profile | Split |
|---|---|---|---|---|
| `10-25-18_04.09 549_Charge.mat` | 857 | 283,969.7 | Charge | exclude |
| `10-25-18_07.39 549_HPPC_25degC_LGHG2.mat` | 48,048 | 80,202.5 | HPPC | exclude |
| `10-26-18_09.29 549_C20DisCh_25degC_LGHG2.mat` | 2,423 | 145,117.0 | C20DisCh | exclude |
| `10-28-18_01.48 549_Dis_0p5C_25degC_LGHG2.mat` | 111 | 6,549.3 | Dis_0p5C | exclude |
| `10-28-18_07.09 549_Dis_2C_25degC_LGHG2.mat` | 27 | 1,548.8 | Dis_2C | exclude |
| `10-28-18_11.04 551_Charge1.mat` | 130 | 7,541.7 | Charge | exclude |
| `10-28-18_13.10 551_Cap_1C_25degC_LGHG2.mat` | 397 | 3,921.5 | Cap_1C | exclude |
| `10-28-18_14.16 551_Charge2.mat` | 213 | 12,455.7 | Charge | **test** |
| `10-28-18_17.44 551_UDDS_25degC_LGHG2.mat` | 159,646 | 15,965.6 | UDDS | **test** |
| `10-28-18_22.11 551_Charge3.mat` | 210 | 12,349.7 | Charge | **test** |
| `10-29-18_01.36 551_HWFET_25degC_LGHG2.mat` | 6,002 | 600.0 | HWFET | **EXCLUDE (FAIL)** |
| `10-29-18_01.47 551_Charge4.mat` | 130 | 7,544.0 | Charge | **test** |
| `10-29-18_03.53 551_LA92_25degC_LGHG2.mat` | 100,764 | 10,083.6 | LA92 | **test** |
| `10-29-18_06.42 551_Charge5.mat` | 210 | 12,314.3 | Charge | **test** |
| `10-29-18_10.08 551_US06_25degC_LGHG2.mat` | 40,155 | 4,015.3 | US06 | **test** |
| `10-29-18_11.15 551_Charge6.mat` | 210 | 12,308.5 | Charge | **test** |
| `10-29-18_14.41 551_Mixed1_25degC_LGHG2.mat` | 77,227 | 7,722.5 | Mixed1 | **full_train, partial_train** |
| `10-29-18_16.50 551_Charge7.mat` | 210 | 12,323.7 | Charge | **full_train, partial_train** |
| `10-29-18_20.16 551_Mixed2_25degC_LGHG2.mat` | 79,172 | 7,916.9 | Mixed2 | **full_train, partial_train** |
| `10-29-18_22.29 551_Charge8.mat` | 211 | 12,318.0 | Charge | **full_train, partial_train** |
| `10-30-18_01.54 552_Cap_1C_25degC_LGHG2.mat` | 393 | 3,880.9 | Cap_1C | exclude |
| `10-30-18_03.00 552_Charge9.mat` | 213 | 12,434.4 | Charge | **full_train, partial_train** |
| `10-30-18_06.27 552_Mixed3_25degC_LGHG2.mat` | 73,827 | 7,382.5 | Mixed3 | **full_train, partial_train** |
| `10-30-18_08.31 552_Charge10.mat` | 209 | 12,280.2 | Charge | **full_train, partial_train** |
| `10-30-18_11.56 552_Mixed4_25degC_LGHG2.mat` | 80,850 | 8,084.8 | Mixed4 | **full_train** |
| `10-30-18_14.12 552_Charge11.mat` | 209 | 12,270.2 | Charge | **full_train** |
| `10-30-18_17.36 552_Mixed5_25degC_LGHG2.mat` | 72,184 | 7,218.1 | Mixed5 | **full_train** |
| `10-30-18_19.38 552_Charge12.mat` | 209 | 12,273.1 | Charge | **full_train** |
| `10-30-18_23.02 552_Mixed6_25degC_LGHG2.mat` | 78,106 | 7,810.4 | Mixed6 | **full_train** |
| `10-31-18_01.13 552_Charge13.mat` | 209 | 12,260.0 | Charge | **full_train** |
| `10-31-18_04.38 552_Mixed7_25degC_LGHG2.mat` | 73,235 | 7,323.3 | Mixed7 | **full_train** |
| `10-31-18_06.41 552_Charge14.mat` | 209 | 12,276.8 | Charge | **full_train** |
| `10-31-18_10.05 552_Mixed8_25degC_LGHG2.mat` | 85,915 | 8,591.3 | Mixed8 | **full_train** |
| `10-31-18_12.30 552_Charge15.mat` | 209 | 12,284.2 | Charge | **full_train** |
| `10-31-18_15.54 552_PausCycl_25degC_LGHG2.mat` | 63,019 | 6,310.1 | PausCycl | exclude |
| `10-31-18_17.41 552_Charge16.mat` | 128 | 7,360.0 | Charge | exclude |

---

## 5. LG Raw-Name to Paper-Profile Mapping

### 5.1 Key Naming Discrepancy

The Vidal 2019 (ITEC) paper calls training profiles **"Mix1" through "Mix8"** (Table I).
The LG HG2 public dataset uses **"Mixed1" through "Mixed8"** in raw filenames.

This is a naming variant — not a change in content. The mapping is inferred from:
- Count correspondence: 8 Mixed files → 8 Mix profiles
- Chronological correspondence: Mixed1 is earliest, Mixed8 is latest
- README confirmation: "A series of eight drive cycles (mix 1-8) consist of random mix of UDDS, HWFET, LA92, US06"

**Verdict:** `Mixed{N} = Mix{N}` is the most plausible paper-to-dataset mapping, supported by README and count. Not directly stated as "Mixed = Mix" in either source.

| Raw filename profile | Paper profile | Confidence |
|---|---|---|
| `Mixed1` | Mix1 | Inferred (count + README) |
| `Mixed2` | Mix2 | Inferred |
| `Mixed3` | Mix3 | Inferred |
| `Mixed4` | Mix4 | Inferred |
| `Mixed5` | Mix5 | Inferred |
| `Mixed6` | Mix6 | Inferred |
| `Mixed7` | Mix7 | Inferred |
| `Mixed8` | Mix8 | Inferred |
| `UDDS` | UDDS | Direct name match |
| `LA92` | LA92 | Direct name match |
| `US06` | US06 | Direct name match |
| `HWFET` | — | Not in paper test set; excluded as invalid file |
| `Charge` | CC | Semantic match (1C CC-CV charge) |

### 5.2 LG HG2 has More Mix Profiles Than Panasonic

The paper confirms: Panasonic has Mix(1-5), LG HG2 has **Mix(1-8)** — three more mix cycles. This is consistent with the larger LG HG2 nominal capacity (3 Ah vs 2.9 Ah) and longer test session.

---

## 6. Full Training File List (`full_train=1`)

### 6.1 Drive Cycles (full_train)

| Filename | Paper Profile | Duration (min) | Ah end | T mean (°C) |
|---|---|---|---|---|
| `551_Mixed1_25degC_LGHG2.mat` | Mix1 | 128.7 | −2.590 | 24.2 |
| `551_Mixed2_25degC_LGHG2.mat` | Mix2 | 131.9 | −2.590 | 24.3 |
| `552_Mixed3_25degC_LGHG2.mat` | Mix3 | 123.0 | −2.558 | 24.3 |
| `552_Mixed4_25degC_LGHG2.mat` | Mix4 | 134.7 | −2.558 | 24.2 |
| `552_Mixed5_25degC_LGHG2.mat` | Mix5 | 120.3 | −2.558 | 24.2 |
| `552_Mixed6_25degC_LGHG2.mat` | Mix6 | 130.2 | −2.558 | 24.3 |
| `552_Mixed7_25degC_LGHG2.mat` | Mix7 | 122.1 | −2.558 | 24.3 |
| `552_Mixed8_25degC_LGHG2.mat` | Mix8 | 143.2 | −2.558 | 24.2 |

### 6.2 Associated Charge Files (full_train)

| Filename | Role | Ah_max | Split basis |
|---|---|---|---|
| `551_Charge7.mat` | Post-Mixed1 / Pre-Mixed2 | 2.6155 | full_train, partial_train |
| `551_Charge8.mat` | Post-Mixed2 | 2.6144 | full_train, partial_train |
| `552_Charge9.mat` | Pre-Mixed3 (post-Cap_1C, following-cycle fallback) | 2.7086 | full_train, partial_train |
| `552_Charge10.mat` | Post-Mixed3 / Pre-Mixed4 | 2.5869 | full_train, partial_train |
| `552_Charge11.mat` | Post-Mixed4 / Pre-Mixed5 | 2.5825 | full_train only |
| `552_Charge12.mat` | Post-Mixed5 / Pre-Mixed6 | 2.5816 | full_train only |
| `552_Charge13.mat` | Post-Mixed6 / Pre-Mixed7 | 2.5831 | full_train only |
| `552_Charge14.mat` | Post-Mixed7 / Pre-Mixed8 | 2.5823 | full_train only |
| `552_Charge15.mat` | Post-Mixed8 | 2.5879 | full_train only |

**full_train count: 8 drive cycles + 9 charge files = 17 files total**

---

## 7. Partial Training File List (`partial_train=1`)

These are a strict subset of `full_train`.

### 7.1 Drive Cycles (partial_train)

| Filename | Paper Profile |
|---|---|
| `551_Mixed1_25degC_LGHG2.mat` | Mix1 |
| `551_Mixed2_25degC_LGHG2.mat` | Mix2 |
| `552_Mixed3_25degC_LGHG2.mat` | Mix3 |

### 7.2 Associated Charge Files (partial_train)

| Filename | Role |
|---|---|
| `551_Charge7.mat` | Post-Mixed1 |
| `551_Charge8.mat` | Post-Mixed2 |
| `552_Charge9.mat` | Pre-Mixed3 (following-cycle fallback) |
| `552_Charge10.mat` | Post-Mixed3 |

**partial_train count: 3 drive cycles + 4 charge files = 7 files total**

---

## 8. Test File List (`test=1`)

### 8.1 Drive Cycles (test)

| Filename | Profile | Duration (min) | Ah end | T mean (°C) |
|---|---|---|---|---|
| `551_UDDS_25degC_LGHG2.mat` | UDDS | 266.1 | −2.590 | 24.0 |
| `551_LA92_25degC_LGHG2.mat` | LA92 | 168.1 | −2.590 | 24.2 |
| `551_US06_25degC_LGHG2.mat` | US06 | 66.9 | −2.590 | 25.0 |

### 8.2 Associated Charge Files (test)

| Filename | Role | Ah_max | Split basis |
|---|---|---|---|
| `551_Charge2.mat` | Pre-UDDS (post-Cap_1C, following-cycle fallback) | 2.7319 | test |
| `551_Charge3.mat` | Post-UDDS / Pre-HWFET-rest | 2.6330 | test |
| `551_Charge4.mat` | Post-HWFET-rest / Pre-LA92 | 0.0148 | test (tiny top-off) |
| `551_Charge5.mat` | Post-LA92 / Pre-US06 | 2.6092 | test |
| `551_Charge6.mat` | Post-US06 / Pre-Mixed1 | 2.6090 | test |

**test count: 3 drive cycles + 5 charge files = 8 files total**

**Note on Charge6:** This charge (11:15, post-US06) is classified `test=1`. Mixed1 (14:41, training) does NOT load this charge file. Mixed1 uses `soc_anchor_source = "full_charge_protocol"` with `soc_init = 1.0`, based on its Ah_start≈0. No cross-split dependency.

---

## 9. Excluded File List

| Filename | Reason |
|---|---|
| `549_Charge.mat` | Pre-HPPC initial/conditioning charge — not paired with drive cycle |
| `549_HPPC_25degC_LGHG2.mat` | HPPC characterization |
| `549_C20DisCh_25degC_LGHG2.mat` | C/20 discharge characterization |
| `549_Dis_0p5C_25degC_LGHG2.mat` | 0.5C discharge characterization |
| `549_Dis_2C_25degC_LGHG2.mat` | 2C discharge characterization |
| `551_Charge1.mat` | Tiny top-off (0.235A, 0.0144Ah) before Cap_1C; not paired with drive cycle |
| `551_Cap_1C_25degC_LGHG2.mat` | 1C capacity test (first, before UDDS) |
| `551_HWFET_25degC_LGHG2.mat` | **DATA QUALITY FAILURE** — zero-current rest file (see §11) |
| `552_Cap_1C_25degC_LGHG2.mat` | 1C capacity test (second, before Mixed3) |
| `552_PausCycl_25degC_LGHG2.mat` | Post-session pause/rest cycle |
| `552_Charge16.mat` | Post-PausCycl; no following drive cycle; V_end=0.000V (QC warning) |

**exclude count: 11 files**

---

## 10. Charge Chronology and Assignment

### 10.1 Split-Assignment Rule

Each charge file is assigned to the split of the drive cycle that **immediately precedes it** chronologically ("preceding-cycle-wins"). When the immediately preceding drive cycle is excluded (Cap_1C, HWFET), the charge is assigned to the **following non-excluded drive cycle's split** ("following-cycle fallback").

All charge assignments in this manifest are **chronologically resolved under the documented reconstruction rule**. They are not independently confirmed by the paper (which does not specify exact file-level charge pairings).

### 10.2 Complete Charge Sequence

| Charge File | Position | Assigned split | Basis |
|---|---|---|---|
| `549_Charge` | Pre-HPPC | exclude | No preceding drive cycle |
| `551_Charge1` | Post-Dis_2C / Pre-Cap_1C | exclude | Preceded by excluded file; tiny top-off |
| `551_Charge2` | Post-Cap_1C / **Pre-UDDS** | **test** | Cap_1C excluded → following-cycle fallback: UDDS |
| `551_Charge3` | Post-**UDDS** / Pre-HWFET-rest | **test** | UDDS precedes |
| `551_Charge4` | Post-**HWFET-rest** / **Pre-LA92** | **test** | HWFET excluded → fallback: LA92 |
| `551_Charge5` | Post-**LA92** / Pre-US06 | **test** | LA92 precedes |
| `551_Charge6` | Post-**US06** / Pre-Mixed1 | **test** | US06 precedes |
| `551_Charge7` | Post-**Mixed1** / Pre-Mixed2 | **full_train, partial_train** | Mixed1 precedes |
| `551_Charge8` | Post-**Mixed2** / Pre-[Cap_1C] | **full_train, partial_train** | Mixed2 precedes |
| `552_Charge9` | Post-Cap_1C / **Pre-Mixed3** | **full_train, partial_train** | Cap_1C excluded → fallback: Mixed3 |
| `552_Charge10` | Post-**Mixed3** / Pre-Mixed4 | **full_train, partial_train** | Mixed3 precedes |
| `552_Charge11` | Post-**Mixed4** / Pre-Mixed5 | **full_train** | Mixed4 precedes |
| `552_Charge12` | Post-**Mixed5** / Pre-Mixed6 | **full_train** | Mixed5 precedes |
| `552_Charge13` | Post-**Mixed6** / Pre-Mixed7 | **full_train** | Mixed6 precedes |
| `552_Charge14` | Post-**Mixed7** / Pre-Mixed8 | **full_train** | Mixed7 precedes |
| `552_Charge15` | Post-**Mixed8** / Pre-PausCycl | **full_train** | Mixed8 precedes |
| `552_Charge16` | Post-PausCycl | exclude | PausCycl excluded; no following selected drive cycle |

### 10.3 SOC Anchoring for Mixed1

**Mixed1** is the only drive cycle whose immediately preceding charge (Charge6, post-US06) is classified `test=1`. Mixed1 does **not** load Charge6. Instead:

- `soc_anchor_source = "full_charge_protocol"`
- `soc_init = 1.0`
- Evidence: Mixed1 `Ah_start ≈ 0.0000` confirms the battery was fully charged before the run

No cross-split dependency exists for Mixed1.

### 10.4 Charge Ah Counter Reset

Several charge files show `Ah_end = 0` but `Ah_max > 0`. This is a cycler behavior: the Ah counter resets to 0 at the **end** of the recording step. Affected files:

| File | Ah_max | Ah_end | Status |
|---|---|---|---|
| `549_Charge` | 2.8180 | 0.0000 | Counter reset |
| `551_Charge2` | 2.7319 | 0.0000 | Counter reset |
| `551_Charge8` | 2.6144 | 0.0000 | Counter reset |
| `552_Charge9` | 2.7086 | 0.0000 | Counter reset |

The canonical "Ah charged" is `Ah_max`. Terminal V~4.186V and I→0 confirm full CC-CV charge in all cases.

---

## 11. Duplicate / Combined-File Findings

### 11.1 HWFET 25°C — Data Quality Failure

**`10-29-18_01.36 551_HWFET_25degC_LGHG2.mat`**: the public HWFET-labelled file contains no drive-cycle excitation:

| Metric | Value |
|---|---|
| n_rows | 6,002 |
| duration_s | 600.0 (10 minutes) |
| Current min | 0.000 A |
| Current max | 0.000 A |
| Ah_start | 0.0000 |
| Ah_end | 0.0000 |
| Voltage | 4.185 – 4.186 V (near full charge, resting) |
| Temperature mean | 23.8 °C |

**Interpretation:** The public HWFET-labelled file records only a 10-minute rest at full charge voltage — no current was applied. Whether the HWFET test was aborted, not scheduled, or recorded into a different file is unknown. The file is excluded both as invalid data (`qc_status = FAIL`) and per protocol (HWFET not in Vidal 2019 test set).

**Impact:** LA92's pre-cycle charge (Charge4) is a tiny top-off (Ah_max=0.0148Ah) because the HWFET rest caused no discharge. Charge4 is test=1. LA92 confirms SOC_init=1.0 via Ah_start≈0.

### 11.2 CSV Duplicates

All `.mat` files in `data/raw/LG_HG2/25degC/` have corresponding `.csv` files. Per the README: "The original CSV files saved by the cycler are also included, they have data identical to that in the .mat files." CSV files are excluded from the manifest; only `.mat` files are used.

### 11.3 No Combined Multi-Cycle Files

At 25°C there are no combined multi-cycle `.mat` files (each profile is stored in a separate file). This contrasts with some other temperatures in the LG HG2 dataset.

### 11.4 Filename Uniqueness in Selected Set

All 25 selected files (17 full_train + 8 test) have unique filenames and paths. `is_duplicate = 0` for all 36 manifest rows.

**Signal-level duplicate detection was not performed.** No hash comparison or signal-trajectory fingerprinting was applied across files. Uniqueness was verified only at the filename and path level. If signal-level deduplication is required in future, apply hash-based comparison before adapter implementation.

---

## 12. Data Quality Results

### 12.1 All Selected Files Pass QC

All 25 selected files pass the required-field and basic QC checks:

- `scipy.io.loadmat` loads without error ✓
- `meas` struct exists ✓
- Required fields present: `Time`, `TimeStamp`, `Voltage`, `Current`, `Ah`, `Battery_Temp_degC` ✓
- Extra fields present (LG HG2 only): `Wh`, `Power`
- Array lengths consistent ✓
- No NaN in `Voltage`, `Current`, or `Ah` ✓
- `Time` monotonically non-decreasing ✓
- Temperature mean in 22–26°C range for drive cycles ✓

### 12.2 HWFET QC Failure (Excluded)

`551_HWFET_25degC_LGHG2.mat` is the only file that fails QC (`qc_status = FAIL`). It is excluded per §11.1. This failure does not affect the training or test sets.

### 12.3 LG HG2 Schema vs. Panasonic

| Field | Panasonic NCR18650PF | LG HG2 | Notes |
|---|---|---|---|
| `Time` | ✓ | ✓ | Seconds from start |
| `TimeStamp` | ✓ | ✓ | MM/DD/YYYY HH:MM:SS |
| `Voltage` | ✓ | ✓ | Cell terminal voltage (V) |
| `Current` | ✓ | ✓ | Negative = discharge |
| `Ah` | ✓ | ✓ | Ah counter (resets each file) |
| `Battery_Temp_degC` | ✓ | ✓ | Cell case temperature |
| `Wh` | ✗ | ✓ | Extra field in LG HG2 |
| `Power` | ✗ | ✓ | Extra field in LG HG2 |

### 12.4 Sampling Rate

Per the LG HG2 README: "Tests considered more important, such as drive cycles, were saved with a 0.1 second time step."

- Drive cycles: ~10 Hz (median dt ≈ 0.1 s), consistent with paper Table IV ("1Hz") — same discrepancy as Panasonic
- Charge files: lower rate (~58 s/sample average), consistent with README

### 12.5 Charge4 Anomaly

`551_Charge4` (post-HWFET-rest, pre-LA92) is a tiny top-off charge:
- Ah_max = 0.0148 Ah (negligible)
- I_max = 0.238 A (0.08C)
- Duration = 7,544 s (~2.1 hours)

The low charge amount is consistent with the HWFET file containing zero current — the battery was never discharged during the "HWFET" recording, so only a minimal top-off was needed before LA92.

---

## 13. Leakage Checks

| Check | Result |
|---|---|
| No file has `full_train=1` AND `test=1` | **PASS** |
| No file has `partial_train=1` AND `test=1` | **PASS** |
| No file has `exclude=1` AND any selected split flag | **PASS** |
| `partial_train` is a strict subset of `full_train` | **PASS** |
| Mixed1 does not load test-classified Charge6 | **PASS** |
| Count/order consistency (split-flag sums equal total rows) | **PASS** |
| Signal-level duplicate detection | **Not performed** (filename/path uniqueness only) |

**Leakage check: PASS** (verified by script output 2026-07-12).

---

## 14. Paper-Confirmed Facts

(Directly quoted or unambiguously derived from Vidal 2019 ITEC)

1. **LG HG2 capacity**: 3 Ah (Table I)
2. **Training profiles**: Mix(1-8) (Table I)
3. **Test profiles**: UDDS, LA92, US06 (§IV.B; also visible in Table I profile column)
4. **Charges included**: "accompanying charges (CC) for each mixed drive cycle" and "corresponding charges (CC)" for test (§IV.B)
5. **Partial training**: Mix1-3 for Case 3 (§IV.B)
6. **Temperature**: "All tests were performed in a thermal chamber set to 25°C" (§II)
7. **SOC reference**: "amp-hours calculated by the battery cycler" used to compute SOC (§IV.A)
8. **Features**: Voltage, Current, Temperature (V, I, T) (Table III)
9. **HWFET**: Not listed in test profiles (Table I profile column); excluded regardless of data quality
10. **Format**: Cylindrical 18650 (Table I)

---

## 15. Dataset-Confirmed Facts

(Confirmed by direct `.mat` file inspection)

1. **LG HG2 naming**: Files use "Mixed1" through "Mixed8" (not "Mix1-8" as in paper)
2. **Drive cycles at 10 Hz**: n_rows × 0.1 ≈ duration_s for all drive cycle files
3. **Charges at ~58 s/sample**: Consistent with README
4. **All selected drive cycles start at Ah ≈ 0**: Confirms Ah counter reset before each test
5. **All full charges terminate at ~4.186 V with I → 0**: Consistent with 1C CC-CV to 4.2V cutoff
6. **HWFET file contains zero current**: 600.0s rest, not a drive cycle
7. **Charge files have 6 extra-over-Panasonic fields**: `Wh`, `Power` (and `TimeStamp`, etc.)
8. **Ah counter resets at end of some charges**: Ah_end = 0 but Ah_max > 0 in 4 charge files
9. **PausCycl is a rest/top-off**: I_min=0, I_max=0.669A, Ah_end=0.0153, V=4.184–4.200V
10. **Test sequence order**: UDDS → HWFET(rest) → LA92 → US06 (Oct 28–29, 2018)
11. **Training sequence order**: Mixed1 → Mixed2 → Mixed3 → Mixed4 → Mixed5 → Mixed6 → Mixed7 → Mixed8 (Oct 29–31, 2018)
12. **Two 1C capacity tests**: One before UDDS (551_Cap_1C) and one before Mixed3 (552_Cap_1C); both excluded per protocol

---

## 16. Reconstruction Assumptions

1. **Mixed{N} = Mix{N} mapping**: Assumed from count and name correspondence. Not directly quoted from paper or README.

2. **Mixed1 SOC_init=1.0**: Charge6 (test=1) precedes Mixed1. Mixed1 does not load Charge6. SOC_init=1.0 assumed from: (a) paper protocol (drive cycles start after full CC-CV charge), (b) Mixed1 Ah_start=−0.0000 ≈ 0. This assumption is well-supported.

3. **Charge2 as pre-UDDS charge**: The sequence Cap_1C → Charge2 → UDDS is chronologically clear. Cap_1C discharges ~2.7264 Ah; Charge2 recharges Ah_max=2.7319 Ah (restores to full). Charge2 is the operational pre-UDDS CC-CV charge. Classified test via following-cycle fallback (Cap_1C excluded).

4. **Charge9 as pre-Mixed3 charge**: The sequence Charge8 → Cap_1C → Charge9 → Mixed3 is clear from timestamps. Cap_1C (second) discharges ~2.6928 Ah; Charge9 recharges Ah_max=2.7086 Ah. The README explicitly notes: "second 1C discharge test is performed before the Mix3 cycle." Charge9 is the operational pre-Mixed3 CC-CV charge. Classified full_train+partial_train via following-cycle fallback.

5. **HWFET rest file classification**: Zero current, constant voltage, 600s duration — unanimous evidence that this is a rest/pause, not a drive cycle. Excluded.

---

## 17. Unresolved Questions

1. **HWFET excitation absent**: The public HWFET-labelled file at 25°C contains no drive-cycle excitation (zero current, 600s rest). Possible explanations: (a) equipment failure, (b) test aborted at start, (c) HWFET intentionally skipped and a placeholder rest recorded. The file exists but no drive-cycle data is present. This is an **open question** — no authoritative explanation found.

2. **Charge16 V_end=0.000V**: The last row of `552_Charge16.mat` shows Voltage=0.000V. This is physically impossible (cell voltage cannot be 0V under normal conditions). Possible causes: cycler recording error at session end, or data corruption in the last row. The file is excluded; no impact on training or test sets.

3. **Charge2 Ah_end=0**: Whether this is a consistent cycler behavior (reset at end of step) or a data recording anomaly is not confirmed from the README alone. The Ah_max=2.7319 value and V_end=4.186V provide sufficient evidence for a full charge regardless.

4. **Cap_1C first vs. second**: The README states "first 1C discharge test is performed before the UDDS cycle, and the second is performed before the Mix3 cycle." This perfectly matches `551_Cap_1C` (before UDDS) and `552_Cap_1C` (before Mixed3). Both excluded per protocol.

---

## 18. Differences Between Paper and Public Dataset

| Item | This Reconstruction | Vidal et al. (2019 ITEC) |
|---|---|---|
| Training profile names | `Mixed1` through `Mixed8` (LG dataset names) | `Mix1` through `Mix8` (paper names) |
| Mapping basis | Inferred from count + README | Not stated verbatim in paper |
| Number of Mix profiles | 8 | 8 (confirmed match) |
| Test profiles | UDDS, LA92, US06 | UDDS, LA92, US06 (confirmed match) |
| HWFET at 25°C | Public HWFET-labelled file contains no drive-cycle excitation | Not in test set per Table I profile column |
| Charge files with Ah reset | Ah_end=0 for 4 charges (use Ah_max) | Not documented in paper |
| Data rate | Drive: ~10 Hz; Charges: ~0.017 Hz | Paper Table IV: "1Hz" (rounded/resampled) |
| Extra fields vs. Panasonic | `Wh`, `Power` present in LG HG2 | Not specified |

---

## 19. Split Status: PROVISIONAL

This split is **PROVISIONAL** for the following reasons:

1. **Mixed{N} = Mix{N} mapping is inferred, not directly quoted.** The correspondence is supported by count, README, and chronological order, but no source explicitly states `Mixed1 = Mix1`.

2. **HWFET excitation absent — cause unknown.** The public HWFET-labelled file contains no drive-cycle excitation. While this does not affect the paper-defined split (HWFET is not in the test set), it is a dataset quality gap.

3. **Charge file assignment rule is a reconstruction.** The paper does not specify which raw charge files map to which drive cycle sequences. The "preceding-cycle-wins" rule is the most defensible approach, but it is not confirmed from the paper.

The split can be considered **FINAL** once:
- [ ] The Panasonic adapter is implemented and its split verified end-to-end
- [ ] The Mixed{N} → Mix{N} mapping is cross-referenced against any available supplementary materials or code from the original authors
- [ ] The HWFET data quality anomaly is investigated (check HWFET files at other temperatures, or contact original dataset authors)

---

*End of report. Generated 2026-07-12 by `scripts/inspect_lg_hg2_25c_split.py` (read-only inspection). No raw data files were modified.*
