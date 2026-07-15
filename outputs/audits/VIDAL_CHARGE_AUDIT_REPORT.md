# Vidal Charge Protocol Audit Report

Generated: 2026-07-15T13:57:29Z UTC

## 1. Evidence paths used

- Vidal PDF: `/Users/yue/Zotero/storage/DUPS5B62/Vidal et al. - 2019 - Li-ion Battery State of Charge Estimation Using Long Short-Term Memory Recurrent Neural Network with.pdf`
- panasonic_18650pf README: `/Users/yue/Research/battery-soc-transfer-benchmark/data/raw/Panasonic_18650PF/Readme file - desc of tests performed.txt`
- lg_hg2 README: `/Users/yue/Research/battery-soc-transfer-benchmark/data/raw/LG_HG2/Readme file - Description of Experimental Tests.txt`
- samsung_30t README: `/Users/yue/Research/battery-soc-transfer-benchmark/data/raw/Samsung_30T/Readme IN21700-30T.txt`
- panasonic_18650pf manifest: `/Users/yue/Research/battery-soc-transfer-benchmark/data/manifests/panasonic_vidal2019_25c_manifest.csv`
- lg_hg2 manifest: `/Users/yue/Research/battery-soc-transfer-benchmark/data/manifests/lg_hg2_vidal2019_25c_manifest.csv`
- samsung_30t manifest: `/Users/yue/Research/battery-soc-transfer-benchmark/data/manifests/samsung_30t_vidal2019_25c_manifest.csv`
- panasonic_18650pf raw root: `/Users/yue/Research/battery-soc-transfer-benchmark/data/raw/Panasonic_18650PF/25degC`
- lg_hg2 raw root: `/Users/yue/Research/battery-soc-transfer-benchmark/data/raw/LG_HG2/25degC`
- samsung_30t raw root: `/Users/yue/Research/battery-soc-transfer-benchmark/data/raw/Samsung_30T/25degC`

## 2. Vidal paper protocol summary

Table I (p.1): Each dataset lists Mix profiles, UDDS/LA92/US06, and CC.
Section IV.B (p.4): FULL training = all Mixed drive cycles + **accompanying charges (CC)**.
Test dataset = UDDS, LA92, US06 + **corresponding charges (CC)**.
Table V (p.4): Case 1 FULL training; Case 3 PARTIAL Mix 1-3 (+ Samsung 48G Mix 1,2,6).
Section IV.A: SOC reference from **cycler-calculated Ah** during training.
Figure 8 (p.5): Samsung T30 example shows drive + charge in continuous input (Case 2 TL).

## 3–5. README summaries

**Panasonic:** Nine drive cycles (Cycle1-4, US06, HWFET, UDDS, LA92, NN); charged after each test; Ah reset before each segment; filename sort for pairing.

**LG HG2:** UDDS→HWFET→LA92→US06→Mix1-8; two Cap_1C before UDDS and Mix3; charged after each test.

**Samsung 30T:** 16 charges; Charge1 initial top-up before Cap_1C; charges before/after every drive; two Cap_1C before UDDS and Mix3.

## 6. Four-way consistency

| Source | Agreement |
|--------|-----------|
| Paper vs README | Both expect CC with mix/test drives; README adds HWFET/Cap_1C not in paper test set |
| README vs chronology | File timestamps match README order for all three datasets |
| Chronology vs manifest | Pairing mostly confirmed; LG Charge4 tiny top-off and Samsung Charge6 cross-split flagged |
| Manifest vs paper | Selected charges marked model_input align with accompanying/corresponding CC intent; exceptions below |

## 7. Chronology summary

- **panasonic_18650pf**: 61 files; timestamp conflicts=0; time regressions=0
- **lg_hg2**: 36 files; timestamp conflicts=0; time regressions=0
- **samsung_30t**: 34 files; timestamp conflicts=0; time regressions=0

## 8–14. Charge counts

### panasonic_18650pf
- Total charges: 22
- model_input charges: 9
- KEEP_MODEL_INPUT: 5
- CHANGE_TO_ANCHOR: 0
- CHANGE_TO_EXCLUDED: 0
- MANUAL_REVIEW: 4
- proposed_charge_role: excluded_profile_related=11, normal_intercycle_charge=4, cross_split_bridge=3, pre_capacity_test_topup=2, leading_full_charge_anchor=1, trailing_charge=1

### lg_hg2
- Total charges: 17
- model_input charges: 14
- KEEP_MODEL_INPUT: 7
- CHANGE_TO_ANCHOR: 0
- CHANGE_TO_EXCLUDED: 1
- MANUAL_REVIEW: 6
- proposed_charge_role: normal_intercycle_charge=8, pre_capacity_test_topup=2, post_capacity_test_full_recharge=2, cross_split_bridge=2, excluded_profile_related=1, tiny_topoff=1, ambiguous=1

### samsung_30t
- Total charges: 16
- model_input charges: 14
- KEEP_MODEL_INPUT: 8
- CHANGE_TO_ANCHOR: 0
- CHANGE_TO_EXCLUDED: 0
- MANUAL_REVIEW: 6
- proposed_charge_role: normal_intercycle_charge=9, pre_capacity_test_topup=2, cross_split_bridge=2, ambiguous=1, post_capacity_test_full_recharge=1, trailing_charge=1

## 15. Tiny top-offs

- `04-27-17_08.52 3541_Charge1a.mat` (panasonic_18650pf): manifest excluded
- `07-22-17_22.09 4020_Charge1.mat` (panasonic_18650pf): manifest excluded
- `10-28-18_11.04 551_Charge1.mat` (lg_hg2): manifest excluded
- `10-29-18_01.47 551_Charge4.mat` (lg_hg2): charge_ah_fraction=0.005 <= 0.05; manifest model_input conflicts with charge role
- `10-31-18_17.41 552_Charge16.mat` (lg_hg2): manifest excluded
- `03-18-19_13.03 734_Charge1_25degC_IN21700_30T.mat` (samsung_30t): manifest excluded
- `03-20-19_17.52 740_Charge9_25degC_IN21700_30T.mat` (samsung_30t): manifest excluded

## 16. Ah reset charges

- `10-25-18_04.09 549_Charge.mat` (lg_hg2): manifest excluded
- `10-28-18_14.16 551_Charge2.mat` (lg_hg2): Post Cap_1C full CC-CV recharge before UDDS; Ah counter reset detected
- `10-29-18_22.29 551_Charge8.mat` (lg_hg2): Between selected drives; Ah counter reset detected
- `10-30-18_03.00 552_Charge9.mat` (lg_hg2): Post Cap_1C full CC-CV recharge before UDDS; Ah counter reset detected
- `03-18-19_14.47 734_Charge2_25degC_IN21700_30T.mat` (samsung_30t): Ah counter reset detected
- `03-19-19_06.45 734_Charge8_25degC_IN21700_30T.mat` (samsung_30t): Ah counter reset detected
- `03-20-19_17.52 740_Charge9_25degC_IN21700_30T.mat` (samsung_30t): manifest excluded
- `03-20-19_19.37 740_Charge10_25degC_IN21700_30T.mat` (samsung_30t): Post inferred Cap_1C before Mixed3; Ah counter reset detected
- `03-20-19_22.00 740_Charge11_25degC_IN21700_30T.mat` (samsung_30t): bridges splits {'full_train', 'partial_train'} -> {'full_train'}; Ah counter reset detected

## 17. Capacity-test-related

- `04-27-17_08.52 3541_Charge1a.mat` (panasonic_18650pf): manifest excluded
- `07-22-17_22.09 4020_Charge1.mat` (panasonic_18650pf): manifest excluded
- `10-28-18_11.04 551_Charge1.mat` (lg_hg2): manifest excluded
- `10-28-18_14.16 551_Charge2.mat` (lg_hg2): Post Cap_1C full CC-CV recharge before UDDS; Ah counter reset detected
- `10-30-18_03.00 552_Charge9.mat` (lg_hg2): Post Cap_1C full CC-CV recharge before UDDS; Ah counter reset detected
- `10-31-18_17.41 552_Charge16.mat` (lg_hg2): manifest excluded
- `03-18-19_13.03 734_Charge1_25degC_IN21700_30T.mat` (samsung_30t): manifest excluded
- `03-20-19_17.52 740_Charge9_25degC_IN21700_30T.mat` (samsung_30t): manifest excluded
- `03-20-19_19.37 740_Charge10_25degC_IN21700_30T.mat` (samsung_30t): Post inferred Cap_1C before Mixed3; Ah counter reset detected

## 18. Excluded-profile-related

- `03-20-17_08.04 3415_Charge2.mat` (panasonic_18650pf): manifest excluded
- `03-09-17_15.13 3349_Charge1.mat` (panasonic_18650pf): manifest excluded
- `03-09-17_19.03 3349_Charge2.mat` (panasonic_18650pf): manifest excluded
- `03-09-17_21.57 3349_ChargeRp.mat` (panasonic_18650pf): manifest excluded
- `03-11-17_00.39 3349_Charge3.mat` (panasonic_18650pf): manifest excluded
- `03-28-17_15.29 3423_Charge1.mat` (panasonic_18650pf): manifest excluded
- `03-12-17_11.58 3390_Charge_2.mat` (panasonic_18650pf): manifest excluded
- `04-29-17_13.07 3541_Charge2a.mat` (panasonic_18650pf): manifest excluded
- `07-22-17_23.40 4020_Charge2.mat` (panasonic_18650pf): manifest excluded
- `07-23-17_02.36 4020_ChargeRp.mat` (panasonic_18650pf): manifest excluded
- `07-24-17_07.55 4020_Charge3.mat` (panasonic_18650pf): manifest excluded
- `10-25-18_04.09 549_Charge.mat` (lg_hg2): manifest excluded

## 19. Cross-split charges

- `03-19-17_11.59 3406_Charge3.mat` (panasonic_18650pf): Between selected drives 03-19-17_09.07 25degC_Cycle_3_Pan18650PF -> 03-19-17_14.31 25degC_Cycle_4_Pan18650PF; bridges splits {'full_train', 'partial_train'} -> {'full_train'}
- `03-19-17_17.53 3406_Charge4.mat` (panasonic_18650pf): Between selected drives 03-19-17_14.31 25degC_Cycle_4_Pan18650PF -> 03-20-17_01.43 25degC_US06_Pan18650PF; bridges splits {'full_train'} -> {'test'}
- `03-21-17_13.34 3416_Charge4.mat` (panasonic_18650pf): Between selected drives 03-21-17_09.38 25degC_LA92_Pan18650PF -> 03-21-17_16.27 25degC_NN_Pan18650PF; bridges splits {'test'} -> {'full_train'}
- `10-29-18_11.15 551_Charge6.mat` (lg_hg2): Between selected drives; bridges splits {'test'} -> {'full_train', 'partial_train'}
- `10-30-18_08.31 552_Charge10.mat` (lg_hg2): Between selected drives; bridges splits {'full_train', 'partial_train'} -> {'full_train'}
- `03-19-19_01.36 734_Charge6_25degC_IN21700_30T.mat` (samsung_30t): bridges splits {'test'} -> {'full_train', 'partial_train'}
- `03-20-19_22.00 740_Charge11_25degC_IN21700_30T.mat` (samsung_30t): bridges splits {'full_train', 'partial_train'} -> {'full_train'}; Ah counter reset detected

## 20. Likely mis-tagged model_input charges

- `03-19-17_11.59 3406_Charge3.mat`: MANUAL_REVIEW — Between selected drives 03-19-17_09.07 25degC_Cycle_3_Pan18650PF -> 03-19-17_14.31 25degC_Cycle_4_Pan18650PF; bridges splits {'full_train', 'partial_train'} -> {'full_train'}
- `03-19-17_17.53 3406_Charge4.mat`: MANUAL_REVIEW — Between selected drives 03-19-17_14.31 25degC_Cycle_4_Pan18650PF -> 03-20-17_01.43 25degC_US06_Pan18650PF; bridges splits {'full_train'} -> {'test'}
- `03-21-17_13.34 3416_Charge4.mat`: MANUAL_REVIEW — Between selected drives 03-21-17_09.38 25degC_LA92_Pan18650PF -> 03-21-17_16.27 25degC_NN_Pan18650PF; bridges splits {'test'} -> {'full_train'}
- `03-21-17_19.43 3416_Charge5.mat`: MANUAL_REVIEW — Trailing charge after last training mix — paper silent on inclusion
- `10-28-18_14.16 551_Charge2.mat`: MANUAL_REVIEW — Ah reset may invalidate SOC label while sequence membership is plausible
- `10-29-18_01.47 551_Charge4.mat`: CHANGE_TO_EXCLUDED — Role is top-up/tiny but marked model_input
- `10-29-18_11.15 551_Charge6.mat`: MANUAL_REVIEW — Between selected drives; bridges splits {'test'} -> {'full_train', 'partial_train'}
- `10-29-18_22.29 551_Charge8.mat`: MANUAL_REVIEW — Ah reset may invalidate SOC label while sequence membership is plausible
- `10-30-18_03.00 552_Charge9.mat`: MANUAL_REVIEW — Ah reset may invalidate SOC label while sequence membership is plausible
- `10-30-18_08.31 552_Charge10.mat`: MANUAL_REVIEW — Between selected drives; bridges splits {'full_train', 'partial_train'} -> {'full_train'}
- `10-31-18_12.30 552_Charge15.mat`: MANUAL_REVIEW — Chronology: after Mixed8, before PausCycl
- `03-18-19_14.47 734_Charge2_25degC_IN21700_30T.mat`: MANUAL_REVIEW — Ah reset may invalidate SOC label while sequence membership is plausible
- `03-19-19_01.36 734_Charge6_25degC_IN21700_30T.mat`: MANUAL_REVIEW — bridges splits {'test'} -> {'full_train', 'partial_train'}
- `03-19-19_06.45 734_Charge8_25degC_IN21700_30T.mat`: MANUAL_REVIEW — Ah reset may invalidate SOC label while sequence membership is plausible
- `03-20-19_19.37 740_Charge10_25degC_IN21700_30T.mat`: MANUAL_REVIEW — Ah reset may invalidate SOC label while sequence membership is plausible
- `03-20-19_22.00 740_Charge11_25degC_IN21700_30T.mat`: MANUAL_REVIEW — Ah reset may invalidate SOC label while sequence membership is plausible
- `03-21-19_10.49 740_Charge16_25degC_IN21700_30T.mat`: MANUAL_REVIEW — Trailing charge after last training mix — paper silent on inclusion

## 21–22. Disputed charge timelines

### 03-19-17_11.59 3406_Charge3.mat
- Previous: `Cycle_3` (drive_cycle)
- Next: `Pause` (characterization)
- Evidence: PAPER=True README=False TIMESTAMP=True SIGNAL=False MANIFEST=False

### 03-19-17_17.53 3406_Charge4.mat
- Previous: `Cycle_4` (drive_cycle)
- Next: `Pause` (characterization)
- Evidence: PAPER=True README=False TIMESTAMP=True SIGNAL=False MANIFEST=False

### 03-21-17_13.34 3416_Charge4.mat
- Previous: `LA92` (drive_cycle)
- Next: `Pause` (characterization)
- Evidence: PAPER=True README=False TIMESTAMP=True SIGNAL=False MANIFEST=False

### 03-21-17_19.43 3416_Charge5.mat
- Previous: `NN` (drive_cycle)
- Next: `Charge` (charge)
- Evidence: PAPER=False README=False TIMESTAMP=True SIGNAL=False MANIFEST=True

### 10-28-18_14.16 551_Charge2.mat
- Previous: `Cap_1C` (characterization)
- Next: `UDDS` (drive_cycle)
- Evidence: PAPER=False README=True TIMESTAMP=True SIGNAL=True MANIFEST=False

### 10-29-18_01.47 551_Charge4.mat
- Previous: `HWFET` (drive_cycle)
- Next: `LA92` (drive_cycle)
- Evidence: PAPER=False README=True TIMESTAMP=False SIGNAL=True MANIFEST=True

### 10-29-18_11.15 551_Charge6.mat
- Previous: `US06` (drive_cycle)
- Next: `Mixed1` (drive_cycle)
- Evidence: PAPER=True README=False TIMESTAMP=True SIGNAL=False MANIFEST=False

### 10-29-18_22.29 551_Charge8.mat
- Previous: `Mixed2` (drive_cycle)
- Next: `Cap_1C` (characterization)
- Evidence: PAPER=True README=False TIMESTAMP=True SIGNAL=True MANIFEST=False

### 10-30-18_03.00 552_Charge9.mat
- Previous: `Cap_1C` (characterization)
- Next: `Mixed3` (drive_cycle)
- Evidence: PAPER=False README=True TIMESTAMP=True SIGNAL=True MANIFEST=False

### 10-30-18_08.31 552_Charge10.mat
- Previous: `Mixed3` (drive_cycle)
- Next: `Mixed4` (drive_cycle)
- Evidence: PAPER=True README=False TIMESTAMP=True SIGNAL=False MANIFEST=False

### 10-31-18_12.30 552_Charge15.mat
- Previous: `Mixed8` (drive_cycle)
- Next: `PausCycl` (pause)
- Evidence: PAPER=False README=False TIMESTAMP=False SIGNAL=False MANIFEST=False

### 03-18-19_14.47 734_Charge2_25degC_IN21700_30T.mat
- Previous: `Cap_1C` (characterization)
- Next: `UDDS` (drive_cycle)
- Evidence: PAPER=False README=False TIMESTAMP=False SIGNAL=True MANIFEST=False

### 03-19-19_01.36 734_Charge6_25degC_IN21700_30T.mat
- Previous: `US06` (drive_cycle)
- Next: `Mixed1` (drive_cycle)
- Evidence: PAPER=True README=False TIMESTAMP=True SIGNAL=False MANIFEST=False

### 03-19-19_06.45 734_Charge8_25degC_IN21700_30T.mat
- Previous: `Mixed2` (drive_cycle)
- Next: `Charge9` (charge)
- Evidence: PAPER=True README=False TIMESTAMP=True SIGNAL=True MANIFEST=False

### 03-20-19_19.37 740_Charge10_25degC_IN21700_30T.mat
- Previous: `Charge9` (charge)
- Next: `Mixed3` (drive_cycle)
- Evidence: PAPER=False README=True TIMESTAMP=True SIGNAL=True MANIFEST=False

### 03-20-19_22.00 740_Charge11_25degC_IN21700_30T.mat
- Previous: `Mixed3` (drive_cycle)
- Next: `Mixed4` (drive_cycle)
- Evidence: PAPER=True README=False TIMESTAMP=True SIGNAL=True MANIFEST=False

### 03-21-19_10.49 740_Charge16_25degC_IN21700_30T.mat
- Previous: `Mixed8` (drive_cycle)
- Next: `PausCycl` (other)
- Evidence: PAPER=False README=False TIMESTAMP=True SIGNAL=False MANIFEST=False

## Recommended actions — panasonic_18650pf

| filename | current role | splits | prev | next | proposed | action | reason | confidence |
|----------|--------------|--------|------|------|----------|--------|--------|------------|
| 03-18-17_00.05 3406_Charge1.mat | model_input | full_train,partial_train | Charge | Pause | leading_full_charge_anchor | KEEP_MODEL_INPUT | Vidal FULL training includes accompanying CC before first Mi | high |
| 03-19-17_06.31 3406_Charge2.mat | model_input | full_train,partial_train | Cycle_2 | Pause | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 03-19-17_11.59 3406_Charge3.mat | model_input | full_train,partial_train | Cycle_3 | Pause | cross_split_bridge | MANUAL_REVIEW | Between selected drives 03-19-17_09.07 25degC_Cycle_3_Pan186 | high |
| 03-19-17_17.53 3406_Charge4.mat | model_input | full_train | Cycle_4 | Pause | cross_split_bridge | MANUAL_REVIEW | Between selected drives 03-19-17_14.31 25degC_Cycle_4_Pan186 | high |
| 03-20-17_03.05 3415_Charge1.mat | model_input | test | US06 | Pause | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 03-20-17_21.35 3416_Charge2.mat | model_input | test | HWFET | Pause | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 03-21-17_06.44 3416_Charge3.mat | model_input | test | UDDS | Pause | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 03-21-17_13.34 3416_Charge4.mat | model_input | test | LA92 | Pause | cross_split_bridge | MANUAL_REVIEW | Between selected drives 03-21-17_09.38 25degC_LA92_Pan18650P | high |
| 03-21-17_19.43 3416_Charge5.mat | model_input | full_train | NN | Charge | trailing_charge | MANUAL_REVIEW | Trailing charge after last training mix — paper silent on in | medium |

## Recommended actions — lg_hg2

| filename | current role | splits | prev | next | proposed | action | reason | confidence |
|----------|--------------|--------|------|------|----------|--------|--------|------------|
| 10-28-18_14.16 551_Charge2.mat | model_input | test | Cap_1C | UDDS | post_capacity_test_full_recharge | MANUAL_REVIEW | Ah reset may invalidate SOC label while sequence membership  | medium |
| 10-28-18_22.11 551_Charge3.mat | model_input | test | UDDS | HWFET | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 10-29-18_01.47 551_Charge4.mat | model_input | test | HWFET | LA92 | tiny_topoff | CHANGE_TO_EXCLUDED | Role is top-up/tiny but marked model_input | high |
| 10-29-18_06.42 551_Charge5.mat | model_input | test | LA92 | US06 | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 10-29-18_11.15 551_Charge6.mat | model_input | test | US06 | Mixed1 | cross_split_bridge | MANUAL_REVIEW | Between selected drives; bridges splits {'test'} -> {'full_t | high |
| 10-29-18_16.50 551_Charge7.mat | model_input | full_train,partial_train | Mixed1 | Mixed2 | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 10-29-18_22.29 551_Charge8.mat | model_input | full_train,partial_train | Mixed2 | Cap_1C | normal_intercycle_charge | MANUAL_REVIEW | Ah reset may invalidate SOC label while sequence membership  | high |
| 10-30-18_03.00 552_Charge9.mat | model_input | full_train,partial_train | Cap_1C | Mixed3 | post_capacity_test_full_recharge | MANUAL_REVIEW | Ah reset may invalidate SOC label while sequence membership  | medium |
| 10-30-18_08.31 552_Charge10.mat | model_input | full_train,partial_train | Mixed3 | Mixed4 | cross_split_bridge | MANUAL_REVIEW | Between selected drives; bridges splits {'full_train', 'part | high |
| 10-30-18_14.12 552_Charge11.mat | model_input | full_train | Mixed4 | Mixed5 | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 10-30-18_19.38 552_Charge12.mat | model_input | full_train | Mixed5 | Mixed6 | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 10-31-18_01.13 552_Charge13.mat | model_input | full_train | Mixed6 | Mixed7 | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 10-31-18_06.41 552_Charge14.mat | model_input | full_train | Mixed7 | Mixed8 | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 10-31-18_12.30 552_Charge15.mat | model_input | full_train | Mixed8 | PausCycl | ambiguous | MANUAL_REVIEW | Chronology: after Mixed8, before PausCycl | medium |

## Recommended actions — samsung_30t

| filename | current role | splits | prev | next | proposed | action | reason | confidence |
|----------|--------------|--------|------|------|----------|--------|--------|------------|
| 03-18-19_14.47 734_Charge2_25degC_IN21700_30T.mat | model_input | test | Cap_1C | UDDS | ambiguous | MANUAL_REVIEW | Ah reset may invalidate SOC label while sequence membership  | medium |
| 03-18-19_18.16 734_Charge3_25degC_IN21700_30T.mat | model_input | test | UDDS | HWFET | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 03-18-19_20.36 734_Charge4_25degC_IN21700_30T.mat | model_input | test | HWFET | LA92 | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 03-18-19_23.25 734_Charge5_25degC_IN21700_30T.mat | model_input | test | LA92 | US06 | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 03-19-19_01.36 734_Charge6_25degC_IN21700_30T.mat | model_input | test | US06 | Mixed1 | cross_split_bridge | MANUAL_REVIEW | bridges splits {'test'} -> {'full_train', 'partial_train'} | high |
| 03-19-19_04.07 734_Charge7_25degC_IN21700_30T.mat | model_input | full_train,partial_train | Mixed1 | Mixed2 | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 03-19-19_06.45 734_Charge8_25degC_IN21700_30T.mat | model_input | full_train,partial_train | Mixed2 | Charge9 | normal_intercycle_charge | MANUAL_REVIEW | Ah reset may invalidate SOC label while sequence membership  | high |
| 03-20-19_19.37 740_Charge10_25degC_IN21700_30T.mat | model_input | full_train,partial_train | Charge9 | Mixed3 | post_capacity_test_full_recharge | MANUAL_REVIEW | Ah reset may invalidate SOC label while sequence membership  | medium |
| 03-20-19_22.00 740_Charge11_25degC_IN21700_30T.mat | model_input | full_train,partial_train | Mixed3 | Mixed4 | cross_split_bridge | MANUAL_REVIEW | Ah reset may invalidate SOC label while sequence membership  | high |
| 03-21-19_00.39 740_Charge12_25degC_IN21700_30T.mat | model_input | full_train | Mixed4 | Mixed5 | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 03-21-19_03.00 740_Charge13_25degC_IN21700_30T.mat | model_input | full_train | Mixed5 | Mixed6 | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 03-21-19_05.35 740_Charge14_25degC_IN21700_30T.mat | model_input | full_train | Mixed6 | Mixed7 | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 03-21-19_08.01 740_Charge15_25degC_IN21700_30T.mat | model_input | full_train | Mixed7 | Mixed8 | normal_intercycle_charge | KEEP_MODEL_INPUT | Accompanying/corresponding CC per Vidal Section IV.B | high |
| 03-21-19_10.49 740_Charge16_25degC_IN21700_30T.mat | model_input | full_train | Mixed8 | PausCycl | trailing_charge | MANUAL_REVIEW | Trailing charge after last training mix — paper silent on in | medium |
