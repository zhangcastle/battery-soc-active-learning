# Vidal Charge Protocol Audit Report

Generated: 2026-07-15T14:30:55Z UTC

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

## 2. Charge ownership rule

Default: **selected drive → corresponding/accompanying charge** (Figure 8: UDDS → CC → LA92 → CC → US06).
Pause files do not break ownership. Cap_1C/characterization before first selected drive → leading anchor.
Do not assign charge to the *following* drive.

## 3. Partial is a subset of Full

`partial_train` ⊂ `full_train`. Mix3 → Charge → Mix4 is **partial_subset_boundary**, not cross-split conflict.

## 4. Protocol action vs label action

- **Protocol action**: should this file be model input, anchor-only, or excluded from Vidal protocol?
- **Label action**: given v1 per-file Ah normalization, is the SOC label OK, reset-aware, or sequence-aware?
- Ah reset affects **label action only**, not protocol role.

## 5. HWFET-owned charge exclusions

Vidal test set = UDDS, LA92, US06 + corresponding CC. HWFET excluded → its charges → CHANGE_TO_EXCLUDED.

## 6. Leading anchor handling

Pre-first-drive and post-Cap_1C full recharges → CHANGE_TO_ANCHOR; stored in `leading_anchors`, not ordered sequence.

## 7. Trailing corresponding charge handling

Charge after last selected drive still **corresponds to that drive** → KEEP_MODEL_INPUT if profile is in Vidal set.

## 8. Vidal paper protocol summary

Section IV.B: FULL = Mix + accompanying CC; TEST = UDDS/LA92/US06 + corresponding CC.

## 9. Chronology summary

- **panasonic_18650pf**: 61 files; timestamp conflicts=0
- **lg_hg2**: 36 files; timestamp conflicts=0
- **samsung_30t**: 34 files; timestamp conflicts=0

## 10. Protocol action counts (37 model_input charges)

- KEEP_MODEL_INPUT: 29
- CHANGE_TO_ANCHOR: 5
- CHANGE_TO_EXCLUDED: 3
- MANUAL_REVIEW: 0

## 11. Label action counts

- LABEL_OK: 26
- REGENERATE_RESET_AWARE: 3
- REGENERATE_SEQUENCE_AWARE: 5
- EXCLUDE_FROM_LABEL_EXPORT: 3
- MANUAL_REVIEW: 0

## 12. Per-dataset breakdown

### panasonic_18650pf (9 model_input charges)
- Protocol KEEP_MODEL_INPUT: 7
- Protocol CHANGE_TO_ANCHOR: 1
- Protocol CHANGE_TO_EXCLUDED: 1
- Protocol MANUAL_REVIEW: 0
- Label REGENERATE_RESET_AWARE: 0
- Label REGENERATE_SEQUENCE_AWARE: 1
- Label EXCLUDE_FROM_LABEL_EXPORT: 1

### lg_hg2 (14 model_input charges)
- Protocol KEEP_MODEL_INPUT: 11
- Protocol CHANGE_TO_ANCHOR: 2
- Protocol CHANGE_TO_EXCLUDED: 1
- Protocol MANUAL_REVIEW: 0
- Label REGENERATE_RESET_AWARE: 1
- Label REGENERATE_SEQUENCE_AWARE: 2
- Label EXCLUDE_FROM_LABEL_EXPORT: 1

### samsung_30t (14 model_input charges)
- Protocol KEEP_MODEL_INPUT: 11
- Protocol CHANGE_TO_ANCHOR: 2
- Protocol CHANGE_TO_EXCLUDED: 1
- Protocol MANUAL_REVIEW: 0
- Label REGENERATE_RESET_AWARE: 2
- Label REGENERATE_SEQUENCE_AWARE: 2
- Label EXCLUDE_FROM_LABEL_EXPORT: 1

## 13. HWFET-owned charges

- `03-20-17_21.35 3416_Charge2.mat` (panasonic_18650pf): excluded_profile_corresponding_charge / CHANGE_TO_EXCLUDED
- `10-29-18_01.47 551_Charge4.mat` (lg_hg2): tiny_topoff / CHANGE_TO_EXCLUDED
- `03-18-19_20.36 734_Charge4_25degC_IN21700_30T.mat` (samsung_30t): excluded_profile_corresponding_charge / CHANGE_TO_EXCLUDED

## 14. Leading anchors

- `03-18-17_00.05 3406_Charge1.mat` (panasonic_18650pf): leading_full_charge_anchor / CHANGE_TO_ANCHOR
- `10-28-18_14.16 551_Charge2.mat` (lg_hg2): post_capacity_test_full_recharge_anchor / CHANGE_TO_ANCHOR
- `10-30-18_03.00 552_Charge9.mat` (lg_hg2): post_capacity_test_full_recharge_anchor / CHANGE_TO_ANCHOR
- `03-18-19_14.47 734_Charge2_25degC_IN21700_30T.mat` (samsung_30t): post_capacity_test_full_recharge_anchor / CHANGE_TO_ANCHOR
- `03-20-19_19.37 740_Charge10_25degC_IN21700_30T.mat` (samsung_30t): post_capacity_test_full_recharge_anchor / CHANGE_TO_ANCHOR

## 15. Trailing corresponding charges

- `03-21-17_19.43 3416_Charge5.mat` (panasonic_18650pf): trailing_corresponding_charge / KEEP_MODEL_INPUT
- `10-31-18_12.30 552_Charge15.mat` (lg_hg2): trailing_corresponding_charge / KEEP_MODEL_INPUT
- `03-21-19_10.49 740_Charge16_25degC_IN21700_30T.mat` (samsung_30t): trailing_corresponding_charge / KEEP_MODEL_INPUT

## 16. Reset-aware label charges

- `10-29-18_22.29 551_Charge8.mat` (lg_hg2): normal_corresponding_charge / KEEP_MODEL_INPUT
- `03-19-19_06.45 734_Charge8_25degC_IN21700_30T.mat` (samsung_30t): normal_corresponding_charge / KEEP_MODEL_INPUT
- `03-20-19_22.00 740_Charge11_25degC_IN21700_30T.mat` (samsung_30t): normal_corresponding_charge / KEEP_MODEL_INPUT

## 17. Partial subset boundary charges

- `03-19-17_11.59 3406_Charge3.mat` (panasonic_18650pf): normal_corresponding_charge / KEEP_MODEL_INPUT
- `10-30-18_08.31 552_Charge10.mat` (lg_hg2): normal_corresponding_charge / KEEP_MODEL_INPUT
- `03-20-19_22.00 740_Charge11_25degC_IN21700_30T.mat` (samsung_30t): normal_corresponding_charge / KEEP_MODEL_INPUT


## 18. Changes vs previous audit

- `10-30-18_03.00 552_Charge9.mat`: protocol KEEP_MODEL_INPUT → CHANGE_TO_ANCHOR; role normal_corresponding_charge → post_capacity_test_full_recharge_anchor
- `03-20-19_19.37 740_Charge10_25degC_IN21700_30T.mat`: protocol KEEP_MODEL_INPUT → CHANGE_TO_ANCHOR; role normal_corresponding_charge → post_capacity_test_full_recharge_anchor
## Recommended actions — panasonic_18650pf

| filename | owned_by | role | protocol | label | split_relation |
|----------|----------|------|----------|-------|----------------|
| 03-18-17_00.05 3406_Charge1.mat |  | leading_full_charge_anchor | CHANGE_TO_ANCHOR | REGENERATE_SEQUENCE_AWARE | leading_anchor |
| 03-19-17_06.31 3406_Charge2.mat | Cycle_2 | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | same_protocol_split |
| 03-19-17_11.59 3406_Charge3.mat | Cycle_3 | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | partial_subset_boundary |
| 03-19-17_17.53 3406_Charge4.mat | Cycle_4 | true_test_train_boundary_charge | KEEP_MODEL_INPUT | LABEL_OK | true_test_train_boundary |
| 03-20-17_03.05 3415_Charge1.mat | US06 | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | same_protocol_split |
| 03-20-17_21.35 3416_Charge2.mat | HWFET | excluded_profile_corresponding_charge | CHANGE_TO_EXCLUDED | EXCLUDE_FROM_LABEL_EXPORT | unrelated |
| 03-21-17_06.44 3416_Charge3.mat | UDDS | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | same_protocol_split |
| 03-21-17_13.34 3416_Charge4.mat | LA92 | true_test_train_boundary_charge | KEEP_MODEL_INPUT | LABEL_OK | true_test_train_boundary |
| 03-21-17_19.43 3416_Charge5.mat | NN | trailing_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | trailing_after_selected_drive |

## Recommended actions — lg_hg2

| filename | owned_by | role | protocol | label | split_relation |
|----------|----------|------|----------|-------|----------------|
| 10-28-18_14.16 551_Charge2.mat |  | post_capacity_test_full_recharge_anchor | CHANGE_TO_ANCHOR | REGENERATE_SEQUENCE_AWARE | leading_anchor |
| 10-28-18_22.11 551_Charge3.mat | UDDS | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | same_protocol_split |
| 10-29-18_01.47 551_Charge4.mat | HWFET | tiny_topoff | CHANGE_TO_EXCLUDED | EXCLUDE_FROM_LABEL_EXPORT | unrelated |
| 10-29-18_06.42 551_Charge5.mat | LA92 | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | same_protocol_split |
| 10-29-18_11.15 551_Charge6.mat | US06 | true_test_train_boundary_charge | KEEP_MODEL_INPUT | LABEL_OK | true_test_train_boundary |
| 10-29-18_16.50 551_Charge7.mat | Mixed1 | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | same_protocol_split |
| 10-29-18_22.29 551_Charge8.mat | Mixed2 | normal_corresponding_charge | KEEP_MODEL_INPUT | REGENERATE_RESET_AWARE | same_protocol_split |
| 10-30-18_03.00 552_Charge9.mat |  | post_capacity_test_full_recharge_anchor | CHANGE_TO_ANCHOR | REGENERATE_SEQUENCE_AWARE | leading_anchor |
| 10-30-18_08.31 552_Charge10.mat | Mixed3 | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | partial_subset_boundary |
| 10-30-18_14.12 552_Charge11.mat | Mixed4 | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | same_protocol_split |
| 10-30-18_19.38 552_Charge12.mat | Mixed5 | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | same_protocol_split |
| 10-31-18_01.13 552_Charge13.mat | Mixed6 | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | same_protocol_split |
| 10-31-18_06.41 552_Charge14.mat | Mixed7 | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | same_protocol_split |
| 10-31-18_12.30 552_Charge15.mat | Mixed8 | trailing_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | trailing_after_selected_drive |

## Recommended actions — samsung_30t

| filename | owned_by | role | protocol | label | split_relation |
|----------|----------|------|----------|-------|----------------|
| 03-18-19_14.47 734_Charge2_25degC_IN21700_30T.mat |  | post_capacity_test_full_recharge_anchor | CHANGE_TO_ANCHOR | REGENERATE_SEQUENCE_AWARE | leading_anchor |
| 03-18-19_18.16 734_Charge3_25degC_IN21700_30T.mat | UDDS | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | same_protocol_split |
| 03-18-19_20.36 734_Charge4_25degC_IN21700_30T.mat | HWFET | excluded_profile_corresponding_charge | CHANGE_TO_EXCLUDED | EXCLUDE_FROM_LABEL_EXPORT | unrelated |
| 03-18-19_23.25 734_Charge5_25degC_IN21700_30T.mat | LA92 | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | same_protocol_split |
| 03-19-19_01.36 734_Charge6_25degC_IN21700_30T.mat | US06 | true_test_train_boundary_charge | KEEP_MODEL_INPUT | LABEL_OK | true_test_train_boundary |
| 03-19-19_04.07 734_Charge7_25degC_IN21700_30T.mat | Mixed1 | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | same_protocol_split |
| 03-19-19_06.45 734_Charge8_25degC_IN21700_30T.mat | Mixed2 | normal_corresponding_charge | KEEP_MODEL_INPUT | REGENERATE_RESET_AWARE | same_protocol_split |
| 03-20-19_19.37 740_Charge10_25degC_IN21700_30T.mat |  | post_capacity_test_full_recharge_anchor | CHANGE_TO_ANCHOR | REGENERATE_SEQUENCE_AWARE | leading_anchor |
| 03-20-19_22.00 740_Charge11_25degC_IN21700_30T.mat | Mixed3 | normal_corresponding_charge | KEEP_MODEL_INPUT | REGENERATE_RESET_AWARE | partial_subset_boundary |
| 03-21-19_00.39 740_Charge12_25degC_IN21700_30T.mat | Mixed4 | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | same_protocol_split |
| 03-21-19_03.00 740_Charge13_25degC_IN21700_30T.mat | Mixed5 | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | same_protocol_split |
| 03-21-19_05.35 740_Charge14_25degC_IN21700_30T.mat | Mixed6 | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | same_protocol_split |
| 03-21-19_08.01 740_Charge15_25degC_IN21700_30T.mat | Mixed7 | normal_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | same_protocol_split |
| 03-21-19_10.49 740_Charge16_25degC_IN21700_30T.mat | Mixed8 | trailing_corresponding_charge | KEEP_MODEL_INPUT | LABEL_OK | trailing_after_selected_drive |
