# Vidal Charge Protocol Audit Report

Generated: 2026-07-15T14:51:34Z UTC

## 1. Evidence paths used

- Vidal PDF: `<external Vidal PDF>` (status: NOT_AVAILABLE)
- panasonic_18650pf README: `data/raw/Panasonic_18650PF/Readme file - desc of tests performed.txt`
- lg_hg2 README: `data/raw/LG_HG2/Readme file - Description of Experimental Tests.txt`
- samsung_30t README: `data/raw/Samsung_30T/Readme IN21700-30T.txt`
- panasonic_18650pf manifest: `data/manifests/panasonic_vidal2019_25c_manifest.csv`
- lg_hg2 manifest: `data/manifests/lg_hg2_vidal2019_25c_manifest.csv`
- samsung_30t manifest: `data/manifests/samsung_30t_vidal2019_25c_manifest.csv`
- panasonic_18650pf raw root: `data/raw/Panasonic_18650PF/25degC`
- lg_hg2 raw root: `data/raw/LG_HG2/25degC`
- samsung_30t raw root: `data/raw/Samsung_30T/25degC`

## 2. Path portability

All paths above are repo-relative unless marked `<external Vidal PDF>`.
Pass `--vidal-pdf` to supply the paper PDF explicitly.

## 3. Current v1 label positioning

Current v1 charge SOC is **open_reference_per_record_v1** (per-file normalized audit label):
`SOC_file = Ah / max(Ah)` within each charge file.
It must **not** be treated as formal continuous Vidal SOC (Figure 8).
All KEEP model-input charges require **sequence-aware reconstruction**.

## 4. Sequence-aware label definition

For corresponding charges, future SOC reconstruction must ensure:

1. Start SOC continuous with ownership drive end SOC
2. End SOC ≈ 1.0 at confirmed full charge
3. Do not force SOC start to 0 because Ah counter starts at 0 in-file
4. Do not allow SOC 1→0 jump when Ah counter resets at file end
5. Use effective cumulative Ah within charge, not per-file normalization

Reconstruction metadata is recorded in the review CSV (`expected_soc_start_source`, etc.).

## 5. Charge ownership rule (unchanged)

Default: **selected drive → corresponding/accompanying charge** (Figure 8).
Pause does not break ownership. Barrier-aware lookup stops at charge/characterization.

## 6. Partial is a subset of Full

`partial_train` ⊂ `full_train`. Mix3 → Charge → Mix4 is **partial_subset_boundary**.

## 7. Protocol action vs label action

- **Protocol action**: model input / anchor / excluded
- **Label action**: sequence reconstruction requirement for v1 labels
- Ah reset adds `ah_unwrap_required` but does not change protocol role

## 8. Protocol action counts (37 model_input charges)

- KEEP_MODEL_INPUT: 29
- CHANGE_TO_ANCHOR: 5
- CHANGE_TO_EXCLUDED: 3
- MANUAL_REVIEW: 0

## 9. Label action counts

- REGENERATE_SEQUENCE_AWARE: 26
- REGENERATE_RESET_AND_SEQUENCE_AWARE: 3
- ANCHOR_ONLY_NO_POINTWISE_LABEL: 5
- EXCLUDE_FROM_LABEL_EXPORT: 3
- MANUAL_REVIEW: 0

## 10. Per-dataset breakdown

### panasonic_18650pf (9 model_input charges)
- Protocol KEEP_MODEL_INPUT: 7
- Protocol CHANGE_TO_ANCHOR: 1
- Protocol CHANGE_TO_EXCLUDED: 1
- Protocol MANUAL_REVIEW: 0
- Label REGENERATE_SEQUENCE_AWARE: 7
- Label REGENERATE_RESET_AND_SEQUENCE_AWARE: 0
- Label ANCHOR_ONLY_NO_POINTWISE_LABEL: 1
- Label EXCLUDE_FROM_LABEL_EXPORT: 1

### lg_hg2 (14 model_input charges)
- Protocol KEEP_MODEL_INPUT: 11
- Protocol CHANGE_TO_ANCHOR: 2
- Protocol CHANGE_TO_EXCLUDED: 1
- Protocol MANUAL_REVIEW: 0
- Label REGENERATE_SEQUENCE_AWARE: 10
- Label REGENERATE_RESET_AND_SEQUENCE_AWARE: 1
- Label ANCHOR_ONLY_NO_POINTWISE_LABEL: 2
- Label EXCLUDE_FROM_LABEL_EXPORT: 1

### samsung_30t (14 model_input charges)
- Protocol KEEP_MODEL_INPUT: 11
- Protocol CHANGE_TO_ANCHOR: 2
- Protocol CHANGE_TO_EXCLUDED: 1
- Protocol MANUAL_REVIEW: 0
- Label REGENERATE_SEQUENCE_AWARE: 9
- Label REGENERATE_RESET_AND_SEQUENCE_AWARE: 2
- Label ANCHOR_ONLY_NO_POINTWISE_LABEL: 2
- Label EXCLUDE_FROM_LABEL_EXPORT: 1

## 11. Reset + sequence-aware charges

- `10-29-18_22.29 551_Charge8.mat` (lg_hg2): normal_corresponding_charge / KEEP_MODEL_INPUT / REGENERATE_RESET_AND_SEQUENCE_AWARE
- `03-19-19_06.45 734_Charge8_25degC_IN21700_30T.mat` (samsung_30t): normal_corresponding_charge / KEEP_MODEL_INPUT / REGENERATE_RESET_AND_SEQUENCE_AWARE
- `03-20-19_22.00 740_Charge11_25degC_IN21700_30T.mat` (samsung_30t): normal_corresponding_charge / KEEP_MODEL_INPUT / REGENERATE_RESET_AND_SEQUENCE_AWARE

## 12. Anchor-only charges

- `03-18-17_00.05 3406_Charge1.mat` (panasonic_18650pf): leading_full_charge_anchor / CHANGE_TO_ANCHOR / ANCHOR_ONLY_NO_POINTWISE_LABEL
- `10-28-18_14.16 551_Charge2.mat` (lg_hg2): post_capacity_test_full_recharge_anchor / CHANGE_TO_ANCHOR / ANCHOR_ONLY_NO_POINTWISE_LABEL
- `10-30-18_03.00 552_Charge9.mat` (lg_hg2): post_capacity_test_full_recharge_anchor / CHANGE_TO_ANCHOR / ANCHOR_ONLY_NO_POINTWISE_LABEL
- `03-18-19_14.47 734_Charge2_25degC_IN21700_30T.mat` (samsung_30t): post_capacity_test_full_recharge_anchor / CHANGE_TO_ANCHOR / ANCHOR_ONLY_NO_POINTWISE_LABEL
- `03-20-19_19.37 740_Charge10_25degC_IN21700_30T.mat` (samsung_30t): post_capacity_test_full_recharge_anchor / CHANGE_TO_ANCHOR / ANCHOR_ONLY_NO_POINTWISE_LABEL

## 13. HWFET-owned excluded charges

- `03-20-17_21.35 3416_Charge2.mat` (panasonic_18650pf): excluded_profile_corresponding_charge / CHANGE_TO_EXCLUDED / EXCLUDE_FROM_LABEL_EXPORT
- `10-29-18_01.47 551_Charge4.mat` (lg_hg2): tiny_topoff / CHANGE_TO_EXCLUDED / EXCLUDE_FROM_LABEL_EXPORT
- `03-18-19_20.36 734_Charge4_25degC_IN21700_30T.mat` (samsung_30t): excluded_profile_corresponding_charge / CHANGE_TO_EXCLUDED / EXCLUDE_FROM_LABEL_EXPORT


## 14. Ownership barrier check

No unexpected barrier stops for KEEP corresponding charges.

## 15. Load failures

All model_input charges: load_status == PASS

## 16. Candidate sequence split summary

### panasonic_18650pf
- **full_train**: ordered=9 items, leading_anchors=1, hwfet=0
- **partial_train**: ordered=5 items, leading_anchors=1, hwfet=0
- **test**: ordered=6 items, leading_anchors=0, hwfet=1

### lg_hg2
- **full_train**: ordered=16 items, leading_anchors=1, hwfet=0
- **partial_train**: ordered=6 items, leading_anchors=1, hwfet=0
- **test**: ordered=6 items, leading_anchors=1, hwfet=1

### samsung_30t
- **full_train**: ordered=16 items, leading_anchors=1, hwfet=0
- **partial_train**: ordered=6 items, leading_anchors=1, hwfet=0
- **test**: ordered=6 items, leading_anchors=1, hwfet=1

## Recommended actions — panasonic_18650pf

| filename | owned_by | protocol | label | split_relation | load |
|----------|----------|----------|-------|----------------|------|
| 03-18-17_00.05 3406_Charge1.mat |  | CHANGE_TO_ANCHOR | ANCHOR_ONLY_NO_POINTWISE_LABEL | leading_anchor | PASS |
| 03-19-17_06.31 3406_Charge2.mat | Cycle_2 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |
| 03-19-17_11.59 3406_Charge3.mat | Cycle_3 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | partial_subset_boundary | PASS |
| 03-19-17_17.53 3406_Charge4.mat | Cycle_4 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | true_test_train_boundary | PASS |
| 03-20-17_03.05 3415_Charge1.mat | US06 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |
| 03-20-17_21.35 3416_Charge2.mat | HWFET | CHANGE_TO_EXCLUDED | EXCLUDE_FROM_LABEL_EXPORT | unrelated | PASS |
| 03-21-17_06.44 3416_Charge3.mat | UDDS | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |
| 03-21-17_13.34 3416_Charge4.mat | LA92 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | true_test_train_boundary | PASS |
| 03-21-17_19.43 3416_Charge5.mat | NN | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |

## Recommended actions — lg_hg2

| filename | owned_by | protocol | label | split_relation | load |
|----------|----------|----------|-------|----------------|------|
| 10-28-18_14.16 551_Charge2.mat |  | CHANGE_TO_ANCHOR | ANCHOR_ONLY_NO_POINTWISE_LABEL | leading_anchor | PASS |
| 10-28-18_22.11 551_Charge3.mat | UDDS | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |
| 10-29-18_01.47 551_Charge4.mat | HWFET | CHANGE_TO_EXCLUDED | EXCLUDE_FROM_LABEL_EXPORT | unrelated | PASS |
| 10-29-18_06.42 551_Charge5.mat | LA92 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |
| 10-29-18_11.15 551_Charge6.mat | US06 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | true_test_train_boundary | PASS |
| 10-29-18_16.50 551_Charge7.mat | Mixed1 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |
| 10-29-18_22.29 551_Charge8.mat | Mixed2 | KEEP_MODEL_INPUT | REGENERATE_RESET_AND_SEQUENCE_AWARE | same_protocol_split | PASS |
| 10-30-18_03.00 552_Charge9.mat |  | CHANGE_TO_ANCHOR | ANCHOR_ONLY_NO_POINTWISE_LABEL | leading_anchor | PASS |
| 10-30-18_08.31 552_Charge10.mat | Mixed3 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | partial_subset_boundary | PASS |
| 10-30-18_14.12 552_Charge11.mat | Mixed4 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |
| 10-30-18_19.38 552_Charge12.mat | Mixed5 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |
| 10-31-18_01.13 552_Charge13.mat | Mixed6 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |
| 10-31-18_06.41 552_Charge14.mat | Mixed7 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |
| 10-31-18_12.30 552_Charge15.mat | Mixed8 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |

## Recommended actions — samsung_30t

| filename | owned_by | protocol | label | split_relation | load |
|----------|----------|----------|-------|----------------|------|
| 03-18-19_14.47 734_Charge2_25degC_IN21700_30T.mat |  | CHANGE_TO_ANCHOR | ANCHOR_ONLY_NO_POINTWISE_LABEL | leading_anchor | PASS |
| 03-18-19_18.16 734_Charge3_25degC_IN21700_30T.mat | UDDS | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |
| 03-18-19_20.36 734_Charge4_25degC_IN21700_30T.mat | HWFET | CHANGE_TO_EXCLUDED | EXCLUDE_FROM_LABEL_EXPORT | unrelated | PASS |
| 03-18-19_23.25 734_Charge5_25degC_IN21700_30T.mat | LA92 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |
| 03-19-19_01.36 734_Charge6_25degC_IN21700_30T.mat | US06 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | true_test_train_boundary | PASS |
| 03-19-19_04.07 734_Charge7_25degC_IN21700_30T.mat | Mixed1 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |
| 03-19-19_06.45 734_Charge8_25degC_IN21700_30T.mat | Mixed2 | KEEP_MODEL_INPUT | REGENERATE_RESET_AND_SEQUENCE_AWARE | same_protocol_split | PASS |
| 03-20-19_19.37 740_Charge10_25degC_IN21700_30T.mat |  | CHANGE_TO_ANCHOR | ANCHOR_ONLY_NO_POINTWISE_LABEL | leading_anchor | PASS |
| 03-20-19_22.00 740_Charge11_25degC_IN21700_30T.mat | Mixed3 | KEEP_MODEL_INPUT | REGENERATE_RESET_AND_SEQUENCE_AWARE | partial_subset_boundary | PASS |
| 03-21-19_00.39 740_Charge12_25degC_IN21700_30T.mat | Mixed4 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |
| 03-21-19_03.00 740_Charge13_25degC_IN21700_30T.mat | Mixed5 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |
| 03-21-19_05.35 740_Charge14_25degC_IN21700_30T.mat | Mixed6 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |
| 03-21-19_08.01 740_Charge15_25degC_IN21700_30T.mat | Mixed7 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | same_protocol_split | PASS |
| 03-21-19_10.49 740_Charge16_25degC_IN21700_30T.mat | Mixed8 | KEEP_MODEL_INPUT | REGENERATE_SEQUENCE_AWARE | trailing_after_selected_drive | PASS |
