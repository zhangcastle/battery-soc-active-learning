#!/usr/bin/env python3
"""
inspect_panasonic_25c_split.py
Read-only inspection script for Panasonic NCR18650PF 25°C files.

Usage:
    python scripts/inspect_panasonic_25c_split.py

Outputs:
    data/manifests/panasonic_vidal2019_25c_manifest.csv
    docs/panasonic_vidal2019_25c_split_report.md

Does NOT modify any raw data files.
Does NOT compute SOC. Does NOT train a model.
"""

import os
import sys
import csv
import datetime
import traceback
from pathlib import Path

import numpy as np
import scipy.io

# ---------------------------------------------------------------------------
# Repo root (two levels up from this script)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_BASE = REPO_ROOT / "data" / "raw" / "Panasonic_18650PF" / "25degC"
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
DOCS_DIR = REPO_ROOT / "docs"

# ---------------------------------------------------------------------------
# File catalogue: every 25°C .mat file with its known classification.
# Relative subfolder paths within RAW_BASE.
# ---------------------------------------------------------------------------
# fmt: off
FILE_CATALOGUE = [
    # ── Drive cycles ──────────────────────────────────────────────────────
    {
        "subfolder": "Drive cycles",
        "filename": "03-18-17_02.17 25degC_Cycle_1_Pan18650PF.mat",
        "profile": "Cycle_1",
        "profile_number": 1,
        "file_type": "drive_cycle",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 1,
        "partial_train": 1,
        "test": 0,
        "exclude": 0,
    },
    {
        "subfolder": "Drive cycles",
        "filename": "03-19-17_03.25 25degC_Cycle_2_Pan18650PF.mat",
        "profile": "Cycle_2",
        "profile_number": 2,
        "file_type": "drive_cycle",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 1,
        "partial_train": 1,
        "test": 0,
        "exclude": 0,
    },
    {
        "subfolder": "Drive cycles",
        "filename": "03-19-17_09.07 25degC_Cycle_3_Pan18650PF.mat",
        "profile": "Cycle_3",
        "profile_number": 3,
        "file_type": "drive_cycle",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 1,
        "partial_train": 1,
        "test": 0,
        "exclude": 0,
    },
    {
        "subfolder": "Drive cycles",
        "filename": "03-19-17_14.31 25degC_Cycle_4_Pan18650PF.mat",
        "profile": "Cycle_4",
        "profile_number": 4,
        "file_type": "drive_cycle",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 1,
        "partial_train": 0,
        "test": 0,
        "exclude": 0,
    },
    {
        "subfolder": "Drive cycles",
        "filename": "03-20-17_01.43 25degC_US06_Pan18650PF.mat",
        "profile": "US06",
        "profile_number": "",
        "file_type": "drive_cycle",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 1,
        "exclude": 0,
    },
    {
        "subfolder": "Drive cycles",
        "filename": "03-20-17_05.56 25degC_HWFTa_Pan18650PF.mat",
        "profile": "HWFET",
        "profile_number": "",
        "file_type": "drive_cycle",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "HWFET-a: first part of split HWFET session; HWFET excluded per protocol",
    },
    {
        "subfolder": "Drive cycles",
        "filename": "03-20-17_19.27 25degC_HWFTb_Pan18650PF.mat",
        "profile": "HWFET",
        "profile_number": "",
        "file_type": "drive_cycle",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "HWFET-b: second part of split HWFET session; HWFET excluded per protocol",
    },
    {
        "subfolder": "Drive cycles",
        "filename": "03-21-17_00.29 25degC_UDDS_Pan18650PF.mat",
        "profile": "UDDS",
        "profile_number": "",
        "file_type": "drive_cycle",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 1,
        "exclude": 0,
    },
    {
        "subfolder": "Drive cycles",
        "filename": "03-21-17_09.38 25degC_LA92_Pan18650PF.mat",
        "profile": "LA92",
        "profile_number": "",
        "file_type": "drive_cycle",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 1,
        "exclude": 0,
    },
    {
        "subfolder": "Drive cycles",
        "filename": "03-21-17_16.27 25degC_NN_Pan18650PF.mat",
        "profile": "NN",
        "profile_number": 5,
        "file_type": "drive_cycle",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 1,
        "partial_train": 0,
        "test": 0,
        "exclude": 0,
        "notes": "NN = Mix5; Neural Network evaluation cycle composed of portions of US06 and LA92",
    },
    # ── Charges paired with Mix cycles and test drive cycles ──────────────
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-18-17_00.05 3406_Charge1.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        # Inherits from Cycle_1 (full_train, partial_train)
        "full_train": 1,
        "partial_train": 1,
        "test": 0,
        "exclude": 0,
        "notes": "Pre-Cycle_1 charge; gives SOC_init for Cycle_1",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-19-17_06.31 3406_Charge2.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        # After Cycle_2 / Before Cycle_3 — inherits from both (full_train)
        "full_train": 1,
        "partial_train": 1,
        "test": 0,
        "exclude": 0,
        "notes": "Post-Cycle_2 and pre-Cycle_3 charge",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-19-17_11.59 3406_Charge3.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        # After Cycle_3 / Before Cycle_4
        "full_train": 1,
        "partial_train": 1,
        "test": 0,
        "exclude": 0,
        "notes": "Post-Cycle_3 and pre-Cycle_4 charge",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-19-17_17.53 3406_Charge4.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        # After Cycle_4 / Before US06
        "full_train": 1,
        "partial_train": 0,
        "test": 0,
        "exclude": 0,
        "notes": "Post-Cycle_4 charge; classified full_train (Cycle_4 immediately precedes, per split rule); also serves as pre-US06 SOC anchor — see selected_charge_for_sequence in US06 row",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-20-17_03.05 3415_Charge1.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        # After US06 / Before HWFTa
        "full_train": 0,
        "partial_train": 0,
        "test": 1,
        "exclude": 0,
        "notes": "Post-US06 charge; paired with US06 test cycle",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-20-17_08.04 3415_Charge2.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        # After HWFTa — HWFET excluded so charge inherits exclude
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Post-HWFTa charge; HWFET excluded so this charge is also excluded",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-20-17_21.35 3416_Charge2.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        # Before UDDS (and after HWFTb which is excluded)
        "full_train": 0,
        "partial_train": 0,
        "test": 1,
        "exclude": 0,
        "notes": "Pre-UDDS charge; paired with UDDS test cycle",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-21-17_06.44 3416_Charge3.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        # After UDDS / Before LA92
        "full_train": 0,
        "partial_train": 0,
        "test": 1,
        "exclude": 0,
        "notes": "Post-UDDS and pre-LA92 charge",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-21-17_13.34 3416_Charge4.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        # After LA92 (test) / Before NN (full_train).
        # Split rule: charge assigned to split of IMMEDIATELY PRECEDING drive cycle.
        # LA92 is immediately preceding → test=1.
        # 3416_Charge4 also serves as selected_charge_for_sequence for NN; see pairing columns.
        "full_train": 0,
        "partial_train": 0,
        "test": 1,
        "exclude": 0,
        "notes": "Post-LA92 charge; classified test (LA92 immediately precedes, per split rule); NN does not load this file — NN uses full_charge_protocol assumption (soc_init=1.0)",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-21-17_19.43 3416_Charge5.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        # After NN
        "full_train": 1,
        "partial_train": 0,
        "test": 0,
        "exclude": 0,
        "notes": "Post-NN (Mix5) charge",
    },
    # ── EXCLUDED files (still included in manifest for completeness) ───────
    # HPPC
    {
        "subfolder": "5 pulse disch",
        "filename": "03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat",
        "profile": "HPPC",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "HPPC characterization — excluded",
    },
    {
        "subfolder": "5 pulse disch",
        "filename": "03-11-17_10.10 3390_dis5_10p.mat",
        "profile": "HPPC_dis",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "HPPC inter-pulse discharge — excluded",
    },
    # OCV / C20
    {
        "subfolder": "C20 OCV and 1C discharge tests_start_of_tests",
        "filename": "05-08-17_13.26 C20 OCV Test_C20_25dC.mat",
        "profile": "OCV",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "C20 OCV characterization — excluded",
    },
    # 1C discharge tests at start of tests (characterization charges)
    {
        "subfolder": "C20 OCV and 1C discharge tests_start_of_tests",
        "filename": "03-09-17_15.12 3349_TS002973.mat",
        "profile": "Telemetry",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Telemetry/setup file — excluded",
    },
    {
        "subfolder": "C20 OCV and 1C discharge tests_start_of_tests",
        "filename": "03-09-17_15.13 3349_Charge1.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pre-characterization charge — excluded (not paired with any training/test cycle)",
    },
    {
        "subfolder": "C20 OCV and 1C discharge tests_start_of_tests",
        "filename": "03-09-17_17.59 3349_Dis1C_1.mat",
        "profile": "Discharge_1C",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "1C discharge characterization — excluded",
    },
    {
        "subfolder": "C20 OCV and 1C discharge tests_start_of_tests",
        "filename": "03-09-17_17.59 3349_Pause_1.mat",
        "profile": "Pause",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pause/rest file — excluded",
    },
    {
        "subfolder": "C20 OCV and 1C discharge tests_start_of_tests",
        "filename": "03-09-17_19.03 3349_Charge2.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pre-characterization charge — excluded",
    },
    {
        "subfolder": "C20 OCV and 1C discharge tests_start_of_tests",
        "filename": "03-09-17_21.03 3349_Dis1C_Rp.mat",
        "profile": "Discharge_1C",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "1C discharge (repeat) — excluded",
    },
    {
        "subfolder": "C20 OCV and 1C discharge tests_start_of_tests",
        "filename": "03-09-17_21.03 3349_Pause_Rp.mat",
        "profile": "Pause",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pause/rest file — excluded",
    },
    {
        "subfolder": "C20 OCV and 1C discharge tests_start_of_tests",
        "filename": "03-09-17_21.57 3349_ChargeRp.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pre-characterization charge (repeat) — excluded",
    },
    {
        "subfolder": "C20 OCV and 1C discharge tests_start_of_tests",
        "filename": "03-10-17_23.36 3349_Dis1C_2.mat",
        "profile": "Discharge_1C",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "1C discharge characterization — excluded",
    },
    {
        "subfolder": "C20 OCV and 1C discharge tests_start_of_tests",
        "filename": "03-10-17_23.36 3349_Pause_2.mat",
        "profile": "Pause",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pause/rest file — excluded",
    },
    {
        "subfolder": "C20 OCV and 1C discharge tests_start_of_tests",
        "filename": "03-11-17_00.39 3349_Charge3.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pre-characterization charge — excluded",
    },
    {
        "subfolder": "C20 OCV and 1C discharge tests_start_of_tests",
        "filename": "03-28-17_15.29 3423_Charge1.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Charge from 10degC session (file mis-filed or cross-referenced under 25degC); excluded",
    },
    # Charges and Pauses: telemetry, pauses, HPPC, non-relevant charges
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-11-17_08.25 3390_TS003014.mat",
        "profile": "Telemetry",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Telemetry/setup file — excluded",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-12-17_11.58 3390_Charge_2.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Post-HPPC charge (before drive cycles); not paired with any train/test cycle",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-18-17_00.23 3406_Pause1.mat",
        "profile": "Pause",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pause before Cycle_1 — excluded",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-19-17_02.26 3406_Pause2.mat",
        "profile": "Pause",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pause before Cycle_2 (no separate charge file visible between Cycle_1 and Cycle_2) — excluded",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-19-17_08.08 3406_Pause3.mat",
        "profile": "Pause",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pause before Cycle_3 — excluded",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-19-17_13.32 3406_Pause4.mat",
        "profile": "Pause",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pause before Cycle_4 — excluded",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-20-17_00.44 3415_Pause1.mat",
        "profile": "Pause",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pause before US06 — excluded",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-20-17_04.57 3415_Pause2.mat",
        "profile": "Pause",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pause before HWFTa — excluded",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-20-17_18.28 3416_Pause2.mat",
        "profile": "Pause",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pause before HWFTb — excluded",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-20-17_23.30 3416_Pause3.mat",
        "profile": "Pause",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pause before UDDS — excluded",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-21-17_08.39 3416_Pause4.mat",
        "profile": "Pause",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pause before LA92 — excluded",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "03-21-17_15.28 3416_Pause5.mat",
        "profile": "Pause",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pause before NN (Mix5) — excluded",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "04-27-17_08.52 3541_Charge1a.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Pre-HPPC charge (April session) — excluded",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "04-27-17_11.53 3541_dis5_10p.mat",
        "profile": "HPPC_dis",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "HPPC inter-pulse discharge (April) — excluded",
    },
    {
        "subfolder": "Charges and Pauses",
        "filename": "04-29-17_13.07 3541_Charge2a.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "Post-HPPC charge (April session) — excluded",
    },
    # End-of-tests
    {
        "subfolder": "1C discharge tests_end_of_tests",
        "filename": "07-22-17_22.08 4020_TS003628.mat",
        "profile": "Telemetry",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "End-of-tests telemetry — excluded",
    },
    {
        "subfolder": "1C discharge tests_end_of_tests",
        "filename": "07-22-17_22.09 4020_Charge1.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "End-of-tests charge — excluded",
    },
    {
        "subfolder": "1C discharge tests_end_of_tests",
        "filename": "07-22-17_22.44 4020_Dis1C_1.mat",
        "profile": "Discharge_1C",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "End-of-tests 1C discharge — excluded",
    },
    {
        "subfolder": "1C discharge tests_end_of_tests",
        "filename": "07-22-17_22.44 4020_Pause_1.mat",
        "profile": "Pause",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "End-of-tests pause — excluded",
    },
    {
        "subfolder": "1C discharge tests_end_of_tests",
        "filename": "07-22-17_23.40 4020_Charge2.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "End-of-tests charge — excluded",
    },
    {
        "subfolder": "1C discharge tests_end_of_tests",
        "filename": "07-23-17_01.42 4020_Dis1C_Rp.mat",
        "profile": "Discharge_1C",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "End-of-tests 1C discharge repeat — excluded",
    },
    {
        "subfolder": "1C discharge tests_end_of_tests",
        "filename": "07-23-17_01.42 4020_Pause_Rp.mat",
        "profile": "Pause",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "End-of-tests pause — excluded",
    },
    {
        "subfolder": "1C discharge tests_end_of_tests",
        "filename": "07-23-17_02.36 4020_ChargeRp.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "End-of-tests charge repeat — excluded",
    },
    {
        "subfolder": "1C discharge tests_end_of_tests",
        "filename": "07-24-17_07.00 4020_Dis1C_2.mat",
        "profile": "Discharge_1C",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "End-of-tests 1C discharge — excluded",
    },
    {
        "subfolder": "1C discharge tests_end_of_tests",
        "filename": "07-24-17_07.00 4020_Pause_2.mat",
        "profile": "Pause",
        "profile_number": "",
        "file_type": "characterization",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "End-of-tests pause — excluded",
    },
    {
        "subfolder": "1C discharge tests_end_of_tests",
        "filename": "07-24-17_07.55 4020_Charge3.mat",
        "profile": "Charge",
        "profile_number": "",
        "file_type": "charge",
        "is_standalone": 1,
        "is_duplicate": 0,
        "duplicate_of": "",
        "full_train": 0,
        "partial_train": 0,
        "test": 0,
        "exclude": 1,
        "notes": "End-of-tests charge — excluded",
    },
]
# fmt: on

# ---------------------------------------------------------------------------
# Drive-cycle sequence info: previous_charge, next_charge,
# selected_charge_for_sequence, soc_anchor_evidence, pairing_status.
#
# Split-assignment rule (documented here as canonical):
#   Each charge file is assigned to the split of the drive cycle that
#   IMMEDIATELY PRECEDES it in chronological order (preceding cycle wins).
#   When the immediately preceding drive cycle is excluded (HWFET), the
#   charge is assigned to the split of the next non-excluded drive cycle
#   that will use it as its SOC anchor (following-cycle fallback).
#   Pre-session charges (no preceding drive cycle) inherit from the
#   immediately following drive cycle.
# ---------------------------------------------------------------------------
DRIVE_CYCLE_SEQUENCE = {
    "03-18-17_02.17 25degC_Cycle_1_Pan18650PF.mat": {
        "previous_charge": "03-18-17_00.05 3406_Charge1.mat",
        # No charge between Cycle_1 and Cycle_2; next available is Charge2
        "next_charge": "03-19-17_06.31 3406_Charge2.mat",
        "selected_charge_for_sequence": "03-18-17_00.05 3406_Charge1.mat",
        "soc_anchor_evidence": (
            "Preceding charge 3406_Charge1 terminates at 4.200V, 0.00A, "
            "Ah_end=2.670Ah → SOC_init=1.0"
        ),
        "confidence": "high",
        "evidence": (
            "3406_Charge1 (00:05 Mar 18) ends before Cycle_1 (02:17 Mar 18) "
            "via Pause1 (00:23); chronological order confirmed by filename dates"
        ),
        "pairing_status": "confirmed",
    },
    "03-19-17_03.25 25degC_Cycle_2_Pan18650PF.mat": {
        # No charge file found in dataset between Cycle_1 end and Cycle_2 start.
        "previous_charge": "",
        "next_charge": "03-19-17_06.31 3406_Charge2.mat",
        # selected_charge_for_sequence = "" per protocol: pre-charge absent.
        "selected_charge_for_sequence": "",
        "soc_anchor_evidence": (
            "No preceding charge file available; Cycle_2 starts with Ah≈0 "
            "suggesting full charge occurred but file is absent from dataset; "
            "SOC_init assumed 1.0 with low confidence"
        ),
        "confidence": "medium",
        "evidence": (
            "3406_Charge2 starts 3h 6min after Cycle_2; "
            "no pre-Cycle_2 charge file present in Charges and Pauses folder "
            "(24h gap since Cycle_1 with only 3406_Pause2 at 02:26 before Cycle_2); "
            "pre-Cycle_2 charge assumed to have occurred during the 24h gap but file absent"
        ),
        "pairing_status": "unresolved",
    },
    "03-19-17_09.07 25degC_Cycle_3_Pan18650PF.mat": {
        "previous_charge": "03-19-17_06.31 3406_Charge2.mat",
        "next_charge": "03-19-17_11.59 3406_Charge3.mat",
        "selected_charge_for_sequence": "03-19-17_06.31 3406_Charge2.mat",
        "soc_anchor_evidence": (
            "Preceding charge 3406_Charge2 terminates at 4.200V, 0.00A, "
            "Ah_end=2.677Ah → SOC_init=1.0"
        ),
        "confidence": "high",
        "evidence": (
            "3406_Charge2 ends before Cycle_3 (02:36h gap via Pause3 at 08:08); "
            "3406_Charge3 follows Cycle_3 at 11:59; "
            "chronological sequence: Charge2 → Pause3 → Cycle_3 → Charge3"
        ),
        "pairing_status": "confirmed",
    },
    "03-19-17_14.31 25degC_Cycle_4_Pan18650PF.mat": {
        "previous_charge": "03-19-17_11.59 3406_Charge3.mat",
        "next_charge": "03-19-17_17.53 3406_Charge4.mat",
        "selected_charge_for_sequence": "03-19-17_11.59 3406_Charge3.mat",
        "soc_anchor_evidence": (
            "Preceding charge 3406_Charge3 terminates at 4.200V, 0.00A, "
            "Ah_end=2.497Ah → SOC_init=1.0"
        ),
        "confidence": "high",
        "evidence": (
            "3406_Charge3 ends before Cycle_4 (via Pause4 at 13:32); "
            "3406_Charge4 follows Cycle_4 at 17:53; "
            "chronological sequence: Charge3 → Pause4 → Cycle_4 → Charge4"
        ),
        "pairing_status": "confirmed",
    },
    "03-20-17_01.43 25degC_US06_Pan18650PF.mat": {
        "previous_charge": "03-19-17_17.53 3406_Charge4.mat",
        "next_charge": "03-20-17_03.05 3415_Charge1.mat",
        "selected_charge_for_sequence": "03-19-17_17.53 3406_Charge4.mat",
        "soc_anchor_evidence": (
            "Preceding charge 3406_Charge4 terminates at 4.200V, 0.00A, "
            "Ah_end=2.760Ah → SOC_init=1.0"
        ),
        "confidence": "high",
        "evidence": (
            "3406_Charge4 (17:53 Mar 19) precedes US06 (01:43 Mar 20) by ~7.8h; "
            "3415_Pause1 at 00:44 Mar 20 is rest before US06; "
            "3415_Charge1 at 03:05 Mar 20 follows US06 as post-cycle charge"
        ),
        "pairing_status": "confirmed",
    },
    "03-20-17_05.56 25degC_HWFTa_Pan18650PF.mat": {
        "previous_charge": "03-20-17_03.05 3415_Charge1.mat",
        "next_charge": "03-20-17_08.04 3415_Charge2.mat",
        "selected_charge_for_sequence": "",  # excluded drive cycle
        "soc_anchor_evidence": "",
        "confidence": "",
        "evidence": "HWFET excluded per protocol",
        "pairing_status": "",
    },
    "03-20-17_19.27 25degC_HWFTb_Pan18650PF.mat": {
        "previous_charge": "03-20-17_08.04 3415_Charge2.mat",
        "next_charge": "03-20-17_21.35 3416_Charge2.mat",
        "selected_charge_for_sequence": "",  # excluded drive cycle
        "soc_anchor_evidence": "",
        "confidence": "",
        "evidence": "HWFET excluded per protocol",
        "pairing_status": "",
    },
    "03-21-17_00.29 25degC_UDDS_Pan18650PF.mat": {
        # 3416_Charge2 follows HWFTb (excluded); assigned to UDDS (test) via
        # following-cycle fallback since HWFTb is excluded.
        "previous_charge": "03-20-17_21.35 3416_Charge2.mat",
        "next_charge": "03-21-17_06.44 3416_Charge3.mat",
        "selected_charge_for_sequence": "03-20-17_21.35 3416_Charge2.mat",
        "soc_anchor_evidence": (
            "Preceding charge 3416_Charge2 terminates at 4.200V, 0.00A, "
            "Ah_end=2.675Ah → SOC_init=1.0"
        ),
        "confidence": "high",
        "evidence": (
            "3416_Charge2 (21:35 Mar 20) precedes UDDS (00:29 Mar 21) "
            "via Pause3 (23:30 Mar 20); 3416_Charge3 (06:44 Mar 21) follows UDDS"
        ),
        "pairing_status": "confirmed",
    },
    "03-21-17_09.38 25degC_LA92_Pan18650PF.mat": {
        "previous_charge": "03-21-17_06.44 3416_Charge3.mat",
        "next_charge": "03-21-17_13.34 3416_Charge4.mat",
        "selected_charge_for_sequence": "03-21-17_06.44 3416_Charge3.mat",
        "soc_anchor_evidence": (
            "Preceding charge 3416_Charge3 terminates at 4.200V, 0.00A, "
            "Ah_end=2.635Ah → SOC_init=1.0"
        ),
        "confidence": "high",
        "evidence": (
            "3416_Charge3 (06:44 Mar 21) precedes LA92 (09:38 Mar 21) "
            "via Pause4 (08:39); 3416_Charge4 (13:34 Mar 21) follows LA92"
        ),
        "pairing_status": "confirmed",
    },
    "03-21-17_16.27 25degC_NN_Pan18650PF.mat": {
        # 3416_Charge4 physically precedes NN but is classified test=1 by
        # split rule (LA92 precedes Charge4). NN does NOT load this charge.
        # SOC_init=1.0 is derived from full-charge protocol (NN Ah_start≈0).
        "previous_charge": "03-21-17_13.34 3416_Charge4.mat",
        "next_charge": "03-21-17_19.43 3416_Charge5.mat",
        "selected_charge_for_sequence": "",
        "soc_anchor_evidence": (
            "soc_anchor_source=full_charge_protocol; soc_init=1.0; "
            "NN Ah_start≈0 confirms battery was fully charged before run; "
            "3416_Charge4 is classified test=1 and is NOT loaded for NN training"
        ),
        "confidence": "high",
        "evidence": (
            "3416_Charge4 (13:34 Mar 21) precedes NN (16:27 Mar 21) "
            "via Pause5 (15:28); 3416_Charge5 (19:43 Mar 21) follows NN; "
            "no cross-split dependency"
        ),
        "pairing_status": "confirmed",
    },
}

# ---------------------------------------------------------------------------
# Terminal values of each selected charge file (for soc_anchor_evidence).
# ---------------------------------------------------------------------------
CHARGE_SOC_EVIDENCE = {
    "03-18-17_00.05 3406_Charge1.mat": (
        "Charge file: V_end=4.200V, I_end=0.00A, Ah_end=2.670Ah"
    ),
    "03-19-17_06.31 3406_Charge2.mat": (
        "Charge file: V_end=4.200V, I_end=0.00A, Ah_end=2.677Ah"
    ),
    "03-19-17_11.59 3406_Charge3.mat": (
        "Charge file: V_end=4.200V, I_end=0.00A, Ah_end=2.497Ah"
    ),
    "03-19-17_17.53 3406_Charge4.mat": (
        "Charge file: V_end=4.200V, I_end=0.00A, Ah_end=2.760Ah"
    ),
    "03-20-17_03.05 3415_Charge1.mat": (
        "Charge file: V_end=4.200V, I_end=0.00A, Ah_end=2.569Ah"
    ),
    "03-20-17_21.35 3416_Charge2.mat": (
        "Charge file: V_end=4.200V, I_end=0.00A, Ah_end=2.675Ah"
    ),
    "03-21-17_06.44 3416_Charge3.mat": (
        "Charge file: V_end=4.200V, I_end=0.00A, Ah_end=2.635Ah"
    ),
    "03-21-17_13.34 3416_Charge4.mat": (
        "Charge file: V_end=4.200V, I_end=0.00A, Ah_end=2.544Ah"
    ),
    "03-21-17_19.43 3416_Charge5.mat": (
        "Charge file: V_end=4.200V, I_end=0.00A, Ah_end=2.513Ah"
    ),
}


def parse_filename_datetime(filename: str) -> tuple[str, str, str]:
    """Extract start_date, start_time, start_datetime from filename prefix."""
    try:
        parts = filename.split(" ", 1)
        dt_part = parts[0]  # e.g. "03-18-17_02.17"
        date_part, time_part = dt_part.split("_")
        month, day, year_2d = date_part.split("-")
        year = f"20{year_2d}"
        hour, minute = time_part.split(".")
        start_date = f"{year}-{month}-{day}"
        start_time = f"{hour}:{minute}"
        start_datetime = f"{year}-{month}-{day}T{hour}:{minute}:00"
        return start_date, start_time, start_datetime
    except Exception:
        return "", "", ""


def load_mat_metadata(path: Path) -> dict:
    """Load a .mat file and extract metadata from the meas struct."""
    result = {
        "load_ok": False,
        "has_meas": False,
        "fields_ok": False,
        "n_rows": None,
        "duration_s": None,
        "first_timestamp": None,
        "last_timestamp": None,
        "voltage_min": None,
        "voltage_max": None,
        "current_min": None,
        "current_max": None,
        "ah_start": None,
        "ah_end": None,
        "temp_mean": None,
        "has_nan_voltage": None,
        "has_nan_current": None,
        "has_nan_ah": None,
        "has_time_reset": None,
        "array_lengths_consistent": None,
        "qc_pass": False,
        "qc_notes": [],
    }

    try:
        mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except Exception as e:
        result["qc_notes"].append(f"loadmat failed: {e}")
        return result

    result["load_ok"] = True

    if "meas" not in mat:
        result["qc_notes"].append("'meas' struct not found in file")
        return result

    result["has_meas"] = True
    meas = mat["meas"]

    required_fields = ["Time", "TimeStamp", "Voltage", "Current", "Ah", "Battery_Temp_degC"]
    missing = [f for f in required_fields if not hasattr(meas, f)]
    if missing:
        result["qc_notes"].append(f"Missing fields: {missing}")
        return result

    result["fields_ok"] = True

    try:
        time = np.atleast_1d(meas.Time).astype(float)
        voltage = np.atleast_1d(meas.Voltage).astype(float)
        current = np.atleast_1d(meas.Current).astype(float)
        ah = np.atleast_1d(meas.Ah).astype(float)
        temp = np.atleast_1d(meas.Battery_Temp_degC).astype(float)
        timestamp_raw = meas.TimeStamp

        n = len(time)
        result["n_rows"] = n

        # Check consistency
        lengths = {
            "Time": len(time),
            "Voltage": len(voltage),
            "Current": len(current),
            "Ah": len(ah),
            "Battery_Temp_degC": len(temp),
        }
        consistent = len(set(lengths.values())) == 1
        result["array_lengths_consistent"] = consistent
        if not consistent:
            result["qc_notes"].append(f"Inconsistent array lengths: {lengths}")

        result["duration_s"] = float(time[-1] - time[0]) if n > 1 else 0.0

        # TimeStamp
        if isinstance(timestamp_raw, np.ndarray):
            ts_arr = timestamp_raw.flatten()
            if ts_arr.dtype.kind in ("U", "S", "O"):
                result["first_timestamp"] = str(ts_arr[0])
                result["last_timestamp"] = str(ts_arr[-1])
            else:
                result["first_timestamp"] = str(ts_arr[0])
                result["last_timestamp"] = str(ts_arr[-1])
        else:
            result["first_timestamp"] = str(timestamp_raw)
            result["last_timestamp"] = ""

        result["voltage_min"] = float(np.nanmin(voltage))
        result["voltage_max"] = float(np.nanmax(voltage))
        result["current_min"] = float(np.nanmin(current))
        result["current_max"] = float(np.nanmax(current))
        result["ah_start"] = float(ah[0])
        result["ah_end"] = float(ah[-1])
        result["temp_mean"] = float(np.nanmean(temp))

        result["has_nan_voltage"] = bool(np.any(np.isnan(voltage)))
        result["has_nan_current"] = bool(np.any(np.isnan(current)))
        result["has_nan_ah"] = bool(np.any(np.isnan(ah)))

        # Time monotonicity
        if n > 1:
            diffs = np.diff(time)
            result["has_time_reset"] = bool(np.any(diffs < 0))
        else:
            result["has_time_reset"] = False

        # QC
        notes = result["qc_notes"]
        if result["has_nan_voltage"]:
            notes.append("WARN: NaN voltage present")
        if result["has_nan_current"]:
            notes.append("WARN: NaN current present")
        if result["has_nan_ah"]:
            notes.append("WARN: NaN Ah present")
        if result["has_time_reset"]:
            notes.append("WARN: time reset detected (non-monotonic)")
        if not consistent:
            notes.append("FAIL: array length mismatch")
        if result["temp_mean"] is not None and not (15 <= result["temp_mean"] <= 35):
            notes.append(
                f"WARN: mean temp {result['temp_mean']:.1f}°C outside expected 15-35°C range"
            )

        result["qc_pass"] = (
            result["load_ok"]
            and result["has_meas"]
            and result["fields_ok"]
            and consistent
            and not result["has_nan_voltage"]
            and not result["has_nan_current"]
            and not result["has_nan_ah"]
            and not result["has_time_reset"]
        )

    except Exception as e:
        result["qc_notes"].append(f"Extraction error: {e}\n{traceback.format_exc()}")

    return result


def build_manifest_rows(catalogue: list[dict]) -> list[dict]:
    """Build one manifest row per file, loading metadata from .mat."""
    rows = []
    print("\n=== Loading .mat files ===")

    for entry in catalogue:
        subfolder = entry["subfolder"]
        filename = entry["filename"]
        path = RAW_BASE / subfolder / filename

        # Parse datetime from filename
        start_date, start_time, start_datetime = parse_filename_datetime(filename)

        # Relative path from repo root
        rel_path = path.relative_to(REPO_ROOT)

        # Load metadata
        if path.exists():
            meta = load_mat_metadata(path)
            exists = True
        else:
            meta = {
                "load_ok": False,
                "has_meas": False,
                "fields_ok": False,
                "n_rows": None,
                "duration_s": None,
                "first_timestamp": None,
                "last_timestamp": None,
                "voltage_min": None,
                "voltage_max": None,
                "current_min": None,
                "current_max": None,
                "ah_start": None,
                "ah_end": None,
                "temp_mean": None,
                "has_nan_voltage": None,
                "has_nan_current": None,
                "has_nan_ah": None,
                "has_time_reset": None,
                "array_lengths_consistent": None,
                "qc_pass": False,
                "qc_notes": ["FILE NOT FOUND"],
            }
            exists = False

        # Sequence / pairing columns (new schema: replaces paired_charge/charge_position)
        file_type = entry.get("file_type", "")
        if file_type == "drive_cycle":
            seq = DRIVE_CYCLE_SEQUENCE.get(filename, {})
            previous_charge = seq.get("previous_charge", "")
            next_charge = seq.get("next_charge", "")
            selected_charge_for_sequence = seq.get("selected_charge_for_sequence", "")
            soc_anchor_evidence = seq.get("soc_anchor_evidence", "")
            confidence = seq.get("confidence", "")
            evidence = seq.get("evidence", "")
            pairing_status = seq.get("pairing_status", "")
        elif file_type == "charge":
            previous_charge = ""
            next_charge = ""
            selected_charge_for_sequence = ""
            soc_anchor_evidence = CHARGE_SOC_EVIDENCE.get(filename, "")
            confidence = "high" if entry.get("exclude", 1) == 0 else ""
            evidence = ""
            pairing_status = ""
        else:
            previous_charge = ""
            next_charge = ""
            selected_charge_for_sequence = ""
            soc_anchor_evidence = ""
            confidence = ""
            evidence = ""
            pairing_status = ""

        # QC status string
        qc_notes = "; ".join(meta["qc_notes"]) if meta["qc_notes"] else ""
        if meta["qc_pass"]:
            qc_status = "PASS"
        elif not exists:
            qc_status = "MISSING"
        elif not meta["load_ok"]:
            qc_status = "FAIL_LOAD"
        elif not meta["has_meas"]:
            qc_status = "FAIL_STRUCT"
        elif not meta["fields_ok"]:
            qc_status = "FAIL_FIELDS"
        elif meta["qc_notes"]:
            qc_status = "WARN"
        else:
            qc_status = "FAIL"

        # Status print
        status_icon = "✓" if meta["qc_pass"] else ("?" if qc_status == "WARN" else "✗")
        print(f"  {status_icon} {filename[:60]:<60}  {qc_status}")
        if qc_notes:
            for note in meta["qc_notes"]:
                print(f"      → {note}")

        row = {
            "battery": "panasonic_18650pf",
            "temperature_c": 25,
            "filename": filename,
            "relative_path": str(rel_path),
            "start_datetime": start_datetime,
            "profile": entry.get("profile", ""),
            "profile_number": entry.get("profile_number", ""),
            "file_type": entry.get("file_type", ""),
            "is_standalone": entry.get("is_standalone", 1),
            "is_duplicate": entry.get("is_duplicate", 0),
            "duplicate_of": entry.get("duplicate_of", ""),
            "previous_charge": previous_charge,
            "next_charge": next_charge,
            "selected_charge_for_sequence": selected_charge_for_sequence,
            "soc_anchor_evidence": soc_anchor_evidence,
            "pairing_status": pairing_status,
            "full_train": entry.get("full_train", 0),
            "partial_train": entry.get("partial_train", 0),
            "test": entry.get("test", 0),
            "exclude": entry.get("exclude", 0),
            "confidence": confidence,
            "evidence": evidence,
            "notes": entry.get("notes", ""),
            # QC fields (extra)
            "qc_status": qc_status,
            "qc_notes": qc_notes,
            "n_rows": meta["n_rows"],
            "duration_s": (
                f"{meta['duration_s']:.1f}" if meta["duration_s"] is not None else ""
            ),
            "first_timestamp": meta["first_timestamp"] or "",
            "last_timestamp": meta["last_timestamp"] or "",
            "voltage_min": (
                f"{meta['voltage_min']:.4f}" if meta["voltage_min"] is not None else ""
            ),
            "voltage_max": (
                f"{meta['voltage_max']:.4f}" if meta["voltage_max"] is not None else ""
            ),
            "current_min": (
                f"{meta['current_min']:.4f}" if meta["current_min"] is not None else ""
            ),
            "current_max": (
                f"{meta['current_max']:.4f}" if meta["current_max"] is not None else ""
            ),
            "ah_start": f"{meta['ah_start']:.4f}" if meta["ah_start"] is not None else "",
            "ah_end": f"{meta['ah_end']:.4f}" if meta["ah_end"] is not None else "",
            "temp_mean": (
                f"{meta['temp_mean']:.1f}" if meta["temp_mean"] is not None else ""
            ),
        }
        rows.append(row)

    return rows


MANIFEST_COLS = [
    "battery",
    "temperature_c",
    "filename",
    "relative_path",
    "start_datetime",
    "profile",
    "profile_number",
    "file_type",
    "is_standalone",
    "is_duplicate",
    "duplicate_of",
    # Charge-sequence columns (replace old paired_charge / charge_position)
    "previous_charge",
    "next_charge",
    "selected_charge_for_sequence",
    "soc_anchor_evidence",
    "pairing_status",
    # Split flags
    "full_train",
    "partial_train",
    "test",
    "exclude",
    "confidence",
    "evidence",
    "notes",
    "qc_status",
    "qc_notes",
    "n_rows",
    "duration_s",
    "first_timestamp",
    "last_timestamp",
    "voltage_min",
    "voltage_max",
    "current_min",
    "current_max",
    "ah_start",
    "ah_end",
    "temp_mean",
]


def write_manifest_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] Manifest written: {output_path}  ({len(rows)} rows)")


def _as_int(val) -> int:
    """Safely convert a manifest field to int (handles str '1'/'0' and int)."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def run_validation(rows: list[dict]) -> bool:
    """
    Run all split-integrity and QC validation checks.
    Returns True if all checks pass.
    """
    print("\n" + "=" * 70)
    print("VALIDATION — Panasonic 25°C Vidal 2019 Split")
    print("=" * 70)

    all_pass = True

    # (a) Manifest loadability — confirmed (rows were already built from the file catalogue)
    print(f"\n[a] Manifest loadable: PASS ({len(rows)} rows)")

    # (b) All selected files exist on disk
    selected = [r for r in rows if _as_int(r.get("full_train")) == 1
                or _as_int(r.get("partial_train")) == 1
                or _as_int(r.get("test")) == 1]
    missing = [r for r in selected if r.get("qc_status") == "MISSING"]
    if missing:
        print(f"\n[b] Files on disk: FAIL — {len(missing)} selected file(s) missing:")
        for r in missing:
            print(f"    ✗ {r['filename']}")
        all_pass = False
    else:
        print(f"[b] Files on disk: PASS — all {len(selected)} selected files found")

    # (c) No full_train=1 AND test=1
    conflict_ft_test = [
        r for r in rows
        if _as_int(r.get("full_train")) == 1 and _as_int(r.get("test")) == 1
    ]
    if conflict_ft_test:
        print(f"\n[c] full_train ∩ test: FAIL — {len(conflict_ft_test)} conflict(s):")
        for r in conflict_ft_test:
            print(f"    ✗ {r['filename']}")
        all_pass = False
    else:
        print("[c] full_train ∩ test: PASS — no overlap")

    # (d) No partial_train=1 AND test=1
    conflict_pt_test = [
        r for r in rows
        if _as_int(r.get("partial_train")) == 1 and _as_int(r.get("test")) == 1
    ]
    if conflict_pt_test:
        print(f"\n[d] partial_train ∩ test: FAIL — {len(conflict_pt_test)} conflict(s):")
        for r in conflict_pt_test:
            print(f"    ✗ {r['filename']}")
        all_pass = False
    else:
        print("[d] partial_train ∩ test: PASS — no overlap")

    # (e) No exclude=1 with any other split flag
    conflict_excl = [
        r for r in rows
        if _as_int(r.get("exclude")) == 1 and (
            _as_int(r.get("full_train")) == 1
            or _as_int(r.get("partial_train")) == 1
            or _as_int(r.get("test")) == 1
        )
    ]
    if conflict_excl:
        print(f"\n[e] exclude ∩ split-flag: FAIL — {len(conflict_excl)} conflict(s):")
        for r in conflict_excl:
            print(f"    ✗ {r['filename']}")
        all_pass = False
    else:
        print("[e] exclude ∩ split-flag: PASS — no overlap")

    # (f) Per-selected-file QC
    qc_failures = [
        r for r in selected
        if r.get("qc_status") not in ("PASS", "WARN")
    ]
    qc_warns = [r for r in selected if r.get("qc_status") == "WARN"]
    print(f"\n[f] Per-file QC ({len(selected)} selected files):")
    for r in selected:
        status = r.get("qc_status", "?")
        icon = "✓" if status == "PASS" else ("⚠" if status == "WARN" else "✗")
        line = f"    {icon} [{status:10s}]  {r['filename']}"
        print(line)
        if r.get("qc_notes"):
            print(f"              → {r['qc_notes']}")
    if qc_failures:
        print(f"    → FAIL: {len(qc_failures)} file(s) failed QC")
        all_pass = False
    elif qc_warns:
        print(f"    → WARN: {len(qc_warns)} file(s) with warnings (non-fatal)")
    else:
        print("    → PASS: all selected files pass QC")

    # Unresolved pairings
    drive_rows = [r for r in rows if r.get("file_type") == "drive_cycle"
                  and _as_int(r.get("exclude")) == 0]
    unresolved = [r for r in drive_rows if r.get("pairing_status") == "unresolved"]
    inferred = [r for r in drive_rows if r.get("pairing_status") == "inferred"]
    print(f"\n--- Pairing Status (non-excluded drive cycles) ---")
    for r in drive_rows:
        ps = r.get("pairing_status", "")
        sc = r.get("selected_charge_for_sequence", "") or "NONE"
        print(f"  {r['profile']:12s}  [{ps:10s}]  selected_charge: {sc}")
    if unresolved:
        print(f"\n  ⚠ UNRESOLVED pairings: {len(unresolved)}")
        for r in unresolved:
            print(f"    - {r['profile']}: {r.get('soc_anchor_evidence', '')}")
    if inferred:
        print(f"\n  ℹ INFERRED pairings: {len(inferred)}")
        for r in inferred:
            print(f"    - {r['profile']}: {r.get('soc_anchor_evidence', '')}")

    print(f"\n{'=' * 70}")
    if all_pass:
        print("OVERALL VALIDATION: PASS")
    else:
        print("OVERALL VALIDATION: FAIL  (see details above)")
    print("=" * 70)
    return all_pass


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY — Panasonic 25°C Vidal 2019 Split")
    print("=" * 70)

    def flag(r, field):
        return _as_int(r.get(field, 0)) == 1

    full_train = [r for r in rows if flag(r, "full_train")]
    partial_train = [r for r in rows if flag(r, "partial_train")]
    test = [r for r in rows if flag(r, "test")]
    excluded = [r for r in rows if flag(r, "exclude")]

    ft_drive = [r for r in full_train if r.get("file_type") == "drive_cycle"]
    ft_charge = [r for r in full_train if r.get("file_type") == "charge"]
    pt_drive = [r for r in partial_train if r.get("file_type") == "drive_cycle"]
    pt_charge = [r for r in partial_train if r.get("file_type") == "charge"]
    t_drive = [r for r in test if r.get("file_type") == "drive_cycle"]
    t_charge = [r for r in test if r.get("file_type") == "charge"]

    print(f"\nTotal files in manifest                     : {len(rows)}")
    print(f"  full_train=1                              : {len(full_train)}"
          f"  ({len(ft_drive)} drive + {len(ft_charge)} charge)")
    print(f"  partial_train=1  (subset of full_train)   : {len(partial_train)}"
          f"  ({len(pt_drive)} drive + {len(pt_charge)} charge)")
    print(f"  test=1                                    : {len(test)}"
          f"  ({len(t_drive)} drive + {len(t_charge)} charge)")
    print(f"  exclude=1                                 : {len(excluded)}")
    total_accounted = len(full_train) + len(test) + len(excluded)
    print(f"  (full_train + test + exclude)             : {total_accounted}")

    print("\n--- Cycle → Mix Mapping (most plausible paper-to-dataset mapping) ---")
    readme_order = [
        "Cycle_1 → Mix1",
        "Cycle_2 → Mix2",
        "Cycle_3 → Mix3",
        "Cycle_4 → Mix4",
        "NN      → Mix5",
    ]
    for m in readme_order:
        print(f"  {m}")
    print(
        "  Inference: Panasonic README names test sequence as 'Cycle 1, Cycle 2,\n"
        "  Cycle 3, Cycle 4, US06, HWFET, UDDS, LA92, Neural Network (NN)'.\n"
        "  Vidal 2019 (ITEC) calls training profiles Mix1–5. The mapping\n"
        "  Cycle_N → MixN and NN → Mix5 is the most plausible correspondence;\n"
        "  the paper does not use the dataset's 'Cycle_N' naming directly."
    )

    print("\n--- Charge Sequence Summary (non-excluded drive cycles) ---")
    drive_only = [r for r in rows if r.get("file_type") == "drive_cycle"
                  and not flag(r, "exclude")]
    for r in drive_only:
        sc_raw = r.get("selected_charge_for_sequence", "")
        ps = r.get("pairing_status", "")
        # Distinguish intentional protocol-based empty from genuinely unresolved
        if not sc_raw:
            if ps == "confirmed":
                sc = "NONE (full_charge_protocol)"
            else:
                sc = "UNRESOLVED"
        else:
            sc = sc_raw
        split = ("full_train" if flag(r, "full_train")
                 else "test" if flag(r, "test") else "?")
        print(f"  {r['profile']:12s} [{split:10s}]  sel_charge: {sc}  [{ps}]")

    print()


def main() -> None:
    print(f"Repo root : {REPO_ROOT}")
    print(f"Data root : {RAW_BASE}")

    if not RAW_BASE.exists():
        print(f"ERROR: Raw data directory not found: {RAW_BASE}")
        sys.exit(1)

    rows = build_manifest_rows(FILE_CATALOGUE)
    print_summary(rows)
    validation_ok = run_validation(rows)

    manifest_path = MANIFESTS_DIR / "panasonic_vidal2019_25c_manifest.csv"
    write_manifest_csv(rows, manifest_path)

    if not validation_ok:
        print("\nWARNING: Validation failures detected — inspect output above.")
        sys.exit(1)
    print("\nDone. Manifest and validation PASS.")


if __name__ == "__main__":
    main()
