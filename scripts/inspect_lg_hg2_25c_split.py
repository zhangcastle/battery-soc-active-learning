#!/usr/bin/env python3
"""
Inspection and split-reconstruction script for the LG HG2 dataset at 25°C.

Reproduces the Vidal et al. (2019 ITEC) file-level train/test split for the
LG INR18650HG2 battery. All decisions and evidences are documented inline.

Paper:
    C. Vidal, P. Kollmeyer, E. Chemali, A. Emadi,
    "Li-ion Battery State of Charge Estimation Using Long Short-Term Memory
     Recurrent Neural Network with Transfer Learning,"
    2019 IEEE Transportation Electrification Conference and Expo (ITEC), 2019.

Paper protocol for LG HG2 (Table I):
    Battery:   LG HG2, 3 Ah, cylindrical 18650
    Profiles:  Mix (1-8), UDDS, LA92, US06, CC
    Training:  Full = all Mix1-8 + accompanying CC charges
    Partial:   Mix1-3 + accompanying CC charges (Case 3 / Case 4)
    Test:      UDDS, LA92, US06 + corresponding CC charges

Raw dataset location:
    data/raw/LG_HG2/25degC/

Raw file naming convention:
    Files use "Mixed1" through "Mixed8" (not "Mix1-8" as in the paper).
    Paper uses "Mix1-8"; public dataset uses "Mixed1-8".
    This mapping is documented as inferred from count and name correspondence.

HWFET note (CRITICAL):
    The 25°C HWFET file (551_HWFET_25degC_LGHG2.mat) contains ONLY 600 seconds
    of ZERO CURRENT at ~4.186V. The public HWFET-labelled file contains no
    drive-cycle excitation — it records only a rest at full-charge voltage.
    This file is EXCLUDED as a data quality failure (exclude=1).

Charge Ah counter note:
    Some charge files show Ah_end=0 although Ah_max>0. This is a cycler
    behavior: the Ah counter resets to 0 at the END of the recording step.
    The canonical "Ah charged" is Ah_max (peak value), not Ah_end.
    For SOC anchoring, terminal V~4.186V and I→0 confirm a full CC-CV charge.

Usage:
    .venv/bin/python scripts/inspect_lg_hg2_25c_split.py

Exit status:
    0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import csv
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_BASE  = REPO_ROOT / "data" / "raw" / "LG_HG2" / "25degC"
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "lg_hg2_vidal2019_25c_manifest.csv"

# ---------------------------------------------------------------------------
# Profile classification rules
# ---------------------------------------------------------------------------
# These patterns are matched against the stem of each .mat filename.
# Order matters: more specific patterns first.
PROFILE_RULES: list[tuple[str, str, str]] = [
    # (regex_pattern, profile_name, file_type)
    # Drive cycles — test
    (r"UDDS",       "UDDS",     "drive_cycle"),
    (r"LA92",       "LA92",     "drive_cycle"),
    (r"US06",       "US06",     "drive_cycle"),
    # Drive cycles — HWFET (rest file at 25°C — see note above)
    (r"HWFET",      "HWFET",    "drive_cycle"),
    # Drive cycles — Mixed (training)
    (r"Mixed(\d+)", "Mixed",    "drive_cycle"),
    # Pause/cycle rest
    (r"PausCycl",   "PausCycl", "pause"),
    # Charges
    (r"Charge",     "Charge",   "charge"),
    # Characterization
    (r"HPPC",       "HPPC",     "characterization"),
    (r"C20DisCh",   "C20DisCh", "characterization"),
    (r"Dis_0p5C",   "Dis_0p5C", "characterization"),
    (r"Dis_2C",     "Dis_2C",   "characterization"),
    (r"Cap_1C",     "Cap_1C",   "characterization"),
]

# ---------------------------------------------------------------------------
# Split assignment table — keyed by file stem (no extension).
# These are derived from paper protocol + raw file timeline analysis.
# See __main__ section for full evidence chain.
# ---------------------------------------------------------------------------

# Full list of 25°C .mat filenames classified by split role.
# Rule: each charge is assigned to the split of the drive cycle that
# IMMEDIATELY PRECEDES it chronologically (preceding-cycle-wins rule).
# When the preceding drive cycle is excluded, the FOLLOWING non-excluded
# drive cycle's split is used as a fallback (following-cycle fallback).
# Mixed1's preceding charge (Charge6) is test=1, so Mixed1 uses
# soc_anchor_source="full_charge_protocol" with no cross-split dependency.

SPLIT_ASSIGNMENTS: dict[str, dict[str, Any]] = {
    # ── Characterization (exclude) ─────────────────────────────────────────
    "10-25-18_04.09 549_Charge": {
        "full_train": 0, "partial_train": 0, "test": 0, "exclude": 1,
        "paper_profile": "",
        "previous_drive_cycle": "",
        "next_drive_cycle": "",
        "soc_anchor_source": "",
        "soc_init": "",
        "split_assignment_basis": "Pre-HPPC initial/conditioning charge; not paired with any drive cycle",
        "pairing_status": "not_applicable",
        "confidence": "high",
        "evidence": "Precedes HPPC (10-25-18_07.39); 283969.7s duration (conditioning); excluded",
        "notes": "Initial conditioning charge before HPPC test — excluded",
    },
    "10-25-18_07.39 549_HPPC_25degC_LGHG2": {
        "full_train": 0, "partial_train": 0, "test": 0, "exclude": 1,
        "paper_profile": "",
        "previous_drive_cycle": "", "next_drive_cycle": "",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "HPPC characterization — excluded per protocol",
        "pairing_status": "not_applicable", "confidence": "high",
        "evidence": "Four-pulse discharge HPPC test; characterization only",
        "notes": "HPPC characterization — excluded",
    },
    "10-26-18_09.29 549_C20DisCh_25degC_LGHG2": {
        "full_train": 0, "partial_train": 0, "test": 0, "exclude": 1,
        "paper_profile": "",
        "previous_drive_cycle": "", "next_drive_cycle": "",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "C/20 discharge characterization — excluded",
        "pairing_status": "not_applicable", "confidence": "high",
        "evidence": "C20DisCh test; characterization",
        "notes": "C/20 discharge characterization — excluded",
    },
    "10-28-18_01.48 549_Dis_0p5C_25degC_LGHG2": {
        "full_train": 0, "partial_train": 0, "test": 0, "exclude": 1,
        "paper_profile": "",
        "previous_drive_cycle": "", "next_drive_cycle": "",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "0.5C discharge characterization — excluded",
        "pairing_status": "not_applicable", "confidence": "high",
        "evidence": "0.5C fixed-rate discharge; characterization",
        "notes": "0.5C discharge characterization — excluded",
    },
    "10-28-18_07.09 549_Dis_2C_25degC_LGHG2": {
        "full_train": 0, "partial_train": 0, "test": 0, "exclude": 1,
        "paper_profile": "",
        "previous_drive_cycle": "", "next_drive_cycle": "",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "2C discharge characterization — excluded",
        "pairing_status": "not_applicable", "confidence": "high",
        "evidence": "2C fixed-rate discharge; characterization",
        "notes": "2C discharge characterization — excluded",
    },
    "10-28-18_11.04 551_Charge1": {
        "full_train": 0, "partial_train": 0, "test": 0, "exclude": 1,
        "paper_profile": "",
        "previous_drive_cycle": "", "next_drive_cycle": "551_Cap_1C_25degC_LGHG2",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": (
            "Top-off charge (I_max=0.235A, Ah_max=0.0144) before Cap_1C; "
            "preceded by Dis_2C (characterization, excluded); not paired with drive cycle"
        ),
        "pairing_status": "not_applicable", "confidence": "high",
        "evidence": "Charge1 at 11:04, Dis_2C before (~07:09+26min), Cap_1C at 13:10; tiny top-off only",
        "notes": "Pre-Cap_1C top-off charge (0.235A, 0.0144Ah) — excluded; not a training or test input",
    },
    "10-28-18_13.10 551_Cap_1C_25degC_LGHG2": {
        "full_train": 0, "partial_train": 0, "test": 0, "exclude": 1,
        "paper_profile": "",
        "previous_drive_cycle": "", "next_drive_cycle": "",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "1C capacity test — characterization; excluded",
        "pairing_status": "not_applicable", "confidence": "high",
        "evidence": "Cap_1C discharge; README: '1C discharge test is performed before the UDDS cycle'",
        "notes": "1C capacity test (first, before UDDS) — excluded",
    },
    # ── Test drive cycles ──────────────────────────────────────────────────
    "10-28-18_17.44 551_UDDS_25degC_LGHG2": {
        "full_train": 0, "partial_train": 0, "test": 1, "exclude": 0,
        "paper_profile": "UDDS",
        "previous_drive_cycle": "",
        "next_drive_cycle": "551_HWFET_25degC_LGHG2",
        "soc_anchor_source": "preceding_full_charge",
        "soc_init": 1.0,
        "split_assignment_basis": "UDDS is a test profile per Vidal 2019 (ITEC) §IV.B",
        "pairing_status": "confirmed",
        "confidence": "high",
        "evidence": (
            "Charge2 (14:16, pre-UDDS) terminates at 4.186V with I→0 "
            "(Ah_max=2.7319Ah); UDDS Ah_start≈0 → SOC_init=1.0; "
            "Charge2 classified test by following-cycle rule (Cap_1C excluded)"
        ),
        "notes": "UDDS test cycle; preceded by full CC-CV charge2 (post-Cap_1C fallback)",
    },
    "10-29-18_01.36 551_HWFET_25degC_LGHG2": {
        "full_train": 0, "partial_train": 0, "test": 0, "exclude": 1,
        "paper_profile": "",
        "previous_drive_cycle": "551_UDDS_25degC_LGHG2",
        "next_drive_cycle": "551_LA92_25degC_LGHG2",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": (
            "The public HWFET-labelled file contains no drive-cycle excitation: "
            "600s zero current at ~4.186V — treated as rest/pause and excluded"
        ),
        "pairing_status": "not_applicable", "confidence": "high",
        "evidence": (
            "n=6002, duration=600.0s, I_min=0, I_max=0, Ah_start=0, Ah_end=0, "
            "V=4.185-4.186V → zero-current rest, not HWFET drive; "
            "HWFET is also excluded per Vidal 2019 protocol (not in test set)"
        ),
        "notes": (
            "DATA QUALITY FAILURE: the public HWFET-labelled file at 25°C contains "
            "only 600s of zero current — no drive-cycle excitation present. "
            "Excluded both as invalid data and per paper protocol (HWFET not in test set)."
        ),
    },
    "10-29-18_03.53 551_LA92_25degC_LGHG2": {
        "full_train": 0, "partial_train": 0, "test": 1, "exclude": 0,
        "paper_profile": "LA92",
        "previous_drive_cycle": "",
        "next_drive_cycle": "551_US06_25degC_LGHG2",
        "soc_anchor_source": "preceding_full_charge",
        "soc_init": 1.0,
        "split_assignment_basis": "LA92 is a test profile per Vidal 2019 (ITEC) §IV.B",
        "pairing_status": "confirmed",
        "confidence": "high",
        "evidence": (
            "Charge4 (01:47, post-HWFET rest) is a tiny top-off (0.238A, Ah_max=0.0148); "
            "HWFET excluded (rest file); Charge4 classified test by following-cycle fallback; "
            "LA92 Ah_start≈0 → confirms full charge before run → SOC_init=1.0"
        ),
        "notes": "LA92 test cycle; pre-LA92 charge (Charge4) is tiny top-off after HWFET rest",
    },
    "10-29-18_10.08 551_US06_25degC_LGHG2": {
        "full_train": 0, "partial_train": 0, "test": 1, "exclude": 0,
        "paper_profile": "US06",
        "previous_drive_cycle": "551_LA92_25degC_LGHG2",
        "next_drive_cycle": "551_Mixed1_25degC_LGHG2",
        "soc_anchor_source": "preceding_full_charge",
        "soc_init": 1.0,
        "split_assignment_basis": "US06 is a test profile per Vidal 2019 (ITEC) §IV.B",
        "pairing_status": "confirmed",
        "confidence": "high",
        "evidence": (
            "Charge5 (06:42, post-LA92) terminates at 4.186V, Ah_max=2.6092Ah; "
            "wait from Charge5 end to US06 start; US06 Ah_start≈0 → SOC_init=1.0"
        ),
        "notes": "US06 test cycle; post-LA92/pre-US06 charge (Charge5) classified test",
    },
    # ── Test charges ───────────────────────────────────────────────────────
    "10-28-18_14.16 551_Charge2": {
        "full_train": 0, "partial_train": 0, "test": 1, "exclude": 0,
        "paper_profile": "CC",
        "previous_drive_cycle": "",
        "next_drive_cycle": "551_UDDS_25degC_LGHG2",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": (
            "Immediately precedes UDDS (test); Cap_1C is excluded → "
            "following-cycle fallback: UDDS test → Charge2 = test"
        ),
        "pairing_status": "confirmed", "confidence": "high",
        "evidence": (
            "Cap_1C (excluded) at 13:10 precedes Charge2 (14:16) which precedes UDDS (17:44); "
            "Ah_max=2.7319Ah, I_max=3.001A, V_end=4.186V → full CC-CV charge; "
            "Ah_end=0 due to cycler counter reset (Ah_max is canonical)"
        ),
        "notes": (
            "Pre-UDDS charge; Ah counter reset at end (Ah_end=0, Ah_max=2.7319); "
            "full CC-CV charge confirmed by V_end=4.186V and I→0"
        ),
    },
    "10-28-18_22.11 551_Charge3": {
        "full_train": 0, "partial_train": 0, "test": 1, "exclude": 0,
        "paper_profile": "CC",
        "previous_drive_cycle": "551_UDDS_25degC_LGHG2",
        "next_drive_cycle": "551_HWFET_25degC_LGHG2",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "UDDS (test) immediately precedes Charge3 → test",
        "pairing_status": "confirmed", "confidence": "high",
        "evidence": "Charge3 at 22:11; UDDS ends before this; HWFET at 01:36; Ah_max=2.6330Ah",
        "notes": "Post-UDDS charge; Ah_end=2.6330 (no reset); V_end=4.186V",
    },
    "10-29-18_01.47 551_Charge4": {
        "full_train": 0, "partial_train": 0, "test": 1, "exclude": 0,
        "paper_profile": "CC",
        "previous_drive_cycle": "551_HWFET_25degC_LGHG2",
        "next_drive_cycle": "551_LA92_25degC_LGHG2",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": (
            "HWFET is excluded (zero-current rest file); "
            "following-cycle fallback: LA92 is test → Charge4 = test"
        ),
        "pairing_status": "confirmed", "confidence": "high",
        "evidence": (
            "Charge4 at 01:47, HWFET at 01:36 (rest, 600s, 0A); "
            "LA92 at 03:53; Ah_max=0.0148Ah, I_max=0.238A → tiny top-off "
            "(HWFET rest caused no discharge → cell nearly full)"
        ),
        "notes": (
            "Post-HWFET-rest/pre-LA92 top-off charge; tiny (0.0148Ah) because "
            "HWFET did not discharge; HWFET excluded as zero-current rest"
        ),
    },
    "10-29-18_06.42 551_Charge5": {
        "full_train": 0, "partial_train": 0, "test": 1, "exclude": 0,
        "paper_profile": "CC",
        "previous_drive_cycle": "551_LA92_25degC_LGHG2",
        "next_drive_cycle": "551_US06_25degC_LGHG2",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "LA92 (test) immediately precedes Charge5 → test",
        "pairing_status": "confirmed", "confidence": "high",
        "evidence": "Charge5 at 06:42; LA92 at 03:53; US06 at 10:08; Ah_max=2.6092Ah",
        "notes": "Post-LA92/pre-US06 charge; Ah_end=2.6092 (no reset); V_end=4.186V",
    },
    "10-29-18_11.15 551_Charge6": {
        "full_train": 0, "partial_train": 0, "test": 1, "exclude": 0,
        "paper_profile": "CC",
        "previous_drive_cycle": "551_US06_25degC_LGHG2",
        "next_drive_cycle": "551_Mixed1_25degC_LGHG2",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "US06 (test) immediately precedes Charge6 → test",
        "pairing_status": "confirmed", "confidence": "high",
        "evidence": "Charge6 at 11:15; US06 at 10:08; Mixed1 at 14:41; Ah_max=2.6090Ah",
        "notes": (
            "Post-US06/pre-Mixed1 charge; classified test (US06 precedes); "
            "Mixed1 uses full_charge_protocol SOC_init=1.0 — no cross-split dependency"
        ),
    },
    # ── Training drive cycles (full_train and partial_train) ───────────────
    "10-29-18_14.41 551_Mixed1_25degC_LGHG2": {
        "full_train": 1, "partial_train": 1, "test": 0, "exclude": 0,
        "paper_profile": "Mix1",
        "previous_drive_cycle": "551_US06_25degC_LGHG2",
        "next_drive_cycle": "551_Mixed2_25degC_LGHG2",
        "soc_anchor_source": "full_charge_protocol",
        "soc_init": 1.0,
        "split_assignment_basis": "Mix1 training per Vidal 2019 (ITEC) Table I; §IV.B",
        "pairing_status": "confirmed",
        "confidence": "high",
        "evidence": (
            "Preceding charge Charge6 (11:15) is test=1 (US06 precedes it); "
            "Mixed1 Ah_start≈0 → battery was fully charged before run; "
            "soc_anchor_source=full_charge_protocol, soc_init=1.0; no cross-split dependency"
        ),
        "notes": (
            "Mixed1 = Mix1 (paper). Preceding charge (Charge6) is test=1; "
            "Mixed1 uses full_charge_protocol for SOC_init — no test file loaded."
        ),
    },
    "10-29-18_20.16 551_Mixed2_25degC_LGHG2": {
        "full_train": 1, "partial_train": 1, "test": 0, "exclude": 0,
        "paper_profile": "Mix2",
        "previous_drive_cycle": "551_Mixed1_25degC_LGHG2",
        "next_drive_cycle": "552_Mixed3_25degC_LGHG2",
        "soc_anchor_source": "preceding_full_charge",
        "soc_init": 1.0,
        "split_assignment_basis": "Mix2 training per Vidal 2019 (ITEC) Table I; §IV.B",
        "pairing_status": "confirmed",
        "confidence": "high",
        "evidence": (
            "Charge7 (16:50, post-Mixed1) terminates at 4.186V, Ah_max=2.6155Ah; "
            "Mixed2 Ah_start≈0 → SOC_init=1.0"
        ),
        "notes": "Mixed2 = Mix2 (paper). SOC anchor: Charge7 (full_train).",
    },
    "10-30-18_06.27 552_Mixed3_25degC_LGHG2": {
        "full_train": 1, "partial_train": 1, "test": 0, "exclude": 0,
        "paper_profile": "Mix3",
        "previous_drive_cycle": "551_Mixed2_25degC_LGHG2",
        "next_drive_cycle": "552_Mixed4_25degC_LGHG2",
        "soc_anchor_source": "preceding_full_charge",
        "soc_init": 1.0,
        "split_assignment_basis": "Mix3 training per Vidal 2019 (ITEC) Table I; §IV.B",
        "pairing_status": "confirmed",
        "confidence": "high",
        "evidence": (
            "Charge9 (03:00, post-Cap_1C) terminates at 4.186V, Ah_max=2.7086Ah; "
            "Cap_1C (excluded) precedes Charge9 → following-cycle fallback: Mixed3 → full_train; "
            "Mixed3 Ah_start≈0 → SOC_init=1.0"
        ),
        "notes": (
            "Mixed3 = Mix3 (paper). Pre-Mixed3 charge (Charge9) assigned full_train "
            "via following-cycle fallback (Cap_1C excluded before it)."
        ),
    },
    "10-30-18_11.56 552_Mixed4_25degC_LGHG2": {
        "full_train": 1, "partial_train": 0, "test": 0, "exclude": 0,
        "paper_profile": "Mix4",
        "previous_drive_cycle": "552_Mixed3_25degC_LGHG2",
        "next_drive_cycle": "552_Mixed5_25degC_LGHG2",
        "soc_anchor_source": "preceding_full_charge",
        "soc_init": 1.0,
        "split_assignment_basis": "Mix4 full-training per Vidal 2019 (ITEC) Table I; §IV.B",
        "pairing_status": "confirmed",
        "confidence": "high",
        "evidence": (
            "Charge10 (08:31, post-Mixed3) terminates at 4.186V, Ah_max=2.5869Ah; "
            "Mixed4 Ah_start≈0 → SOC_init=1.0"
        ),
        "notes": "Mixed4 = Mix4 (paper). Full-training only.",
    },
    "10-30-18_17.36 552_Mixed5_25degC_LGHG2": {
        "full_train": 1, "partial_train": 0, "test": 0, "exclude": 0,
        "paper_profile": "Mix5",
        "previous_drive_cycle": "552_Mixed4_25degC_LGHG2",
        "next_drive_cycle": "552_Mixed6_25degC_LGHG2",
        "soc_anchor_source": "preceding_full_charge",
        "soc_init": 1.0,
        "split_assignment_basis": "Mix5 full-training per Vidal 2019 (ITEC) Table I; §IV.B",
        "pairing_status": "confirmed",
        "confidence": "high",
        "evidence": (
            "Charge11 (14:12, post-Mixed4) terminates at 4.186V, Ah_max=2.5825Ah; "
            "Mixed5 Ah_start≈0 → SOC_init=1.0"
        ),
        "notes": "Mixed5 = Mix5 (paper).",
    },
    "10-30-18_23.02 552_Mixed6_25degC_LGHG2": {
        "full_train": 1, "partial_train": 0, "test": 0, "exclude": 0,
        "paper_profile": "Mix6",
        "previous_drive_cycle": "552_Mixed5_25degC_LGHG2",
        "next_drive_cycle": "552_Mixed7_25degC_LGHG2",
        "soc_anchor_source": "preceding_full_charge",
        "soc_init": 1.0,
        "split_assignment_basis": "Mix6 full-training per Vidal 2019 (ITEC) Table I; §IV.B",
        "pairing_status": "confirmed",
        "confidence": "high",
        "evidence": (
            "Charge12 (19:38, post-Mixed5) terminates at 4.186V, Ah_max=2.5816Ah; "
            "Mixed6 Ah_start≈0 → SOC_init=1.0"
        ),
        "notes": "Mixed6 = Mix6 (paper).",
    },
    "10-31-18_04.38 552_Mixed7_25degC_LGHG2": {
        "full_train": 1, "partial_train": 0, "test": 0, "exclude": 0,
        "paper_profile": "Mix7",
        "previous_drive_cycle": "552_Mixed6_25degC_LGHG2",
        "next_drive_cycle": "552_Mixed8_25degC_LGHG2",
        "soc_anchor_source": "preceding_full_charge",
        "soc_init": 1.0,
        "split_assignment_basis": "Mix7 full-training per Vidal 2019 (ITEC) Table I; §IV.B",
        "pairing_status": "confirmed",
        "confidence": "high",
        "evidence": (
            "Charge13 (01:13, post-Mixed6) terminates at 4.186V, Ah_max=2.5831Ah; "
            "Mixed7 Ah_start≈0 → SOC_init=1.0"
        ),
        "notes": "Mixed7 = Mix7 (paper).",
    },
    "10-31-18_10.05 552_Mixed8_25degC_LGHG2": {
        "full_train": 1, "partial_train": 0, "test": 0, "exclude": 0,
        "paper_profile": "Mix8",
        "previous_drive_cycle": "552_Mixed7_25degC_LGHG2",
        "next_drive_cycle": "",
        "soc_anchor_source": "preceding_full_charge",
        "soc_init": 1.0,
        "split_assignment_basis": "Mix8 full-training per Vidal 2019 (ITEC) Table I; §IV.B",
        "pairing_status": "confirmed",
        "confidence": "high",
        "evidence": (
            "Charge14 (06:41, post-Mixed7) terminates at 4.186V, Ah_max=2.5823Ah; "
            "Mixed8 Ah_start≈0 → SOC_init=1.0"
        ),
        "notes": "Mixed8 = Mix8 (paper).",
    },
    # ── Training charges ───────────────────────────────────────────────────
    "10-29-18_16.50 551_Charge7": {
        "full_train": 1, "partial_train": 1, "test": 0, "exclude": 0,
        "paper_profile": "CC",
        "previous_drive_cycle": "551_Mixed1_25degC_LGHG2",
        "next_drive_cycle": "551_Mixed2_25degC_LGHG2",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "Mixed1 (full_train, partial_train) immediately precedes Charge7",
        "pairing_status": "confirmed", "confidence": "high",
        "evidence": "Charge7 at 16:50; Mixed1 at 14:41; Mixed2 at 20:16; Ah_max=2.6155Ah, V_end=4.186V",
        "notes": "Post-Mixed1/pre-Mixed2 charge; full_train and partial_train",
    },
    "10-29-18_22.29 551_Charge8": {
        "full_train": 1, "partial_train": 1, "test": 0, "exclude": 0,
        "paper_profile": "CC",
        "previous_drive_cycle": "551_Mixed2_25degC_LGHG2",
        "next_drive_cycle": "552_Mixed3_25degC_LGHG2",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "Mixed2 (full_train, partial_train) immediately precedes Charge8",
        "pairing_status": "confirmed", "confidence": "high",
        "evidence": (
            "Charge8 at 22:29; Mixed2 at 20:16; Cap_1C (excl) then Mixed3 next day; "
            "Ah_max=2.6144Ah; Ah_end=0 (counter reset at end — cycler behavior)"
        ),
        "notes": "Post-Mixed2 charge; full_train and partial_train; Ah counter reset at end",
    },
    "10-30-18_03.00 552_Charge9": {
        "full_train": 1, "partial_train": 1, "test": 0, "exclude": 0,
        "paper_profile": "CC",
        "previous_drive_cycle": "551_Mixed2_25degC_LGHG2",
        "next_drive_cycle": "552_Mixed3_25degC_LGHG2",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": (
            "Cap_1C (characterization, excluded) precedes Charge9; "
            "following-cycle fallback: Mixed3 is full_train+partial_train → Charge9 = same"
        ),
        "pairing_status": "confirmed", "confidence": "high",
        "evidence": (
            "Charge9 at 03:00; Cap_1C (excl) at 01:54; Mixed3 at 06:27; "
            "Ah_max=2.7086Ah (recharges ~2.69Ah discharged by Cap_1C=2.6928Ah); "
            "Ah_end=0 (counter reset); V_end=4.186V"
        ),
        "notes": (
            "Pre-Mixed3 charge (post-Cap_1C); "
            "second 1C capacity test (README: 'second is performed before the Mix3 cycle'); "
            "full_train and partial_train; Ah counter reset at end"
        ),
    },
    "10-30-18_08.31 552_Charge10": {
        "full_train": 1, "partial_train": 1, "test": 0, "exclude": 0,
        "paper_profile": "CC",
        "previous_drive_cycle": "552_Mixed3_25degC_LGHG2",
        "next_drive_cycle": "552_Mixed4_25degC_LGHG2",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "Mixed3 (full_train, partial_train) immediately precedes Charge10",
        "pairing_status": "confirmed", "confidence": "high",
        "evidence": "Charge10 at 08:31; Mixed3 at 06:27; Mixed4 at 11:56; Ah_max=2.5869Ah",
        "notes": "Post-Mixed3/pre-Mixed4 charge; full_train and partial_train",
    },
    "10-30-18_14.12 552_Charge11": {
        "full_train": 1, "partial_train": 0, "test": 0, "exclude": 0,
        "paper_profile": "CC",
        "previous_drive_cycle": "552_Mixed4_25degC_LGHG2",
        "next_drive_cycle": "552_Mixed5_25degC_LGHG2",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "Mixed4 (full_train) immediately precedes Charge11",
        "pairing_status": "confirmed", "confidence": "high",
        "evidence": "Charge11 at 14:12; Mixed4 at 11:56; Mixed5 at 17:36; Ah_max=2.5825Ah",
        "notes": "Post-Mixed4/pre-Mixed5 charge; full_train only",
    },
    "10-30-18_19.38 552_Charge12": {
        "full_train": 1, "partial_train": 0, "test": 0, "exclude": 0,
        "paper_profile": "CC",
        "previous_drive_cycle": "552_Mixed5_25degC_LGHG2",
        "next_drive_cycle": "552_Mixed6_25degC_LGHG2",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "Mixed5 (full_train) immediately precedes Charge12",
        "pairing_status": "confirmed", "confidence": "high",
        "evidence": "Charge12 at 19:38; Mixed5 at 17:36; Mixed6 at 23:02; Ah_max=2.5816Ah",
        "notes": "Post-Mixed5/pre-Mixed6 charge; full_train only",
    },
    "10-31-18_01.13 552_Charge13": {
        "full_train": 1, "partial_train": 0, "test": 0, "exclude": 0,
        "paper_profile": "CC",
        "previous_drive_cycle": "552_Mixed6_25degC_LGHG2",
        "next_drive_cycle": "552_Mixed7_25degC_LGHG2",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "Mixed6 (full_train) immediately precedes Charge13",
        "pairing_status": "confirmed", "confidence": "high",
        "evidence": "Charge13 at 01:13; Mixed6 at 23:02 (prev day); Mixed7 at 04:38; Ah_max=2.5831Ah",
        "notes": "Post-Mixed6/pre-Mixed7 charge; full_train only",
    },
    "10-31-18_06.41 552_Charge14": {
        "full_train": 1, "partial_train": 0, "test": 0, "exclude": 0,
        "paper_profile": "CC",
        "previous_drive_cycle": "552_Mixed7_25degC_LGHG2",
        "next_drive_cycle": "552_Mixed8_25degC_LGHG2",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "Mixed7 (full_train) immediately precedes Charge14",
        "pairing_status": "confirmed", "confidence": "high",
        "evidence": "Charge14 at 06:41; Mixed7 at 04:38; Mixed8 at 10:05; Ah_max=2.5823Ah",
        "notes": "Post-Mixed7/pre-Mixed8 charge; full_train only",
    },
    "10-31-18_12.30 552_Charge15": {
        "full_train": 1, "partial_train": 0, "test": 0, "exclude": 0,
        "paper_profile": "CC",
        "previous_drive_cycle": "552_Mixed8_25degC_LGHG2",
        "next_drive_cycle": "",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "Mixed8 (full_train) immediately precedes Charge15",
        "pairing_status": "confirmed", "confidence": "high",
        "evidence": "Charge15 at 12:30; Mixed8 at 10:05; Ah_max=2.5879Ah",
        "notes": "Post-Mixed8 charge; full_train only",
    },
    # ── Post-session files (exclude) ───────────────────────────────────────
    "10-30-18_01.54 552_Cap_1C_25degC_LGHG2": {
        "full_train": 0, "partial_train": 0, "test": 0, "exclude": 1,
        "paper_profile": "",
        "previous_drive_cycle": "", "next_drive_cycle": "",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "1C capacity test (second) — characterization; excluded",
        "pairing_status": "not_applicable", "confidence": "high",
        "evidence": "README: second 1C discharge test performed before Mix3 cycle",
        "notes": "1C capacity test (second, before Mixed3) — excluded",
    },
    "10-31-18_15.54 552_PausCycl_25degC_LGHG2": {
        "full_train": 0, "partial_train": 0, "test": 0, "exclude": 1,
        "paper_profile": "",
        "previous_drive_cycle": "552_Mixed8_25degC_LGHG2",
        "next_drive_cycle": "",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "PausCycl is a rest/pause file — excluded per protocol",
        "pairing_status": "not_applicable", "confidence": "high",
        "evidence": "PausCycl: n=63019, I_min=0, I_max=0.669A (tiny), Ah_end=0.0153; rest/top-off",
        "notes": "Post-Mixed8 pause/rest cycle — excluded",
    },
    "10-31-18_17.41 552_Charge16": {
        "full_train": 0, "partial_train": 0, "test": 0, "exclude": 1,
        "paper_profile": "",
        "previous_drive_cycle": "552_PausCycl_25degC_LGHG2",
        "next_drive_cycle": "",
        "soc_anchor_source": "", "soc_init": "",
        "split_assignment_basis": "PausCycl (excluded) precedes Charge16; no following drive cycle",
        "pairing_status": "not_applicable", "confidence": "high",
        "evidence": "Charge16 at 17:41; PausCycl ends before this; no subsequent drive cycle; V_end=0.000V (possible data quality issue)",
        "notes": "Post-session charge; excluded. V_end=0.000V is suspicious — possible data recording error at file end.",
    },
}

# ---------------------------------------------------------------------------
# Required meas fields (same as Panasonic plus Wh, Power)
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = ["Time", "TimeStamp", "Voltage", "Current", "Ah", "Battery_Temp_degC"]
EXTRA_FIELDS    = ["Wh", "Power"]  # present in LG HG2 but not required

# ---------------------------------------------------------------------------
# Manifest column order
# ---------------------------------------------------------------------------
MANIFEST_COLS = [
    "battery", "temperature_c", "filename", "relative_path",
    "start_datetime", "profile", "profile_number", "paper_profile",
    "file_type", "is_standalone", "is_combined", "is_duplicate", "duplicate_of",
    "previous_drive_cycle", "next_drive_cycle",
    "soc_anchor_source", "soc_init",
    "split_assignment_basis", "pairing_status",
    "full_train", "partial_train", "test", "exclude",
    "confidence", "evidence", "notes",
    "qc_status", "qc_notes",
    "n_rows", "duration_s", "median_dt_s",
    "first_timestamp", "last_timestamp",
    "voltage_min", "voltage_max",
    "current_min", "current_max",
    "ah_start", "ah_end", "ah_max",
    "temp_min", "temp_max", "temp_mean",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flag(row: dict, col: str) -> bool:
    return bool(int(row.get(col, 0)))


def parse_filename_datetime(stem: str) -> str:
    """Extract ISO datetime from filename stem like 10-29-18_14.41 ..."""
    m = re.match(r"(\d{2})-(\d{2})-(\d{2})_(\d{2})\.(\d{2})", stem)
    if not m:
        return ""
    mo, dy, yr, hh, mm = m.groups()
    try:
        dt = datetime(2000 + int(yr), int(mo), int(dy), int(hh), int(mm))
        return dt.isoformat()
    except ValueError:
        return ""


def parse_profile_number(stem: str) -> str:
    """Extract numeric suffix from Mixed/profile name."""
    m = re.search(r"Mixed(\d+)", stem)
    if m:
        return m.group(1)
    return ""


def detect_profile(stem: str) -> tuple[str, str]:
    """Return (profile_name, file_type) for a file stem."""
    for pattern, name, ftype in PROFILE_RULES:
        if re.search(pattern, stem, re.IGNORECASE):
            return name, ftype
    return "Unknown", "unknown"


def load_mat_qc(path: Path) -> dict:
    """Load .mat file and extract QC metrics. Returns dict."""
    result: dict[str, Any] = {
        "qc_status": "PASS", "qc_notes": "",
        "n_rows": "", "duration_s": "", "median_dt_s": "",
        "first_timestamp": "", "last_timestamp": "",
        "voltage_min": "", "voltage_max": "",
        "current_min": "", "current_max": "",
        "ah_start": "", "ah_end": "", "ah_max": "",
        "temp_min": "", "temp_max": "", "temp_mean": "",
    }
    warnings: list[str] = []
    try:
        mat = scipy.io.loadmat(str(path), simplify_cells=True)
    except Exception as e:
        result["qc_status"] = "FAIL"
        result["qc_notes"] = f"loadmat error: {e}"
        return result

    if "meas" not in mat:
        result["qc_status"] = "FAIL"
        result["qc_notes"] = "No meas struct found"
        return result

    m = mat["meas"]

    # Check required fields
    missing = [f for f in REQUIRED_FIELDS if f not in m]
    if missing:
        result["qc_status"] = "FAIL"
        result["qc_notes"] = f"Missing fields: {missing}"
        return result

    t   = np.atleast_1d(m["Time"]).flatten().astype(float)
    v   = np.atleast_1d(m["Voltage"]).flatten().astype(float)
    i   = np.atleast_1d(m["Current"]).flatten().astype(float)
    ah  = np.atleast_1d(m["Ah"]).flatten().astype(float)
    tmp = np.atleast_1d(m["Battery_Temp_degC"]).flatten().astype(float)
    ts  = np.atleast_1d(m["TimeStamp"]).flatten()

    n = len(t)
    result["n_rows"] = n

    # Check consistent lengths
    lengths = {"Time": n, "Voltage": len(v), "Current": len(i),
               "Ah": len(ah), "Battery_Temp_degC": len(tmp)}
    if len(set(lengths.values())) > 1:
        warnings.append(f"Inconsistent field lengths: {lengths}")

    # Duration and sampling
    dur = float(t[-1]) - float(t[0])
    result["duration_s"] = round(dur, 3)
    if n > 1:
        dts = np.diff(t)
        result["median_dt_s"] = round(float(np.median(dts)), 4)
        if np.any(dts < 0):
            warnings.append("Time not monotonically non-decreasing")

    # NaN checks
    for name_, arr in [("Voltage", v), ("Current", i), ("Ah", ah)]:
        if np.any(np.isnan(arr)):
            warnings.append(f"NaN found in {name_}")

    # Timestamps
    if len(ts) > 0:
        result["first_timestamp"] = str(ts[0]) if ts[0] != "" else ""
        result["last_timestamp"]  = str(ts[-1]) if ts[-1] != "" else ""

    # Signal stats
    result["voltage_min"]  = round(float(np.nanmin(v)),  4)
    result["voltage_max"]  = round(float(np.nanmax(v)),  4)
    result["current_min"]  = round(float(np.nanmin(i)),  4)
    result["current_max"]  = round(float(np.nanmax(i)),  4)
    result["ah_start"]     = round(float(ah[0]),          4)
    result["ah_end"]       = round(float(ah[-1]),         4)
    result["ah_max"]       = round(float(np.nanmax(ah)),  4)
    result["temp_min"]     = round(float(np.nanmin(tmp)), 2)
    result["temp_max"]     = round(float(np.nanmax(tmp)), 2)
    result["temp_mean"]    = round(float(np.nanmean(tmp)), 2)

    # Temperature range check for selected files (informational)
    if result["temp_mean"] < 15 or result["temp_mean"] > 40:
        warnings.append(f"WARN: mean temp {result['temp_mean']}°C outside expected 15-40°C range")

    # HWFET-specific check
    if "HWFET" in path.name:
        if result["current_min"] == 0.0 and result["current_max"] == 0.0:
            warnings.append(
                "DATA QUALITY FAILURE: the public HWFET-labelled file contains no "
                "drive-cycle excitation (zero current throughout — rest/pause only)"
            )
            result["qc_status"] = "FAIL"

    if warnings:
        if result["qc_status"] == "PASS":
            # Promote WARN notes but keep status as PASS unless already FAIL
            if any("FAILURE" in w for w in warnings):
                result["qc_status"] = "FAIL"
            else:
                result["qc_status"] = "WARN"
        result["qc_notes"] = "; ".join(warnings)

    return result


# ---------------------------------------------------------------------------
# Core discovery and manifest generation
# ---------------------------------------------------------------------------

def discover_files(raw_base: Path) -> list[Path]:
    """Return sorted list of all .mat files in the 25°C directory."""
    return sorted(raw_base.glob("*.mat"))


def build_manifest(raw_base: Path) -> list[dict]:
    """Scan all .mat files, classify, QC-check, and return manifest rows."""
    rows: list[dict] = []
    mat_files = discover_files(raw_base)

    if not mat_files:
        print(f"ERROR: No .mat files found in {raw_base}", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Loading .mat files ===")
    for p in mat_files:
        stem = p.stem
        profile_raw, file_type = detect_profile(stem)
        profile_num  = parse_profile_number(stem)
        # For Mixed files, include the number in the profile name
        if profile_raw == "Mixed" and profile_num:
            profile_raw = f"Mixed{profile_num}"
        start_dt     = parse_filename_datetime(stem)
        qc           = load_mat_qc(p)

        # Look up split assignment
        sa = SPLIT_ASSIGNMENTS.get(stem, {})
        if not sa:
            # File not in catalogue — exclude by default, flag as unknown
            sa = {
                "full_train": 0, "partial_train": 0, "test": 0, "exclude": 1,
                "paper_profile": "",
                "previous_drive_cycle": "", "next_drive_cycle": "",
                "soc_anchor_source": "", "soc_init": "",
                "split_assignment_basis": "Not in split catalogue — excluded by default",
                "pairing_status": "unknown", "confidence": "low",
                "evidence": "No explicit rule defined for this file",
                "notes": f"WARN: {stem} not in SPLIT_ASSIGNMENTS; excluded by default",
            }

        status_sym = "✓" if qc["qc_status"] in ("PASS", "WARN") else "✗"
        status_str = qc["qc_status"]
        print(f"  {status_sym} {p.name:<52} {status_str}")
        if qc.get("qc_notes"):
            # Print warnings for any selected or QC-FAIL files
            if qc["qc_status"] in ("WARN", "FAIL") or sa.get("full_train") or sa.get("test"):
                print(f"      → {qc['qc_notes']}")

        rel = p.relative_to(REPO_ROOT)
        row = {
            "battery":          "lg_hg2",
            "temperature_c":    25,
            "filename":         p.name,
            "relative_path":    str(rel),
            "start_datetime":   start_dt,
            "profile":          profile_raw,
            "profile_number":   profile_num,
            "paper_profile":    sa.get("paper_profile", ""),
            "file_type":        file_type,
            "is_standalone":    1,
            "is_combined":      0,
            "is_duplicate":     0,
            "duplicate_of":     "",
            "previous_drive_cycle":  sa.get("previous_drive_cycle", ""),
            "next_drive_cycle":      sa.get("next_drive_cycle", ""),
            "soc_anchor_source":     sa.get("soc_anchor_source", ""),
            "soc_init":              sa.get("soc_init", ""),
            "split_assignment_basis": sa.get("split_assignment_basis", ""),
            "pairing_status":        sa.get("pairing_status", ""),
            "full_train":       int(sa.get("full_train", 0)),
            "partial_train":    int(sa.get("partial_train", 0)),
            "test":             int(sa.get("test", 0)),
            "exclude":          int(sa.get("exclude", 0)),
            "confidence":       sa.get("confidence", ""),
            "evidence":         sa.get("evidence", ""),
            "notes":            sa.get("notes", ""),
            **qc,
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def run_validation(rows: list[dict]) -> bool:
    """Run all 15 validation checks. Returns True if all pass."""
    passed = True
    errors: list[str] = []

    print("\n" + "=" * 70)
    print("VALIDATION — LG HG2 25°C Vidal 2019 Split")
    print("=" * 70)

    # 1. Unique row paths
    paths_all = [r["relative_path"] for r in rows]
    if len(paths_all) != len(set(paths_all)):
        errors.append("[1] Duplicate relative_path entries found")
    else:
        print("[1] Unique paths: PASS")

    # 2. Files on disk
    missing_files = [r["filename"] for r in rows
                     if (sa := SPLIT_ASSIGNMENTS.get(Path(r["filename"]).stem))
                     and (sa.get("full_train") or sa.get("partial_train") or sa.get("test"))
                     and not (RAW_BASE / r["filename"]).exists()]
    selected_rows = [r for r in rows if flag(r, "full_train") or flag(r, "test")]
    missing_on_disk = [r["filename"] for r in selected_rows
                       if not (RAW_BASE / r["filename"]).exists()]
    if missing_on_disk:
        errors.append(f"[2] Files missing on disk: {missing_on_disk}")
    else:
        print(f"[2] Files on disk: PASS — all {len(selected_rows)} selected files found")

    # 3. Temperature (all rows are from 25degC subdirectory — verified by path)
    non_25c = [r["filename"] for r in selected_rows
               if "25degC" not in r["relative_path"] and "Charge" not in r["filename"]
               and "PausCycl" not in r["filename"]]
    if non_25c:
        errors.append(f"[3] Non-25°C files in selected set: {non_25c}")
    else:
        print("[3] Temperature 25°C: PASS")

    # 4. All expected Mix profiles found
    train_profiles = {r["profile"] for r in rows if flag(r, "full_train")
                      and r["file_type"] == "drive_cycle"}
    expected_mixed = {f"Mixed{i}" for i in range(1, 9)}
    missing_mixed = expected_mixed - train_profiles
    if missing_mixed:
        errors.append(f"[4] Missing training profiles: {missing_mixed}")
    else:
        print(f"[4] All Mix1–8 profiles found: PASS ({sorted(train_profiles)})")

    # 5. UDDS, LA92, US06 found in test
    test_profiles = {r["profile"] for r in rows if flag(r, "test")
                     and r["file_type"] == "drive_cycle"}
    expected_test = {"UDDS", "LA92", "US06"}
    missing_test = expected_test - test_profiles
    if missing_test:
        errors.append(f"[5] Missing test drive cycles: {missing_test}")
    else:
        print(f"[5] UDDS, LA92, US06 present in test: PASS")

    # 6. No selected HWFET
    selected_hwfet = [r["filename"] for r in rows
                      if r["profile"] == "HWFET"
                      and (flag(r, "full_train") or flag(r, "test"))]
    if selected_hwfet:
        errors.append(f"[6] HWFET appears in selected set (should be excluded): {selected_hwfet}")
    else:
        print("[6] HWFET excluded: PASS")

    # 7. No full_train ∩ test
    overlap_ft_test = [r["filename"] for r in rows
                       if flag(r, "full_train") and flag(r, "test")]
    if overlap_ft_test:
        errors.append(f"[7] full_train ∩ test overlap: {overlap_ft_test}")
    else:
        print("[7] full_train ∩ test: PASS — no overlap")

    # 8. No partial_train ∩ test
    overlap_pt_test = [r["filename"] for r in rows
                       if flag(r, "partial_train") and flag(r, "test")]
    if overlap_pt_test:
        errors.append(f"[8] partial_train ∩ test overlap: {overlap_pt_test}")
    else:
        print("[8] partial_train ∩ test: PASS — no overlap")

    # 9. No excluded file has any selected split flag
    bad_exclude = [r["filename"] for r in rows
                   if flag(r, "exclude") and (flag(r, "full_train") or flag(r, "test"))]
    if bad_exclude:
        errors.append(f"[9] Excluded file with split flag: {bad_exclude}")
    else:
        print("[9] exclude ∩ split-flag: PASS — no overlap")

    # 10. partial_train is strict subset of full_train
    pt_only = [r["filename"] for r in rows
               if flag(r, "partial_train") and not flag(r, "full_train")]
    if pt_only:
        errors.append(f"[10] partial_train but not full_train: {pt_only}")
    else:
        print("[10] partial_train ⊆ full_train: PASS")

    # 11. Filename uniqueness across train and test.
    # NOTE: Only filename/path uniqueness is checked here. Signal-level duplicate
    # detection (e.g. hash or trajectory comparison across files) was NOT performed.
    # See §11 of the split report for the rationale.
    all_selected = [r["filename"] for r in rows
                    if flag(r, "full_train") or flag(r, "test")]
    if len(all_selected) != len(set(all_selected)):
        errors.append("[11] Duplicate filenames in selected set")
    else:
        print(f"[11] Filename uniqueness across train/test: PASS"
              f" (signal-level duplicate detection not performed)")

    # 12. Selected files pass required-field QC
    qc_fail = [r["filename"] for r in rows
               if (flag(r, "full_train") or flag(r, "test"))
               and r.get("qc_status") == "FAIL"]
    qc_warn = [r["filename"] for r in rows
               if (flag(r, "full_train") or flag(r, "test"))
               and r.get("qc_status") == "WARN"]
    if qc_fail:
        errors.append(f"[12] Selected files FAIL QC: {qc_fail}")
    else:
        warn_note = f" ({len(qc_warn)} WARN)" if qc_warn else ""
        print(f"[12] Selected files QC: PASS{warn_note}")
    if qc_warn:
        for fn in qc_warn:
            r = next(x for x in rows if x["filename"] == fn)
            print(f"      WARN: {fn}: {r.get('qc_notes','')}")

    # 13. Charge assignments have evidence
    charge_rows = [r for r in rows
                   if r["file_type"] == "charge"
                   and (flag(r, "full_train") or flag(r, "test"))]
    missing_evidence = [r["filename"] for r in charge_rows
                        if not r.get("evidence")]
    if missing_evidence:
        errors.append(f"[13] Charge files without evidence: {missing_evidence}")
    else:
        print(f"[13] All charge assignments have evidence: PASS ({len(charge_rows)} charges)")

    # 14. Unresolved charge relationships remain unresolved
    unresolved = [r for r in rows
                  if r.get("pairing_status") == "unresolved"
                  and (flag(r, "full_train") or flag(r, "test"))]
    if unresolved:
        print(f"[14] Unresolved charge relationships: {len(unresolved)}")
        for r in unresolved:
            print(f"      - {r['filename']}: {r.get('evidence', 'no evidence')}")
    else:
        print("[14] Unresolved charge relationships: NONE"
              " (all chronologically resolved under the documented reconstruction rule)")

    # 15. Count/order consistency.
    # Verifies that the sum of split-flag counts equals the total manifest row count.
    # A hash-based repeated-run check is NOT included; to verify full determinism,
    # run the script twice and diff the manifest CSVs.
    n_total = len(rows)
    n_full_train = sum(1 for r in rows if flag(r, "full_train"))
    n_test       = sum(1 for r in rows if flag(r, "test"))
    n_exclude    = sum(1 for r in rows if flag(r, "exclude"))
    n_sum = n_full_train + n_test + n_exclude
    if n_sum != n_total:
        errors.append(
            f"[15] Count mismatch: full_train({n_full_train}) + test({n_test}) "
            f"+ exclude({n_exclude}) = {n_sum} ≠ total({n_total})"
        )
    else:
        print(f"[15] Count/order consistency: PASS ({n_total} = {n_full_train}+{n_test}+{n_exclude})"
              f" — no hash check")

    if errors:
        print(f"\n{'='*70}")
        print(f"VALIDATION FAILURES ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")
        passed = False
    return passed


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict]) -> None:
    """Print human-readable split summary."""
    print("\n" + "=" * 70)
    print("SUMMARY — LG HG2 25°C Vidal 2019 Split")
    print("=" * 70)

    n_total      = len(rows)
    ft_drive     = [r for r in rows if flag(r, "full_train") and r["file_type"] == "drive_cycle"]
    ft_charge    = [r for r in rows if flag(r, "full_train") and r["file_type"] == "charge"]
    pt_drive     = [r for r in rows if flag(r, "partial_train") and r["file_type"] == "drive_cycle"]
    pt_charge    = [r for r in rows if flag(r, "partial_train") and r["file_type"] == "charge"]
    tst_drive    = [r for r in rows if flag(r, "test") and r["file_type"] == "drive_cycle"]
    tst_charge   = [r for r in rows if flag(r, "test") and r["file_type"] == "charge"]
    excl         = [r for r in rows if flag(r, "exclude")]
    unresolved   = [r for r in rows if r.get("pairing_status") == "unresolved"]

    print(f"\nTotal files in manifest                     : {n_total}")
    print(f"  full_train=1                              : {len(ft_drive)+len(ft_charge)}  "
          f"({len(ft_drive)} drive + {len(ft_charge)} charge)")
    print(f"  partial_train=1  (subset of full_train)   : {len(pt_drive)+len(pt_charge)}  "
          f"({len(pt_drive)} drive + {len(pt_charge)} charge)")
    print(f"  test=1                                    : {len(tst_drive)+len(tst_charge)}  "
          f"({len(tst_drive)} drive + {len(tst_charge)} charge)")
    print(f"  exclude=1                                 : {len(excl)}")
    print(f"  (full_train + test + exclude)             : "
          f"{len(ft_drive)+len(ft_charge)+len(tst_drive)+len(tst_charge)+len(excl)}")

    print("\n--- Raw name → Paper profile mapping (inferred from count + README) ---")
    print("  Raw dataset uses 'Mixed1' through 'Mixed8' (not 'Mix1-8' as in paper)")
    print("  Mapping:")
    for i in range(1, 9):
        print(f"    Mixed{i} → Mix{i}")
    print("  Vidal 2019 (ITEC) Table I: LG HG2 profiles = Mix(1-8), UDDS, LA92, US06, CC")

    print("\n--- HWFET 25°C STATUS ---")
    hwfet = next((r for r in rows if "HWFET" in r["filename"]), None)
    if hwfet:
        print(f"  {hwfet['filename']}: {hwfet['qc_status']}")
        print(f"  → {hwfet.get('qc_notes','')}")
        print("  Decision: EXCLUDED (the public HWFET-labelled file contains no drive-cycle excitation)")

    print("\n--- Charge Sequence Summary (non-excluded drive cycles) ---")
    drive_rows = [r for r in rows if r["file_type"] == "drive_cycle"
                  and not flag(r, "exclude")]
    for r in drive_rows:
        sc_raw = r.get("soc_anchor_source", "")
        ps     = r.get("pairing_status", "")
        split  = ("full_train" if flag(r, "full_train")
                  else "test" if flag(r, "test") else "?")
        print(f"  {r['profile']:10s}  [{split:10s}]  soc_anchor: {sc_raw or 'n/a':30s}  [{ps}]")

    if unresolved:
        print(f"\n  ⚠ UNRESOLVED pairings: {len(unresolved)}")
        for r in unresolved:
            print(f"    - {r['filename']}: {r.get('evidence','')}")
    else:
        print("\n  ✓ Unresolved pairings: NONE")


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------

def write_manifest(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] Manifest written: {out_path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"Repo root : {REPO_ROOT}")
    print(f"Data root : {RAW_BASE}")

    if not RAW_BASE.exists():
        print(f"ERROR: Raw base directory not found: {RAW_BASE}", file=sys.stderr)
        return 1

    rows    = build_manifest(RAW_BASE)
    ok      = run_validation(rows)
    print_summary(rows)
    write_manifest(rows, MANIFEST_PATH)

    if ok:
        print("\nDone. Manifest and validation PASS.")
        return 0
    else:
        print("\nDone. Manifest written but validation FAILED — review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
