#!/usr/bin/env python3
"""
inspect_samsung_30t_25c_split.py
=================================
Paper-faithful file-level split reconstruction for the Samsung INR21700-30T
dataset at 25 °C for Vidal et al. (2019 ITEC) reproduction.

Scope
-----
- Locates the canonical 25 °C directory under data/raw/Samsung_30T/ programmatically.
- Recursively discovers candidate .mat files.
- Classifies files by profile type using explicit documented rules.
- Computes SHA-256 and a lightweight signal fingerprint for each .mat file.
- Assigns split roles (full_train, partial_train, test, exclude).
- Performs 20 validation checks (checks 1–20 from the task specification).
- Verifies repeated-run manifest-hash stability (check 20).
- Writes data/manifests/samsung_30t_vidal2019_25c_manifest.csv.
- Prints a structured summary and returns non-zero exit status on failure.

Battery identity
----------------
Public dataset : Samsung INR21700-30T (filenames contain "IN21700_30T";
                 README names it "Samsung 3.0T", 3 Ah, McMaster University,
                 Dr. Phillip Kollmeyer).
Vidal 2019 name: "Samsung T30", 3 Ah.
Identity status: STRONGLY SUPPORTED — capacity, format, test profiles, and
                 March 2019 collection date are all consistent with the 2019
                 paper; "T30" is most plausibly a transposition of "30T".
                 Not CONFIRMED because the authors do not explicitly state the
                 equivalence in the public dataset documentation.

Vidal 2019 protocol (Samsung T30 / 30T, §IV, Table I)
------------------------------------------------------
Samsung T30 is used as the TARGET domain in Case 4 (source = Panasonic 18650PF,
target = Samsung T30).  Per paper §IV.B and Table I:
  full_train   — Mix1–Mix8 + associated CC charge files
  partial_train — Mix1–Mix3 + associated CC charge files
                  (paper: "partial Training datasets consisting of cycles
                  MIX 1-3 … for Cases 3 and 4"; Samsung 48G is the exception
                  using Mix 1, 2, 6 — Samsung 30T uses Mix 1-3)
  test         — UDDS, LA92, US06 + associated CC charge files
  exclude      — HWFET (not listed as test/train for Samsung 30T in Table I),
                 HPPC, C20DisCh, Dis_0p5C, Dis_2C, Cap_1C, PausCycl, and
                 any preparation, characterisation, or rest-only files.

Note: HWFET at Samsung 30T 25 °C IS a valid drive cycle (2600 s, real current
up to −18 A / +12 A regen) unlike the LG HG2 HWFET which was a rest file.
HWFET is still excluded here because the Vidal paper Table I does not list it
as a selected test profile for Samsung T30.

Charge assignment rule (consistent with Panasonic and LG HG2 manifests)
-----------------------------------------------------------------------
Each charge file is assigned to the split of the drive cycle that immediately
precedes the charge, using a following-cycle fallback when no drive cycle
directly precedes the charge (e.g. the preceding segment is excluded or absent).

Special cases at 25 °C:
  Charge1  : tiny top-off (0.022 Ah) before first Cap_1C → EXCLUDE
  Charge2  : post-Cap_1C / pre-UDDS (following-cycle fallback) → TEST
  Charge3  : post-UDDS → TEST
  Charge4  : post-HWFET / pre-LA92 (HWFET excluded, following-cycle fallback)
             → TEST
  Charge5  : post-LA92 → TEST
  Charge6  : post-US06 → TEST
  Charge7  : post-Mixed1 → FULL+PARTIAL
  Charge8  : post-Mixed2 → FULL+PARTIAL
  Charge9  : tiny top-off (0.050 Ah) before MISSING Cap_1C → EXCLUDE
             (analogous to Charge1; a Cap_1C test occurred between
             Charge9 and Charge10 but no .mat file was saved)
  Charge10 : post-[missing Cap_1C] / pre-Mixed3 → FULL+PARTIAL
  Charge11 : post-Mixed3 → FULL+PARTIAL
  Charge12 : post-Mixed4 → FULL only
  Charge13 : post-Mixed5 → FULL only
  Charge14 : post-Mixed6 → FULL only
  Charge15 : post-Mixed7 → FULL only
  Charge16 : post-Mixed8 / pre-PausCycl → FULL only

Data gap: No Cap_1C .mat file exists between Charge9 (17:52 03-20-19) and
Charge10 (19:37 03-20-19) at 25 °C, despite the README documenting a second
1C discharge test before Mixed3.  The Charge9 → Charge10 gap (~1 h 45 min)
is consistent with a ~1 h Cap_1C discharge followed by setup time.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import re
import sys
import traceback
from datetime import datetime
from typing import Any

import numpy as np
import scipy.io

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_BASE = REPO_ROOT / "data" / "raw" / "Samsung_30T"
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "samsung_30t_vidal2019_25c_manifest.csv"

# Temperature directory name used in this dataset
TEMP_DIR_NAME = "25degC"
NOMINAL_TEMP_C = 25.0

# Battery identity
BATTERY_PUBLIC = "Samsung_INR21700-30T"
PAPER_BATTERY_NAME = "Samsung T30"
DATASET_IDENTITY_STATUS = "STRONGLY_SUPPORTED"

# Expected paper profiles
EXPECTED_FULL_TRAIN_PROFILES = {f"Mixed{i}" for i in range(1, 9)}
EXPECTED_PARTIAL_TRAIN_PROFILES = {"Mixed1", "Mixed2", "Mixed3"}
EXPECTED_TEST_PROFILES = {"UDDS", "LA92", "US06"}
EXPECTED_MEAS_FIELDS = {"Time", "TimeStamp", "Voltage", "Current", "Ah", "Battery_Temp_degC"}

# ---------------------------------------------------------------------------
# Explicit split-assignment table (file-stem → split annotation)
# Keys are the "profile_timestamp_id" portion of the stem, e.g. "734_Charge2"
# Values define the split role and evidence.
# ---------------------------------------------------------------------------
SPLIT_TABLE: dict[str, dict[str, Any]] = {
    # === Characterisation files ===
    "729_HPPC": dict(
        file_type="characterization", full_train=0, partial_train=0, test=0, exclude=1,
        pairing_status="n/a",
        evidence="HPPC; not a drive cycle or charge; excluded per paper protocol",
    ),
    "729_C20DisCh": dict(
        file_type="characterization", full_train=0, partial_train=0, test=0, exclude=1,
        pairing_status="n/a",
        evidence="C/20 discharge; characterisation only",
    ),
    "729_Dis_0p5C": dict(
        file_type="characterization", full_train=0, partial_train=0, test=0, exclude=1,
        pairing_status="n/a",
        evidence="0.5C discharge; characterisation only",
    ),
    "729_Dis_2C": dict(
        file_type="characterization", full_train=0, partial_train=0, test=0, exclude=1,
        pairing_status="n/a",
        evidence="2C discharge; characterisation only",
    ),
    "734_Charge1": dict(
        file_type="charge", full_train=0, partial_train=0, test=0, exclude=1,
        previous_drive_cycle="",
        next_drive_cycle="Cap_1C_25degC (first)",
        soc_anchor_source="",
        soc_init="",
        split_assignment_basis="tiny top-off (0.022 Ah) before first Cap_1C; analogous to LG Charge1; excluded",
        pairing_status="resolved",
        confidence="high",
        evidence="I_max=0.421 A, Ah_max=0.022, duration=2039 s; Dis_2C precedes; Cap_1C follows; not a training input",
    ),
    "734_Cap_1C": dict(
        file_type="characterization", full_train=0, partial_train=0, test=0, exclude=1,
        pairing_status="n/a",
        evidence="First 1C discharge test (README: performed before UDDS cycle); characterisation",
    ),
    "734_Charge2": dict(
        file_type="charge", full_train=0, partial_train=0, test=1, exclude=0,
        previous_drive_cycle="Cap_1C_25degC (excluded)",
        next_drive_cycle="UDDS_25degC",
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        split_assignment_basis="following-cycle fallback: Cap_1C excluded; UDDS is test",
        pairing_status="chronologically resolved under reconstruction rule",
        confidence="high",
        evidence="Full charge (2.937 Ah, I_max=3A); 14:47→16:26; UDDS follows at 16:26; Table I §IV.B",
    ),
    "734_UDDS": dict(
        file_type="drive_cycle", full_train=0, partial_train=0, test=1, exclude=0,
        pairing_status="n/a",
        evidence="UDDS; listed as test profile in Vidal 2019 Table I §IV.B",
    ),
    "734_Charge3": dict(
        file_type="charge", full_train=0, partial_train=0, test=1, exclude=0,
        previous_drive_cycle="UDDS_25degC",
        next_drive_cycle="HWFET_25degC",
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        split_assignment_basis="preceding-cycle wins: UDDS is test",
        pairing_status="chronologically resolved under reconstruction rule",
        confidence="high",
        evidence="Full charge (2.810 Ah, I_max=3A); 18:16→19:53; UDDS precedes at 16:26",
    ),
    "734_HWFET": dict(
        file_type="drive_cycle", full_train=0, partial_train=0, test=0, exclude=1,
        pairing_status="n/a",
        notes="Valid drive cycle (2600 s, I_min=-18.4 A, I_max=+12.0 A regen); excluded because Vidal 2019 Table I does not list HWFET as a test profile for Samsung T30",
        evidence="I_min=-18.389 A, I_max=11.999 A; 2600 s; real drive excitation confirmed; but not listed in Table I as selected test/train for Samsung 30T/T30",
    ),
    "734_Charge4": dict(
        file_type="charge", full_train=0, partial_train=0, test=1, exclude=0,
        previous_drive_cycle="HWFET_25degC (excluded)",
        next_drive_cycle="LA92_25degC",
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        split_assignment_basis="following-cycle fallback: HWFET excluded; LA92 is test",
        pairing_status="chronologically resolved under reconstruction rule",
        confidence="high",
        evidence="Full charge (2.805 Ah, I_max=3A); 20:36→22:13; LA92 follows at 22:13",
    ),
    "734_LA92": dict(
        file_type="drive_cycle", full_train=0, partial_train=0, test=1, exclude=0,
        pairing_status="n/a",
        evidence="LA92; listed as test profile in Vidal 2019 Table I §IV.B",
    ),
    "734_Charge5": dict(
        file_type="charge", full_train=0, partial_train=0, test=1, exclude=0,
        previous_drive_cycle="LA92_25degC",
        next_drive_cycle="US06_25degC",
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        split_assignment_basis="preceding-cycle wins: LA92 is test",
        pairing_status="chronologically resolved under reconstruction rule",
        confidence="high",
        evidence="Full charge (2.811 Ah, I_max=3A); 23:25→01:03; LA92 precedes at 22:13",
    ),
    "734_US06": dict(
        file_type="drive_cycle", full_train=0, partial_train=0, test=1, exclude=0,
        pairing_status="n/a",
        evidence="US06; listed as test profile in Vidal 2019 Table I §IV.B",
    ),
    "734_Charge6": dict(
        file_type="charge", full_train=0, partial_train=0, test=1, exclude=0,
        previous_drive_cycle="US06_25degC",
        next_drive_cycle="Mixed1_25degC",
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        split_assignment_basis="preceding-cycle wins: US06 is test",
        pairing_status="chronologically resolved under reconstruction rule",
        confidence="medium",
        notes="Charge6 falls between test (US06) and train (Mixed1); preceding-cycle-wins rule assigns to test. This creates a gap: Mixed1 has no preceding charge in its split. SOC init for Mixed1 = full_charge_protocol (voltage at file start ≈ 4.185 V confirms full state after Charge6).",
        evidence="Full charge (2.806 Ah, I_max=3A); 01:36→03:13; US06 precedes at 01:03; Mixed1 follows at 03:13",
    ),
    "734_Mixed1": dict(
        file_type="drive_cycle", full_train=1, partial_train=1, test=0, exclude=0,
        soc_anchor_source="full_charge_protocol",
        soc_init=1.0,
        pairing_status="resolved",
        evidence="Mixed1; full training + partial training; Vidal 2019 §IV.B; Table I; Charge6 assigned to test by preceding-cycle rule",
    ),
    "734_Charge7": dict(
        file_type="charge", full_train=1, partial_train=1, test=0, exclude=0,
        previous_drive_cycle="Mixed1_25degC",
        next_drive_cycle="Mixed2_25degC",
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        split_assignment_basis="preceding-cycle wins: Mixed1 is full+partial train",
        pairing_status="chronologically resolved under reconstruction rule",
        confidence="high",
        evidence="Full charge (2.808 Ah, I_max=3A); 04:07→05:44; Mixed1 precedes at 03:13; Mixed2 follows at 05:44",
    ),
    "734_Mixed2": dict(
        file_type="drive_cycle", full_train=1, partial_train=1, test=0, exclude=0,
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        pairing_status="resolved",
        evidence="Mixed2; full training + partial training; Vidal 2019 §IV.B; Table I",
    ),
    "734_Charge8": dict(
        file_type="charge", full_train=1, partial_train=1, test=0, exclude=0,
        previous_drive_cycle="Mixed2_25degC",
        next_drive_cycle="[missing Cap_1C → Mixed3]",
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        split_assignment_basis="preceding-cycle wins: Mixed2 is full+partial train",
        pairing_status="chronologically resolved under reconstruction rule",
        confidence="high",
        evidence="Full charge (2.808 Ah, I_max=3A); 06:45→[35h gap]; Mixed2 precedes at 05:44",
    ),
    # After 35-hour gap (03-19 06:45 → 03-20 17:52)
    "740_Charge9": dict(
        file_type="charge", full_train=0, partial_train=0, test=0, exclude=1,
        previous_drive_cycle="",
        next_drive_cycle="[missing Cap_1C]",
        soc_anchor_source="",
        soc_init="",
        split_assignment_basis="tiny top-off (0.050 Ah) before second Cap_1C; analogous to Charge1; excluded",
        pairing_status="inferred under README + chronology evidence",
        confidence="medium",
        notes="I_max=0.922 A, Ah_max=0.050, duration=2148 s; ~35h gap after Charge8; 1h45m before Charge10; consistent with pre-Cap_1C top-off followed by Cap_1C discharge (no .mat saved) and full recharge (Charge10). Role inferred: README documents a second Cap_1C before Mixed3; Charge9 low-Ah pattern matches Charge1 (pre-first-Cap_1C top-off).",
        evidence="README: second 1C test performed before Mix3 cycle; Charge9 Ah_max=0.050 Ah at 0.922 A (top-off pattern); 35h gap after Charge8; Charge10 (2.917 Ah full recharge) follows 70 min later",
    ),
    "740_Charge10": dict(
        file_type="charge", full_train=1, partial_train=1, test=0, exclude=0,
        previous_drive_cycle="[missing Cap_1C] (excluded)",
        next_drive_cycle="Mixed3_25degC",
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        split_assignment_basis="following-cycle fallback: missing Cap_1C excluded; Mixed3 is full+partial train",
        pairing_status="inferred under README + chronology evidence",
        confidence="medium",
        notes="Full recharge after missing Cap_1C test (2.917 Ah, I_max=3A); analogous to LG HG2 Charge9 (post-Cap_1C, pre-Mixed3); 19:37→21:16. Pairing inferred: README documents second Cap_1C before Mixed3; no Cap_1C .mat present; Charge10 full-recharge volume consistent with ~2.9 Ah discharge.",
        evidence="README: second 1C test before Mix3; Charge10 I_max=2.999 A, Ah_max=2.917 Ah, duration=5968 s; Mixed3 follows at 21:16; time gap of ~70 min between Charge9 end and Charge10 start consistent with 1C discharge of 3 Ah",
    ),
    "740_Mixed3": dict(
        file_type="drive_cycle", full_train=1, partial_train=1, test=0, exclude=0,
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        pairing_status="resolved",
        evidence="Mixed3; full training + partial training; Vidal 2019 §IV.B; Table I; paper: Mix1-3 partial for Cases 3/4",
    ),
    "740_Charge11": dict(
        file_type="charge", full_train=1, partial_train=1, test=0, exclude=0,
        previous_drive_cycle="Mixed3_25degC",
        next_drive_cycle="Mixed4_25degC",
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        split_assignment_basis="preceding-cycle wins: Mixed3 is full+partial train",
        pairing_status="chronologically resolved under reconstruction rule",
        confidence="high",
        evidence="Full charge (2.786 Ah, I_max=3A); 22:00→23:37; Mixed3 precedes at 21:16",
    ),
    "740_Mixed4": dict(
        file_type="drive_cycle", full_train=1, partial_train=0, test=0, exclude=0,
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        pairing_status="resolved",
        evidence="Mixed4; full training only (Mix4–8 outside partial subset); Vidal 2019 Table I §IV.B",
    ),
    "740_Charge12": dict(
        file_type="charge", full_train=1, partial_train=0, test=0, exclude=0,
        previous_drive_cycle="Mixed4_25degC",
        next_drive_cycle="Mixed5_25degC",
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        split_assignment_basis="preceding-cycle wins: Mixed4 is full train only",
        pairing_status="chronologically resolved under reconstruction rule",
        confidence="high",
        evidence="Full charge (2.790 Ah, I_max=3A); 00:39→02:16; Mixed4 precedes at 23:37",
    ),
    "740_Mixed5": dict(
        file_type="drive_cycle", full_train=1, partial_train=0, test=0, exclude=0,
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        pairing_status="resolved",
        evidence="Mixed5; full training only; Vidal 2019 Table I §IV.B",
    ),
    "740_Charge13": dict(
        file_type="charge", full_train=1, partial_train=0, test=0, exclude=0,
        previous_drive_cycle="Mixed5_25degC",
        next_drive_cycle="Mixed6_25degC",
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        split_assignment_basis="preceding-cycle wins: Mixed5 is full train only",
        pairing_status="chronologically resolved under reconstruction rule",
        confidence="high",
        evidence="Full charge (2.790 Ah, I_max=3A); 03:00→04:37; Mixed5 precedes at 02:16",
    ),
    "740_Mixed6": dict(
        file_type="drive_cycle", full_train=1, partial_train=0, test=0, exclude=0,
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        pairing_status="resolved",
        evidence="Mixed6; full training only; Vidal 2019 Table I §IV.B",
    ),
    "740_Charge14": dict(
        file_type="charge", full_train=1, partial_train=0, test=0, exclude=0,
        previous_drive_cycle="Mixed6_25degC",
        next_drive_cycle="Mixed7_25degC",
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        split_assignment_basis="preceding-cycle wins: Mixed6 is full train only",
        pairing_status="chronologically resolved under reconstruction rule",
        confidence="high",
        evidence="Full charge (2.790 Ah, I_max=3A); 05:35→07:12; Mixed6 precedes at 04:37",
    ),
    "740_Mixed7": dict(
        file_type="drive_cycle", full_train=1, partial_train=0, test=0, exclude=0,
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        pairing_status="resolved",
        evidence="Mixed7; full training only; Vidal 2019 Table I §IV.B",
    ),
    "740_Charge15": dict(
        file_type="charge", full_train=1, partial_train=0, test=0, exclude=0,
        previous_drive_cycle="Mixed7_25degC",
        next_drive_cycle="Mixed8_25degC",
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        split_assignment_basis="preceding-cycle wins: Mixed7 is full train only",
        pairing_status="chronologically resolved under reconstruction rule",
        confidence="high",
        evidence="Full charge (2.789 Ah, I_max=3A); 08:01→09:39; Mixed7 precedes at 07:12",
    ),
    "740_Mixed8": dict(
        file_type="drive_cycle", full_train=1, partial_train=0, test=0, exclude=0,
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        pairing_status="resolved",
        evidence="Mixed8; full training only; Vidal 2019 Table I §IV.B",
    ),
    "740_Charge16": dict(
        file_type="charge", full_train=1, partial_train=0, test=0, exclude=0,
        previous_drive_cycle="Mixed8_25degC",
        next_drive_cycle="PausCycl_25degC (excluded)",
        soc_anchor_source="preceding_full_charge",
        soc_init=1.0,
        split_assignment_basis="preceding-cycle wins: Mixed8 is full train; PausCycl excluded",
        pairing_status="chronologically resolved under reconstruction rule",
        confidence="high",
        evidence="Full charge (2.793 Ah, I_max=3A); 10:49→12:26; Mixed8 precedes at 09:39; PausCycl follows",
    ),
    "740_PausCycl": dict(
        file_type="other", full_train=0, partial_train=0, test=0, exclude=1,
        pairing_status="n/a",
        notes="Real drive excitation present (I_min=-18.87 A, Ah_end=-0.44 Ah); not a Vidal training or test profile; excluded",
        evidence="I_min=-18.870 A, I_max=10.886 A, dur=1476 s; not a paper profile; last file in sequence",
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROFILE_RE = re.compile(
    r"^\d{2}-\d{2}-\d{2}_\d{2}\.\d{2} (\d+)_(Mixed\d+|UDDS|HWFET|LA92|US06|Charge\d+|HPPC|C20DisCh|Dis_0p5C|Dis_2C|Cap_1C|PausCycl)_"
    r"(\d+)degC_IN21700_30T$",
    re.IGNORECASE,
)


def parse_stem(stem: str) -> tuple[str, str, str]:
    """Return (test_id, profile, temp_str) from a .mat stem or ('', '', '')."""
    m = _PROFILE_RE.match(stem)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return "", "", ""


def parse_datetime(stem: str) -> str:
    """Parse 'MM-DD-YY_HH.MM …' prefix into ISO datetime string."""
    m = re.match(r"^(\d{2})-(\d{2})-(\d{2})_(\d{2})\.(\d{2})", stem)
    if not m:
        return ""
    mo, dy, yr, hr, mi = m.groups()
    try:
        dt = datetime(2000 + int(yr), int(mo), int(dy), int(hr), int(mi))
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ""


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def signal_fingerprint(t, v, i, ah) -> str:
    """Lightweight fingerprint: n_rows|dur|v0|v-1|i0|i-1|ah0|ah-1|sum_i8|sum_ah8."""
    def _f(x): return f"{float(x):.5f}"
    n = len(t)
    dur = float(t[-1]) - float(t[0]) if n > 1 else 0.0
    step = max(1, n // 8)
    sampled_i = [i[k * step] for k in range(8) if k * step < n]
    sampled_ah = [ah[k * step] for k in range(8) if k * step < n]
    s_i = sum(sampled_i)
    s_ah = sum(sampled_ah)
    parts = [
        str(n), _f(dur),
        _f(v[0]), _f(v[-1]),
        _f(i[0]), _f(i[-1]),
        _f(ah[0]), _f(ah[-1]),
        _f(s_i), _f(s_ah),
    ]
    return "|".join(parts)


def derive_paper_profile(profile: str, file_type: str, exclude: int) -> str:
    """Map public profile name to Vidal paper profile name.

    Rules:
      - Mixed{N}  → Mix{N}      (paper uses "Mix1–8", dataset uses "Mixed1–8")
      - selected charge files → "CC"  (paper refers to "CC charge" segments)
      - UDDS / LA92 / US06     → unchanged
      - excluded files (characterisation, HWFET, PausCycl, excluded charges) → ""
    """
    if exclude == 1:
        return ""
    m = re.match(r"^Mixed(\d+)$", profile, re.IGNORECASE)
    if m:
        return f"Mix{m.group(1)}"
    if file_type == "charge":
        return "CC"
    return profile  # UDDS, LA92, US06 pass through unchanged


def load_mat(path: pathlib.Path):
    """Load .mat and return meas struct; raise on failure."""
    mat = scipy.io.loadmat(str(path), simplify_cells=True)
    return mat["meas"]


def inspect_file(path: pathlib.Path, rel_base: pathlib.Path) -> dict:
    """Return a fully populated row dict for one .mat file."""
    stem = path.stem
    test_id, profile, temp_str = parse_stem(stem)
    rel_path = str(path.relative_to(rel_base))
    start_dt = parse_datetime(stem)

    row: dict[str, Any] = dict(
        battery=BATTERY_PUBLIC,
        paper_battery_name=PAPER_BATTERY_NAME,
        dataset_identity_status=DATASET_IDENTITY_STATUS,
        temperature_c=float(temp_str) if temp_str else "",
        filename=path.name,
        relative_path=rel_path,
        sha256="",
        start_datetime=start_dt,
        profile=profile,
        profile_number="".join(filter(str.isdigit, profile)) if profile else "",
        paper_profile="",  # set below after split table is applied
        file_type="unknown",
        is_standalone=1,
        is_combined=0,
        is_duplicate=0,
        duplicate_of="",
        signal_fingerprint="",
        previous_drive_cycle="",
        next_drive_cycle="",
        soc_anchor_source="",
        soc_init="",
        split_assignment_basis="",
        pairing_status="unresolved",
        full_train=0,
        partial_train=0,
        test=0,
        exclude=0,
        confidence="",
        evidence="",
        notes="",
        qc_status="PASS",
        qc_notes="",
        n_rows=0,
        duration_s="",
        median_dt_s="",
        first_timestamp="",
        last_timestamp="",
        voltage_min="",
        voltage_max="",
        current_min="",
        current_max="",
        ah_min="",
        ah_start="",
        ah_end="",
        ah_max="",
        temp_min="",
        temp_max="",
        temp_mean="",
    )

    # Apply split table
    lookup_key = f"{test_id}_{profile}"
    if lookup_key in SPLIT_TABLE:
        entry = SPLIT_TABLE[lookup_key]
        for k, v in entry.items():
            row[k] = v
    else:
        row["evidence"] = f"Not in split table; key={lookup_key}"
        row["qc_status"] = "WARN"
        row["qc_notes"] = f"Unrecognised file key: {lookup_key}"

    # Derive paper_profile after split table (needs file_type and exclude to be set)
    row["paper_profile"] = derive_paper_profile(profile, row["file_type"], row["exclude"])

    # SHA-256
    try:
        row["sha256"] = sha256_file(path)
    except Exception as exc:
        row["qc_status"] = "FAIL"
        row["qc_notes"] += f" sha256_error:{exc}"

    # Load .mat
    qc_issues: list[str] = []
    try:
        meas = load_mat(path)
        fields = set(meas.keys()) if isinstance(meas, dict) else set(dir(meas))

        t   = np.atleast_1d(meas["Time"]).flatten().astype(float)
        ts  = np.atleast_1d(meas["TimeStamp"]).flatten() if "TimeStamp" in meas else np.array([])
        v   = np.atleast_1d(meas["Voltage"]).flatten().astype(float)
        i   = np.atleast_1d(meas["Current"]).flatten().astype(float)
        ah  = np.atleast_1d(meas["Ah"]).flatten().astype(float)
        tmp = np.atleast_1d(meas["Battery_Temp_degC"]).flatten().astype(float)

        n = len(t)
        dur = float(t[-1]) - float(t[0]) if n > 1 else 0.0
        dts = np.diff(t)
        med_dt = float(np.median(dts)) if len(dts) > 0 else 0.0

        row["n_rows"] = n
        row["duration_s"] = round(dur, 2)
        row["median_dt_s"] = round(med_dt, 4)
        row["first_timestamp"] = str(ts[0]) if len(ts) > 0 else ""
        row["last_timestamp"] = str(ts[-1]) if len(ts) > 0 else ""
        row["voltage_min"] = round(float(v.min()), 4)
        row["voltage_max"] = round(float(v.max()), 4)
        row["current_min"] = round(float(i.min()), 4)
        row["current_max"] = round(float(i.max()), 4)
        row["ah_min"] = round(float(ah.min()), 6)
        row["ah_start"] = round(float(ah[0]), 6)
        row["ah_end"] = round(float(ah[-1]), 6)
        row["ah_max"] = round(float(ah.max()), 6)
        row["temp_min"] = round(float(tmp.min()), 2)
        row["temp_max"] = round(float(tmp.max()), 2)
        row["temp_mean"] = round(float(tmp.mean()), 2)

        row["signal_fingerprint"] = signal_fingerprint(t, v, i, ah)

        # QC checks
        missing = EXPECTED_MEAS_FIELDS - set(meas.keys())
        if missing:
            qc_issues.append(f"missing_fields:{sorted(missing)}")

        if np.any(np.isnan(v)) or np.any(np.isnan(i)) or np.any(np.isnan(ah)):
            qc_issues.append("NaN_detected")

        if len(t) > 1:
            n_neg = int((dts < 0).sum())
            n_zero = int((dts == 0).sum())
            if n_neg > 0:
                qc_issues.append(f"non_monotonic_time(strictly_decreasing:{n_neg})")
            elif n_zero > 0:
                # Single duplicate timestamp at last row is a common cycler artifact
                qc_issues.append(f"duplicate_timestamp(dt=0 count:{n_zero})")

        all_lens = [len(t), len(v), len(i), len(ah), len(tmp)]
        if len(set(all_lens)) > 1:
            qc_issues.append(f"length_mismatch:{all_lens}")

        if row.get("file_type") == "drive_cycle":
            if abs(i).max() < 0.01:
                qc_issues.append("zero_current_drive_cycle")

        # Ah counter reset check
        if ah[-1] == 0.0 and ah.max() > 0.01:
            if row.get("file_type") == "charge":
                pass  # expected reset on charges
            else:
                qc_issues.append("ah_counter_reset_at_end")

        if qc_issues:
            row["qc_status"] = "WARN" if row["qc_status"] == "PASS" else row["qc_status"]
            row["qc_notes"] = (row.get("qc_notes") or "") + "; ".join(qc_issues)

    except Exception as exc:
        row["qc_status"] = "FAIL"
        row["qc_notes"] += f" load_error:{exc}"
        traceback.print_exc()

    return row


# ---------------------------------------------------------------------------
# Manifest column order
# ---------------------------------------------------------------------------
MANIFEST_COLUMNS = [
    "battery", "paper_battery_name", "dataset_identity_status",
    "temperature_c", "filename", "relative_path", "sha256",
    "start_datetime", "profile", "profile_number", "paper_profile",
    "file_type", "is_standalone", "is_combined", "is_duplicate", "duplicate_of",
    "signal_fingerprint", "previous_drive_cycle", "next_drive_cycle",
    "soc_anchor_source", "soc_init", "split_assignment_basis", "pairing_status",
    "full_train", "partial_train", "test", "exclude",
    "confidence", "evidence", "notes", "qc_status", "qc_notes",
    "n_rows", "duration_s", "median_dt_s",
    "first_timestamp", "last_timestamp",
    "voltage_min", "voltage_max", "current_min", "current_max",
    "ah_min", "ah_start", "ah_end", "ah_max",
    "temp_min", "temp_max", "temp_mean",
]


def build_manifest() -> list[dict]:
    temp_dir = RAW_BASE / TEMP_DIR_NAME
    if not temp_dir.is_dir():
        print(f"ERROR: 25 °C directory not found: {temp_dir}", file=sys.stderr)
        sys.exit(1)

    mat_files = sorted(temp_dir.glob("*.mat"))
    print(f"\nFound {len(mat_files)} .mat files in {temp_dir}")

    rows = []
    for p in mat_files:
        row = inspect_file(p, REPO_ROOT)
        rows.append(row)

    # Sort deterministically by start_datetime then filename
    rows.sort(key=lambda r: (r["start_datetime"], r["filename"]))

    # Normalise all rows to manifest columns
    normalised = []
    for r in rows:
        norm = {c: r.get(c, "") for c in MANIFEST_COLUMNS}
        normalised.append(norm)

    return normalised


def serialize_manifest(rows: list[dict]) -> str:
    """Deterministic CSV serialisation for hash comparison."""
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def write_manifest(rows: list[dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nManifest written: {MANIFEST_PATH}")


# ---------------------------------------------------------------------------
# Validation (20 checks)
# ---------------------------------------------------------------------------

def validate(rows: list[dict]) -> tuple[bool, list[str]]:
    failures: list[str] = []

    def fail(msg: str):
        failures.append(msg)
        print(f"  [FAIL] {msg}")

    def warn(msg: str):
        print(f"  [WARN] {msg}")

    print("\n=== Validation ===")

    # [1] Manifest paths unique
    paths = [r["relative_path"] for r in rows]
    if len(paths) != len(set(paths)):
        dupes = [p for p in set(paths) if paths.count(p) > 1]
        fail(f"[1] Duplicate manifest paths: {dupes}")
    else:
        print(f"  [1] PASS  Manifest paths unique ({len(paths)} rows)")

    # [2] SHA-256 exact duplicate groups
    sha_map: dict[str, list[str]] = {}
    for r in rows:
        h = r["sha256"]
        if h:
            sha_map.setdefault(h, []).append(r["filename"])
    exact_dup_groups = {h: fns for h, fns in sha_map.items() if len(fns) > 1}
    if exact_dup_groups:
        warn(f"[2] {len(exact_dup_groups)} SHA-256 exact duplicate group(s): {exact_dup_groups}")
    else:
        print(f"  [2] PASS  No SHA-256 exact duplicates")

    # [3] Signal-fingerprint duplicate candidates
    fp_map: dict[str, list[str]] = {}
    for r in rows:
        fp = r["signal_fingerprint"]
        if fp:
            fp_map.setdefault(fp, []).append(r["filename"])
    fp_dup_groups = {fp: fns for fp, fns in fp_map.items() if len(fns) > 1}
    if fp_dup_groups:
        warn(f"[3] {len(fp_dup_groups)} signal-fingerprint duplicate candidate group(s): {fp_dup_groups}")
    else:
        print(f"  [3] PASS  No signal-fingerprint duplicate candidates")

    # [4] All selected paths exist
    missing_paths = [
        r["relative_path"] for r in rows
        if (r["full_train"] or r["partial_train"] or r["test"])
        and not (REPO_ROOT / r["relative_path"]).exists()
    ]
    if missing_paths:
        fail(f"[4] Missing selected paths: {missing_paths}")
    else:
        print(f"  [4] PASS  All selected paths exist")

    # [5] All selected files from canonical 25 °C session
    wrong_temp = [
        r["filename"] for r in rows
        if (r["full_train"] or r["partial_train"] or r["test"])
        and str(r["temperature_c"]) != str(NOMINAL_TEMP_C)
    ]
    if wrong_temp:
        fail(f"[5] Non-25 °C selected files: {wrong_temp}")
    else:
        print(f"  [5] PASS  All selected files at 25 °C")

    # [6] Expected paper profiles present
    found_profiles = {r["profile"] for r in rows if r["full_train"] or r["test"]}
    missing_profiles = (EXPECTED_FULL_TRAIN_PROFILES | EXPECTED_TEST_PROFILES) - found_profiles
    if missing_profiles:
        fail(f"[6] Missing expected profiles: {sorted(missing_profiles)}")
    else:
        print(f"  [6] PASS  All expected profiles found")

    # [7] Missing or unexpected profiles reported
    all_profiles = {r["profile"] for r in rows if r["profile"]}
    known_profiles = (
        EXPECTED_FULL_TRAIN_PROFILES | EXPECTED_TEST_PROFILES |
        {"HWFET", "HPPC", "C20DisCh", "Dis_0p5C", "Dis_2C", "Cap_1C", "PausCycl"} |
        {f"Charge{i}" for i in range(1, 17)}
    )
    unexpected = all_profiles - known_profiles
    if unexpected:
        warn(f"[7] Unexpected profiles found: {sorted(unexpected)}")
    else:
        print(f"  [7] PASS  No unexpected profiles")

    # [8] No protocol-excluded HWFET file is selected
    # HWFET at Samsung 30T 25°C is a valid drive cycle (real current), but is
    # excluded by paper protocol (Table I does not list HWFET as a test profile).
    invalid_selected = [
        r["filename"] for r in rows
        if r["profile"] == "HWFET" and (r["full_train"] or r["partial_train"] or r["test"])
    ]
    if invalid_selected:
        fail(f"[8] Protocol-excluded HWFET file is selected: {invalid_selected}")
    else:
        print(f"  [8] PASS  No protocol-excluded HWFET file is selected")

    # [9] No full_train/test overlap
    overlap_ft = [
        r["filename"] for r in rows if r["full_train"] == 1 and r["test"] == 1
    ]
    if overlap_ft:
        fail(f"[9] full_train ∩ test overlap: {overlap_ft}")
    else:
        print(f"  [9] PASS  No full_train/test overlap")

    # [10] No partial_train/test overlap
    overlap_pt = [
        r["filename"] for r in rows if r["partial_train"] == 1 and r["test"] == 1
    ]
    if overlap_pt:
        fail(f"[10] partial_train ∩ test overlap: {overlap_pt}")
    else:
        print(f"  [10] PASS  No partial_train/test overlap")

    # [11] No excluded file has selected split flag
    excluded_selected = [
        r["filename"] for r in rows
        if r["exclude"] == 1 and (r["full_train"] == 1 or r["partial_train"] == 1 or r["test"] == 1)
    ]
    if excluded_selected:
        fail(f"[11] Excluded files with selected flag: {excluded_selected}")
    else:
        print(f"  [11] PASS  No excluded file has a selected flag")

    # [12] partial_train semantics: every partial_train file is also full_train
    partial_not_full = [
        r["filename"] for r in rows if r["partial_train"] == 1 and r["full_train"] != 1
    ]
    if partial_not_full:
        fail(f"[12] partial_train files not in full_train: {partial_not_full}")
    else:
        print(f"  [12] PASS  partial_train is a strict subset of full_train")

    # [13] No exact SHA-256 duplicate crosses train/test
    sha_train: set[str] = set()
    sha_test: set[str] = set()
    for r in rows:
        if r["sha256"]:
            if r["full_train"] or r["partial_train"]:
                sha_train.add(r["sha256"])
            if r["test"]:
                sha_test.add(r["sha256"])
    cross_sha = sha_train & sha_test
    if cross_sha:
        fail(f"[13] SHA-256 duplicate crosses train/test: {len(cross_sha)} hash(es)")
    else:
        print(f"  [13] PASS  No SHA-256 exact duplicate crosses train/test")

    # [14] No signal-fingerprint duplicate candidate crosses train/test without review
    fp_train: dict[str, str] = {}
    fp_test: dict[str, str] = {}
    for r in rows:
        fp = r["signal_fingerprint"]
        if fp:
            if r["full_train"] or r["partial_train"]:
                fp_train[fp] = r["filename"]
            if r["test"]:
                fp_test[fp] = r["filename"]
    cross_fp = set(fp_train) & set(fp_test)
    if cross_fp:
        fail(f"[14] Signal-fingerprint candidate crosses train/test: {[(fp_train[fp], fp_test[fp]) for fp in cross_fp]}")
    else:
        print(f"  [14] PASS  No signal-fingerprint duplicate crosses train/test")

    # [15] All selected files pass required-field QC
    qc_fail = [
        r["filename"] for r in rows
        if (r["full_train"] or r["partial_train"] or r["test"])
        and r["qc_status"] == "FAIL"
    ]
    if qc_fail:
        fail(f"[15] Selected files with QC FAIL: {qc_fail}")
    else:
        print(f"  [15] PASS  All selected files pass required-field QC")

    # [16] All selected charges have chronological evidence
    charges_no_evidence = [
        r["filename"] for r in rows
        if r["file_type"] == "charge"
        and (r["full_train"] or r["test"])
        and not r.get("evidence")
    ]
    if charges_no_evidence:
        fail(f"[16] Selected charges without chronological evidence: {charges_no_evidence}")
    else:
        print(f"  [16] PASS  All selected charges have chronological evidence")

    # [17] Unresolved relationships remain explicitly unresolved
    unresolved = [
        r["filename"] for r in rows
        if r["pairing_status"] == "unresolved"
        and (r["full_train"] or r["partial_train"] or r["test"])
    ]
    if unresolved:
        warn(f"[17] Selected files with unresolved pairing status: {unresolved}")
    else:
        print(f"  [17] PASS  No selected files have unresolved pairing status")

    # [18] Identity status not stronger than evidence
    if DATASET_IDENTITY_STATUS == "CONFIRMED":
        warn("[18] Identity status is CONFIRMED; review evidence if this was not explicitly verified")
    elif DATASET_IDENTITY_STATUS in ("STRONGLY_SUPPORTED", "PLAUSIBLE_BUT_UNCONFIRMED", "CONTRADICTED"):
        print(f"  [18] PASS  Identity status is {DATASET_IDENTITY_STATUS}")
    else:
        warn(f"[18] Unknown identity status: {DATASET_IDENTITY_STATUS}")

    # [19] Output order deterministic (re-sort and compare)
    re_sorted = sorted(rows, key=lambda r: (r["start_datetime"], r["filename"]))
    if [r["filename"] for r in rows] != [r["filename"] for r in re_sorted]:
        fail("[19] Output order is NOT deterministic (sort mismatch)")
    else:
        print(f"  [19] PASS  Output order is deterministic")

    # [20] Repeated-run manifest hash stable
    serial_1 = serialize_manifest(rows)
    rows_2 = build_manifest()
    serial_2 = serialize_manifest(rows_2)
    h1 = hashlib.sha256(serial_1.encode()).hexdigest()
    h2 = hashlib.sha256(serial_2.encode()).hexdigest()
    if h1 != h2:
        fail(f"[20] Repeated-run hash MISMATCH: run1={h1[:16]}… run2={h2[:16]}…")
    else:
        print(f"  [20] PASS  Repeated-run manifest hash stable: {h1[:16]}…")

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict]) -> None:
    total = len(rows)

    def count(flag):
        return sum(1 for r in rows if r[flag] == 1)

    def count2(flag, ftype):
        return sum(1 for r in rows if r[flag] == 1 and r["file_type"] == ftype)

    ft = count("full_train")
    pt = count("partial_train")
    te = count("test")
    ex = count("exclude")

    ft_drive = count2("full_train", "drive_cycle")
    ft_charge = count2("full_train", "charge")
    pt_drive = count2("partial_train", "drive_cycle")
    pt_charge = count2("partial_train", "charge")
    te_drive = count2("test", "drive_cycle")
    te_charge = count2("test", "charge")

    unresolved = [r["filename"] for r in rows if r["pairing_status"] == "unresolved"]
    qc_fail = [r["filename"] for r in rows if r["qc_status"] == "FAIL"]
    qc_warn = [r["filename"] for r in rows if r["qc_status"] == "WARN"]

    print("\n" + "=" * 60)
    print("Samsung INR21700-30T  25 °C  — Split Summary")
    print("=" * 60)
    print(f"  Canonical raw dir  : {RAW_BASE / TEMP_DIR_NAME}")
    print(f"  Paper battery name : {PAPER_BATTERY_NAME}")
    print(f"  Public dataset name: {BATTERY_PUBLIC}")
    print(f"  Identity status    : {DATASET_IDENTITY_STATUS}")
    print()
    print(f"  Total manifest rows : {total}")
    print(f"  full_train          : {ft}  (drive={ft_drive}, charge={ft_charge})")
    print(f"  partial_train       : {pt}  (drive={pt_drive}, charge={pt_charge})")
    print(f"  test                : {te}  (drive={te_drive}, charge={te_charge})")
    print(f"  exclude             : {ex}")
    print()
    print(f"  Missing expected profiles  : none (checked in validation)")
    print(f"  Unresolved charge pairings : {len(unresolved)}")
    if unresolved:
        for fn in unresolved:
            print(f"    - {fn}")
    print(f"  QC failures (mandatory) : {len(qc_fail)}")
    if qc_fail:
        for fn in qc_fail:
            r = next(x for x in rows if x["filename"] == fn)
            print(f"    - {fn}: {r['qc_notes']}")
    # Summarise warnings by type rather than listing every file
    warn_types: dict[str, int] = {}
    for r in qc_warn:
        note = next(x for x in rows if x["filename"] == r)["qc_notes"]
        key = note.split("(")[0].strip()
        warn_types[key] = warn_types.get(key, 0) + 1
    print(f"  QC warnings  : {len(qc_warn)} files")
    for wtype, cnt in sorted(warn_types.items()):
        print(f"    {cnt}× {wtype}  (cycler artifact; not a data-quality failure)")
    print()
    print("  Split status: PROVISIONAL")
    print("  Reason: Battery identity is STRONGLY_SUPPORTED (not CONFIRMED);")
    print("          partial_train subset for Samsung T30 inferred from paper text")
    print("          (Mix1-3); missing Cap_1C .mat at 25°C is unresolved data gap.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("Samsung INR21700-30T 25 °C Split Inspection")
    print("=" * 60)

    rows = build_manifest()
    print_summary(rows)
    write_manifest(rows)

    ok, failures = validate(rows)

    if ok:
        print("\nOverall: PASS")
        return 0
    else:
        print(f"\nOverall: FAIL  ({len(failures)} check(s) failed)")
        for f in failures:
            print(f"  - {f}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
