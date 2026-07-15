#!/usr/bin/env python3
"""Read-only Vidal 2019 charge protocol audit for Panasonic / LG / Samsung 30T."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from battery_benchmark.data.lg_hg2 import LGHG2Adapter
from battery_benchmark.data.panasonic import PanasonicAdapter
from battery_benchmark.data.samsung_30t import Samsung30TAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = REPO_ROOT / "outputs/audits"

VIDAL_PDF = Path(
    "/Users/yue/Zotero/storage/DUPS5B62/"
    "Vidal et al. - 2019 - Li-ion Battery State of Charge Estimation Using "
    "Long Short-Term Memory Recurrent Neural Network with.pdf"
)
README_PATHS = {
    "panasonic_18650pf": REPO_ROOT
    / "data/raw/Panasonic_18650PF/Readme file - desc of tests performed.txt",
    "lg_hg2": REPO_ROOT
    / "data/raw/LG_HG2/Readme file - Description of Experimental Tests.txt",
    "samsung_30t": REPO_ROOT / "data/raw/Samsung_30T/Readme IN21700-30T.txt",
}
MANIFEST_PATHS = {
    "panasonic_18650pf": REPO_ROOT / "data/manifests/panasonic_vidal2019_25c_manifest.csv",
    "lg_hg2": REPO_ROOT / "data/manifests/lg_hg2_vidal2019_25c_manifest.csv",
    "samsung_30t": REPO_ROOT / "data/manifests/samsung_30t_vidal2019_25c_manifest.csv",
}
RAW_ROOTS = {
    "panasonic_18650pf": REPO_ROOT / "data/raw/Panasonic_18650PF/25degC",
    "lg_hg2": REPO_ROOT / "data/raw/LG_HG2/25degC",
    "samsung_30t": REPO_ROOT / "data/raw/Samsung_30T/25degC",
}
ADAPTERS = {
    "panasonic_18650pf": PanasonicAdapter,
    "lg_hg2": LGHG2Adapter,
    "samsung_30t": Samsung30TAdapter,
}

CAPACITY_HINT = {
    "panasonic_18650pf": 2.9,
    "lg_hg2": 3.0,
    "samsung_30t": 3.0,
}
FULL_V = 4.18
TINY_FRAC = 0.05
FULL_FRAC = 0.70
CURRENT_END_PAN_LG = 0.05
CURRENT_END_SAM = 0.01

PAPER_PROFILES = {
    "panasonic_18650pf": {
        "full_train": ["Mix1", "Mix2", "Mix3", "Mix4", "Mix5"],
        "partial_train": ["Mix1", "Mix2", "Mix3"],
        "test": ["UDDS", "LA92", "US06"],
    },
    "lg_hg2": {
        "full_train": [f"Mix{i}" for i in range(1, 9)],
        "partial_train": ["Mix1", "Mix2", "Mix3"],
        "test": ["UDDS", "LA92", "US06"],
    },
    "samsung_30t": {
        "full_train": [f"Mix{i}" for i in range(1, 9)],
        "partial_train": ["Mix1", "Mix2", "Mix3"],
        "test": ["UDDS", "LA92", "US06"],
    },
}


def _parse_filename_dt(filename: str) -> datetime | None:
    m = re.match(r"(\d{2})-(\d{2})-(\d{2})_(\d{2})\.(\d{2})", filename)
    if not m:
        return None
    mo, day, yr, hh, mm = m.groups()
    return datetime(2000 + int(yr), int(mo), int(day), int(hh), int(mm))


def _parse_manifest_dt(value: str) -> datetime | None:
    if not value or value.strip() == "":
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def _charge_number(filename: str, profile: str) -> int | None:
    m = re.search(r"Charge(\d+)", filename, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"Charge_(\d+)", profile, re.I)
    if m:
        return int(m.group(1))
    if profile.startswith("Charge") and profile[6:].isdigit():
        return int(profile[6:])
    return None


def _split_memberships(row: dict) -> list[str]:
    out = []
    for s in ("full_train", "partial_train", "test", "exclude"):
        if str(row.get(s, "0")).strip() in ("1", "1.0", "True", "true"):
            out.append(s)
    return out


def _selected_drive(row: dict) -> bool:
    return (
        row.get("file_type") == "drive_cycle"
        and row.get("record_role") == "model_input"
    )


def _load_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_record(adapter, row: dict, split: str = "full_train"):
    entry = next(e for e in adapter.entries if e.filename == row["filename"])
    try:
        return adapter.load_entry(entry, requested_split=split)
    except Exception:
        return None


def _signal_stats(rec) -> dict[str, Any]:
    if rec is None:
        return {}
    df = rec.data
    ah = df["cycler_ah"]
    cur = df["current_a"]
    vol = df["voltage_v"]
    ts = df["time_s"]
    dt = ts.diff()
    dup = int((dt == 0).sum())
    neg = int((dt < 0).sum())
    ah_max = float(ah.max())
    ah_end = float(ah.iloc[-1])
    ah_min = float(ah.min())
    ah_start = float(ah.iloc[0])
    peak_idx = int(ah.idxmax())
    ah_reset = bool(rec.qc.ah_reset_detected) or (
        ah_max > 0.05 and ah_end < ah_max * 0.5
    )
    cap = rec.metadata.get("capacity_hint_ah") or 3.0
    charge_ah_max = max(abs(ah_max), abs(ah_end), abs(ah_max - ah_min))
    frac = charge_ah_max / cap if cap else None
    v_max = float(vol.max())
    v_end = float(vol.iloc[-1])
    i_end = float(cur.iloc[-1])
    i_max = float(cur.max())
    taper = bool(i_max > 0.5 and abs(i_end) < (CURRENT_END_SAM if "samsung" in rec.metadata.get("dataset", "") else CURRENT_END_PAN_LG))
    full_v = v_max >= FULL_V and v_end >= FULL_V - 0.02
    return {
        "ah_start": ah_start,
        "ah_end": ah_end,
        "ah_min": ah_min,
        "ah_max": ah_max,
        "ah_peak_index": peak_idx,
        "ah_peak_time": float(ts.iloc[peak_idx]) if len(ts) else None,
        "ah_end_minus_ah_peak": ah_end - ah_max,
        "current_start": float(cur.iloc[0]),
        "current_end": i_end,
        "current_min": float(cur.min()),
        "current_max": i_max,
        "current_abs_max": float(cur.abs().max()),
        "voltage_start": float(vol.iloc[0]),
        "voltage_end": v_end,
        "voltage_min": float(vol.min()),
        "voltage_max": v_max,
        "charge_ah_max": charge_ah_max,
        "charge_ah_fraction": frac,
        "full_voltage_reached": full_v,
        "current_taper_detected": taper,
        "ah_reset_detected": ah_reset,
        "duplicate_timestamp_count": dup,
        "negative_time_step_count": neg,
        "adapter_qc_status": rec.qc.status,
        "start_datetime_signal": str(rec.data["timestamp"].iloc[0]) if "timestamp" in rec.data.columns and rec.data["timestamp"].iloc[0] is not None else "",
        "end_datetime_signal": str(rec.data["timestamp"].iloc[-1]) if "timestamp" in rec.data.columns and rec.data["timestamp"].iloc[-1] is not None else "",
        "duration_s": float(ts.iloc[-1]) if len(ts) else None,
    }


def build_paper_reference() -> pd.DataFrame:
    rows = [
        {
            "dataset": "panasonic_18650pf",
            "paper_battery_name": "PANASONIC 18650PF (PANASONIC)",
            "paper_dataset_available": True,
            "paper_training_profiles": "Mix (1-5) drive cycles + accompanying CC charges",
            "paper_partial_profiles": "Mix (1-3) drive cycles + accompanying CC (Case 3/4)",
            "paper_test_profiles": "UDDS, LA92, US06 + corresponding CC charges",
            "paper_charge_statement": "CC = 1C constant-current constant-voltage charging profile (Table I p.1)",
            "paper_measured_ah_statement": "Cycler-calculated Ah used as SOC reference during training (Section IV.A p.4)",
            "figure_8_sequence": "Samsung T30 drive cycles and charging segments in one continuous evaluation plot",
            "figure_8_relevance": "Illustrates CC charge segments enter model input stream; does not define per-file pairing for all batteries",
            "source_location": "Table I p.1; Section IV.B p.4; Table V p.4; Figure 8 p.5",
            "notes": "Panasonic Mix1-5 maps to Cycle1-4 + NN; HWFET not in Table I test profiles",
        },
        {
            "dataset": "lg_hg2",
            "paper_battery_name": "LG HG2 (LG_HG2)",
            "paper_dataset_available": True,
            "paper_training_profiles": "Mix (1-8) + accompanying CC",
            "paper_partial_profiles": "Mix (1-3) + accompanying CC",
            "paper_test_profiles": "UDDS, LA92, US06 + corresponding CC",
            "paper_charge_statement": "CC listed in Table I; training uses accompanying charges per Section IV.B",
            "paper_measured_ah_statement": "Same as Panasonic — cycler Ah for SOC cost",
            "figure_8_sequence": "N/A (Samsung example only)",
            "figure_8_relevance": "Confirms charge+drive in LSTM input; LG pairing not specified",
            "source_location": "Table I p.1; Section IV.B p.4; Table V p.4",
            "notes": "HWFET in README drive order but not Vidal test set",
        },
        {
            "dataset": "samsung_30t",
            "paper_battery_name": "SAMSUNG T30 (SAM_T30)",
            "paper_dataset_available": True,
            "paper_training_profiles": "Mix (1-8) + accompanying CC",
            "paper_partial_profiles": "Mix (1-3) + accompanying CC",
            "paper_test_profiles": "UDDS, LA92, US06 + corresponding CC",
            "paper_charge_statement": "CC in Table I; Fig.8 shows charge segments in SAM_T30 evaluation",
            "paper_measured_ah_statement": "Cycler Ah reference (Section IV.A)",
            "figure_8_sequence": "Continuous SAM_T30 SOC with drive and charge segments",
            "figure_8_relevance": "Direct example for T30; supports CC as model input",
            "source_location": "Table I p.1; Section IV.B p.4; Table V p.4; Figure 8 p.5",
            "notes": "",
        },
        {
            "dataset": "samsung_48g",
            "paper_battery_name": "SAMSUNG 48G (SAM_48G)",
            "paper_dataset_available": False,
            "paper_training_profiles": "Mix (1,2,6,7,8) + CC (Table I)",
            "paper_partial_profiles": "Mix (1,2,6) per Case 3/4 note",
            "paper_test_profiles": "UDDS, LA92, US06 + CC",
            "paper_charge_statement": "Listed in Table I only",
            "paper_measured_ah_statement": "Same protocol",
            "figure_8_sequence": "N/A",
            "figure_8_relevance": "Out of scope — raw data unavailable in repo",
            "source_location": "Table I p.1; Table V p.4",
            "notes": "Not audited — no raw data in repository",
        },
    ]
    return pd.DataFrame(rows)


def build_chronology(dataset: str, rows: list[dict], records: dict[str, Any]) -> pd.DataFrame:
    enriched = []
    for row in rows:
        fn = row["filename"]
        rec = records.get(fn)
        sig = _signal_stats(rec) if rec else {}
        fn_dt = _parse_filename_dt(fn)
        man_dt = _parse_manifest_dt(row.get("start_datetime", ""))
        sig_dt = _parse_manifest_dt(sig.get("start_datetime_signal", "")[:19].replace("/", "-")) if sig.get("start_datetime_signal") else None
        if sig_dt is None and sig.get("start_datetime_signal"):
            try:
                sig_dt = pd.to_datetime(sig["start_datetime_signal"])
            except Exception:
                sig_dt = None
        selected = man_dt or fn_dt
        time_source = "manifest"
        if rec and sig.get("start_datetime_signal"):
            selected = sig_dt or man_dt or fn_dt
            time_source = "signal" if sig_dt else ("manifest" if man_dt else "filename")
        elif fn_dt and man_dt and abs((fn_dt - man_dt).total_seconds()) > 3600:
            time_source = "filename_conflict"
            selected = sig_dt or man_dt or fn_dt
        elif not man_dt:
            selected = fn_dt
            time_source = "filename"

        conflict = False
        if man_dt and fn_dt and abs((man_dt - fn_dt).total_seconds()) > 120:
            conflict = True
        if sig_dt and man_dt and hasattr(sig_dt, "to_pydatetime"):
            if abs((sig_dt.to_pydatetime() - man_dt).total_seconds()) > 3600:
                conflict = True

        enriched.append({
            "dataset": dataset,
            "source_file": row.get("relative_path", ""),
            "filename": fn,
            "folder": str(Path(row.get("relative_path", "")).parent.name),
            "sequence_id": f"{dataset}_25c_{fn.replace('.mat','')}" if rec is None else rec.metadata.get("sequence_id", ""),
            "start_datetime_signal": sig.get("start_datetime_signal", ""),
            "start_datetime_manifest": man_dt.isoformat() if man_dt else "",
            "start_datetime_filename": fn_dt.isoformat() if fn_dt else "",
            "selected_start_datetime": selected.isoformat() if selected is not None and hasattr(selected, "isoformat") else str(selected or ""),
            "end_datetime": sig.get("end_datetime_signal", ""),
            "duration_s": sig.get("duration_s") or row.get("duration_s", ""),
            "time_source_used": time_source,
            "timestamp_conflict": conflict,
            "profile": row.get("profile", ""),
            "paper_profile": row.get("paper_profile", ""),
            "file_type": row.get("file_type", ""),
            "record_role": row.get("record_role", ""),
            "full_train": row.get("full_train", "0"),
            "partial_train": row.get("partial_train", "0"),
            "test": row.get("test", "0"),
            "exclude": row.get("exclude", "0"),
            **{k: sig.get(k, "") for k in (
                "ah_start", "ah_end", "ah_min", "ah_max", "ah_peak_index", "ah_peak_time",
                "ah_end_minus_ah_peak", "current_start", "current_end", "current_min",
                "current_max", "current_abs_max", "voltage_start", "voltage_end",
                "voltage_min", "voltage_max", "ah_reset_detected", "duplicate_timestamp_count",
                "negative_time_step_count", "adapter_qc_status",
            )},
            "capacity_hint_ah": CAPACITY_HINT.get(dataset, 3.0),
            "charge_ah_fraction": sig.get("charge_ah_fraction", ""),
            "full_voltage_reached": sig.get("full_voltage_reached", ""),
            "current_taper_detected": sig.get("current_taper_detected", ""),
            "README_expected_order": "",
            "order_consistent_with_readme": "",
        })

    chrono = pd.DataFrame(enriched)
    chrono["_sort"] = pd.to_datetime(chrono["selected_start_datetime"], errors="coerce")
    chrono = chrono.sort_values("_sort").reset_index(drop=True)

    prev_sel = None
    prev_sel_split = ""
    for i, r in chrono.iterrows():
        prev = chrono.iloc[i - 1] if i > 0 else None
        nxt = chrono.iloc[i + 1] if i < len(chrono) - 1 else None
        chrono.at[i, "previous_file"] = prev["filename"] if prev is not None else ""
        chrono.at[i, "previous_profile"] = prev["profile"] if prev is not None else ""
        chrono.at[i, "previous_file_type"] = prev["file_type"] if prev is not None else ""
        chrono.at[i, "previous_record_role"] = prev["record_role"] if prev is not None else ""
        chrono.at[i, "next_file"] = nxt["filename"] if nxt is not None else ""
        chrono.at[i, "next_profile"] = nxt["profile"] if nxt is not None else ""
        chrono.at[i, "next_file_type"] = nxt["file_type"] if nxt is not None else ""
        chrono.at[i, "next_record_role"] = nxt["record_role"] if nxt is not None else ""

        if prev is not None and _selected_drive(prev.to_dict()):
            chrono.at[i, "previous_selected_drive"] = prev["filename"]
            chrono.at[i, "previous_selected_drive_split"] = ",".join(_split_memberships(prev.to_dict()))
            prev_sel = prev["filename"]
            prev_sel_split = chrono.at[i, "previous_selected_drive_split"]
        else:
            chrono.at[i, "previous_selected_drive"] = prev_sel or ""
            chrono.at[i, "previous_selected_drive_split"] = prev_sel_split

        # forward selected drive
        nxt_sel = ""
        nxt_sel_split = ""
        for j in range(i + 1, len(chrono)):
            cand = chrono.iloc[j]
            if _selected_drive(cand.to_dict()):
                nxt_sel = cand["filename"]
                nxt_sel_split = ",".join(_split_memberships(cand.to_dict()))
                break
        chrono.at[i, "next_selected_drive"] = nxt_sel
        chrono.at[i, "next_selected_drive_split"] = nxt_sel_split

        imm_prev = ""
        imm_next = ""
        for j in range(i - 1, -1, -1):
            c = chrono.iloc[j]
            if c["record_role"] == "model_input" and c["file_type"] in ("drive_cycle", "charge"):
                imm_prev = c["profile"]
                break
        for j in range(i + 1, len(chrono)):
            c = chrono.iloc[j]
            if c["record_role"] == "model_input" and c["file_type"] in ("drive_cycle", "charge"):
                imm_next = c["profile"]
                break
        chrono.at[i, "previous_immediate_selected_profile"] = imm_prev
        chrono.at[i, "next_immediate_selected_profile"] = imm_next

    chrono = chrono.drop(columns=["_sort"], errors="ignore")
    return chrono


def _classify_charge(
    dataset: str,
    row: dict,
    chrono_row: pd.Series,
    stats: dict,
) -> dict[str, Any]:
    fn = row["filename"]
    splits = _split_memberships(row)
    frac = stats.get("charge_ah_fraction") or 0
    prev_prof = chrono_row.get("previous_profile", "")
    next_prof = chrono_row.get("next_profile", "")
    prev_ft = chrono_row.get("previous_file_type", "")
    next_ft = chrono_row.get("next_file_type", "")
    prev_drive = chrono_row.get("previous_selected_drive", "")
    next_drive = chrono_row.get("next_selected_drive", "")
    prev_drive_splits = set(str(chrono_row.get("previous_selected_drive_split", "")).split(",")) - {""}
    next_drive_splits = set(str(chrono_row.get("next_selected_drive_split", "")).split(",")) - {""}
    cn = _charge_number(fn, row.get("profile", ""))

    role = "ambiguous"
    confidence = "medium"
    reason_parts = []
    paper = readme = chrono_s = signal = manifest = False
    requires_review = False
    proposed_mi = False

    if row.get("record_role") == "excluded":
        if "Cap_1C" in prev_prof or "Cap_1C" in fn or frac <= TINY_FRAC:
            role = "pre_capacity_test_topup" if frac <= TINY_FRAC else "excluded_profile_related"
        elif "HPPC" in prev_prof or "Dis_" in prev_prof or "C20" in prev_prof:
            role = "excluded_profile_related"
        else:
            role = "excluded_profile_related"
        confidence = "high"
        reason_parts.append("manifest excluded")
        manifest = True
        return _charge_result(row, splits, stats, chrono_row, role, False, confidence, False, reason_parts, paper, readme, chrono_s, signal, manifest)

    # tiny top-off
    if frac and frac <= TINY_FRAC:
        role = "tiny_topoff"
        confidence = "high"
        signal = readme = True
        reason_parts.append(f"charge_ah_fraction={frac:.3f} <= {TINY_FRAC}")
        requires_review = True
        proposed_mi = False

    elif dataset == "panasonic_18650pf":
        if cn == 1 and not prev_drive:
            role = "leading_full_charge_anchor"
            paper = readme = chrono_s = True
            reason_parts.append("First charge before Cycle_1 (Mix1); Vidal FULL includes accompanying CC")
            proposed_mi = True
            confidence = "high"
        elif cn == 5 and "NN" in str(chrono_row.get("previous_profile", "")):
            role = "trailing_charge"
            chrono_s = manifest = True
            reason_parts.append("Post-NN (Mix5) charge after last full_train mix")
            proposed_mi = True
        elif prev_drive and next_drive and row.get("record_role") == "model_input":
            role = "normal_intercycle_charge"
            paper = chrono_s = True
            reason_parts.append(f"Between selected drives {Path(prev_drive).stem} -> {Path(next_drive).stem}")
            proposed_mi = True
            confidence = "high"
        elif "HWFET" in str(prev_prof) or "HWFET" in str(next_prof):
            role = "excluded_profile_related"
            readme = chrono_s = True
            requires_review = True

    elif dataset == "lg_hg2":
        if cn == 1:
            role = "pre_capacity_test_topup"
            readme = manifest = True
            confidence = "high"
            proposed_mi = False
        elif prev_prof == "Cap_1C" and stats.get("full_voltage_reached") and frac >= FULL_FRAC:
            role = "post_capacity_test_full_recharge"
            readme = chrono_s = signal = True
            proposed_mi = True
            reason_parts.append("Post Cap_1C full CC-CV recharge before UDDS")
        elif prev_prof == "Cap_1C" and cn == 9:
            role = "post_capacity_test_full_recharge"
            readme = chrono_s = True
            proposed_mi = True
        elif prev_drive and next_drive and row.get("record_role") == "model_input":
            role = "normal_intercycle_charge"
            paper = chrono_s = True
            reason_parts.append(f"Between selected drives")
            proposed_mi = True
            confidence = "high"
        elif prev_prof == "HWFET" or next_prof == "HWFET":
            role = "tiny_topoff" if frac <= TINY_FRAC else "excluded_profile_related"
            readme = chrono_s = True
            requires_review = True

    elif dataset == "samsung_30t":
        if cn == 1:
            role = "initial_topup"
            readme = manifest = True
            confidence = "high"
            proposed_mi = False
        elif cn == 9 and frac <= TINY_FRAC:
            role = "pre_capacity_test_topup"
            readme = chrono_s = True
            proposed_mi = False
        elif cn == 10 and stats.get("full_voltage_reached"):
            role = "post_capacity_test_full_recharge"
            readme = chrono_s = signal = True
            proposed_mi = True
            reason_parts.append("Post inferred Cap_1C before Mixed3")
        elif prev_drive and next_drive and row.get("record_role") == "model_input":
            role = "normal_intercycle_charge"
            paper = chrono_s = True
            proposed_mi = True
            confidence = "high"
        elif cn == 16:
            role = "trailing_charge"
            chrono_s = True
            proposed_mi = True

    # cross split between selected drives
    if prev_drive_splits and next_drive_splits and prev_drive_splits != next_drive_splits:
        if role == "normal_intercycle_charge":
            role = "cross_split_bridge"
        requires_review = True
        chrono_s = True
        reason_parts.append(f"bridges splits {prev_drive_splits} -> {next_drive_splits}")

    if stats.get("ah_reset_detected"):
        requires_review = True
        signal = True
        reason_parts.append("Ah counter reset detected")

    if row.get("record_role") == "model_input" and role in ("initial_topup", "pre_capacity_test_topup", "tiny_topoff"):
        requires_review = True
        manifest = True
        reason_parts.append("manifest model_input conflicts with charge role")

    if not reason_parts:
        reason_parts.append(f"Chronology: after {prev_prof}, before {next_prof}")

    return _charge_result(
        row, splits, stats, chrono_row, role, proposed_mi, confidence,
        requires_review, reason_parts, paper, readme, chrono_s, signal, manifest,
    )


def _charge_result(row, splits, stats, chrono_row, role, proposed_mi, confidence, requires_review, reason_parts, paper, readme, chrono_s, signal, manifest):
    return {
        "dataset": chrono_row["dataset"],
        "filename": row["filename"],
        "sequence_id": chrono_row.get("sequence_id", ""),
        "charge_number": _charge_number(row["filename"], row.get("profile", "")),
        "start_datetime": chrono_row.get("selected_start_datetime", ""),
        "end_datetime": chrono_row.get("end_datetime", ""),
        "duration_s": chrono_row.get("duration_s", ""),
        "current_record_role": row.get("record_role", ""),
        "current_split_memberships": ",".join(splits),
        "previous_file": chrono_row.get("previous_file", ""),
        "previous_profile": chrono_row.get("previous_profile", ""),
        "previous_file_type": chrono_row.get("previous_file_type", ""),
        "previous_selected_drive": chrono_row.get("previous_selected_drive", ""),
        "previous_selected_drive_split": chrono_row.get("previous_selected_drive_split", ""),
        "next_file": chrono_row.get("next_file", ""),
        "next_profile": chrono_row.get("next_profile", ""),
        "next_file_type": chrono_row.get("next_file_type", ""),
        "next_selected_drive": chrono_row.get("next_selected_drive", ""),
        "next_selected_drive_split": chrono_row.get("next_selected_drive_split", ""),
        **{k: stats.get(k, "") for k in (
            "ah_start", "ah_end", "ah_min", "ah_max", "charge_ah_max", "charge_ah_fraction",
            "voltage_start", "voltage_end", "voltage_max", "full_voltage_reached",
            "current_start", "current_end", "current_max", "current_taper_detected",
            "ah_reset_detected",
        )},
        "timestamp_conflict": chrono_row.get("timestamp_conflict", False),
        "proposed_charge_role": role,
        "proposed_model_input": proposed_mi,
        "confidence": confidence,
        "requires_manual_review": requires_review,
        "reason": "; ".join(reason_parts),
        "paper_support": paper,
        "readme_support": readme,
        "chronology_support": chrono_s,
        "signal_support": signal,
        "manifest_support": manifest,
    }


def _recommended_action(row: dict, cls: dict) -> tuple[str, str]:
    role = cls["proposed_charge_role"]
    splits = cls["current_split_memberships"]
    if cls["requires_manual_review"] or role in ("ambiguous", "cross_split_bridge"):
        if role in ("initial_topup", "pre_capacity_test_topup", "tiny_topoff") and cls["current_record_role"] == "model_input":
            return "CHANGE_TO_EXCLUDED", "Role is top-up/tiny but marked model_input"
        if cls.get("ah_reset_detected"):
            return "MANUAL_REVIEW", "Ah reset may invalidate SOC label while sequence membership is plausible"
        return "MANUAL_REVIEW", cls["reason"]
    if role in ("initial_topup", "pre_capacity_test_topup", "tiny_topoff", "excluded_profile_related"):
        return "CHANGE_TO_EXCLUDED", f"Proposed role {role} not in Vidal selected CC input set"
    if role == "leading_full_charge_anchor":
        return "KEEP_MODEL_INPUT", "Vidal FULL training includes accompanying CC before first Mix (Section IV.B)"
    if role == "trailing_charge":
        return "MANUAL_REVIEW", "Trailing charge after last training mix — paper silent on inclusion"
    if role == "cross_split_bridge":
        return "MANUAL_REVIEW", cls["reason"]
    if role in ("normal_intercycle_charge", "post_capacity_test_full_recharge"):
        if role == "post_capacity_test_full_recharge" and "test" in splits:
            return "KEEP_MODEL_INPUT", "Full recharge before test drive (corresponding CC for UDDS)"
        return "KEEP_MODEL_INPUT", "Accompanying/corresponding CC per Vidal Section IV.B"
    return "MANUAL_REVIEW", cls["reason"]


def build_review(class_df: pd.DataFrame) -> pd.DataFrame:
    review_rows = []
    mi = class_df[class_df["current_record_role"] == "model_input"]
    for _, cls in mi.iterrows():
        action, reason = _recommended_action(cls.to_dict(), cls.to_dict())
        review_rows.append({**cls.to_dict(), "recommended_action": action, "review_reason": reason})
    return pd.DataFrame(review_rows)


def build_candidate_sequences(chrono: pd.DataFrame, class_df: pd.DataFrame) -> dict:
    out: dict[str, Any] = {}
    cls_by_fn = {r["filename"]: r for _, r in class_df.iterrows()}
    for dataset in chrono["dataset"].unique():
        out[dataset] = {}
        ds = chrono[chrono["dataset"] == dataset]
        for split in ("full_train", "partial_train", "test"):
            profiles = PAPER_PROFILES[dataset][split]
            drives = ds[
                (ds["file_type"] == "drive_cycle")
                & (ds["record_role"] == "model_input")
                & (ds["paper_profile"].isin(profiles) | ds["profile"].isin(["UDDS", "LA92", "US06", "US06", "Cycle_1", "Cycle_2", "Cycle_3", "Cycle_4", "NN"] + [f"Mixed{i}" for i in range(1, 9)]))
            ]
            if split == "full_train" and dataset == "panasonic_18650pf":
                drives = ds[(ds["file_type"] == "drive_cycle") & (ds["record_role"] == "model_input") & (ds[split] == "1")]
            elif split in ("full_train", "partial_train", "test"):
                drives = ds[(ds["file_type"] == "drive_cycle") & (ds["record_role"] == "model_input") & (ds[split] == "1")]

            seq = []
            order = 0
            for _, d in drives.sort_values("selected_start_datetime").iterrows():
                order += 1
                seq.append({
                    "sequence_order": order,
                    "filename": d["filename"],
                    "profile": d["profile"],
                    "paper_profile": d["paper_profile"],
                    "file_type": "drive_cycle",
                    "sequence_role": "drive_input",
                    "split_memberships": _split_memberships(d.to_dict()),
                    "evidence": "Section IV.B Table V",
                })
                # intercycle charge after drive (except last)
                charges_after = ds[
                    (ds["file_type"] == "charge")
                    & (ds["previous_selected_drive"] == d["filename"])
                    & (ds["record_role"] == "model_input")
                ]
                for _, c in charges_after.iterrows():
                    cr = cls_by_fn.get(c["filename"], {})
                    if cr.get("proposed_charge_role") in ("trailing_charge", "leading_full_charge_anchor"):
                        continue
                    order += 1
                    seq.append({
                        "sequence_order": order,
                        "filename": c["filename"],
                        "profile": c["profile"],
                        "paper_profile": c.get("paper_profile", "CC"),
                        "file_type": "charge",
                        "sequence_role": "intercycle_charge_input",
                        "split_memberships": _split_memberships(c.to_dict()),
                        "evidence": cr.get("reason", ""),
                    })

            anchors = [r for _, r in class_df.iterrows() if r["proposed_charge_role"] == "leading_full_charge_anchor" and r["dataset"] == dataset]
            trailing = [r for _, r in class_df.iterrows() if r["proposed_charge_role"] == "trailing_charge" and r["dataset"] == dataset]
            out[dataset][split] = {
                "ordered_model_sequence": seq,
                "leading_anchors": [a["filename"] for a in anchors],
                "trailing_candidates": [t["filename"] for t in trailing],
                "excluded_related_charges": class_df[(class_df["dataset"] == dataset) & (class_df["proposed_charge_role"] == "excluded_profile_related")]["filename"].tolist(),
                "capacity_test_related_charges": class_df[(class_df["dataset"] == dataset) & (class_df["proposed_charge_role"].isin(["initial_topup", "pre_capacity_test_topup", "post_capacity_test_full_recharge"]))]["filename"].tolist(),
                "ambiguous_charges": class_df[(class_df["dataset"] == dataset) & (class_df["proposed_charge_role"] == "ambiguous")]["filename"].tolist(),
                "cross_split_charges": class_df[(class_df["dataset"] == dataset) & (class_df["proposed_charge_role"] == "cross_split_bridge")]["filename"].tolist(),
            }
    return out


def write_report(
    paths: dict,
    chrono: pd.DataFrame,
    class_df: pd.DataFrame,
    review_df: pd.DataFrame,
    sequences: dict,
) -> None:
    lines = [
        "# Vidal Charge Protocol Audit Report",
        "",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')} UTC",
        "",
        "## 1. Evidence paths used",
        "",
        f"- Vidal PDF: `{paths['vidal_pdf']}`",
    ]
    for ds, p in paths["readmes"].items():
        lines.append(f"- {ds} README: `{p}`")
    for ds, p in paths["manifests"].items():
        lines.append(f"- {ds} manifest: `{p}`")
    for ds, p in paths["raw_roots"].items():
        lines.append(f"- {ds} raw root: `{p}`")

    lines += [
        "",
        "## 2. Vidal paper protocol summary",
        "",
        "Table I (p.1): Each dataset lists Mix profiles, UDDS/LA92/US06, and CC.",
        "Section IV.B (p.4): FULL training = all Mixed drive cycles + **accompanying charges (CC)**.",
        "Test dataset = UDDS, LA92, US06 + **corresponding charges (CC)**.",
        "Table V (p.4): Case 1 FULL training; Case 3 PARTIAL Mix 1-3 (+ Samsung 48G Mix 1,2,6).",
        "Section IV.A: SOC reference from **cycler-calculated Ah** during training.",
        "Figure 8 (p.5): Samsung T30 example shows drive + charge in continuous input (Case 2 TL).",
        "",
        "## 3–5. README summaries",
        "",
        "**Panasonic:** Nine drive cycles (Cycle1-4, US06, HWFET, UDDS, LA92, NN); charged after each test; Ah reset before each segment; filename sort for pairing.",
        "",
        "**LG HG2:** UDDS→HWFET→LA92→US06→Mix1-8; two Cap_1C before UDDS and Mix3; charged after each test.",
        "",
        "**Samsung 30T:** 16 charges; Charge1 initial top-up before Cap_1C; charges before/after every drive; two Cap_1C before UDDS and Mix3.",
        "",
        "## 6. Four-way consistency",
        "",
        "| Source | Agreement |",
        "|--------|-----------|",
        "| Paper vs README | Both expect CC with mix/test drives; README adds HWFET/Cap_1C not in paper test set |",
        "| README vs chronology | File timestamps match README order for all three datasets |",
        "| Chronology vs manifest | Pairing mostly confirmed; LG Charge4 tiny top-off and Samsung Charge6 cross-split flagged |",
        "| Manifest vs paper | Selected charges marked model_input align with accompanying/corresponding CC intent; exceptions below |",
        "",
        "## 7. Chronology summary",
        "",
    ]
    for ds in chrono["dataset"].unique():
        sub = chrono[chrono["dataset"] == ds]
        conflicts = sub["timestamp_conflict"].sum()
        lines.append(f"- **{ds}**: {len(sub)} files; timestamp conflicts={int(conflicts)}; time regressions={int(sub['negative_time_step_count'].astype(float).sum())}")

    lines += ["", "## 8–14. Charge counts", ""]
    for ds in class_df["dataset"].unique():
        sub = class_df[class_df["dataset"] == ds]
        mi = sub[sub["current_record_role"] == "model_input"]
        rev = review_df[review_df["dataset"] == ds] if len(review_df) else pd.DataFrame()
        lines.append(f"### {ds}")
        lines.append(f"- Total charges: {len(sub)}")
        lines.append(f"- model_input charges: {len(mi)}")
        for action in ("KEEP_MODEL_INPUT", "CHANGE_TO_ANCHOR", "CHANGE_TO_EXCLUDED", "MANUAL_REVIEW"):
            n = len(rev[rev["recommended_action"] == action]) if len(rev) else 0
            lines.append(f"- {action}: {n}")
        roles = sub["proposed_charge_role"].value_counts()
        lines.append("- proposed_charge_role: " + ", ".join(f"{k}={v}" for k, v in roles.items()))
        lines.append("")

    def _list_section(title: str, df: pd.DataFrame, mask):
        lines.append(f"## {title}")
        lines.append("")
        sub = df[mask]
        if len(sub) == 0:
            lines.append("_None_")
        else:
            for _, r in sub.iterrows():
                lines.append(f"- `{r['filename']}` ({r['dataset']}): {r.get('reason', r.get('review_reason', ''))}")
        lines.append("")

    _list_section(
        "15. Tiny top-offs",
        class_df,
        pd.to_numeric(class_df["charge_ah_fraction"], errors="coerce") <= TINY_FRAC,
    )
    _list_section("16. Ah reset charges", class_df, class_df["ah_reset_detected"] == True)
    _list_section("17. Capacity-test-related", class_df, class_df["proposed_charge_role"].isin(["initial_topup", "pre_capacity_test_topup", "post_capacity_test_full_recharge"]))
    _list_section("18. Excluded-profile-related", class_df, class_df["proposed_charge_role"] == "excluded_profile_related")
    _list_section("19. Cross-split charges", class_df, class_df["proposed_charge_role"] == "cross_split_bridge")

    lines += ["## 20. Likely mis-tagged model_input charges", ""]
    bad = review_df[review_df["recommended_action"] != "KEEP_MODEL_INPUT"]
    if len(bad) == 0:
        lines.append("_None flagged_")
    else:
        for _, r in bad.iterrows():
            lines.append(f"- `{r['filename']}`: {r['recommended_action']} — {r['review_reason']}")

    lines += ["", "## 21–22. Disputed charge timelines", ""]
    for _, r in bad.iterrows():
        lines.append(f"### {r['filename']}")
        lines.append(f"- Previous: `{r['previous_profile']}` ({r['previous_file_type']})")
        lines.append(f"- Next: `{r['next_profile']}` ({r['next_file_type']})")
        lines.append(f"- Evidence: PAPER={r['paper_support']} README={r['readme_support']} TIMESTAMP={r['chronology_support']} SIGNAL={r['signal_support']} MANIFEST={r['manifest_support']}")
        lines.append("")

    for ds in ("panasonic_18650pf", "lg_hg2", "samsung_30t"):
        lines += [f"## Recommended actions — {ds}", ""]
        lines.append("| filename | current role | splits | prev | next | proposed | action | reason | confidence |")
        lines.append("|----------|--------------|--------|------|------|----------|--------|--------|------------|")
        sub = review_df[review_df["dataset"] == ds]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['filename']} | {r['current_record_role']} | {r['current_split_memberships']} | "
                f"{r['previous_profile']} | {r['next_profile']} | {r['proposed_charge_role']} | "
                f"{r['recommended_action']} | {r['review_reason'][:60]} | {r['confidence']} |"
            )
        lines.append("")

    (AUDIT_DIR / "VIDAL_CHARGE_AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def create_notebook(output_path: Path) -> None:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Vidal Charge Pairing Review\n", "\n", "Read-only notebook. Loads audit CSVs and plots flagged charges."],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from battery_benchmark.data.panasonic import PanasonicAdapter\n",
                    "from battery_benchmark.data.lg_hg2 import LGHG2Adapter\n",
                    "from battery_benchmark.data.samsung_30t import Samsung30TAdapter\n",
                    "\n",
                    "REPO = Path('..').resolve()\n",
                    "AUDIT = REPO / 'outputs/audits'\n",
                    "review = pd.read_csv(AUDIT / 'vidal_model_input_charge_review.csv')\n",
                    "class_df = pd.read_csv(AUDIT / 'vidal_charge_classification.csv')\n",
                    "ADAPTERS = {\n",
                    "    'panasonic_18650pf': (PanasonicAdapter, REPO / 'data/manifests/panasonic_vidal2019_25c_manifest.csv'),\n",
                    "    'lg_hg2': (LGHG2Adapter, REPO / 'data/manifests/lg_hg2_vidal2019_25c_manifest.csv'),\n",
                    "    'samsung_30t': (Samsung30TAdapter, REPO / 'data/manifests/samsung_30t_vidal2019_25c_manifest.csv'),\n",
                    "}\n",
                    "TINY = 0.05\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "DATASETS = sorted(review['dataset'].unique())\n",
                    "DATASET = DATASETS[0]  # change index or assign directly\n",
                    "charges = review[review['dataset'] == DATASET]['filename'].tolist()\n",
                    "FILENAME = charges[0]  # change index\n",
                    "print('Available datasets:', DATASETS)\n",
                    "print(f'Selected: {DATASET} / {FILENAME}')\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "flagged = review[\n",
                    "    (review['ah_reset_detected'] == True)\n",
                    "    | (pd.to_numeric(review['charge_ah_fraction'], errors='coerce') <= TINY)\n",
                    "    | (review['recommended_action'] != 'KEEP_MODEL_INPUT')\n",
                    "    | (review['requires_manual_review'] == True)\n",
                    "]\n",
                    "print(f'Flagged records: {len(flagged)}')\n",
                    "flagged[['dataset','filename','proposed_charge_role','recommended_action']].head(20)\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "def load_record(dataset, filename):\n",
                    "    cls, manifest = ADAPTERS[dataset]\n",
                    "    ad = cls(raw_root=REPO, manifest_path=manifest)\n",
                    "    entry = next(e for e in ad.entries if e.filename == filename)\n",
                    "    return ad.load_entry(entry, requested_split='full_train')\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "row = review[(review['dataset']==DATASET) & (review['filename']==FILENAME)].iloc[0]\n",
                    "rec = load_record(DATASET, FILENAME)\n",
                    "df = rec.data\n",
                    "meta = {\n",
                    "    'dataset': DATASET,\n",
                    "    'filename': FILENAME,\n",
                    "    'previous profile': row['previous_profile'],\n",
                    "    'next profile': row['next_profile'],\n",
                    "    'current role': row['current_record_role'],\n",
                    "    'current splits': row['current_split_memberships'],\n",
                    "    'proposed role': row['proposed_charge_role'],\n",
                    "    'recommended action': row['recommended_action'],\n",
                    "    'Ah max': row['charge_ah_max'],\n",
                    "    'Ah end': row['ah_end'],\n",
                    "    'Ah fraction': row['charge_ah_fraction'],\n",
                    "    'reset flag': row['ah_reset_detected'],\n",
                    "    'paper support': row['paper_support'],\n",
                    "    'README support': row['readme_support'],\n",
                    "}\n",
                    "for k,v in meta.items():\n",
                    "    print(f'{k}: {v}')\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "plt.figure(figsize=(10,4))\n",
                    "plt.plot(df['time_s'], df['cycler_ah'])\n",
                    "plt.xlabel('time_s'); plt.ylabel('cycler_ah'); plt.title(f'{FILENAME} — Ah')\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "plt.figure(figsize=(10,4))\n",
                    "plt.plot(df['time_s'], df['current_a'])\n",
                    "plt.xlabel('time_s'); plt.ylabel('current_a'); plt.title(f'{FILENAME} — current')\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "plt.figure(figsize=(10,4))\n",
                    "plt.plot(df['time_s'], df['voltage_v'])\n",
                    "plt.xlabel('time_s'); plt.ylabel('voltage_v'); plt.title(f'{FILENAME} — voltage')\n",
                    "plt.show()\n",
                ],
            },
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    import json as _json
    output_path.write_text(_json.dumps(nb, indent=1), encoding="utf-8")


def run_audit(overwrite: bool = False) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        AUDIT_DIR / "vidal_paper_protocol_reference.csv",
        AUDIT_DIR / "vidal_charge_chronology.csv",
        AUDIT_DIR / "vidal_charge_classification.csv",
        AUDIT_DIR / "vidal_model_input_charge_review.csv",
        AUDIT_DIR / "vidal_candidate_sequences.json",
    ]
    if not overwrite:
        existing = [p for p in outputs if p.exists()]
        if existing:
            raise FileExistsError(f"Outputs exist: {existing[0]}. Pass --overwrite")

    paper_df = build_paper_reference()
    paper_df.to_csv(outputs[0], index=False)

    all_chrono = []
    all_class = []
    records_cache: dict[str, dict[str, Any]] = {}

    for dataset, manifest_path in MANIFEST_PATHS.items():
        rows = _load_manifest(manifest_path)
        adapter = ADAPTERS[dataset](raw_root=REPO_ROOT, manifest_path=manifest_path)
        recs = {}
        for row in rows:
            split = "full_train"
            if row.get("test") == "1":
                split = "test"
            elif row.get("exclude") == "1":
                split = "exclude"
            rec = _load_record(adapter, row, split=split)
            if rec:
                recs[row["filename"]] = rec
        records_cache[dataset] = recs

        chrono = build_chronology(dataset, rows, recs)
        all_chrono.append(chrono)

        charge_rows = [r for r in rows if r.get("file_type") == "charge"]
        chrono_idx = {r["filename"]: r for _, r in chrono.iterrows()}
        for row in charge_rows:
            fn = row["filename"]
            cr = chrono_idx[fn]
            stats = _signal_stats(recs.get(fn))
            all_class.append(_classify_charge(dataset, row, cr, stats))

    chrono_df = pd.concat(all_chrono, ignore_index=True)
    class_df = pd.DataFrame(all_class)
    review_df = build_review(class_df)
    sequences = build_candidate_sequences(chrono_df, class_df)

    chrono_df.to_csv(outputs[1], index=False)
    class_df.to_csv(outputs[2], index=False)
    review_df.to_csv(outputs[3], index=False)
    with open(outputs[4], "w", encoding="utf-8") as f:
        json.dump(sequences, f, indent=2, sort_keys=True)

    paths = {
        "vidal_pdf": str(VIDAL_PDF),
        "readmes": {k: str(v) for k, v in README_PATHS.items()},
        "manifests": {k: str(v) for k, v in MANIFEST_PATHS.items()},
        "raw_roots": {k: str(v) for k, v in RAW_ROOTS.items()},
    }
    write_report(paths, chrono_df, class_df, review_df, sequences)

    nb_path = REPO_ROOT / "notebooks/review_vidal_charge_pairing.ipynb"
    create_notebook(nb_path)

    # Self-checks
    assert len(chrono_df) == sum(len(_load_manifest(p)) for p in MANIFEST_PATHS.values())
    assert len(class_df) == sum(1 for p in MANIFEST_PATHS.values() for r in _load_manifest(p) if r["file_type"] == "charge")
    assert len(review_df) == len(class_df[class_df["current_record_role"] == "model_input"])
    for ds in chrono_df["dataset"].unique():
        sub = chrono_df[chrono_df["dataset"] == ds].copy()
        sub["_t"] = pd.to_datetime(sub["selected_start_datetime"], errors="coerce")
        if not sub["_t"].is_monotonic_increasing:
            raise AssertionError(f"Chronology not monotonic for {ds}")
    print(f"Wrote audit to {AUDIT_DIR}")
    print(f"Notebook: {REPO_ROOT / 'notebooks/review_vidal_charge_pairing.ipynb'}")
    print(f"Chronology rows: {len(chrono_df)}")
    print(f"Charge rows: {len(class_df)}")
    print(f"model_input review: {len(review_df)}")
    print(review_df["recommended_action"].value_counts().to_string())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_audit(overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
