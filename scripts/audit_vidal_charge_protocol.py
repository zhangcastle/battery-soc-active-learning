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


def _is_hwfet_profile(profile: str) -> bool:
    p = str(profile).upper()
    return "HWFET" in p or "HWFT" in p


def _skippable_gap(row: dict | pd.Series) -> bool:
    """Pause/rest files that do not break charge ownership lookup."""
    prof = str(row.get("profile", ""))
    ft = str(row.get("file_type", ""))
    if ft == "pause" or prof in ("Pause", "PausCycl"):
        return True
    return False


def _is_actual_drive(row: dict | pd.Series) -> bool:
    return str(row.get("file_type", "")) == "drive_cycle"


def _is_selected_drive(row: dict | pd.Series) -> bool:
    return _is_actual_drive(row) and str(row.get("record_role", "")) == "model_input"


def _protocol_split_set(row: dict | pd.Series | None) -> set[str]:
    if row is None:
        return set()
    out = set()
    for s in ("full_train", "partial_train", "test"):
        if str(row.get(s, "0")).strip() in ("1", "1.0"):
            out.add(s)
    return out


def _find_previous_actual_drive(chrono: pd.DataFrame, idx: int) -> pd.Series | None:
    for j in range(idx - 1, -1, -1):
        row = chrono.iloc[j]
        if _skippable_gap(row):
            continue
        if _is_actual_drive(row):
            return row
    return None


def _find_next_selected_drive(chrono: pd.DataFrame, idx: int) -> pd.Series | None:
    for j in range(idx + 1, len(chrono)):
        row = chrono.iloc[j]
        if _skippable_gap(row):
            continue
        if _is_selected_drive(row):
            return row
        if _is_actual_drive(row):
            continue
    return None


def _find_immediate_prev_non_skippable(chrono: pd.DataFrame, idx: int) -> pd.Series | None:
    for j in range(idx - 1, -1, -1):
        row = chrono.iloc[j]
        if not _skippable_gap(row):
            return row
    return None


def _leading_context(chrono: pd.DataFrame, idx: int) -> tuple[bool, str]:
    """True if charge follows Cap_1C/characterization with no selected drive before."""
    for j in range(idx - 1, -1, -1):
        row = chrono.iloc[j]
        if _skippable_gap(row):
            continue
        if _is_selected_drive(row):
            return False, ""
        prof = str(row.get("profile", ""))
        if prof == "Cap_1C" or row.get("file_type") == "characterization":
            return True, prof
        if _is_actual_drive(row):
            return False, ""
        if row.get("file_type") == "charge":
            continue
        return False, ""
    return True, "session_start"


def _compute_split_relation(
    owner_splits: set[str],
    charge_splits: set[str],
    next_selected_splits: set[str],
    is_leading: bool,
    is_trailing: bool,
) -> str:
    if is_leading:
        return "leading_anchor"
    if is_trailing and not next_selected_splits:
        return "trailing_after_selected_drive"

    if owner_splits and next_selected_splits:
        if owner_splits == next_selected_splits:
            return "same_protocol_split"
        if owner_splits == {"test"} and next_selected_splits & {"full_train", "partial_train"}:
            return "true_test_train_boundary"
        if owner_splits & {"full_train", "partial_train"} and next_selected_splits == {"test"}:
            return "true_test_train_boundary"
        if owner_splits & {"partial_train", "full_train"} and next_selected_splits <= {"full_train"}:
            if owner_splits >= next_selected_splits or owner_splits & {"partial_train"}:
                return "partial_subset_boundary"
        if owner_splits & {"full_train"} and next_selected_splits & {"full_train", "partial_train"}:
            if owner_splits == {"full_train"} and next_selected_splits == {"full_train", "partial_train"}:
                return "partial_subset_boundary"

    if owner_splits and charge_splits:
        if owner_splits == charge_splits or owner_splits >= charge_splits:
            return "same_protocol_split"
        if charge_splits & owner_splits:
            return "partial_subset_boundary"

    if owner_splits and not next_selected_splits:
        return "trailing_after_selected_drive"
    return "unrelated"


def _is_cap_1c_row(row: dict | pd.Series) -> bool:
    prof = str(row.get("profile", ""))
    return prof == "Cap_1C" or (
        str(row.get("file_type", "")) == "characterization" and "Cap" in prof
    )


def _infer_post_cap_anchor(chrono: pd.DataFrame, idx: int, stats: dict) -> tuple[bool, str]:
    """Charge immediately after Cap_1C (or missing Cap_1C + pre-cap top-off) before next selected drive."""
    imm = _find_immediate_prev_non_skippable(chrono, idx)
    next_selected = _find_next_selected_drive(chrono, idx)
    if next_selected is None:
        return False, ""
    if imm is not None and _is_cap_1c_row(imm):
        return True, "Cap_1C"
    next_prof = str(next_selected.get("profile", ""))
    if not re.match(r"Mixed?\d", next_prof, re.I):
        return False, ""
    frac = float(stats.get("charge_ah_fraction") or 0)
    if imm is not None and str(imm.get("record_role", "")) == "excluded" and str(imm.get("file_type", "")) == "charge":
        if frac >= 0.5 and bool(stats.get("full_voltage_reached")):
            return True, "missing_Cap_1C_inferred"
    return False, ""


def _resolve_ownership(chrono: pd.DataFrame, idx: int, row: dict, stats: dict | None = None) -> dict[str, Any]:
    prev_actual = _find_previous_actual_drive(chrono, idx)
    next_selected = _find_next_selected_drive(chrono, idx)
    imm_prev = _find_immediate_prev_non_skippable(chrono, idx)
    is_leading_ctx, leading_reason = _leading_context(chrono, idx)
    post_cap, post_cap_reason = _infer_post_cap_anchor(chrono, idx, stats or {})

    owner = prev_actual
    ownership_basis = "previous_actual_drive"
    if post_cap:
        owner = None
        ownership_basis = f"post_capacity_test:{post_cap_reason}"
        is_leading_ctx = True
        leading_reason = post_cap_reason
    elif owner is None and is_leading_ctx:
        ownership_basis = f"leading_context:{leading_reason}"
    elif owner is None:
        ownership_basis = "no_previous_actual_drive"

    owner_splits = _protocol_split_set(owner)
    charge_splits = _protocol_split_set(row)
    next_splits = _protocol_split_set(next_selected)

    is_leading = owner is None and is_leading_ctx
    is_trailing = (
        owner is not None
        and _is_selected_drive(owner)
        and next_selected is None
    )

    split_relation = _compute_split_relation(
        owner_splits, charge_splits, next_splits, is_leading, is_trailing
    )

    return {
        "previous_actual_drive": owner["filename"] if owner is not None else "",
        "previous_actual_drive_selected": _is_selected_drive(owner) if owner is not None else False,
        "previous_actual_drive_profile": owner["profile"] if owner is not None else "",
        "previous_actual_drive_protocol_splits": ",".join(sorted(owner_splits)),
        "next_actual_drive": next_selected["filename"] if next_selected is not None else "",
        "next_actual_drive_selected": next_selected is not None,
        "next_actual_drive_profile": next_selected["profile"] if next_selected is not None else "",
        "next_actual_drive_protocol_splits": ",".join(sorted(next_splits)),
        "ownership_basis": ownership_basis,
        "owned_by_previous_profile": owner["profile"] if owner is not None else "",
        "owned_by_previous_sequence_id": owner["sequence_id"] if owner is not None else "",
        "owned_by_previous_protocol_splits": ",".join(sorted(owner_splits)),
        "is_leading_charge": is_leading,
        "is_trailing_charge": is_trailing,
        "split_relation": split_relation,
        "immediate_previous_profile": imm_prev["profile"] if imm_prev is not None else "",
        "immediate_previous_file_type": imm_prev["file_type"] if imm_prev is not None else "",
    }


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
    chrono: pd.DataFrame,
    chrono_idx: int,
    stats: dict,
) -> dict[str, Any]:
    fn = row["filename"]
    splits = _split_memberships(row)
    frac = float(stats.get("charge_ah_fraction") or 0)
    own = _resolve_ownership(chrono, chrono_idx, row, stats)
    chrono_row = chrono.iloc[chrono_idx]

    role = "ambiguous"
    confidence = "medium"
    reason_parts: list[str] = []
    paper = readme = chrono_s = signal = manifest = False

    owner_prof = own["owned_by_previous_profile"]
    is_leading = own["is_leading_charge"]
    is_trailing = own["is_trailing_charge"]
    split_rel = own["split_relation"]
    imm_prev_prof = own["immediate_previous_profile"]
    cn = _charge_number(fn, row.get("profile", ""))

    if row.get("record_role") == "excluded":
        if frac <= TINY_FRAC and cn in (1, 9):
            role = "pre_capacity_test_topup" if cn != 1 else "initial_topup"
        elif _is_hwfet_profile(owner_prof):
            role = "excluded_profile_corresponding_charge"
        else:
            role = "excluded_profile_corresponding_charge" if owner_prof else "ambiguous"
        readme = manifest = True
        reason_parts.append("manifest excluded")
        return _charge_result(row, splits, stats, chrono_row, own, role, confidence, reason_parts, paper, readme, chrono_s, signal, manifest)

    # --- model_input / reviewable selected charges ---
    post_cap = own["ownership_basis"].startswith("post_capacity_test:")
    if is_leading or post_cap or (not owner_prof and own["ownership_basis"].startswith(("leading_context", "post_capacity_test"))):
        if post_cap or imm_prev_prof == "Cap_1C" or "Cap_1C" in own["ownership_basis"]:
            role = "post_capacity_test_full_recharge_anchor"
            readme = chrono_s = True
            nxt = own["next_actual_drive_profile"]
            reason_parts.append(
                f"Post Cap_1C full recharge before next selected drive ({nxt or 'Mix3 gateway'})"
            )
        else:
            role = "leading_full_charge_anchor"
            paper = readme = chrono_s = True
            reason_parts.append("Leading charge before first selected drive")
        confidence = "high"

    elif owner_prof and _is_hwfet_profile(owner_prof):
        role = "excluded_profile_corresponding_charge"
        readme = paper = chrono_s = True
        if frac <= TINY_FRAC:
            role = "tiny_topoff"
            signal = True
            reason_parts.append(f"HWFET-owned tiny top-off ({frac:.3f} Ah fraction)")
        else:
            reason_parts.append("Charge owned by excluded HWFET drive (not in Vidal test set)")
        confidence = "high"

    elif frac <= TINY_FRAC and not _is_hwfet_profile(owner_prof):
        role = "tiny_topoff"
        signal = True
        reason_parts.append(f"charge_ah_fraction={frac:.3f} <= {TINY_FRAC}")

    elif is_trailing and owner_prof:
        role = "trailing_corresponding_charge"
        chrono_s = readme = True
        reason_parts.append(f"Corresponding charge after last selected drive ({owner_prof})")
        confidence = "high"

    elif owner_prof:
        role = "normal_corresponding_charge"
        paper = readme = chrono_s = True
        reason_parts.append(f"Corresponding charge owned by preceding {owner_prof}")
        confidence = "high"
        if split_rel == "true_test_train_boundary":
            role = "true_test_train_boundary_charge"
            reason_parts.append(f"Split boundary: {own['owned_by_previous_protocol_splits']} -> {own['next_actual_drive_protocol_splits']}")
        elif split_rel == "partial_subset_boundary":
            reason_parts.append("Partial train is subset of full train (not a conflict)")

    if stats.get("ah_reset_detected"):
        signal = True
        reason_parts.append("Ah counter reset detected (label concern only)")

    if not reason_parts:
        reason_parts.append(f"After {imm_prev_prof}, before {own['next_actual_drive_profile']}")

    proto, label, review_reason = _recommended_actions(row, own, role, stats, split_rel)

    return _charge_result(
        row, splits, stats, chrono_row, own, role, confidence, reason_parts,
        paper, readme, chrono_s, signal, manifest,
        proto, label, review_reason,
    )


def _recommended_actions(
    row: dict,
    own: dict,
    role: str,
    stats: dict,
    split_rel: str,
) -> tuple[str, str, str]:
    ah_reset = bool(stats.get("ah_reset_detected"))
    frac = float(stats.get("charge_ah_fraction") or 0)

    if role in ("initial_topup", "pre_capacity_test_topup", "excluded_profile_corresponding_charge"):
        return ("CHANGE_TO_EXCLUDED", "EXCLUDE_FROM_LABEL_EXPORT", f"Role {role} outside Vidal protocol")

    if role == "tiny_topoff":
        if _is_hwfet_profile(own["owned_by_previous_profile"]):
            return (
                "CHANGE_TO_EXCLUDED",
                "EXCLUDE_FROM_LABEL_EXPORT",
                f"HWFET-owned tiny top-off ({own['owned_by_previous_profile']})",
            )
        return ("MANUAL_REVIEW", "MANUAL_REVIEW", f"Tiny top-off ({frac:.3f} Ah fraction) — verify protocol intent")

    if role in ("leading_full_charge_anchor", "post_capacity_test_full_recharge_anchor"):
        label = "REGENERATE_SEQUENCE_AWARE" if ah_reset else "REGENERATE_SEQUENCE_AWARE"
        return (
            "CHANGE_TO_ANCHOR",
            label,
            "Leading/post-Cap anchor for SOC_init evidence; not in ordered drive->charge sequence",
        )

    if role in ("normal_corresponding_charge", "trailing_corresponding_charge", "true_test_train_boundary_charge"):
        proto = "KEEP_MODEL_INPUT"
        reason = f"Corresponding CC owned by {own['owned_by_previous_profile']} (Section IV.B / Figure 8)"
        if ah_reset:
            return proto, "REGENERATE_RESET_AWARE", reason + "; Ah reset requires reset-aware labelling"
        return proto, "LABEL_OK", reason

    return ("MANUAL_REVIEW", "MANUAL_REVIEW", f"Unresolved role {role}")


def _charge_result(
    row, splits, stats, chrono_row, own, role, confidence, reason_parts,
    paper, readme, chrono_s, signal, manifest,
    protocol_action="", label_action="", review_reason="",
):
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
        "next_file": chrono_row.get("next_file", ""),
        "next_profile": chrono_row.get("next_profile", ""),
        "next_file_type": chrono_row.get("next_file_type", ""),
        **own,
        **{k: stats.get(k, "") for k in (
            "ah_start", "ah_end", "ah_min", "ah_max", "charge_ah_max", "charge_ah_fraction",
            "voltage_start", "voltage_end", "voltage_max", "full_voltage_reached",
            "current_start", "current_end", "current_max", "current_taper_detected",
            "ah_reset_detected",
        )},
        "timestamp_conflict": chrono_row.get("timestamp_conflict", False),
        "proposed_charge_role": role,
        "recommended_protocol_action": protocol_action,
        "recommended_label_action": label_action,
        "reason": "; ".join(reason_parts),
        "review_reason": review_reason,
        "confidence": confidence,
        "paper_support": paper,
        "readme_support": readme,
        "chronology_support": chrono_s,
        "signal_support": signal,
        "manifest_support": manifest,
    }


def build_review(class_df: pd.DataFrame) -> pd.DataFrame:
    mi = class_df[class_df["current_record_role"] == "model_input"].copy()
    review_cols = [
        "dataset", "filename", "sequence_id", "charge_number",
        "previous_actual_drive", "previous_actual_drive_selected", "previous_actual_drive_protocol_splits",
        "next_actual_drive", "next_actual_drive_selected", "next_actual_drive_protocol_splits",
        "ownership_basis", "owned_by_previous_profile", "owned_by_previous_protocol_splits",
        "split_relation", "current_record_role", "current_split_memberships",
        "charge_ah_max", "charge_ah_fraction", "ah_reset_detected",
        "full_voltage_reached", "current_taper_detected",
        "proposed_charge_role", "recommended_protocol_action", "recommended_label_action",
        "reason", "confidence",
        "paper_support", "readme_support", "chronology_support", "signal_support", "manifest_support",
        "is_leading_charge", "is_trailing_charge", "review_reason",
    ]
    existing = [c for c in review_cols if c in mi.columns]
    return mi[existing]


def build_candidate_sequences(chrono: pd.DataFrame, class_df: pd.DataFrame) -> dict:
    out: dict[str, Any] = {}
    cls_by_fn = {r["filename"]: r for _, r in class_df.iterrows()}
    chrono_by_ds: dict[str, pd.DataFrame] = {
        ds: chrono[chrono["dataset"] == ds].reset_index(drop=True)
        for ds in chrono["dataset"].unique()
    }

    anchor_roles = {"leading_full_charge_anchor", "post_capacity_test_full_recharge_anchor"}
    exclude_roles = {"excluded_profile_corresponding_charge", "tiny_topoff", "initial_topup", "pre_capacity_test_topup"}

    for dataset, ds in chrono_by_ds.items():
        out[dataset] = {}
        for split in ("full_train", "partial_train", "test"):
            drives = ds[
                (ds["file_type"] == "drive_cycle")
                & (ds["record_role"] == "model_input")
                & (ds[split] == "1")
                & (~ds["profile"].apply(_is_hwfet_profile))
            ].sort_values("selected_start_datetime")

            seq: list[dict] = []
            order = 0
            for _, d in drives.iterrows():
                if split == "test" and d["profile"] not in ("UDDS", "LA92", "US06"):
                    continue
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
                # corresponding charge: owned by this drive
                for _, c in ds[ds["file_type"] == "charge"].iterrows():
                    cr = cls_by_fn.get(c["filename"], {})
                    if cr.get("owned_by_previous_sequence_id") != d["sequence_id"]:
                        continue
                    if cr.get("proposed_charge_role") in anchor_roles | exclude_roles:
                        continue
                    if split == "test":
                        if cr.get("current_record_role") != "model_input":
                            continue
                    elif split == "partial_train":
                        # partial_train ⊂ full_train: include Mix1-3 accompanying charges by ownership
                        pass
                    elif split not in cr.get("current_split_memberships", ""):
                        continue
                    order += 1
                    seq.append({
                        "sequence_order": order,
                        "filename": c["filename"],
                        "profile": c["profile"],
                        "paper_profile": cr.get("paper_profile", "CC") if isinstance(cr, dict) else "CC",
                        "file_type": "charge",
                        "sequence_role": "intercycle_charge_input",
                        "split_memberships": cr.get("current_split_memberships", ""),
                        "evidence": cr.get("reason", ""),
                    })

            anchors = class_df[
                (class_df["dataset"] == dataset)
                & (class_df["proposed_charge_role"].isin(anchor_roles))
            ]["filename"].tolist()
            hwfet_owned = class_df[
                (class_df["dataset"] == dataset)
                & (
                    (class_df["proposed_charge_role"].isin(exclude_roles))
                    | (class_df["owned_by_previous_profile"].apply(_is_hwfet_profile))
                )
            ]["filename"].tolist()
            trailing = class_df[
                (class_df["dataset"] == dataset)
                & (class_df["proposed_charge_role"] == "trailing_corresponding_charge")
            ]["filename"].tolist()

            out[dataset][split] = {
                "ordered_model_sequence": seq,
                "leading_anchors": anchors,
                "trailing_corresponding_charges": trailing,
                "excluded_related_charges": hwfet_owned,
                "capacity_test_related_charges": class_df[
                    (class_df["dataset"] == dataset)
                    & (class_df["proposed_charge_role"].isin([
                        "initial_topup", "pre_capacity_test_topup",
                        "post_capacity_test_full_recharge_anchor",
                    ]))
                ]["filename"].tolist(),
                "ambiguous_charges": class_df[
                    (class_df["dataset"] == dataset)
                    & (class_df["proposed_charge_role"] == "ambiguous")
                ]["filename"].tolist(),
                "partial_subset_boundary_charges": class_df[
                    (class_df["dataset"] == dataset)
                    & (class_df["split_relation"] == "partial_subset_boundary")
                ]["filename"].tolist(),
            }
    return out


def write_report(
    paths: dict,
    chrono: pd.DataFrame,
    class_df: pd.DataFrame,
    review_df: pd.DataFrame,
    sequences: dict,
    prev_review: pd.DataFrame | None = None,
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
        "## 2. Charge ownership rule",
        "",
        "Default: **selected drive → corresponding/accompanying charge** (Figure 8: UDDS → CC → LA92 → CC → US06).",
        "Pause files do not break ownership. Cap_1C/characterization before first selected drive → leading anchor.",
        "Do not assign charge to the *following* drive.",
        "",
        "## 3. Partial is a subset of Full",
        "",
        "`partial_train` ⊂ `full_train`. Mix3 → Charge → Mix4 is **partial_subset_boundary**, not cross-split conflict.",
        "",
        "## 4. Protocol action vs label action",
        "",
        "- **Protocol action**: should this file be model input, anchor-only, or excluded from Vidal protocol?",
        "- **Label action**: given v1 per-file Ah normalization, is the SOC label OK, reset-aware, or sequence-aware?",
        "- Ah reset affects **label action only**, not protocol role.",
        "",
        "## 5. HWFET-owned charge exclusions",
        "",
        "Vidal test set = UDDS, LA92, US06 + corresponding CC. HWFET excluded → its charges → CHANGE_TO_EXCLUDED.",
        "",
        "## 6. Leading anchor handling",
        "",
        "Pre-first-drive and post-Cap_1C full recharges → CHANGE_TO_ANCHOR; stored in `leading_anchors`, not ordered sequence.",
        "",
        "## 7. Trailing corresponding charge handling",
        "",
        "Charge after last selected drive still **corresponds to that drive** → KEEP_MODEL_INPUT if profile is in Vidal set.",
        "",
        "## 8. Vidal paper protocol summary",
        "",
        "Section IV.B: FULL = Mix + accompanying CC; TEST = UDDS/LA92/US06 + corresponding CC.",
        "",
        "## 9. Chronology summary",
        "",
    ]
    for ds in chrono["dataset"].unique():
        sub = chrono[chrono["dataset"] == ds]
        lines.append(f"- **{ds}**: {len(sub)} files; timestamp conflicts={int(sub['timestamp_conflict'].sum())}")

    lines += ["", "## 10. Protocol action counts (37 model_input charges)", ""]
    for action in ("KEEP_MODEL_INPUT", "CHANGE_TO_ANCHOR", "CHANGE_TO_EXCLUDED", "MANUAL_REVIEW"):
        n = len(review_df[review_df["recommended_protocol_action"] == action])
        lines.append(f"- {action}: {n}")

    lines += ["", "## 11. Label action counts", ""]
    for action in (
        "LABEL_OK", "REGENERATE_RESET_AWARE", "REGENERATE_SEQUENCE_AWARE",
        "EXCLUDE_FROM_LABEL_EXPORT", "MANUAL_REVIEW",
    ):
        n = len(review_df[review_df["recommended_label_action"] == action])
        lines.append(f"- {action}: {n}")

    lines += ["", "## 12. Per-dataset breakdown", ""]
    for ds in review_df["dataset"].unique():
        sub = review_df[review_df["dataset"] == ds]
        lines.append(f"### {ds} ({len(sub)} model_input charges)")
        for action in ("KEEP_MODEL_INPUT", "CHANGE_TO_ANCHOR", "CHANGE_TO_EXCLUDED", "MANUAL_REVIEW"):
            lines.append(f"- Protocol {action}: {len(sub[sub['recommended_protocol_action'] == action])}")
        for action in ("REGENERATE_RESET_AWARE", "REGENERATE_SEQUENCE_AWARE", "EXCLUDE_FROM_LABEL_EXPORT"):
            lines.append(f"- Label {action}: {len(sub[sub['recommended_label_action'] == action])}")
        lines.append("")

    def _list_df(title, df, col="filename"):
        lines.append(f"## {title}")
        lines.append("")
        if len(df) == 0:
            lines.append("_None_")
        else:
            for _, r in df.iterrows():
                lines.append(f"- `{r[col]}` ({r['dataset']}): {r.get('proposed_charge_role','')} / {r.get('recommended_protocol_action','')}")
        lines.append("")

    hw = review_df[review_df["owned_by_previous_profile"].apply(_is_hwfet_profile)]
    _list_df("13. HWFET-owned charges", hw)
    anchors = review_df[review_df["recommended_protocol_action"] == "CHANGE_TO_ANCHOR"]
    _list_df("14. Leading anchors", anchors)
    trailing = review_df[review_df["proposed_charge_role"] == "trailing_corresponding_charge"]
    _list_df("15. Trailing corresponding charges", trailing)
    reset = review_df[review_df["recommended_label_action"] == "REGENERATE_RESET_AWARE"]
    _list_df("16. Reset-aware label charges", reset)
    partial = review_df[review_df["split_relation"] == "partial_subset_boundary"]
    _list_df("17. Partial subset boundary charges", partial)

    if prev_review is not None and len(prev_review) > 0:
        lines += ["", "## 18. Changes vs previous audit", ""]
        prev = prev_review.set_index("filename")
        for _, r in review_df.iterrows():
            fn = r["filename"]
            if fn not in prev.index:
                lines.append(f"- NEW `{fn}`")
                continue
            p = prev.loc[fn]
            changes = []
            if p.get("recommended_protocol_action", p.get("recommended_action")) != r["recommended_protocol_action"]:
                old = p.get("recommended_protocol_action", p.get("recommended_action"))
                changes.append(f"protocol {old} → {r['recommended_protocol_action']}")
            if p.get("proposed_charge_role") != r["proposed_charge_role"]:
                changes.append(f"role {p['proposed_charge_role']} → {r['proposed_charge_role']}")
            if changes:
                lines.append(f"- `{fn}`: {'; '.join(changes)}")

    for ds in ("panasonic_18650pf", "lg_hg2", "samsung_30t"):
        lines += [f"## Recommended actions — {ds}", ""]
        lines.append("| filename | owned_by | role | protocol | label | split_relation |")
        lines.append("|----------|----------|------|----------|-------|----------------|")
        sub = review_df[review_df["dataset"] == ds]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['filename']} | {r['owned_by_previous_profile']} | {r['proposed_charge_role']} | "
                f"{r['recommended_protocol_action']} | {r['recommended_label_action']} | {r['split_relation']} |"
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
                    "    | (review['recommended_protocol_action'] != 'KEEP_MODEL_INPUT')\n",
                    "    | (review['recommended_label_action'] != 'LABEL_OK')\n",
                    "]\n",
                    "print(f'Flagged records: {len(flagged)}')\n",
                    "flagged[['dataset','filename','proposed_charge_role','recommended_protocol_action','recommended_label_action']].head(20)\n",
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
                    "    'owned by': row['owned_by_previous_profile'],\n",
                    "    'proposed role': row['proposed_charge_role'],\n",
                    "    'protocol action': row['recommended_protocol_action'],\n",
                    "    'label action': row['recommended_label_action'],\n",
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
        chrono_idx = {r["filename"]: i for i, r in chrono.iterrows()}
        for row in charge_rows:
            fn = row["filename"]
            i = chrono_idx[fn]
            stats = _signal_stats(recs.get(fn))
            all_class.append(_classify_charge(dataset, row, chrono, i, stats))

    chrono_df = pd.concat(all_chrono, ignore_index=True)
    class_df = pd.DataFrame(all_class)
    review_df = build_review(class_df)
    sequences = build_candidate_sequences(chrono_df, class_df)

    review_path = outputs[3]
    prev_review = None
    if review_path.exists():
        prev_review = pd.read_csv(review_path)

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
    write_report(paths, chrono_df, class_df, review_df, sequences, prev_review=prev_review)

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
    print(review_df["recommended_protocol_action"].value_counts().to_string())
    print("--- label actions ---")
    print(review_df["recommended_label_action"].value_counts().to_string())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_audit(overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
