#!/usr/bin/env python3
"""Read-only Vidal 2019 charge protocol audit for Panasonic / LG / Samsung 30T."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from battery_benchmark.data.lg_hg2 import LGHG2Adapter
from battery_benchmark.data.panasonic import PanasonicAdapter
from battery_benchmark.data.samsung_30t import Samsung30TAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_README_PATHS = {
    "panasonic_18650pf": "data/raw/Panasonic_18650PF/Readme file - desc of tests performed.txt",
    "lg_hg2": "data/raw/LG_HG2/Readme file - Description of Experimental Tests.txt",
    "samsung_30t": "data/raw/Samsung_30T/Readme IN21700-30T.txt",
}
DEFAULT_MANIFEST_PATHS = {
    "panasonic_18650pf": "data/manifests/vidal_2019/panasonic_vidal2019_25c_manifest.csv",
    "lg_hg2": "data/manifests/vidal_2019/lg_hg2_vidal2019_25c_manifest.csv",
    "samsung_30t": "data/manifests/vidal_2019/samsung_30t_vidal2019_25c_manifest.csv",
}
DEFAULT_RAW_ROOTS = {
    "panasonic_18650pf": "data/raw/Panasonic_18650PF/25degC",
    "lg_hg2": "data/raw/LG_HG2/25degC",
    "samsung_30t": "data/raw/Samsung_30T/25degC",
}
DEFAULT_OUTPUT_DIR = "outputs/audits"
VIDAL_PDF_SEARCH_DIRS = ("", "docs", "references", "data")


@dataclass
class AuditConfig:
    repo_root: Path = field(default_factory=lambda: REPO_ROOT)
    audit_dir: Path = field(default_factory=lambda: REPO_ROOT / DEFAULT_OUTPUT_DIR)
    vidal_pdf: Path | None = None
    paper_reference_status: str = "NOT_AVAILABLE"
    readme_paths: dict[str, Path] = field(default_factory=dict)
    manifest_paths: dict[str, Path] = field(default_factory=dict)
    raw_roots: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.readme_paths:
            self.readme_paths = {
                k: self.repo_root / v for k, v in DEFAULT_README_PATHS.items()
            }
        if not self.manifest_paths:
            self.manifest_paths = {
                k: self.repo_root / v for k, v in DEFAULT_MANIFEST_PATHS.items()
            }
        if not self.raw_roots:
            self.raw_roots = {k: self.repo_root / v for k, v in DEFAULT_RAW_ROOTS.items()}


def _rel_path(path: Path | None, repo: Path) -> str:
    if path is None:
        return "<external Vidal PDF>"
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def _resolve_path(repo: Path, value: str | None, default: str) -> Path:
    p = Path(value) if value else repo / default
    return p if p.is_absolute() else repo / p


def _find_vidal_pdf(repo: Path, explicit: Path | None) -> tuple[Path | None, str]:
    if explicit is not None:
        if explicit.exists():
            return explicit, "EXPLICIT"
        return None, "EXPLICIT_NOT_FOUND"
    patterns = ("*Vidal*.pdf", "*vidal*.pdf", "*VIDAL*.pdf")
    for sub in VIDAL_PDF_SEARCH_DIRS:
        base = repo / sub if sub else repo
        if not base.is_dir():
            continue
        for pat in patterns:
            matches = sorted(base.glob(pat))
            if matches:
                return matches[0], "DISCOVERED"
    return None, "NOT_AVAILABLE"


def _build_audit_config(args: argparse.Namespace) -> AuditConfig:
    repo = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    explicit_pdf = Path(args.vidal_pdf).resolve() if args.vidal_pdf else None
    vidal_pdf, paper_status = _find_vidal_pdf(repo, explicit_pdf)
    cfg = AuditConfig(
        repo_root=repo,
        audit_dir=_resolve_path(repo, args.output_dir, DEFAULT_OUTPUT_DIR),
        vidal_pdf=vidal_pdf,
        paper_reference_status=paper_status,
        readme_paths={
            "panasonic_18650pf": _resolve_path(
                repo, args.panasonic_readme, DEFAULT_README_PATHS["panasonic_18650pf"]
            ),
            "lg_hg2": _resolve_path(repo, args.lg_readme, DEFAULT_README_PATHS["lg_hg2"]),
            "samsung_30t": _resolve_path(
                repo, args.samsung_readme, DEFAULT_README_PATHS["samsung_30t"]
            ),
        },
        manifest_paths={
            "panasonic_18650pf": repo / DEFAULT_MANIFEST_PATHS["panasonic_18650pf"],
            "lg_hg2": repo / DEFAULT_MANIFEST_PATHS["lg_hg2"],
            "samsung_30t": repo / DEFAULT_MANIFEST_PATHS["samsung_30t"],
        },
        raw_roots={
            "panasonic_18650pf": _resolve_path(
                repo, args.panasonic_root, DEFAULT_RAW_ROOTS["panasonic_18650pf"]
            ),
            "lg_hg2": _resolve_path(repo, args.lg_root, DEFAULT_RAW_ROOTS["lg_hg2"]),
            "samsung_30t": _resolve_path(
                repo, args.samsung_root, DEFAULT_RAW_ROOTS["samsung_30t"]
            ),
        },
    )
    return cfg
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
    return _lookup_previous_actual_drive(chrono, idx)["drive"]


def _find_next_selected_drive(chrono: pd.DataFrame, idx: int) -> pd.Series | None:
    return _lookup_next_selected_drive(chrono, idx)["drive"]


def _ownership_barrier_kind(row: dict | pd.Series) -> str | None:
    """Return barrier category when row blocks backward/forward drive lookup."""
    if _skippable_gap(row):
        return None
    if _is_actual_drive(row):
        return None
    ft = str(row.get("file_type", ""))
    prof = str(row.get("profile", ""))
    if ft == "charge":
        return "CHARGE"
    if _is_cap_1c_row(row):
        return "CHARACTERIZATION"
    if ft == "characterization":
        return "CHARACTERIZATION"
    if any(k in prof.upper() for k in ("HPPC", "EIS", "C20", "DIS_2C", "DISCHARGE", "TELEMETRY")):
        return "CHARACTERIZATION"
    if prof in ("Pause", "PausCycl"):
        return None
    return "UNKNOWN"


def _lookup_previous_actual_drive(chrono: pd.DataFrame, idx: int) -> dict[str, Any]:
    for j in range(idx - 1, -1, -1):
        row = chrono.iloc[j]
        if _skippable_gap(row):
            continue
        if _is_actual_drive(row):
            return {
                "drive": row,
                "ownership_search_status": "FOUND_PREVIOUS_DRIVE",
                "ownership_barrier_type": "",
                "ownership_barrier_filename": "",
            }
        barrier = _ownership_barrier_kind(row)
        if barrier:
            return {
                "drive": None,
                "ownership_search_status": f"STOPPED_BY_{barrier}",
                "ownership_barrier_type": barrier.lower(),
                "ownership_barrier_filename": row["filename"],
            }
    return {
        "drive": None,
        "ownership_search_status": "START_OF_SESSION",
        "ownership_barrier_type": "",
        "ownership_barrier_filename": "",
    }


def _lookup_next_selected_drive(chrono: pd.DataFrame, idx: int) -> dict[str, Any]:
    for j in range(idx + 1, len(chrono)):
        row = chrono.iloc[j]
        if _skippable_gap(row):
            continue
        if _is_selected_drive(row):
            return {
                "drive": row,
                "next_drive_search_status": "FOUND_NEXT_SELECTED_DRIVE",
                "next_drive_barrier_type": "",
                "next_drive_barrier_filename": "",
            }
        if _is_actual_drive(row):
            return {
                "drive": None,
                "next_drive_search_status": "STOPPED_BY_UNSELECTED_DRIVE",
                "next_drive_barrier_type": "unselected_drive",
                "next_drive_barrier_filename": row["filename"],
            }
        barrier = _ownership_barrier_kind(row)
        if barrier:
            return {
                "drive": None,
                "next_drive_search_status": f"STOPPED_BY_{barrier}",
                "next_drive_barrier_type": barrier.lower(),
                "next_drive_barrier_filename": row["filename"],
            }
    return {
        "drive": None,
        "next_drive_search_status": "END_OF_SESSION",
        "next_drive_barrier_type": "",
        "next_drive_barrier_filename": "",
    }


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
    prev_lookup = _lookup_previous_actual_drive(chrono, idx)
    next_lookup = _lookup_next_selected_drive(chrono, idx)
    prev_actual = prev_lookup["drive"]
    next_selected = next_lookup["drive"]
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
        and next_lookup["next_drive_search_status"] == "END_OF_SESSION"
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
        "ownership_search_status": prev_lookup["ownership_search_status"],
        "ownership_barrier_type": prev_lookup["ownership_barrier_type"],
        "ownership_barrier_filename": prev_lookup["ownership_barrier_filename"],
        "next_drive_search_status": next_lookup["next_drive_search_status"],
        "next_drive_barrier_type": next_lookup["next_drive_barrier_type"],
        "next_drive_barrier_filename": next_lookup["next_drive_barrier_filename"],
    }


def _selected_drive(row: dict) -> bool:
    return (
        row.get("file_type") == "drive_cycle"
        and row.get("record_role") == "model_input"
    )


def _load_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_record(adapter, row: dict, split: str = "full_train") -> dict[str, Any]:
    fn = row["filename"]
    try:
        entry = next(e for e in adapter.entries if e.filename == fn)
    except StopIteration:
        return {
            "record": None,
            "load_status": "FAIL",
            "load_error_type": "EntryNotFound",
            "load_error_message": f"Manifest entry not found for {fn}",
        }
    try:
        rec = adapter.load_entry(entry, requested_split=split)
        return {
            "record": rec,
            "load_status": "PASS",
            "load_error_type": "",
            "load_error_message": "",
        }
    except Exception as exc:
        return {
            "record": None,
            "load_status": "FAIL",
            "load_error_type": type(exc).__name__,
            "load_error_message": str(exc),
        }


def _reconstruction_fields(protocol_action: str, ah_reset: bool) -> dict[str, Any]:
    empty = {
        "previous_drive_end_soc_required": False,
        "capacity_reference_required": False,
        "ah_unwrap_required": False,
        "full_charge_anchor_required": False,
        "sequence_continuity_required": False,
        "expected_soc_start_source": "",
        "expected_soc_end_target": "",
    }
    if protocol_action == "KEEP_MODEL_INPUT":
        return {
            "previous_drive_end_soc_required": True,
            "capacity_reference_required": True,
            "ah_unwrap_required": ah_reset,
            "full_charge_anchor_required": False,
            "sequence_continuity_required": True,
            "expected_soc_start_source": "previous_drive_soc_end",
            "expected_soc_end_target": 1.0,
        }
    if protocol_action == "CHANGE_TO_ANCHOR":
        return {
            "previous_drive_end_soc_required": False,
            "capacity_reference_required": True,
            "ah_unwrap_required": ah_reset,
            "full_charge_anchor_required": True,
            "sequence_continuity_required": False,
            "expected_soc_start_source": "",
            "expected_soc_end_target": 1.0,
        }
    if protocol_action in ("CHANGE_TO_EXCLUDED", "MANUAL_REVIEW"):
        return empty
    return empty


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


def build_paper_reference(paper_status: str = "NOT_AVAILABLE") -> pd.DataFrame:
    available = paper_status in ("EXPLICIT", "DISCOVERED")
    rows = [
        {
            "dataset": "panasonic_18650pf",
            "paper_battery_name": "PANASONIC 18650PF (PANASONIC)",
            "paper_dataset_available": available,
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
            "paper_dataset_available": available,
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
            "paper_dataset_available": available,
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


def build_chronology(
    dataset: str,
    rows: list[dict],
    load_results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    enriched = []
    for row in rows:
        fn = row["filename"]
        lr = load_results.get(fn, {})
        rec = lr.get("record")
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
            "load_status": lr.get("load_status", "FAIL" if row.get("file_type") == "charge" else ""),
            "load_error_type": lr.get("load_error_type", ""),
            "load_error_message": lr.get("load_error_message", ""),
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
    load_status: str = "PASS",
) -> dict[str, Any]:
    fn = row["filename"]
    splits = _split_memberships(row)
    frac = float(stats.get("charge_ah_fraction") or 0)
    own = _resolve_ownership(chrono, chrono_idx, row, stats)
    chrono_row = chrono.iloc[chrono_idx]

    if load_status != "PASS" and row.get("record_role") == "model_input":
        return _charge_result(
            row, splits, stats, chrono_row, own,
            role="ambiguous",
            confidence="low",
            reason_parts=[f"Adapter load failed: {chrono_row.get('load_error_type', '')}"],
            paper=False, readme=False, chrono_s=False, signal=False, manifest=True,
            protocol_action="MANUAL_REVIEW",
            label_action="MANUAL_REVIEW",
            review_reason=f"Load failure: {chrono_row.get('load_error_message', '')}",
            load_status=load_status,
        )

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
        return _charge_result(
            row, splits, stats, chrono_row, own, role, confidence, reason_parts,
            paper, readme, chrono_s, signal, manifest,
            protocol_action="CHANGE_TO_EXCLUDED",
            label_action="EXCLUDE_FROM_LABEL_EXPORT",
            review_reason=f"Role {role} outside Vidal protocol",
            load_status=load_status,
        )

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
        load_status=load_status,
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
        return (
            "CHANGE_TO_ANCHOR",
            "ANCHOR_ONLY_NO_POINTWISE_LABEL",
            "Leading/post-Cap anchor for SOC_init evidence; not in ordered drive->charge sequence",
        )

    if role in ("normal_corresponding_charge", "trailing_corresponding_charge", "true_test_train_boundary_charge"):
        proto = "KEEP_MODEL_INPUT"
        reason = f"Corresponding CC owned by {own['owned_by_previous_profile']} (Section IV.B / Figure 8)"
        if ah_reset:
            return (
                proto,
                "REGENERATE_RESET_AND_SEQUENCE_AWARE",
                reason + "; Ah reset and per-file normalization require sequence reconstruction",
            )
        return (
            proto,
            "REGENERATE_SEQUENCE_AWARE",
            reason + "; v1 per-file SOC is not continuous Vidal SOC — sequence reconstruction required",
        )

    return ("MANUAL_REVIEW", "MANUAL_REVIEW", f"Unresolved role {role}")


def _charge_result(
    row, splits, stats, chrono_row, own, role, confidence, reason_parts,
    paper, readme, chrono_s, signal, manifest,
    protocol_action="", label_action="", review_reason="",
    load_status: str = "PASS",
):
    ah_reset = bool(stats.get("ah_reset_detected"))
    recon = _reconstruction_fields(protocol_action, ah_reset)
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
        **recon,
        **{k: stats.get(k, "") for k in (
            "ah_start", "ah_end", "ah_min", "ah_max", "charge_ah_max", "charge_ah_fraction",
            "voltage_start", "voltage_end", "voltage_max", "full_voltage_reached",
            "current_start", "current_end", "current_max", "current_taper_detected",
            "ah_reset_detected",
        )},
        "load_status": load_status,
        "load_error_type": chrono_row.get("load_error_type", ""),
        "load_error_message": chrono_row.get("load_error_message", ""),
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
        "ownership_search_status", "ownership_barrier_type", "ownership_barrier_filename",
        "next_drive_search_status", "next_drive_barrier_type", "next_drive_barrier_filename",
        "split_relation", "current_record_role", "current_split_memberships",
        "charge_ah_max", "charge_ah_fraction", "ah_reset_detected",
        "full_voltage_reached", "current_taper_detected",
        "previous_drive_end_soc_required", "capacity_reference_required",
        "ah_unwrap_required", "full_charge_anchor_required", "sequence_continuity_required",
        "expected_soc_start_source", "expected_soc_end_target",
        "load_status", "load_error_type", "load_error_message",
        "proposed_charge_role", "recommended_protocol_action", "recommended_label_action",
        "reason", "confidence",
        "paper_support", "readme_support", "chronology_support", "signal_support", "manifest_support",
        "is_leading_charge", "is_trailing_charge", "review_reason",
    ]
    existing = [c for c in review_cols if c in mi.columns]
    return mi[existing]


def _row_in_split(row: dict | pd.Series, split: str) -> bool:
    return str(row.get(split, "0")).strip() in ("1", "1.0", "True", "true")


def _charge_splits_for_list(cr: dict) -> set[str]:
    return set(str(cr.get("current_split_memberships", "")).split(",")) - {""}


def _anchor_applies_to_split(cr: dict, split: str) -> bool:
    """Anchor applies to splits of its following selected drive."""
    next_splits = set(str(cr.get("next_actual_drive_protocol_splits", "")).split(",")) - {""}
    if not next_splits:
        return False
    if split in next_splits:
        return True
    if split == "partial_train" and "full_train" in next_splits:
        # partial anchor before Mix1/Mix3 gateway
        nxt_prof = str(cr.get("next_actual_drive_profile", ""))
        if re.match(r"(Cycle_[123]|Mixed?[123])", nxt_prof, re.I):
            return True
    return False


def _owner_drive_in_split(chrono: pd.DataFrame, cr: dict, split: str) -> bool:
    sid = cr.get("owned_by_previous_sequence_id", "")
    if not sid:
        return False
    owners = chrono[chrono["sequence_id"] == sid]
    if len(owners) == 0:
        return False
    return _row_in_split(owners.iloc[0], split)


def build_candidate_sequences(chrono: pd.DataFrame, class_df: pd.DataFrame) -> dict:
    out: dict[str, Any] = {}
    cls_by_fn = {r["filename"]: r for _, r in class_df.iterrows()}
    chrono_by_ds: dict[str, pd.DataFrame] = {
        ds: chrono[chrono["dataset"] == ds].reset_index(drop=True)
        for ds in chrono["dataset"].unique()
    }

    for dataset, ds in chrono_by_ds.items():
        out[dataset] = {}
        all_cls = class_df[class_df["dataset"] == dataset]

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
                for _, c in ds[ds["file_type"] == "charge"].iterrows():
                    cr = cls_by_fn.get(c["filename"], {})
                    if cr.get("owned_by_previous_sequence_id") != d["sequence_id"]:
                        continue
                    if cr.get("recommended_protocol_action") != "KEEP_MODEL_INPUT":
                        continue
                    if split == "test" and cr.get("current_record_role") != "model_input":
                        continue
                    if split == "partial_train":
                        if not _owner_drive_in_split(ds, cr, "partial_train"):
                            continue
                    elif split == "full_train":
                        if split not in _charge_splits_for_list(cr) and not _owner_drive_in_split(ds, cr, split):
                            continue
                    order += 1
                    seq.append({
                        "sequence_order": order,
                        "filename": c["filename"],
                        "profile": c["profile"],
                        "paper_profile": cr.get("paper_profile", "CC"),
                        "file_type": "charge",
                        "sequence_role": "intercycle_charge_input",
                        "split_memberships": cr.get("current_split_memberships", ""),
                        "evidence": cr.get("reason", ""),
                    })

            split_cls = []
            for _, cr in all_cls.iterrows():
                crd = cr.to_dict()
                fn = crd["filename"]
                memberships = _charge_splits_for_list(crd)
                in_split = split in memberships or _owner_drive_in_split(ds, crd, split)
                if split == "partial_train" and _owner_drive_in_split(ds, crd, "partial_train"):
                    in_split = True
                if split == "test" and crd.get("current_record_role") == "model_input" and "test" in memberships:
                    in_split = True
                if not in_split and crd.get("recommended_protocol_action") != "CHANGE_TO_ANCHOR":
                    if not _anchor_applies_to_split(crd, split):
                        continue
                split_cls.append(crd)

            leading = [
                {
                    "filename": r["filename"],
                    "next_selected_drive": r.get("next_actual_drive", ""),
                    "next_selected_profile": r.get("next_actual_drive_profile", ""),
                    "recommended_protocol_action": r.get("recommended_protocol_action", ""),
                }
                for r in split_cls
                if r.get("recommended_protocol_action") == "CHANGE_TO_ANCHOR"
                and _anchor_applies_to_split(r, split)
            ]

            hwfet = [
                {
                    "filename": r["filename"],
                    "owned_by": r.get("owned_by_previous_profile", ""),
                    "current_split_memberships": r.get("current_split_memberships", ""),
                    "recommended_protocol_action": r.get("recommended_protocol_action", ""),
                }
                for r in split_cls
                if _is_hwfet_profile(str(r.get("owned_by_previous_profile", "")))
            ]

            initial_topups = [
                r["filename"] for r in split_cls
                if r.get("proposed_charge_role") == "initial_topup"
            ]
            capacity_test_topups = [
                r["filename"] for r in split_cls
                if r.get("proposed_charge_role") == "pre_capacity_test_topup"
            ]
            other_excluded = [
                r["filename"] for r in split_cls
                if r.get("recommended_protocol_action") == "CHANGE_TO_EXCLUDED"
                and r["filename"] not in {h["filename"] for h in hwfet}
            ]
            trailing = [
                r["filename"] for r in split_cls
                if r.get("proposed_charge_role") == "trailing_corresponding_charge"
                and _owner_drive_in_split(ds, r, split)
            ]
            partial_boundary = [
                r["filename"] for r in split_cls
                if r.get("split_relation") == "partial_subset_boundary"
            ]
            test_train_boundary = [
                r["filename"] for r in split_cls
                if r.get("split_relation") == "true_test_train_boundary"
            ]

            out[dataset][split] = {
                "ordered_model_sequence": seq,
                "leading_anchors": leading,
                "hwfet_owned_charges": hwfet,
                "initial_topups": sorted(set(initial_topups)),
                "capacity_test_topups": sorted(set(capacity_test_topups)),
                "other_excluded_charges": sorted(set(other_excluded)),
                "trailing_corresponding_charges": sorted(set(trailing)),
                "partial_subset_boundary_charges": sorted(set(partial_boundary)),
                "true_test_train_boundary_charges": sorted(set(test_train_boundary)),
            }
    return out


def write_report(
    cfg: AuditConfig,
    paths: dict,
    chrono: pd.DataFrame,
    class_df: pd.DataFrame,
    review_df: pd.DataFrame,
    sequences: dict,
    load_failures: list[dict],
    prev_review: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# Vidal Charge Protocol Audit Report",
        "",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')} UTC",
        "",
        "## 1. Evidence paths used",
        "",
        f"- Vidal PDF: `{paths['vidal_pdf']}` (status: {paths['paper_reference_status']})",
    ]
    for ds, p in paths["readmes"].items():
        lines.append(f"- {ds} README: `{p}`")
    for ds, p in paths["manifests"].items():
        lines.append(f"- {ds} manifest: `{p}`")
    for ds, p in paths["raw_roots"].items():
        lines.append(f"- {ds} raw root: `{p}`")

    lines += [
        "",
        "## 2. Path portability",
        "",
        "All paths above are repo-relative unless marked `<external Vidal PDF>`.",
        "Pass `--vidal-pdf` to supply the paper PDF explicitly.",
        "",
        "## 3. Current v1 label positioning",
        "",
        "Current v1 charge SOC is **open_reference_per_record_v1** (per-file normalized audit label):",
        "`SOC_file = Ah / max(Ah)` within each charge file.",
        "It must **not** be treated as formal continuous Vidal SOC (Figure 8).",
        "All KEEP model-input charges require **sequence-aware reconstruction**.",
        "",
        "## 4. Sequence-aware label definition",
        "",
        "For corresponding charges, future SOC reconstruction must ensure:",
        "",
        "1. Start SOC continuous with ownership drive end SOC",
        "2. End SOC ≈ 1.0 at confirmed full charge",
        "3. Do not force SOC start to 0 because Ah counter starts at 0 in-file",
        "4. Do not allow SOC 1→0 jump when Ah counter resets at file end",
        "5. Use effective cumulative Ah within charge, not per-file normalization",
        "",
        "Reconstruction metadata is recorded in the review CSV (`expected_soc_start_source`, etc.).",
        "",
        "## 5. Charge ownership rule (unchanged)",
        "",
        "Default: **selected drive → corresponding/accompanying charge** (Figure 8).",
        "Pause does not break ownership. Barrier-aware lookup stops at charge/characterization.",
        "",
        "## 6. Partial is a subset of Full",
        "",
        "`partial_train` ⊂ `full_train`. Mix3 → Charge → Mix4 is **partial_subset_boundary**.",
        "",
        "## 7. Protocol action vs label action",
        "",
        "- **Protocol action**: model input / anchor / excluded",
        "- **Label action**: sequence reconstruction requirement for v1 labels",
        "- Ah reset adds `ah_unwrap_required` but does not change protocol role",
        "",
        "## 8. Protocol action counts (37 model_input charges)",
        "",
    ]
    for action in ("KEEP_MODEL_INPUT", "CHANGE_TO_ANCHOR", "CHANGE_TO_EXCLUDED", "MANUAL_REVIEW"):
        n = len(review_df[review_df["recommended_protocol_action"] == action])
        lines.append(f"- {action}: {n}")

    lines += ["", "## 9. Label action counts", ""]
    for action in (
        "REGENERATE_SEQUENCE_AWARE",
        "REGENERATE_RESET_AND_SEQUENCE_AWARE",
        "ANCHOR_ONLY_NO_POINTWISE_LABEL",
        "EXCLUDE_FROM_LABEL_EXPORT",
        "MANUAL_REVIEW",
    ):
        n = len(review_df[review_df["recommended_label_action"] == action])
        lines.append(f"- {action}: {n}")
    if "LABEL_OK" in review_df["recommended_label_action"].values:
        lines.append("- **WARNING**: LABEL_OK still present (should be 0)")

    lines += ["", "## 10. Per-dataset breakdown", ""]
    for ds in review_df["dataset"].unique():
        sub = review_df[review_df["dataset"] == ds]
        lines.append(f"### {ds} ({len(sub)} model_input charges)")
        for action in ("KEEP_MODEL_INPUT", "CHANGE_TO_ANCHOR", "CHANGE_TO_EXCLUDED", "MANUAL_REVIEW"):
            lines.append(f"- Protocol {action}: {len(sub[sub['recommended_protocol_action'] == action])}")
        for action in (
            "REGENERATE_SEQUENCE_AWARE", "REGENERATE_RESET_AND_SEQUENCE_AWARE",
            "ANCHOR_ONLY_NO_POINTWISE_LABEL", "EXCLUDE_FROM_LABEL_EXPORT",
        ):
            lines.append(f"- Label {action}: {len(sub[sub['recommended_label_action'] == action])}")
        lines.append("")

    def _list_df(title, df, extra=""):
        lines.append(f"## {title}")
        lines.append("")
        if len(df) == 0:
            lines.append("_None_")
        else:
            for _, r in df.iterrows():
                lines.append(
                    f"- `{r['filename']}` ({r['dataset']}): {r.get('proposed_charge_role','')} / "
                    f"{r.get('recommended_protocol_action','')} / {r.get('recommended_label_action','')}{extra}"
                )
        lines.append("")

    reset_seq = review_df[review_df["recommended_label_action"] == "REGENERATE_RESET_AND_SEQUENCE_AWARE"]
    _list_df("11. Reset + sequence-aware charges", reset_seq)
    anchors = review_df[review_df["recommended_protocol_action"] == "CHANGE_TO_ANCHOR"]
    _list_df("12. Anchor-only charges", anchors)
    hw = review_df[review_df["owned_by_previous_profile"].apply(_is_hwfet_profile)]
    _list_df("13. HWFET-owned excluded charges", hw)

    barrier_issues = review_df[
        (review_df["ownership_search_status"] != "FOUND_PREVIOUS_DRIVE")
        & (review_df["recommended_protocol_action"] == "KEEP_MODEL_INPUT")
        & (~review_df["ownership_basis"].str.startswith("post_capacity_test", na=False))
        & (review_df["ownership_basis"] != "leading_context:session_start")
    ]
    lines += ["", "## 14. Ownership barrier check", ""]
    if len(barrier_issues) == 0:
        lines.append("No unexpected barrier stops for KEEP corresponding charges.")
    else:
        for _, r in barrier_issues.iterrows():
            lines.append(
                f"- `{r['filename']}`: {r['ownership_search_status']} "
                f"({r['ownership_barrier_type']} @ {r['ownership_barrier_filename']})"
            )

    lines += ["", "## 15. Load failures", ""]
    mi_fail = [f for f in load_failures if f.get("model_input")]
    if not mi_fail:
        lines.append("All model_input charges: load_status == PASS")
    else:
        for f in mi_fail:
            lines.append(
                f"- `{f['filename']}` ({f['dataset']}): {f['load_error_type']}: {f['load_error_message']}"
            )

    lines += ["", "## 16. Candidate sequence split summary", ""]
    for ds, splits in sequences.items():
        lines.append(f"### {ds}")
        for split, data in splits.items():
            n_seq = len(data.get("ordered_model_sequence", []))
            n_anchors = len(data.get("leading_anchors", []))
            lines.append(
                f"- **{split}**: ordered={n_seq} items, leading_anchors={n_anchors}, "
                f"hwfet={len(data.get('hwfet_owned_charges', []))}"
            )
        lines.append("")

    for ds in ("panasonic_18650pf", "lg_hg2", "samsung_30t"):
        lines += [f"## Recommended actions — {ds}", ""]
        lines.append("| filename | owned_by | protocol | label | split_relation | load |")
        lines.append("|----------|----------|----------|-------|----------------|------|")
        sub = review_df[review_df["dataset"] == ds]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['filename']} | {r['owned_by_previous_profile']} | "
                f"{r['recommended_protocol_action']} | {r['recommended_label_action']} | "
                f"{r['split_relation']} | {r.get('load_status','')} |"
            )
        lines.append("")

    (cfg.audit_dir / "VIDAL_CHARGE_AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


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
                    "    'panasonic_18650pf': (PanasonicAdapter, REPO / 'data/manifests/vidal_2019/panasonic_vidal2019_25c_manifest.csv'),\n",
                    "    'lg_hg2': (LGHG2Adapter, REPO / 'data/manifests/vidal_2019/lg_hg2_vidal2019_25c_manifest.csv'),\n",
                    "    'samsung_30t': (Samsung30TAdapter, REPO / 'data/manifests/vidal_2019/samsung_30t_vidal2019_25c_manifest.csv'),\n",
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
                    "    (review['recommended_label_action'] == 'REGENERATE_RESET_AND_SEQUENCE_AWARE')\n",
                    "    | (review['recommended_label_action'] == 'ANCHOR_ONLY_NO_POINTWISE_LABEL')\n",
                    "    | (review['recommended_label_action'] == 'EXCLUDE_FROM_LABEL_EXPORT')\n",
                    "    | (review['recommended_protocol_action'] != 'KEEP_MODEL_INPUT')\n",
                    "    | (review['load_status'] != 'PASS')\n",
                    "    | (review['ownership_search_status'] != 'FOUND_PREVIOUS_DRIVE')\n",
                    "]\n",
                    "print(f'Flagged records: {len(flagged)}')\n",
                    "cols = ['dataset','filename','owned_by_previous_profile','ownership_search_status',\n",
                    "        'recommended_protocol_action','recommended_label_action','expected_soc_start_source',\n",
                    "        'expected_soc_end_target','load_status','ah_unwrap_required']\n",
                    "flagged[cols].head(20)\n",
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
                    "    'ownership barrier': row.get('ownership_barrier_type', ''),\n",
                    "    'load status': row.get('load_status', ''),\n",
                    "    'SOC start source': row.get('expected_soc_start_source', ''),\n",
                    "    'SOC end target': row.get('expected_soc_end_target', ''),\n",
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


def run_audit(cfg: AuditConfig, overwrite: bool = False) -> int:
    audit_dir = cfg.audit_dir
    audit_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        audit_dir / "vidal_paper_protocol_reference.csv",
        audit_dir / "vidal_charge_chronology.csv",
        audit_dir / "vidal_charge_classification.csv",
        audit_dir / "vidal_model_input_charge_review.csv",
        audit_dir / "vidal_candidate_sequences.json",
    ]
    if not overwrite:
        existing = [p for p in outputs if p.exists()]
        if existing:
            raise FileExistsError(f"Outputs exist: {existing[0]}. Pass --overwrite")

    paper_df = build_paper_reference(cfg.paper_reference_status)
    paper_df.to_csv(outputs[0], index=False)

    all_chrono = []
    all_class = []
    load_failures: list[dict] = []

    for dataset, manifest_path in cfg.manifest_paths.items():
        rows = _load_manifest(manifest_path)
        adapter = ADAPTERS[dataset](raw_root=cfg.repo_root, manifest_path=manifest_path)
        load_results: dict[str, dict[str, Any]] = {}
        for row in rows:
            split = "full_train"
            if row.get("test") == "1":
                split = "test"
            elif row.get("exclude") == "1":
                split = "exclude"
            lr = _load_record(adapter, row, split=split)
            load_results[row["filename"]] = lr
            if lr["load_status"] != "PASS" and row.get("file_type") == "charge":
                load_failures.append({
                    "dataset": dataset,
                    "filename": row["filename"],
                    "model_input": row.get("record_role") == "model_input",
                    "load_error_type": lr["load_error_type"],
                    "load_error_message": lr["load_error_message"],
                })

        chrono = build_chronology(dataset, rows, load_results)
        all_chrono.append(chrono)

        charge_rows = [r for r in rows if r.get("file_type") == "charge"]
        chrono_idx = {r["filename"]: i for i, r in chrono.iterrows()}
        for row in charge_rows:
            fn = row["filename"]
            i = chrono_idx[fn]
            lr = load_results[fn]
            rec = lr["record"]
            stats = _signal_stats(rec) if rec else {}
            all_class.append(
                _classify_charge(
                    dataset, row, chrono, i, stats,
                    load_status=lr["load_status"],
                )
            )

    chrono_df = pd.concat(all_chrono, ignore_index=True)
    class_df = pd.DataFrame(all_class)
    review_df = build_review(class_df)
    sequences = build_candidate_sequences(chrono_df, class_df)

    review_path = outputs[3]
    prev_review = pd.read_csv(review_path) if review_path.exists() else None

    chrono_df.to_csv(outputs[1], index=False)
    class_df.to_csv(outputs[2], index=False)
    review_df.to_csv(outputs[3], index=False)
    with open(outputs[4], "w", encoding="utf-8") as f:
        json.dump(sequences, f, indent=2, sort_keys=True)

    paths = {
        "vidal_pdf": _rel_path(cfg.vidal_pdf, cfg.repo_root),
        "paper_reference_status": cfg.paper_reference_status,
        "readmes": {k: _rel_path(v, cfg.repo_root) for k, v in cfg.readme_paths.items()},
        "manifests": {k: _rel_path(v, cfg.repo_root) for k, v in cfg.manifest_paths.items()},
        "raw_roots": {k: _rel_path(v, cfg.repo_root) for k, v in cfg.raw_roots.items()},
    }
    write_report(cfg, paths, chrono_df, class_df, review_df, sequences, load_failures, prev_review)

    nb_path = cfg.repo_root / "notebooks/review_vidal_charge_pairing.ipynb"
    create_notebook(nb_path)

    # Self-checks
    assert len(chrono_df) == sum(len(_load_manifest(p)) for p in cfg.manifest_paths.values())
    assert len(class_df) == sum(
        1 for p in cfg.manifest_paths.values()
        for r in _load_manifest(p) if r["file_type"] == "charge"
    )
    assert len(review_df) == len(class_df[class_df["current_record_role"] == "model_input"])
    assert "LABEL_OK" not in review_df["recommended_label_action"].values

    mi_load_fail = [f for f in load_failures if f.get("model_input")]
    exit_code = 0
    if mi_load_fail:
        exit_code = 1
        print("ERROR: model_input charge load failures:", file=sys.stderr)
        for f in mi_load_fail:
            print(f"  {f['filename']}: {f['load_error_type']}: {f['load_error_message']}", file=sys.stderr)

    print(f"Wrote audit to {audit_dir}")
    print(f"Notebook: {nb_path}")
    print(f"Chronology rows: {len(chrono_df)}")
    print(f"Charge rows: {len(class_df)}")
    print(f"model_input review: {len(review_df)}")
    print(review_df["recommended_protocol_action"].value_counts().to_string())
    print("--- label actions ---")
    print(review_df["recommended_label_action"].value_counts().to_string())
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repo-root", default=None, help="Repository root (default: auto-detect)")
    parser.add_argument("--output-dir", default=None, help=f"Audit output dir (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--vidal-pdf", default=None, help="Path to Vidal 2019 PDF")
    parser.add_argument("--panasonic-root", default=None)
    parser.add_argument("--lg-root", default=None)
    parser.add_argument("--samsung-root", default=None)
    parser.add_argument("--panasonic-readme", default=None)
    parser.add_argument("--lg-readme", default=None)
    parser.add_argument("--samsung-readme", default=None)
    args = parser.parse_args()
    cfg = _build_audit_config(args)
    return run_audit(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
