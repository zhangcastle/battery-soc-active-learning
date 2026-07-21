"""Sequence-aware Vidal SOC v2 reconstruction."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from battery_benchmark.data.sequence_record import SequenceRecord
from battery_benchmark.reproductions.vidal_2019.labels.vidal_ah_unwrap import unwrap_cycler_ah
from battery_benchmark.reproductions.vidal_2019.labels.vidal_capacity_policy import VidalCapacityPolicy
from battery_benchmark.reproductions.vidal_2019.labels.vidal_current_convention import (
    CurrentConvention,
    audit_record_convention,
    resolve_dataset_convention,
)
from battery_benchmark.reproductions.vidal_2019.labels.vidal_sequence_soc_result import (
    ENDPOINT_CORRECTION_THRESHOLD,
    INTEGRATION_METHOD,
    LABEL_POLICY_VERSION,
    AnchorMetadata,
    SequenceLabelRecordMeta,
    SequenceSOCResult,
)
from battery_benchmark.reproductions.vidal_2019.labels.vidal_time_integration import (
    apply_soc_clipping,
    trapezoidal_coulomb_integration,
)

CONTINUITY_TOLERANCE = 1e-6
FULL_VOLTAGE_V = 4.15
MIN_EFFECTIVE_CHARGE_AH = 1e-6
SOC_CLIP_WARNING_FRACTION = 0.001
SOC_CLIP_FAIL_FRACTION = 0.05
SOC_CLIP_FAIL_EXCESS = 0.02

VIDAL_CHARGE_TERMINATION_POLICY = {
    "panasonic_18650pf": {"full_voltage_v": 4.15, "taper_current_a": 0.05},
    "lg_hg2": {"full_voltage_v": 4.15, "taper_current_a": 0.05},
    "samsung_30t": {"full_voltage_v": 4.15, "taper_current_a": 0.01},
}

PANASONIC_CYCLE1 = "03-18-17_02.17 25degC_Cycle_1_Pan18650PF.mat"
PANASONIC_CYCLE2 = "03-19-17_03.25 25degC_Cycle_2_Pan18650PF.mat"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name)


def _charge_signals(record: SequenceRecord) -> tuple[bool, bool]:
    policy = VIDAL_CHARGE_TERMINATION_POLICY.get(
        record.metadata.get("dataset", ""),
        {"full_voltage_v": FULL_VOLTAGE_V, "taper_current_a": 0.05},
    )
    v = record.data["voltage_v"].to_numpy(dtype=float)
    i_arr = record.data["current_a"].to_numpy(dtype=float)
    full_v = bool(np.nanmax(v) >= policy["full_voltage_v"]) if len(v) else False
    tail = i_arr[-min(20, len(i_arr)):] if len(i_arr) else i_arr
    taper = bool(np.nanmin(np.abs(tail)) <= policy["taper_current_a"]) if len(tail) else False
    qc = record.qc
    if hasattr(qc, "details"):
        details = getattr(qc, "details", {}) or {}
        if isinstance(details, dict):
            full_v = full_v or bool(details.get("full_voltage_reached", False))
            taper = taper or bool(details.get("current_taper_detected", False))
    return full_v, taper


def _raw_source_path(record: SequenceRecord) -> str:
    rel = record.metadata.get("relative_path") or record.metadata.get("source_path")
    return str(rel or record.metadata.get("source_file", ""))


def _source_sha(record: SequenceRecord) -> str:
    return str(record.metadata.get("source_sha256", ""))


def _apply_clip_qc(
    *,
    status: str,
    reasons: list[str],
    clip_count: int,
    clip_excess: float,
    row_count: int,
) -> str:
    if not clip_count:
        return status
    clip_fraction = clip_count / max(row_count, 1)
    reasons.append(f"soc_clip_count={clip_count}")
    reasons.append(f"soc_clip_fraction={clip_fraction:.6f}")
    if clip_excess > SOC_CLIP_FAIL_EXCESS or (
        clip_fraction > SOC_CLIP_FAIL_FRACTION and clip_excess > 1e-3
    ):
        reasons.append(f"soc_clip_excess={clip_excess:.6f}")
        return "FAIL"
    return "WARNING" if status == "PASS" else status


def _adapter_reset_flag(record: SequenceRecord) -> bool:
    for note in getattr(record.qc, "notes", []) or []:
        if "ah_reset" in str(note).lower():
            return True
    for warn in getattr(record.qc, "warnings", []) or []:
        if "ah_reset" in str(warn).lower() or "reset" in str(warn).lower():
            return True
    return False


def compute_drive_soc(
    record: SequenceRecord,
    soc_init: float,
    capacity_ah: float,
    convention: CurrentConvention,
) -> tuple[pd.DataFrame, SequenceLabelRecordMeta]:
    """Compute drive SOC from coulomb integration."""
    df = record.data.copy()
    if not np.isfinite(soc_init):
        df_out = df.copy()
        df_out["cycler_ah_raw"] = df.get("cycler_ah", np.nan)
        df_out["cycler_ah_effective"] = df.get("cycler_ah", np.nan)
        df_out["soc_cc_raw"] = np.nan
        df_out["soc_sequence_v2"] = np.nan
        meta = SequenceLabelRecordMeta(
            dataset=record.metadata["dataset"],
            protocol_split=record.metadata["split"],
            sequence_id=record.metadata["sequence_id"],
            sequence_order=0,
            source_filename=record.metadata["source_file"],
            file_type=record.metadata["file_type"],
            profile=record.metadata.get("profile", ""),
            record_role=record.metadata["record_role"],
            soc_init=float("nan"),
            soc_end=float("nan"),
            soc_init_source="unresolved",
            capacity_reference_ah=capacity_ah,
            label_qc_status="FAIL",
            label_qc_reasons=["missing_soc_init"],
            row_count=len(df_out),
            source_sha256=_source_sha(record),
            source_raw_path=_raw_source_path(record),
        )
        return df_out, meta

    integration = trapezoidal_coulomb_integration(
        df["time_s"].to_numpy(dtype=float),
        df["current_a"].to_numpy(dtype=float),
        positive_is_charge=convention.positive_is_charge,
    )
    soc_raw = soc_init + integration.cumulative_signed_ah / capacity_ah
    soc_v2, clip_count, clip_excess = apply_soc_clipping(soc_raw)

    reasons: list[str] = []
    status = "PASS"
    if integration.issues:
        reasons.extend(integration.issues)
    if integration.negative_dt_count:
        status = "FAIL"
    if len(soc_raw) >= 2 and float(soc_raw[-1]) > float(soc_raw[0]) + 1e-4:
        status = "WARNING"
        reasons.append("drive_soc_not_overall_falling")
    status = _apply_clip_qc(
        status=status,
        reasons=reasons,
        clip_count=clip_count,
        clip_excess=clip_excess,
        row_count=len(soc_raw),
    )

    out = pd.DataFrame({
        "time_s": df["time_s"].values,
        "timestamp": df["timestamp"].values if "timestamp" in df.columns else df["time_s"].values,
        "voltage_v": df["voltage_v"].values,
        "current_a": df["current_a"].values,
        "temperature_c": df["temperature_c"].values,
        "cycler_ah_raw": df["cycler_ah"].values,
        "cycler_ah_effective": df["cycler_ah"].values,
        "soc_cc_raw": soc_raw,
        "soc_sequence_v2": soc_v2,
    })

    meta = SequenceLabelRecordMeta(
        dataset=record.metadata["dataset"],
        protocol_split=record.metadata["split"],
        sequence_id=record.metadata["sequence_id"],
        sequence_order=0,
        source_filename=record.metadata["source_file"],
        file_type=record.metadata["file_type"],
        profile=record.metadata.get("profile", ""),
        record_role=record.metadata["record_role"],
        soc_init=soc_init,
        soc_end=float(soc_v2[-1]) if len(soc_v2) else soc_init,
        soc_init_source="",
        capacity_reference_ah=capacity_ah,
        label_qc_status=status,
        label_qc_reasons=reasons,
        soc_clip_count=clip_count,
        soc_clip_max_excess=clip_excess,
        row_count=len(out),
        source_sha256=_source_sha(record),
        source_raw_path=_raw_source_path(record),
    )
    return out, meta


def compute_charge_soc(
    record: SequenceRecord,
    soc_start: float,
    capacity_ah: float,
    convention: CurrentConvention,
    *,
    allow_endpoint_correction: bool,
) -> tuple[pd.DataFrame, SequenceLabelRecordMeta, dict[str, Any]]:
    """Compute sequence-aware charge SOC with optional endpoint correction."""
    df = record.data.copy()
    raw_ah = df["cycler_ah"].to_numpy(dtype=float)
    adapter_reset = _adapter_reset_flag(record)
    unwrap = unwrap_cycler_ah(raw_ah, adapter_reset_detected=adapter_reset)

    integration = trapezoidal_coulomb_integration(
        df["time_s"].to_numpy(dtype=float),
        df["current_a"].to_numpy(dtype=float),
        positive_is_charge=convention.positive_is_charge,
    )

    soc_cc_raw = soc_start + integration.cumulative_signed_ah / capacity_ah
    effective_ah = unwrap.cycler_ah_effective
    delta_effective_ah = effective_ah - effective_ah[0] if len(effective_ah) else effective_ah
    total_effective_charge_ah = float(delta_effective_ah[-1]) if len(delta_effective_ah) else 0.0
    if total_effective_charge_ah > MIN_EFFECTIVE_CHARGE_AH and np.isfinite(total_effective_charge_ah):
        progress = delta_effective_ah / total_effective_charge_ah
        progress = np.clip(progress, 0.0, 1.0)
        soc_seq = soc_start + (1.0 - soc_start) * progress
    else:
        progress = np.full(len(df), np.nan)
        soc_seq = np.full(len(df), np.nan)
    endpoint_applied = False
    raw_end = float(soc_cc_raw[-1]) if len(soc_cc_raw) else soc_start
    corrected_end = float(soc_seq[-1]) if len(soc_seq) else soc_start
    correction_mag = abs(raw_end - 1.0)

    full_v, taper = _charge_signals(record)
    reasons: list[str] = []
    status = "PASS"

    valid_effective_ah = bool(total_effective_charge_ah > MIN_EFFECTIVE_CHARGE_AH and np.isfinite(total_effective_charge_ah))
    if allow_endpoint_correction and full_v and taper and valid_effective_ah:
        endpoint_applied = True
        if correction_mag > ENDPOINT_CORRECTION_THRESHOLD:
            status = "WARNING"
            reasons.append(
                f"endpoint_deviation={correction_mag:.4f}>{ENDPOINT_CORRECTION_THRESHOLD}"
            )
    else:
        soc_seq = soc_cc_raw.copy()
        endpoint_applied = False
        if allow_endpoint_correction:
            status = "FAIL" if not valid_effective_ah else "WARNING"
            if not full_v:
                reasons.append("endpoint_correction_skipped_no_full_voltage")
            if not taper:
                reasons.append("endpoint_correction_skipped_no_taper")
            if not valid_effective_ah:
                reasons.append("invalid_effective_charge_ah")

    soc_seq, clip_count, clip_excess = apply_soc_clipping(soc_seq)
    if integration.issues:
        reasons.extend(integration.issues)
    if integration.negative_dt_count:
        status = "FAIL"
    if len(soc_seq) >= 2 and float(soc_seq[-1]) < float(soc_seq[0]) - 1e-4:
        status = "FAIL"
        reasons.append("charge_soc_not_overall_rising")
    if len(soc_seq) and np.isnan(soc_seq).any():
        status = "FAIL"
        reasons.append("soc_sequence_contains_nan")
    status = _apply_clip_qc(
        status=status,
        reasons=reasons,
        clip_count=clip_count,
        clip_excess=clip_excess,
        row_count=len(soc_seq),
    )

    if unwrap.reset_indices:
        reasons.append(f"reset_unwrap_at={unwrap.reset_indices}")

    out = pd.DataFrame({
        "time_s": df["time_s"].values,
        "timestamp": df["timestamp"].values if "timestamp" in df.columns else df["time_s"].values,
        "voltage_v": df["voltage_v"].values,
        "current_a": df["current_a"].values,
        "temperature_c": df["temperature_c"].values,
        "cycler_ah_raw": unwrap.cycler_ah_raw,
        "cycler_ah_effective": unwrap.cycler_ah_effective,
        "soc_cc_raw": soc_cc_raw,
        "soc_sequence_v2": soc_seq,
    })

    meta = SequenceLabelRecordMeta(
        dataset=record.metadata["dataset"],
        protocol_split=record.metadata["split"],
        sequence_id=record.metadata["sequence_id"],
        sequence_order=0,
        source_filename=record.metadata["source_file"],
        file_type=record.metadata["file_type"],
        profile=record.metadata.get("profile", ""),
        record_role=record.metadata["record_role"],
        soc_init=soc_start,
        soc_end=float(soc_seq[-1]) if len(soc_seq) else soc_start,
        soc_init_source="previous_drive_end",
        capacity_reference_ah=capacity_ah,
        reset_unwrap_applied=bool(unwrap.reset_indices),
        endpoint_correction_applied=endpoint_applied,
        raw_soc_end_before_correction=raw_end,
        corrected_soc_end=corrected_end,
        endpoint_correction_magnitude=correction_mag,
        label_qc_status=status,
        label_qc_reasons=reasons,
        soc_clip_count=clip_count,
        soc_clip_max_excess=clip_excess,
        row_count=len(out),
        source_sha256=_source_sha(record),
        source_raw_path=_raw_source_path(record),
    )
    unwrap_info = {
        "filename": record.metadata["source_file"],
        "reset_indices": unwrap.reset_indices,
        "raw_ah_before": [float(raw_ah[i - 1]) for i in unwrap.reset_indices if i > 0],
        "raw_ah_after": [float(raw_ah[i]) for i in unwrap.reset_indices if i < len(raw_ah)],
        "effective_ah_before": [
            float(unwrap.cycler_ah_effective[i - 1]) for i in unwrap.reset_indices if i > 0
        ],
        "effective_ah_after": [
            float(unwrap.cycler_ah_effective[i]) for i in unwrap.reset_indices if i < len(raw_ah)
        ],
        "unwrap_offset_added": unwrap.unwrap_offsets,
        "status": unwrap.status,
    }
    return out, meta, unwrap_info


def build_anchor_metadata(
    dataset: str,
    anchor_info: dict[str, Any],
    record: SequenceRecord | None,
) -> AnchorMetadata:
    full_v, taper = (False, False)
    reasons: list[str] = []
    status = "PASS"
    if record is not None:
        full_v, taper = _charge_signals(record)
        if not full_v:
            status = "FAIL"
            reasons.append("full_voltage_not_reached")
        if not taper:
            status = "FAIL"
            reasons.append("current_taper_not_detected")
    else:
        status = "FAIL"
        reasons.append("anchor_record_not_loaded")

    return AnchorMetadata(
        dataset=dataset,
        filename=anchor_info["filename"],
        next_selected_drive=anchor_info.get("next_selected_drive", ""),
        next_selected_profile=anchor_info.get("next_selected_profile", ""),
        anchor_soc=1.0,
        full_voltage_reached=full_v,
        current_taper_detected=taper,
        anchor_qc_status=status,
        anchor_qc_reasons=reasons,
    )


def load_candidate_sequences(audit_json: Path) -> dict[str, Any]:
    with open(audit_json, encoding="utf-8") as f:
        return json.load(f)


def validate_sequence_item(
    item: dict[str, Any],
    manifest_by_file: dict[str, dict[str, Any]],
    protocol_split: str,
) -> list[str]:
    """Validate ordered sequence item against manifest."""
    issues: list[str] = []
    fn = item["filename"]
    if fn not in manifest_by_file:
        issues.append(f"missing_manifest_entry:{fn}")
        return issues
    m = manifest_by_file[fn]
    if m["record_role"] not in ("model_input",):
        issues.append(f"unexpected_role:{m['record_role']}")
    split_key = protocol_split
    if split_key == "full_train" and not m.get("full_train"):
        issues.append("split_membership_mismatch")
    if split_key == "partial_train" and not m.get("partial_train"):
        issues.append("split_membership_mismatch")
    if split_key == "test" and not m.get("test"):
        issues.append("split_membership_mismatch")
    if item["file_type"] != m["file_type"]:
        issues.append("file_type_mismatch")
    return issues


def _is_panasonic_cycle2(dataset: str, filename: str) -> bool:
    return dataset == "panasonic_18650pf" and filename == PANASONIC_CYCLE2


def _is_panasonic_cycle1(dataset: str, filename: str) -> bool:
    return dataset == "panasonic_18650pf" and filename == PANASONIC_CYCLE1


def _mark_segment(
    meta: SequenceLabelRecordMeta,
    *,
    segment_id: int,
    segment_order: int,
    segment_start: bool,
    reason: str,
) -> None:
    meta.sequence_segment_id = segment_id
    meta.segment_order = segment_order
    meta.sequence_segment_start = segment_start
    meta.segment_start_reason = reason


def reconstruct_split_sequence(
    dataset: str,
    protocol_split: str,
    ordered_items: list[dict[str, Any]],
    leading_anchors: list[dict[str, Any]],
    record_loader: Any,
    convention: CurrentConvention,
    capacity_policy: VidalCapacityPolicy,
    *,
    boundary_charge_ends: dict[str, float] | None = None,
    missing_corresponding_charge: bool = False,
) -> tuple[list[SequenceSOCResult], list[AnchorMetadata], list[dict], list[dict]]:
    """Walk ordered sequence and produce labelled records."""
    results: list[SequenceSOCResult] = []
    anchors_out: list[AnchorMetadata] = []
    unwrap_rows: list[dict] = []
    boundary_rows: list[dict] = []

    anchor_by_next_drive = {
        a["next_selected_drive"]: a for a in leading_anchors
    }
    anchor_records: dict[str, AnchorMetadata] = {}

    for anc in leading_anchors:
        rec = record_loader.load_by_filename(anc["filename"])
        meta = build_anchor_metadata(dataset, anc, rec)
        anchors_out.append(meta)
        anchor_records[anc["next_selected_drive"]] = meta

    prior_soc_end: float | None = None
    prior_meta: SequenceLabelRecordMeta | None = None
    segment_id = 1
    segment_order = 0

    for item in ordered_items:
        order = int(item["sequence_order"])
        fn = item["filename"]
        rec = record_loader.load_by_filename(fn)
        if rec is None:
            continue
        rec.metadata["split"] = protocol_split

        cap = capacity_policy.capacity_reference_ah

        if item["file_type"] == "drive_cycle":
            soc_init_source = "unresolved"
            soc_init: float | None = None
            segment_start = False
            segment_start_reason = "previous_corresponding_charge"

            if _is_panasonic_cycle2(dataset, fn):
                if prior_meta is not None:
                    prior_meta.missing_following_charge = True
                    prior_meta.missing_charge_reason = (
                        "No public post-Cycle1 corresponding charge found; "
                        "Cycle2 begins a new protocol full-start segment."
                    )
                if results:
                    results[-1].meta = prior_meta if prior_meta is not None else results[-1].meta
                if prior_meta is not None:
                    segment_id += 1
                segment_order = 0
                soc_init = 1.0
                soc_init_source = "protocol_full_start_missing_charge"
                segment_start = True
                segment_start_reason = "protocol_full_start_missing_charge"
            elif fn in anchor_records:
                anc = anchor_records[fn]
                if anc.anchor_qc_status == "PASS":
                    soc_init = 1.0
                    soc_init_source = "full_charge_anchor"
                    if prior_meta is not None:
                        segment_id += 1
                        segment_order = 0
                    segment_start = True
                    segment_start_reason = "full_charge_anchor"
                else:
                    soc_init_source = "unresolved"
            elif prior_soc_end is not None:
                soc_init = prior_soc_end
                soc_init_source = "previous_charge_end"
            elif boundary_charge_ends and fn in boundary_charge_ends:
                soc_init = boundary_charge_ends[fn]
                soc_init_source = "previous_charge_end"
            elif order == 1 and missing_corresponding_charge:
                anc_fn = leading_anchors[0]["filename"] if leading_anchors else ""
                if leading_anchors and leading_anchors[0].get("next_selected_drive") == fn:
                    anc = anchor_records.get(fn)
                    if anc and anc.anchor_qc_status == "PASS":
                        soc_init = 1.0
                        soc_init_source = "full_charge_anchor"

            if soc_init is None:
                df_out = rec.data.copy()
                df_out["soc_cc_raw"] = np.nan
                df_out["soc_sequence_v2"] = np.nan
                meta = SequenceLabelRecordMeta(
                    dataset=dataset,
                    protocol_split=protocol_split,
                    sequence_id=rec.metadata["sequence_id"],
                    sequence_order=order,
                    source_filename=fn,
                    file_type="drive_cycle",
                    profile=item.get("profile", ""),
                    record_role=rec.metadata["record_role"],
                    soc_init=float("nan"),
                    soc_end=float("nan"),
                    soc_init_source="unresolved",
                    capacity_reference_ah=cap,
                    label_qc_status="FAIL",
                    label_qc_reasons=["missing_soc_init"],
                    row_count=len(df_out),
                    source_sha256=_source_sha(rec),
                    source_raw_path=_raw_source_path(rec),
                )
                segment_order += 1
                _mark_segment(
                    meta,
                    segment_id=segment_id,
                    segment_order=segment_order,
                    segment_start=segment_start or segment_order == 1,
                    reason=segment_start_reason if segment_start else "unresolved",
                )
                results.append(SequenceSOCResult(dataframe=df_out, meta=meta))
                prior_soc_end = None
                prior_meta = meta
                continue

            df_out, meta = compute_drive_soc(rec, soc_init, cap, convention)
            meta.sequence_order = order
            meta.soc_init_source = soc_init_source
            segment_order += 1
            _mark_segment(
                meta,
                segment_id=segment_id,
                segment_order=segment_order,
                segment_start=segment_start or segment_order == 1,
                reason=segment_start_reason if (segment_start or segment_order == 1) else "previous_corresponding_charge",
            )
            if _is_panasonic_cycle2(dataset, fn):
                meta.missing_preceding_charge = True
                meta.missing_charge_reason = (
                    "No public pre-Cycle2 charge file; drive starts from full-charge "
                    "protocol state based on manifest chronology and cycler Ah."
                )
                meta.confidence = "medium"
                if meta.label_qc_status == "PASS":
                    meta.label_qc_status = "WARNING"
                meta.label_qc_reasons.append("protocol_full_start_missing_charge")
            if _is_panasonic_cycle1(dataset, fn):
                meta.missing_following_charge = True
                meta.missing_charge_reason = (
                    "No post-Cycle1 corresponding charge found under preceding-drive "
                    "ownership rule; next selected train drive starts a new segment."
                )

            if prior_meta is not None and not meta.sequence_segment_start:
                gap = abs(soc_init - prior_meta.soc_end)
                boundary_rows.append({
                    "dataset": dataset,
                    "protocol_split": protocol_split,
                    "sequence_segment_id": segment_id,
                    "previous_filename": prior_meta.source_filename,
                    "next_filename": fn,
                    "previous_file_type": prior_meta.file_type,
                    "next_file_type": "drive_cycle",
                    "previous_soc_end": prior_meta.soc_end,
                    "next_soc_start": soc_init,
                    "continuity_error": gap,
                    "boundary_type": "charge_to_drive",
                    "status": "PASS" if gap <= CONTINUITY_TOLERANCE else "FAIL",
                })
                if gap > CONTINUITY_TOLERANCE:
                    meta.label_qc_status = "FAIL"
                    meta.label_qc_reasons.append(f"boundary_gap={gap:.2e}")

            results.append(SequenceSOCResult(dataframe=df_out, meta=meta))
            prior_soc_end = meta.soc_end
            prior_meta = meta

        elif item["file_type"] == "charge":
            if prior_soc_end is None:
                meta = SequenceLabelRecordMeta(
                    dataset=dataset,
                    protocol_split=protocol_split,
                    sequence_id=rec.metadata["sequence_id"],
                    sequence_order=order,
                    source_filename=fn,
                    file_type="charge",
                    profile=item.get("profile", ""),
                    record_role=rec.metadata["record_role"],
                    soc_init=float("nan"),
                    soc_end=float("nan"),
                    soc_init_source="unresolved",
                    capacity_reference_ah=cap,
                    label_qc_status="FAIL",
                    label_qc_reasons=["missing_previous_drive_soc_end"],
                    row_count=len(rec.data),
                    source_sha256=_source_sha(rec),
                    source_raw_path=_raw_source_path(rec),
                )
                segment_order += 1
                _mark_segment(
                    meta,
                    segment_id=segment_id,
                    segment_order=segment_order,
                    segment_start=segment_order == 1,
                    reason="unresolved",
                )
                results.append(SequenceSOCResult(dataframe=rec.data, meta=meta))
                prior_meta = meta
                continue

            df_out, meta, unwrap_info = compute_charge_soc(
                rec,
                prior_soc_end,
                cap,
                convention,
                allow_endpoint_correction=True,
            )
            meta.sequence_order = order
            meta.soc_init_source = "previous_drive_end"
            meta.owner_filename = prior_meta.source_filename if prior_meta is not None else ""
            meta.owner_sequence_order = prior_meta.sequence_order if prior_meta is not None else None
            segment_order += 1
            _mark_segment(
                meta,
                segment_id=segment_id,
                segment_order=segment_order,
                segment_start=segment_order == 1,
                reason="previous_corresponding_charge",
            )
            if prior_meta is not None:
                gap = abs(prior_soc_end - prior_meta.soc_end)
                boundary_rows.append({
                    "dataset": dataset,
                    "protocol_split": protocol_split,
                    "sequence_segment_id": segment_id,
                    "previous_filename": prior_meta.source_filename,
                    "next_filename": fn,
                    "previous_file_type": prior_meta.file_type,
                    "next_file_type": "charge",
                    "previous_soc_end": prior_meta.soc_end,
                    "next_soc_start": prior_soc_end,
                    "continuity_error": gap,
                    "boundary_type": "drive_to_charge",
                    "status": "PASS" if gap <= CONTINUITY_TOLERANCE else "FAIL",
                })
            if unwrap_info.get("reset_indices"):
                unwrap_rows.append(unwrap_info)

            results.append(SequenceSOCResult(dataframe=df_out, meta=meta))
            prior_soc_end = meta.soc_end
            prior_meta = meta

    return results, anchors_out, unwrap_rows, boundary_rows


def audit_current_signs(
    dataset: str,
    records: list[SequenceRecord],
    convention: CurrentConvention | None = None,
) -> tuple[list[Any], CurrentConvention]:
    rows = []
    for rec in records:
        if rec.metadata["record_role"] != "model_input":
            continue
        rows.append(
            audit_record_convention(
                dataset,
                rec.metadata["source_file"],
                rec.metadata["file_type"],
                rec.metadata.get("profile", ""),
                rec.data["current_a"].to_numpy(dtype=float),
                dataset_convention=convention,
            )
        )
    conv = resolve_dataset_convention(dataset, rows) if rows else CurrentConvention(
        positive_is_charge=True,
        convention_name="positive_is_charge_default",
        dataset=dataset,
    )
    return rows, conv
