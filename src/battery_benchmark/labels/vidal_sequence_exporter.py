"""Export and QC reporting for Vidal sequence SOC v2."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from battery_benchmark.labels.vidal_sequence_soc_result import (
    INTEGRATION_METHOD,
    LABEL_POLICY_VERSION,
    SequenceSOCResult,
    AnchorMetadata,
)


def _git_commit(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_dirty(repo: Path) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


def _source_tree_hash(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "diff", "--binary"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
        )
        return hashlib.sha256(out).hexdigest()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def export_split_artifacts(
    results: list[SequenceSOCResult],
    *,
    dataset: str,
    protocol_split: str,
    output_dir: Path,
    include_warnings: bool = False,
    pass_only: bool = False,
    source_manifest: Path,
    capacity_policy: Any,
    current_convention: Any,
    missing_corresponding_charge: bool = False,
    warnings: list[str] | None = None,
    repo_root: Path,
) -> dict[str, Any]:
    """Write per-split v2 labelled artifacts."""
    split_dir = output_dir / dataset / protocol_split
    records_dir = split_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict] = []
    concat_parts: list[pd.DataFrame] = []
    global_row = 0

    kept_results: list[SequenceSOCResult] = []
    kept_orders: set[int] = set()
    for res in results:
        meta = res.meta
        if meta.label_qc_status == "FAIL":
            continue
        if pass_only and meta.label_qc_status == "WARNING":
            continue
        kept_results.append(res)
        kept_orders.add(meta.sequence_order)

    # In pass-only mode, do not leave a charge whose owner drive was filtered.
    if pass_only:
        dep_filtered: list[SequenceSOCResult] = []
        for res in kept_results:
            meta = res.meta
            if meta.file_type == "charge" and meta.owner_sequence_order is not None:
                if meta.owner_sequence_order not in kept_orders:
                    continue
            dep_filtered.append(res)
        kept_results = dep_filtered

    prior_meta = None
    for res in kept_results:
        meta = res.meta

        safe = meta.source_filename.replace("/", "_").replace(" ", "_")
        artifact_name = f"{meta.sequence_order:02d}__{safe}.parquet"
        artifact_path = records_dir / artifact_name
        res.dataframe.to_parquet(artifact_path, index=False)
        sha = _sha256_file(artifact_path)
        meta.artifact_path = str(artifact_path.relative_to(repo_root))
        meta.artifact_sha256 = sha

        index_rows.append({
            "dataset": meta.dataset,
            "protocol_split": meta.protocol_split,
            "sequence_order": meta.sequence_order,
            "source_filename": meta.source_filename,
            "profile": meta.profile,
            "file_type": meta.file_type,
            "record_role": meta.record_role,
            "row_count": meta.row_count,
            "soc_start": meta.soc_init,
            "soc_init": meta.soc_init,
            "soc_end": meta.soc_end,
            "soc_init_source": meta.soc_init_source,
            "capacity_reference_ah": meta.capacity_reference_ah,
            "reset_unwrap_applied": meta.reset_unwrap_applied,
            "endpoint_correction_applied": meta.endpoint_correction_applied,
            "endpoint_correction_magnitude": meta.endpoint_correction_magnitude,
            "soc_clip_count": meta.soc_clip_count,
            "soc_clip_max_excess": meta.soc_clip_max_excess,
            "label_qc_status": meta.label_qc_status,
            "artifact_path": meta.artifact_path,
            "artifact_sha256": sha,
            "source_raw_path": meta.source_raw_path,
            "source_sha256": meta.source_sha256,
            "source_manifest": str(source_manifest.relative_to(repo_root)),
            "label_policy_version": LABEL_POLICY_VERSION,
            "code_commit": _git_commit(repo_root),
            "git_dirty": _git_dirty(repo_root),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sequence_segment_id": meta.sequence_segment_id,
            "segment_order": meta.segment_order,
            "sequence_segment_start": meta.sequence_segment_start,
            "segment_start_reason": meta.segment_start_reason,
            "missing_preceding_charge": meta.missing_preceding_charge,
            "missing_following_charge": meta.missing_following_charge,
            "missing_charge_reason": meta.missing_charge_reason,
            "confidence": meta.confidence,
            "owner_filename": meta.owner_filename,
            "owner_sequence_order": meta.owner_sequence_order,
            "label_qc_reasons": ";".join(meta.label_qc_reasons),
        })

        part = res.dataframe.copy()
        part["global_sequence_row"] = range(global_row, global_row + len(part))
        part["source_sequence_order"] = meta.sequence_order
        part["source_segment_id"] = meta.sequence_segment_id
        part["source_row_index"] = range(len(part))
        part["boundary_before"] = False
        part["boundary_type"] = ""
        if len(part):
            part.loc[part.index[0], "boundary_before"] = True
            if prior_meta is None:
                boundary_type = "sequence_start"
            elif meta.sequence_segment_id != prior_meta.sequence_segment_id:
                boundary_type = (
                    "missing_charge_segment_start"
                    if meta.segment_start_reason == "protocol_full_start_missing_charge"
                    else "segment_start"
                )
            else:
                boundary_type = f"{prior_meta.file_type}_to_{meta.file_type}"
            part.loc[part.index[0], "boundary_type"] = boundary_type
        concat_parts.append(part)
        global_row += len(part)
        prior_meta = meta

    index_df = pd.DataFrame(index_rows)
    index_path = split_dir / "sequence_index.csv"
    index_df.to_csv(index_path, index=False)

    if concat_parts:
        concat_df = pd.concat(concat_parts, ignore_index=True)
    else:
        concat_df = pd.DataFrame()
    concat_path = split_dir / "concatenated_sequence.parquet"
    concat_df.to_parquet(concat_path, index=False)

    exported_boundary_rows, orphan_rows, completeness_rows = audit_exported_results(kept_results)

    git_dirty = _git_dirty(repo_root)
    metadata = {
        "dataset": dataset,
        "protocol_split": protocol_split,
        "artifact_status": "validated_for_training" if not orphan_rows else "invalid_for_training",
        "source_manifest": str(source_manifest.relative_to(repo_root)),
        "manifest_sha256": _sha256_file(source_manifest),
        "audit_commit_or_timestamp": datetime.now(timezone.utc).isoformat(),
        "capacity_policy": asdict(capacity_policy) if hasattr(capacity_policy, "__dataclass_fields__") else capacity_policy,
        "current_convention": asdict(current_convention) if hasattr(current_convention, "__dataclass_fields__") else current_convention,
        "integration_method": INTEGRATION_METHOD,
        "endpoint_correction_policy": "progress_to_1_when_full_charge_qc_pass",
        "missing_corresponding_charge": missing_corresponding_charge,
        "warnings": warnings or [],
        "label_policy_version": LABEL_POLICY_VERSION,
        "code_commit": _git_commit(repo_root),
        "git_dirty": git_dirty,
        "source_tree_hash": _source_tree_hash(repo_root) if git_dirty else "",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_hashes": {
            "sequence_index.csv": _sha256_file(index_path) if index_path.exists() else "",
            "concatenated_sequence.parquet": _sha256_file(concat_path) if concat_path.exists() else "",
        },
        "export_policy": "pass_and_warning_excluding_fail" if not pass_only else "pass_only_dependency_aware",
    }
    meta_path = split_dir / "sequence_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "record_count": len(index_rows),
        "index_path": str(index_path),
        "metadata_path": str(meta_path),
        "exported_boundary_rows": exported_boundary_rows,
        "orphan_rows": orphan_rows,
        "completeness_rows": completeness_rows,
    }


def audit_exported_results(
    results: list[SequenceSOCResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate final exported order, ownership, and within-segment continuity."""
    boundary_rows: list[dict[str, Any]] = []
    orphan_rows: list[dict[str, Any]] = []
    completeness_rows: list[dict[str, Any]] = []
    by_order = {r.meta.sequence_order: r for r in results}

    for res in results:
        meta = res.meta
        expected = meta.label_qc_status != "FAIL"
        present = meta.sequence_order in by_order
        completeness_rows.append({
            "dataset": meta.dataset,
            "protocol_split": meta.protocol_split,
            "sequence_segment_id": meta.sequence_segment_id,
            "expected_sequence_order": meta.sequence_order,
            "source_filename": meta.source_filename,
            "file_type": meta.file_type,
            "owner_filename": meta.owner_filename,
            "expected_in_export": expected,
            "present_in_export": present,
            "status": "PASS" if expected == present else "FAIL",
            "reason": "",
        })
        if meta.file_type == "charge":
            owner_order = meta.owner_sequence_order
            if owner_order is None or owner_order not in by_order:
                orphan_rows.append({
                    "dataset": meta.dataset,
                    "protocol_split": meta.protocol_split,
                    "source_filename": meta.source_filename,
                    "sequence_order": meta.sequence_order,
                    "owner_filename": meta.owner_filename,
                    "owner_sequence_order": owner_order,
                    "status": "FAIL",
                    "reason": "charge_owner_not_exported",
                })

    sorted_results = sorted(results, key=lambda r: r.meta.sequence_order)
    for prev, nxt in zip(sorted_results, sorted_results[1:]):
        p = prev.meta
        n = nxt.meta
        same_segment = p.sequence_segment_id == n.sequence_segment_id
        continuity_error = abs(n.soc_init - p.soc_end) if same_segment else 0.0
        boundary_type = (
            f"{p.file_type}_to_{n.file_type}" if same_segment else "SEGMENT_BOUNDARY"
        )
        status = "PASS"
        reason = ""
        if same_segment and continuity_error > 1e-6:
            status = "FAIL"
            reason = "within_segment_soc_gap"
        if same_segment and p.file_type == "charge" and n.file_type == "charge":
            status = "FAIL"
            reason = "unexplained_charge_to_charge"
        boundary_rows.append({
            "dataset": p.dataset,
            "protocol_split": p.protocol_split,
            "sequence_segment_id": n.sequence_segment_id,
            "previous_filename": p.source_filename,
            "next_filename": n.source_filename,
            "previous_file_type": p.file_type,
            "next_file_type": n.file_type,
            "previous_soc_end": p.soc_end,
            "next_soc_start": n.soc_init,
            "continuity_error": continuity_error,
            "boundary_type": boundary_type,
            "status": status,
            "reason": reason,
        })
    return boundary_rows, orphan_rows, completeness_rows


def write_qc_reports(
    qc_root: Path,
    *,
    summary_rows: list[dict],
    failures: list[dict],
    warnings: list[dict],
    boundary_rows: list[dict],
    reset_rows: list[dict],
    anchor_rows: list[dict],
    current_sign_rows: list[dict],
    manifest_log_rows: list[dict],
    exported_boundary_rows: list[dict] | None = None,
    orphan_rows: list[dict] | None = None,
    completeness_rows: list[dict] | None = None,
) -> None:
    qc_root.mkdir(parents=True, exist_ok=True)
    failure_cols = [
        "dataset", "protocol_split", "source_filename", "sequence_order",
        "file_type", "label_qc_status", "soc_init_source", "soc_end",
        "reasons", "issue",
    ]
    orphan_cols = [
        "dataset", "protocol_split", "source_filename", "sequence_order",
        "owner_filename", "owner_sequence_order", "status", "reason",
    ]
    boundary_cols = [
        "dataset", "protocol_split", "sequence_segment_id", "previous_filename",
        "next_filename", "previous_file_type", "next_file_type",
        "previous_soc_end", "next_soc_start", "continuity_error",
        "boundary_type", "status", "reason",
    ]
    completeness_cols = [
        "dataset", "protocol_split", "sequence_segment_id",
        "expected_sequence_order", "source_filename", "file_type",
        "owner_filename", "expected_in_export", "present_in_export",
        "status", "reason",
    ]
    pd.DataFrame(summary_rows).to_csv(qc_root / "summary.csv", index=False)
    pd.DataFrame(failures, columns=failure_cols).to_csv(qc_root / "failures.csv", index=False)
    pd.DataFrame(warnings).to_csv(qc_root / "warnings.csv", index=False)
    pd.DataFrame(boundary_rows, columns=boundary_cols).to_csv(qc_root / "reconstruction_boundary_continuity.csv", index=False)
    pd.DataFrame(boundary_rows, columns=boundary_cols).to_csv(qc_root / "boundary_continuity.csv", index=False)
    pd.DataFrame(exported_boundary_rows or [], columns=boundary_cols).to_csv(qc_root / "exported_boundary_continuity.csv", index=False)
    pd.DataFrame(orphan_rows or [], columns=orphan_cols).to_csv(qc_root / "orphan_records.csv", index=False)
    pd.DataFrame(completeness_rows or [], columns=completeness_cols).to_csv(qc_root / "exported_sequence_completeness.csv", index=False)
    pd.DataFrame(reset_rows).to_csv(qc_root / "reset_unwrap.csv", index=False)
    pd.DataFrame(anchor_rows).to_csv(qc_root / "anchor_qc.csv", index=False)
    pd.DataFrame(current_sign_rows).to_csv(qc_root / "current_sign_audit.csv", index=False)
    action_df = pd.DataFrame(manifest_log_rows)
    action_df.to_csv(qc_root / "manifest_action_log.csv", index=False)
    action_df.to_csv(qc_root / "manifest_change_log.csv", index=False)
    if not action_df.empty:
        changes = action_df[action_df["old_record_role"] != action_df["new_record_role"]]
    else:
        changes = action_df
    changes.to_csv(qc_root / "manifest_changes_only.csv", index=False)
    pd.DataFrame(anchor_rows).to_csv(qc_root / "soc_anchors.csv", index=False)
