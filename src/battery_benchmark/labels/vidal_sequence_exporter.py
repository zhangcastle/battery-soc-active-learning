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

    for res in results:
        meta = res.meta
        if meta.label_qc_status == "FAIL":
            if not include_warnings:
                continue
        elif meta.label_qc_status == "WARNING" and not include_warnings:
            continue

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
            "soc_end": meta.soc_end,
            "soc_init_source": meta.soc_init_source,
            "capacity_reference_ah": meta.capacity_reference_ah,
            "reset_unwrap_applied": meta.reset_unwrap_applied,
            "endpoint_correction_applied": meta.endpoint_correction_applied,
            "label_qc_status": meta.label_qc_status,
            "artifact_path": meta.artifact_path,
            "artifact_sha256": sha,
        })

        part = res.dataframe.copy()
        part["global_sequence_row"] = range(global_row, global_row + len(part))
        part["source_sequence_order"] = meta.sequence_order
        part["source_row_index"] = range(len(part))
        part["boundary_before"] = False
        part["boundary_type"] = ""
        if global_row == 0:
            part.loc[part.index[0], "boundary_before"] = True
            part.loc[part.index[0], "boundary_type"] = "sequence_start"
        concat_parts.append(part)
        global_row += len(part)

    index_df = pd.DataFrame(index_rows)
    index_path = split_dir / "sequence_index.csv"
    index_df.to_csv(index_path, index=False)

    if concat_parts:
        concat_df = pd.concat(concat_parts, ignore_index=True)
    else:
        concat_df = pd.DataFrame()
    concat_path = split_dir / "concatenated_sequence.parquet"
    concat_df.to_parquet(concat_path, index=False)

    metadata = {
        "dataset": dataset,
        "protocol_split": protocol_split,
        "source_manifest": str(source_manifest.relative_to(repo_root)),
        "audit_commit_or_timestamp": datetime.now(timezone.utc).isoformat(),
        "capacity_policy": asdict(capacity_policy) if hasattr(capacity_policy, "__dataclass_fields__") else capacity_policy,
        "current_convention": asdict(current_convention) if hasattr(current_convention, "__dataclass_fields__") else current_convention,
        "integration_method": INTEGRATION_METHOD,
        "endpoint_correction_policy": "progress_to_1_when_full_charge_qc_pass",
        "missing_corresponding_charge": missing_corresponding_charge,
        "warnings": warnings or [],
        "label_policy_version": LABEL_POLICY_VERSION,
        "code_commit": _git_commit(repo_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_hashes": {
            "sequence_index.csv": _sha256_file(index_path) if index_path.exists() else "",
            "concatenated_sequence.parquet": _sha256_file(concat_path) if concat_path.exists() else "",
        },
    }
    meta_path = split_dir / "sequence_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "record_count": len(index_rows),
        "index_path": str(index_path),
        "metadata_path": str(meta_path),
    }


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
) -> None:
    qc_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(qc_root / "summary.csv", index=False)
    pd.DataFrame(failures).to_csv(qc_root / "failures.csv", index=False)
    pd.DataFrame(warnings).to_csv(qc_root / "warnings.csv", index=False)
    pd.DataFrame(boundary_rows).to_csv(qc_root / "boundary_continuity.csv", index=False)
    pd.DataFrame(reset_rows).to_csv(qc_root / "reset_unwrap.csv", index=False)
    pd.DataFrame(anchor_rows).to_csv(qc_root / "anchor_qc.csv", index=False)
    pd.DataFrame(current_sign_rows).to_csv(qc_root / "current_sign_audit.csv", index=False)
    pd.DataFrame(manifest_log_rows).to_csv(qc_root / "manifest_change_log.csv", index=False)
    pd.DataFrame(anchor_rows).to_csv(qc_root / "soc_anchors.csv", index=False)
