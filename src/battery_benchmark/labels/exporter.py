"""Export labelled Vidal-compatible datasets to versioned Parquet + JSON artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from battery_benchmark.data.sequence_record import SequenceRecord
from battery_benchmark.labels.label_qc import SOCLabelQCResult, run_label_qc
from battery_benchmark.labels.soc_calculator import (
    CALCULATOR_NAME,
    LABEL_SOURCE,
    SOC_FORMULA_VERSION,
)

DEFAULT_DATASET_VERSION = "vidal_compatible_v1"

_SPLIT_PRIORITY = {"full_train": 0, "partial_train": 1, "test": 2}

REQUIRED_EXPORT_COLUMNS = (
    "time_s", "timestamp", "voltage_v", "current_a", "temperature_c", "cycler_ah",
    "soc_raw", "soc", "cycler_ah_delta", "current_integrated_ah", "ah_reference_difference",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def label_qc_for_record(record: SequenceRecord) -> SOCLabelQCResult:
    """Recompute label QC from a labelled ``SequenceRecord``."""
    return run_label_qc(
        record.data,
        file_type=record.metadata["file_type"],
        adapter_qc=record.qc,
        denominator_ah=float(record.metadata["soc_denominator_ah"]),
        formula_mode=record.metadata["soc_formula_mode"],
    )


def _label_qc_to_dict(qc: SOCLabelQCResult) -> dict[str, Any]:
    return asdict(qc)


def _dedupe_records_by_sequence_id(
    records_by_dataset_and_split: dict[str, dict[str, list[SequenceRecord]]],
) -> dict[str, dict[str, list[SequenceRecord]]]:
    """Keep one export copy per ``sequence_id`` using split priority."""
    deduped: dict[str, dict[str, list[SequenceRecord]]] = {}

    for dataset_name, splits in records_by_dataset_and_split.items():
        best: dict[str, tuple[int, str, SequenceRecord]] = {}

        for split, records in splits.items():
            if split == "exclude":
                continue
            prio = _SPLIT_PRIORITY.get(split, 99)
            for record in records:
                seq_id = record.metadata["sequence_id"]
                current = best.get(seq_id)
                if current is None or prio < current[0]:
                    best[seq_id] = (prio, split, record)

        by_split: dict[str, list[SequenceRecord]] = {}
        for _prio, split, record in best.values():
            meta = dict(record.metadata)
            meta["split"] = split
            by_split.setdefault(split, []).append(
                SequenceRecord(data=record.data, metadata=meta, qc=record.qc)
            )

        deduped[dataset_name] = {
            split: by_split.get(split, [])
            for split in splits
            if split != "exclude"
        }

    return deduped


def export_labelled_dataset(
    records_by_dataset_and_split: dict[str, dict[str, list[SequenceRecord]]],
    output_root: str | Path,
    *,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    source_manifests: dict[str, Path] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write labelled records to ``output_root / dataset_version /``.

    Only ``model_input`` records from selected splits should be passed.
    Excluded records are not exported.
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Parquet export requires pyarrow. Install with: pip install pyarrow>=14.0"
        ) from exc

    for splits in records_by_dataset_and_split.values():
        if "exclude" in splits:
            raise ValueError("exclude split must not be exported.")

    root = Path(output_root) / dataset_version
    if root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Export directory already exists: {root}. Pass overwrite=True to replace."
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    records_by_dataset_and_split = _dedupe_records_by_sequence_id(
        records_by_dataset_and_split
    )

    manifest_records: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []
    qc_counts: dict[str, dict[str, int]] = {}

    for dataset_name, splits in sorted(records_by_dataset_and_split.items()):
        ds_dir = root / dataset_name
        split_stats: dict[str, Any] = {"dataset_name": dataset_name, "splits": {}}
        qc_counts[dataset_name] = {"PASS": 0, "WARN": 0, "FAIL": 0}
        total_rows = 0

        for split, records in sorted(splits.items()):
            if split == "exclude":
                raise ValueError("exclude split must not be exported.")
            split_dir = ds_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            split_rows = 0

            for record in records:
                if record.metadata.get("record_role") != "model_input":
                    raise ValueError(
                        f"Only model_input records may be exported; "
                        f"got {record.metadata.get('record_role')!r} for "
                        f"{record.metadata['sequence_id']!r}."
                    )

                seq_id = record.metadata["sequence_id"]
                parquet_path = split_dir / f"{seq_id}.parquet"
                meta_path = split_dir / f"{seq_id}.metadata.json"
                qc_path = split_dir / f"{seq_id}.label_qc.json"

                record.data.to_parquet(parquet_path, index=False)
                meta_path.write_text(
                    json.dumps(record.metadata, indent=2, sort_keys=True, default=str),
                    encoding="utf-8",
                )
                lqc = label_qc_for_record(record)
                qc_path.write_text(
                    json.dumps(_label_qc_to_dict(lqc), indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                qc_counts[dataset_name][lqc.status] += 1
                split_rows += record.n_rows

                manifest_records.append({
                    "sequence_id": seq_id,
                    "dataset": dataset_name,
                    "split": split,
                    "profile": record.metadata.get("profile", ""),
                    "paper_profile": record.metadata.get("paper_profile", ""),
                    "file_type": record.metadata.get("file_type", ""),
                    "source_file": record.metadata.get("source_file", ""),
                    "source_sha256": record.metadata.get("source_sha256", ""),
                    "parquet_path": str(parquet_path.relative_to(root)),
                    "metadata_path": str(meta_path.relative_to(root)),
                    "label_qc_path": str(qc_path.relative_to(root)),
                    "n_rows": record.n_rows,
                    "label_qc_status": lqc.status,
                })

            split_stats["splits"][split] = {
                "record_count": len(records),
                "row_count": split_rows,
            }
            total_rows += split_rows

        split_stats["total_records"] = sum(
            s["record_count"] for s in split_stats["splits"].values()
        )
        split_stats["total_rows"] = total_rows
        split_stats["label_qc_counts"] = dict(qc_counts[dataset_name])
        dataset_summaries.append(split_stats)

    manifest: dict[str, Any] = {
        "dataset_version": dataset_version,
        "calculator_name": CALCULATOR_NAME,
        "calculator_version": SOC_FORMULA_VERSION,
        "label_source": LABEL_SOURCE,
        "formulas": {
            "drive_cycle": "(cycler_ah + abs(min(cycler_ah))) / abs(min(cycler_ah))",
            "charge": "cycler_ah / abs(max(cycler_ah))",
        },
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_manifests": [],
        "datasets": dataset_summaries,
        "records": sorted(manifest_records, key=lambda r: r["sequence_id"]),
        "excluded_records_exported": False,
    }

    if source_manifests:
        for name, path in sorted(source_manifests.items()):
            manifest["source_manifests"].append({
                "name": name,
                "path": str(path),
                "sha256": _sha256_file(path),
            })

    manifest_path = root / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return root


def validate_exported_labelled_dataset(output_root: str | Path) -> None:
    """Validate an exported labelled dataset directory."""
    root = Path(output_root)
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset_manifest.json in {root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    seen_ids: set[str] = set()

    for entry in records:
        seq_id = entry["sequence_id"]
        if seq_id in seen_ids:
            raise ValueError(f"Duplicate sequence_id in manifest: {seq_id!r}")
        seen_ids.add(seq_id)

        if entry.get("split") == "exclude":
            raise ValueError(f"Excluded record found in export manifest: {seq_id!r}")

        for key in ("parquet_path", "metadata_path", "label_qc_path"):
            rel = entry[key]
            if not (root / rel).exists():
                raise FileNotFoundError(f"Missing {key} for {seq_id!r}: {rel}")

        df = pd.read_parquet(root / entry["parquet_path"])
        if len(df) != entry["n_rows"]:
            raise ValueError(
                f"Row count mismatch for {seq_id!r}: manifest={entry['n_rows']} "
                f"parquet={len(df)}"
            )

        for col in REQUIRED_EXPORT_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing column {col!r} in {seq_id!r}")

        meta = json.loads((root / entry["metadata_path"]).read_text(encoding="utf-8"))
        if meta["sequence_id"] != seq_id:
            raise ValueError(f"sequence_id mismatch in metadata for {seq_id!r}")
        if meta["source_sha256"] != entry["source_sha256"]:
            raise ValueError(f"source_sha256 mismatch for {seq_id!r}")
        if meta["split"] != entry["split"]:
            raise ValueError(f"split mismatch for {seq_id!r}")
        if meta["file_type"] != entry["file_type"]:
            raise ValueError(f"file_type mismatch for {seq_id!r}")
        if meta.get("record_role") != "model_input":
            raise ValueError(f"record_role must be model_input for {seq_id!r}")

        lqc = json.loads((root / entry["label_qc_path"]).read_text(encoding="utf-8"))
        if lqc.get("status") != entry["label_qc_status"]:
            raise ValueError(f"label_qc_status mismatch for {seq_id!r}")

    declared = len(records)
    on_disk = sum(1 for _ in root.rglob("*.parquet"))
    if declared != on_disk:
        raise ValueError(
            f"Manifest record count ({declared}) != parquet files on disk ({on_disk})"
        )
