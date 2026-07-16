"""Read-only loader for validated Vidal sequence-aware SOC v2 artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


CANONICAL_MAP = {
    "time_s": "timestamp",
    "voltage_v": "voltage",
    "current_a": "current",
    "temperature_c": "temperature",
    "soc_sequence_v2": "soc",
}
REQUIRED_COLUMNS = tuple(CANONICAL_MAP)
ALLOWED_STATUSES = frozenset({"PASS", "WARNING", "FAIL"})
ALLOWED_FILE_TYPES = frozenset({"drive_cycle", "charge"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class VidalV2Record:
    """One indexed record with canonical training columns and provenance."""

    data: pd.DataFrame
    metadata: dict[str, object]


class VidalSequenceV2Dataset:
    """Load one dataset/protocol split without touching raw MAT data."""

    def __init__(
        self,
        root: str | Path,
        dataset: str,
        protocol_split: str,
        allowed_statuses: Iterable[str] = ("PASS", "WARNING"),
        verify_hashes: bool = False,
    ) -> None:
        self.root = Path(root)
        self.dataset = dataset
        self.protocol_split = protocol_split
        self.allowed_statuses = frozenset(allowed_statuses)
        self.verify_hashes = verify_hashes
        if not self.allowed_statuses <= ALLOWED_STATUSES - {"FAIL"}:
            raise ValueError("FAIL labels are never loadable for Case 1")
        self.split_root = self.root / dataset / protocol_split
        self.index_path = self.split_root / "sequence_index.csv"
        self.records_dir = self.split_root / "records"
        self._index = self._load_index()
        self.records = self._load_records()
        self.inventory = self._build_inventory()

    @property
    def index(self) -> pd.DataFrame:
        return self._index.copy()

    def _load_index(self) -> pd.DataFrame:
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Missing sequence index: {self.index_path}")
        if not self.records_dir.is_dir():
            raise FileNotFoundError(f"Missing records directory: {self.records_dir}")
        index = pd.read_csv(self.index_path).sort_values("sequence_order", kind="stable")
        required = {
            "dataset", "protocol_split", "sequence_order", "sequence_segment_id",
            "source_filename", "file_type", "label_qc_status", "artifact_path",
        }
        missing = required - set(index.columns)
        if missing:
            raise ValueError(f"Index missing columns: {sorted(missing)}")
        if index.empty or index["sequence_order"].duplicated().any():
            raise ValueError("sequence_order must be non-empty and unique")
        if set(index["dataset"].astype(str)) != {self.dataset}:
            raise ValueError("Index dataset does not match requested directory")
        if set(index["protocol_split"].astype(str)) != {self.protocol_split}:
            raise ValueError("Index protocol_split does not match requested directory")
        if index["sequence_segment_id"].isna().any():
            raise ValueError("Every record requires sequence_segment_id")
        if index["source_filename"].isna().any() or (index["source_filename"].astype(str) == "").any():
            raise ValueError("Every record requires source_filename")
        bad_types = set(index["file_type"].astype(str)) - ALLOWED_FILE_TYPES
        if bad_types:
            raise ValueError(f"Unsupported file types: {sorted(bad_types)}")
        bad_status = set(index["label_qc_status"].astype(str)) - ALLOWED_STATUSES
        if bad_status:
            raise ValueError(f"Unsupported label statuses: {sorted(bad_status)}")
        disallowed = set(index["label_qc_status"].astype(str)) - self.allowed_statuses
        if disallowed:
            raise ValueError(f"Split contains disallowed label statuses: {sorted(disallowed)}")
        indexed_names = {Path(str(p)).name for p in index["artifact_path"]}
        actual_names = {p.name for p in self.records_dir.glob("*.parquet")}
        if indexed_names != actual_names:
            raise ValueError(
                f"Index/records mismatch: missing={sorted(indexed_names-actual_names)}, "
                f"unregistered={sorted(actual_names-indexed_names)}"
            )
        return index.reset_index(drop=True)

    def _load_records(self) -> list[VidalV2Record]:
        loaded: list[VidalV2Record] = []
        for row in self._index.to_dict("records"):
            path = self.records_dir / Path(str(row["artifact_path"])).name
            frame = pd.read_parquet(path)
            missing = set(REQUIRED_COLUMNS) - set(frame.columns)
            if missing:
                raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
            if frame.empty:
                raise ValueError(f"Empty record: {path.name}")
            numeric = frame[list(REQUIRED_COLUMNS)].to_numpy(dtype=float)
            if not np.isfinite(numeric).all():
                raise ValueError(f"Non-finite feature/target value in {path.name}")
            soc = frame["soc_sequence_v2"].to_numpy(dtype=float)
            if np.any((soc < 0.0) | (soc > 1.0)):
                raise ValueError(f"SOC outside [0, 1] in {path.name}")
            time = frame["time_s"].to_numpy(dtype=float)
            if np.any(np.diff(time) < 0):
                raise ValueError(f"Non-monotonic time in {path.name}")
            if int(row.get("row_count", len(frame))) != len(frame):
                raise ValueError(f"row_count mismatch for {path.name}")
            actual_hash = sha256_file(path) if self.verify_hashes else str(row.get("artifact_sha256", ""))
            expected_hash = str(row.get("artifact_sha256", ""))
            if self.verify_hashes and expected_hash and actual_hash != expected_hash:
                raise ValueError(f"SHA-256 mismatch for {path.name}")
            canonical = frame.copy()
            for source, target in CANONICAL_MAP.items():
                canonical[target] = canonical[source]
            metadata = dict(row)
            metadata.update({
                "paper_profile": row.get("paper_profile", row.get("profile", "")),
                "warning_reasons": row.get("label_warning_reasons", row.get("label_qc_reasons", "")),
                "record_sha256": actual_hash or expected_hash,
                "source_file_sha256": row.get("source_file_sha256", row.get("source_sha256", "")),
                "charge_owner": row.get("charge_owner", row.get("owner_filename", "")),
                "record_path": str(path),
            })
            loaded.append(VidalV2Record(canonical, metadata))
        return loaded

    def _build_inventory(self) -> pd.DataFrame:
        rows = []
        for record in self.records:
            df, meta = record.data, record.metadata
            rows.append({
                "dataset": self.dataset,
                "protocol_split": self.protocol_split,
                "sequence_order": meta["sequence_order"],
                "sequence_segment_id": meta["sequence_segment_id"],
                "source_filename": meta["source_filename"],
                "file_type": meta["file_type"],
                "paper_profile": meta["paper_profile"],
                "n_samples": len(df),
                "duration_s": float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]),
                "soc_start": float(df["soc"].iloc[0]),
                "soc_end": float(df["soc"].iloc[-1]),
                "voltage_min": float(df["voltage"].min()),
                "voltage_max": float(df["voltage"].max()),
                "current_min": float(df["current"].min()),
                "current_max": float(df["current"].max()),
                "temperature_min": float(df["temperature"].min()),
                "temperature_max": float(df["temperature"].max()),
                "label_qc_status": meta["label_qc_status"],
                "warning_reasons": meta["warning_reasons"],
                "record_sha256": meta["record_sha256"],
            })
        return pd.DataFrame(rows).sort_values("sequence_order").reset_index(drop=True)


def validate_case1_roots(
    root: str | Path,
    datasets: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a PASS/FAIL table for requested Vidal v2 inputs.

    The default remains the full three-dataset, three-split preflight used by
    the Colab notebook. Case 1 runs may pass the current dataset plus
    ``full_train``/``test`` so local caches do not need unrelated splits.
    """
    root = Path(root)
    rows = []
    datasets = tuple(datasets or ("panasonic_18650pf", "lg_hg2", "samsung_30t"))
    splits = tuple(splits or ("full_train", "partial_train", "test"))
    for dataset in datasets:
        for split in splits:
            split_root = root / dataset / split
            checks = {
                "index": (split_root / "sequence_index.csv").is_file(),
                "metadata": (split_root / "sequence_metadata.json").is_file(),
                "concatenated": (split_root / "concatenated_sequence.parquet").is_file(),
                "records": (split_root / "records").is_dir(),
            }
            rows.append({"dataset": dataset, "split": split, **checks,
                         "status": "PASS" if all(checks.values()) else "FAIL"})
    return pd.DataFrame(rows)
