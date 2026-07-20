"""Provenance-aware raw ingestion for CMU eVTOL KiltHub Version 3.

This module is deliberately limited to asset inventory, strict raw CSV schema
validation, canonical row provenance, and time-continuity block detection.  It
does not infer missions, phases, RPTs, SOC, SOH, splits, or model sequences.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from battery_benchmark.data.adapter_qc import AdapterSchemaError, QCResult
from battery_benchmark.data.vidal_sequence_v2_dataset import sha256_file
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty


DATASET_NAME = "CMU eVTOL Battery Dataset"
ARTICLE_ID = "14226830"
DATASET_VERSION = "3"
LOADER_SCHEMA_VERSION = "cmu_evtol_kilthub_v3_raw_v0.1"

# Engineering default for raw QC only.  Version 3 reconnaissance found 10 s
# and 30 s nominal cadences; 120 s is four times the slower cadence.  Formal
# benchmark work must review and explicitly pass its chosen threshold.
DEFAULT_GAP_THRESHOLD_S = 120.0

CMU_V3_CELL_IDS = (
    "VAH01",
    "VAH02",
    "VAH05",
    "VAH06",
    "VAH07",
    "VAH09",
    "VAH10",
    "VAH11",
    "VAH12",
    "VAH13",
    "VAH15",
    "VAH16",
    "VAH17",
    "VAH20",
    "VAH22",
    "VAH23",
    "VAH24",
    "VAH25",
    "VAH26",
    "VAH27",
    "VAH28",
    "VAH30",
)

RAW_COLUMNS = (
    "time_s",
    "Ecell_V",
    "I_mA",
    "EnergyCharge_W_h",
    "QCharge_mA_h",
    "EnergyDischarge_W_h",
    "QDischarge_mA_h",
    "Temperature__C",
    "cycleNumber",
    "Ns",
)

CANONICAL_COLUMNS = (
    "cell_id",
    "source_file",
    "source_row",
    *RAW_COLUMNS,
    "continuous_block_id",
    "continuous_block_boundary_reason",
)

ASSET_INVENTORY_COLUMNS = (
    "dataset_name",
    "article_id",
    "dataset_version",
    "cell_id",
    "main_file",
    "main_file_present",
    "impedance_file",
    "impedance_file_present",
    "file_size_bytes",
    "sha256",
    "row_count",
    "observed_columns",
    "impedance_file_size_bytes",
    "impedance_sha256",
    "impedance_row_count",
    "impedance_observed_columns",
)

IMPEDANCE_COLUMNS = (
    "20%_1_second",
    "20%_30_second",
    "60%_1_second",
    "60%_30_second",
    "cycle numbers",
)

RAW_QC_SUMMARY_COLUMNS = (
    "dataset_name",
    "article_id",
    "dataset_version",
    "loader_schema_version",
    "generated_at_utc",
    "git_revision",
    "git_dirty",
    "cell_id",
    "main_file",
    "main_sha256",
    "impedance_file",
    "impedance_file_present",
    "impedance_sha256",
    "row_count",
    "gap_threshold_s",
    "gap_threshold_source",
    "block_count",
    "time_duplicate_count",
    "time_reversal_count",
    "current_time_non_finite_count",
    "previous_time_non_finite_count",
    "gap_exceeded_count",
    "min_positive_delta_s",
    "median_positive_delta_s",
    "p95_positive_delta_s",
    "p99_positive_delta_s",
    "max_positive_delta_s",
)

CONTINUOUS_BLOCK_RULES = (
    "file_start",
    "current_time_non_finite",
    "previous_time_non_finite",
    "time_duplicate",
    "time_reversal",
    "gap_exceeded",
    "source_concat_boundary",
)

UNIT_MAPPING = {
    "time_s": "s",
    "Ecell_V": "V",
    "I_mA": "mA; positive=charge; negative=discharge",
    "EnergyCharge_W_h": "Wh",
    "QCharge_mA_h": "mAh",
    "EnergyDischarge_W_h": "Wh",
    "QDischarge_mA_h": "mAh",
    "Temperature__C": "degC; cell surface temperature",
    "cycleNumber": "tester-native count; not a unique identifier",
    "Ns": "tester-native segment value; no phase semantics assigned in v0.1",
}


class CMURawIngestionError(ValueError):
    """Base error for CMU Version 3 raw ingestion."""


class CMUInvalidCellIDError(CMURawIngestionError):
    """Raised when a main filename does not identify one of the 22 v3 cells."""


class CMUEmptyFileError(CMURawIngestionError):
    """Raised when a main CSV is empty or contains only a header."""


class CMURawSchemaError(AdapterSchemaError, CMURawIngestionError):
    """Raised when a raw CSV header or numeric field violates the v3 schema."""


@dataclass(frozen=True)
class CMURawIngestionResult:
    """Canonical rows plus provenance, boundary counts, and shared QC metadata."""

    data: pd.DataFrame
    metadata: dict[str, object]
    qc: QCResult


def _read_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), None)
    except UnicodeDecodeError as exc:
        raise CMURawSchemaError(f"Cannot decode CSV header for {path}: {exc}") from None
    if not header:
        raise CMUEmptyFileError(f"CMU main CSV is empty: {path}")
    return header


def _duplicate_names(columns: Sequence[str]) -> list[str]:
    counts = Counter(columns)
    return sorted(name for name, count in counts.items() if count > 1)


def validate_raw_header(path: str | Path) -> list[str]:
    """Return the observed header after strict missing/duplicate/extra checks."""
    path = Path(path)
    columns = _read_header(path)
    duplicates = _duplicate_names(columns)
    if duplicates:
        raise CMURawSchemaError(
            f"Duplicate raw CSV column(s) in {path.name}: {duplicates}"
        )

    missing = sorted(set(RAW_COLUMNS) - set(columns))
    unexpected = sorted(set(columns) - set(RAW_COLUMNS))
    if missing or unexpected:
        raise CMURawSchemaError(
            f"Raw CSV schema mismatch for {path.name}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if tuple(columns) != RAW_COLUMNS:
        raise CMURawSchemaError(
            f"Raw CSV column order mismatch for {path.name}: observed={columns}"
        )
    return columns


def validate_impedance_header(path: str | Path) -> list[str]:
    """Strictly validate the Version 3 impedance summary header."""
    path = Path(path)
    columns = _read_header(path)
    duplicates = _duplicate_names(columns)
    if duplicates:
        raise CMURawSchemaError(
            f"Duplicate impedance CSV column(s) in {path.name}: {duplicates}"
        )
    missing = sorted(set(IMPEDANCE_COLUMNS) - set(columns))
    unexpected = sorted(set(columns) - set(IMPEDANCE_COLUMNS))
    if missing or unexpected:
        raise CMURawSchemaError(
            f"Impedance CSV schema mismatch for {path.name}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if tuple(columns) != IMPEDANCE_COLUMNS:
        raise CMURawSchemaError(
            f"Impedance CSV column order mismatch for {path.name}: observed={columns}"
        )
    return columns


def cell_id_from_main_file(path: str | Path) -> str:
    """Validate a ``VAH##.csv`` main filename and return its registered cell ID."""
    path = Path(path)
    if not path.name.endswith(".csv") or path.name.endswith("_impedance.csv"):
        raise CMUInvalidCellIDError(f"Not a CMU main time-series filename: {path.name}")
    cell_id = path.stem
    if cell_id not in CMU_V3_CELL_IDS:
        raise CMUInvalidCellIDError(
            f"Unknown CMU Version 3 cell ID {cell_id!r}; expected one of {CMU_V3_CELL_IDS}"
        )
    return cell_id


def _count_data_rows(path: Path) -> int:
    """Count Version 3 numeric CSV records, excluding the single header row."""
    with path.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    return max(line_count - 1, 0)


def build_asset_inventory(
    raw_root: str | Path,
    cell_ids: Iterable[str] = CMU_V3_CELL_IDS,
) -> pd.DataFrame:
    """Build a deterministic asset inventory for expected Version 3 cells.

    Missing main or impedance files are represented explicitly rather than
    silently omitted.  Main files are read only for SHA-256, row count, and
    observed header; no canonical data conversion is performed here.
    """
    raw_root = Path(raw_root)
    records: list[dict[str, object]] = []

    for cell_id in cell_ids:
        if cell_id not in CMU_V3_CELL_IDS:
            raise CMUInvalidCellIDError(f"Unknown CMU Version 3 cell ID: {cell_id}")
        main_name = f"{cell_id}.csv"
        impedance_name = f"{cell_id}_impedance.csv"
        main_path = raw_root / main_name
        impedance_path = raw_root / impedance_name
        main_present = main_path.is_file()

        observed_columns: list[str] = []
        file_size_bytes = 0
        digest = ""
        row_count = 0
        if main_present:
            file_size_bytes = main_path.stat().st_size
            digest = sha256_file(main_path)
            observed_columns = validate_raw_header(main_path)
            row_count = _count_data_rows(main_path)

        impedance_present = impedance_path.is_file()
        impedance_size: int | None = None
        impedance_digest: str | None = None
        impedance_row_count: int | None = None
        impedance_columns: list[str] | None = None
        if impedance_present:
            impedance_size = impedance_path.stat().st_size
            impedance_digest = sha256_file(impedance_path)
            impedance_columns = validate_impedance_header(impedance_path)
            impedance_row_count = _count_data_rows(impedance_path)

        records.append(
            {
                "dataset_name": DATASET_NAME,
                "article_id": ARTICLE_ID,
                "dataset_version": DATASET_VERSION,
                "cell_id": cell_id,
                "main_file": main_name,
                "main_file_present": main_present,
                "impedance_file": impedance_name,
                "impedance_file_present": impedance_present,
                "file_size_bytes": file_size_bytes,
                "sha256": digest,
                "row_count": row_count,
                "observed_columns": json.dumps(observed_columns, separators=(",", ":")),
                "impedance_file_size_bytes": impedance_size,
                "impedance_sha256": impedance_digest,
                "impedance_row_count": impedance_row_count,
                "impedance_observed_columns": (
                    json.dumps(impedance_columns, separators=(",", ":"))
                    if impedance_columns is not None
                    else None
                ),
            }
        )

    return pd.DataFrame(records, columns=ASSET_INVENTORY_COLUMNS)


def write_asset_inventory(inventory: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a validated inventory CSV and return its path."""
    missing = sorted(set(ASSET_INVENTORY_COLUMNS) - set(inventory.columns))
    if missing:
        raise ValueError(f"Asset inventory missing column(s): {missing}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inventory.loc[:, ASSET_INVENTORY_COLUMNS].to_csv(output_path, index=False)
    return output_path


def _software_version() -> str:
    try:
        return version("battery-benchmark")
    except PackageNotFoundError:
        return "unknown"


def positive_delta_statistics(time_values: Sequence[float]) -> dict[str, float]:
    """Summarise finite, strictly positive adjacent time differences only."""
    values = np.asarray(time_values, dtype=np.float64)
    deltas = np.diff(values)
    positive = deltas[np.isfinite(deltas) & (deltas > 0)]
    if not len(positive):
        return {
            "min_positive_delta_s": math.nan,
            "median_positive_delta_s": math.nan,
            "p95_positive_delta_s": math.nan,
            "p99_positive_delta_s": math.nan,
            "max_positive_delta_s": math.nan,
        }
    return {
        "min_positive_delta_s": float(np.min(positive)),
        "median_positive_delta_s": float(np.median(positive)),
        "p95_positive_delta_s": float(np.percentile(positive, 95)),
        "p99_positive_delta_s": float(np.percentile(positive, 99)),
        "max_positive_delta_s": float(np.max(positive)),
    }


def _normalise_source_boundaries(
    source_boundary_rows: Iterable[int] | None,
    row_count: int,
) -> set[int]:
    boundaries = set(source_boundary_rows or ())
    invalid = sorted(row for row in boundaries if row < 1 or row > row_count)
    if invalid:
        raise ValueError(
            f"source_boundary_rows must be 1-based data rows within [1, {row_count}]; "
            f"invalid={invalid}"
        )
    return boundaries


def _assign_continuous_blocks(
    time_values: np.ndarray,
    gap_threshold_s: float,
    source_boundary_rows: set[int],
) -> tuple[np.ndarray, list[str], Counter[str]]:
    block_ids = np.zeros(len(time_values), dtype=np.int64)
    reason_text: list[str] = []
    reason_counts: Counter[str] = Counter()
    current_block = 0

    for index, current_time in enumerate(time_values):
        source_row = index + 1
        reasons: list[str] = []
        if index == 0:
            reasons.append("file_start")
        if source_row in source_boundary_rows:
            reasons.append("source_concat_boundary")
        if not np.isfinite(current_time):
            reasons.append("current_time_non_finite")

        if index > 0:
            previous_time = time_values[index - 1]
            if not np.isfinite(previous_time):
                reasons.append("previous_time_non_finite")
            if np.isfinite(current_time) and np.isfinite(previous_time):
                delta = current_time - previous_time
                if delta == 0:
                    reasons.append("time_duplicate")
                elif delta < 0:
                    reasons.append("time_reversal")
                elif delta > gap_threshold_s:
                    reasons.append("gap_exceeded")

        # file_start identifies block zero; any later rule hit starts a new block.
        if index > 0 and reasons:
            current_block += 1
        block_ids[index] = current_block
        reason_text.append("|".join(reasons))
        reason_counts.update(reasons)

    return block_ids, reason_text, reason_counts


def load_cmu_v3_main_file(
    path: str | Path,
    *,
    gap_threshold_s: float | None = None,
    source_boundary_rows: Iterable[int] | None = None,
    repo_root: str | Path | None = None,
) -> CMURawIngestionResult:
    """Load one Version 3 main CSV without sorting, filtering, or semantic inference.

    ``source_row`` is the 1-based data-record ordinal after the header.  A
    boundary row is retained as the first row of its new block.  ``cycleNumber``
    and ``Ns`` are preserved but never consulted by block detection.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CMU main CSV not found: {path}")
    cell_id = cell_id_from_main_file(path)
    validate_raw_header(path)

    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        raise CMUEmptyFileError(f"CMU main CSV contains no data rows: {path}")

    for column in RAW_COLUMNS:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise CMURawSchemaError(
                f"Raw CSV column {column!r} contains non-numeric data in {path.name}: {exc}"
            ) from None

    threshold_source = "explicit_argument"
    if gap_threshold_s is None:
        gap_threshold_s = DEFAULT_GAP_THRESHOLD_S
        threshold_source = "exploratory_loader_default"
    if not math.isfinite(gap_threshold_s) or gap_threshold_s <= 0:
        raise ValueError("gap_threshold_s must be a finite positive number")

    source_boundaries = _normalise_source_boundaries(source_boundary_rows, len(frame))
    time_values = frame["time_s"].to_numpy(dtype=np.float64, copy=True)
    block_ids, boundary_reasons, reason_counts = _assign_continuous_blocks(
        time_values,
        float(gap_threshold_s),
        source_boundaries,
    )

    canonical = frame.loc[:, RAW_COLUMNS].copy()
    canonical.insert(0, "source_row", np.arange(1, len(frame) + 1, dtype=np.int64))
    canonical.insert(0, "source_file", path.name)
    canonical.insert(0, "cell_id", cell_id)
    canonical["continuous_block_id"] = block_ids
    canonical["continuous_block_boundary_reason"] = boundary_reasons
    canonical = canonical.loc[:, CANONICAL_COLUMNS]

    qc = QCResult(
        duplicate_timestamp_count=reason_counts["time_duplicate"],
        negative_time_step_count=reason_counts["time_reversal"],
        nan_count=int(frame.loc[:, RAW_COLUMNS].isna().sum().sum()),
        inf_count=int(
            np.isinf(frame.loc[:, RAW_COLUMNS].to_numpy(dtype=np.float64)).sum()
        ),
    )
    if reason_counts["current_time_non_finite"]:
        qc.add_warning(
            f"{reason_counts['current_time_non_finite']} non-finite time_s value(s) retained"
        )
    if reason_counts["time_duplicate"]:
        qc.add_warning(
            f"{reason_counts['time_duplicate']} duplicate time_s boundary/boundaries retained"
        )
    if reason_counts["time_reversal"]:
        qc.add_warning(
            f"{reason_counts['time_reversal']} reversed time_s boundary/boundaries retained"
        )
    if reason_counts["gap_exceeded"]:
        qc.add_note(
            f"{reason_counts['gap_exceeded']} gap boundary/boundaries exceed threshold"
        )

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    repo_root = Path(repo_root)
    impedance_path = path.with_name(f"{cell_id}_impedance.csv")
    impedance_present = impedance_path.is_file()
    if impedance_present:
        validate_impedance_header(impedance_path)
    metadata: dict[str, object] = {
        "dataset_name": DATASET_NAME,
        "article_id": ARTICLE_ID,
        "dataset_version": DATASET_VERSION,
        "cell_id": cell_id,
        "source_file": path.name,
        "source_sha256": sha256_file(path),
        "impedance_file": impedance_path.name,
        "impedance_file_present": impedance_present,
        "impedance_sha256": sha256_file(impedance_path) if impedance_present else None,
        "loader_schema_version": LOADER_SCHEMA_VERSION,
        "gap_threshold_s": float(gap_threshold_s),
        "gap_threshold_unit": "s",
        "gap_threshold_source": threshold_source,
        "continuous_block_rules": list(CONTINUOUS_BLOCK_RULES),
        "source_concat_boundary_detection": "caller_supplied_only",
        "source_concat_boundary_limitation": (
            "Concatenation boundaries with no visible time discontinuity cannot be "
            "recovered automatically from a merged CSV."
        ),
        "input_row_count": len(frame),
        "output_row_count": len(canonical),
        "block_count": int(block_ids.max()) + 1,
        "boundary_reason_counts": {
            rule: int(reason_counts[rule]) for rule in CONTINUOUS_BLOCK_RULES
        },
        "filtered_row_count": 0,
        "filter_reason_counts": {},
        "unit_mapping": dict(UNIT_MAPPING),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software_version": _software_version(),
        "git_revision": get_full_commit_hash(repo_root),
        "git_dirty": is_dirty(repo_root),
    }
    return CMURawIngestionResult(data=canonical, metadata=metadata, qc=qc)


def build_raw_qc_summary(
    raw_root: str | Path,
    *,
    gap_threshold_s: float | None = None,
    cell_ids: Iterable[str] = CMU_V3_CELL_IDS,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    """Run canonical ingestion sequentially and return one QC row per cell.

    Only one cell's canonical frame is retained at a time. Any cell failure is
    re-raised with its cell ID and source path; failures are never skipped.
    """
    raw_root = Path(raw_root)
    generated_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, object]] = []

    for cell_id in cell_ids:
        if cell_id not in CMU_V3_CELL_IDS:
            raise CMUInvalidCellIDError(f"Unknown CMU Version 3 cell ID: {cell_id}")
        main_path = raw_root / f"{cell_id}.csv"
        try:
            result = load_cmu_v3_main_file(
                main_path,
                gap_threshold_s=gap_threshold_s,
                repo_root=repo_root,
            )
        except Exception as exc:
            raise CMURawIngestionError(
                f"Raw QC failed for cell {cell_id} at {main_path}: {exc}"
            ) from exc

        metadata = result.metadata
        counts = metadata["boundary_reason_counts"]
        delta_stats = positive_delta_statistics(result.data["time_s"].to_numpy())
        records.append(
            {
                "dataset_name": DATASET_NAME,
                "article_id": ARTICLE_ID,
                "dataset_version": DATASET_VERSION,
                "loader_schema_version": metadata["loader_schema_version"],
                "generated_at_utc": generated_at,
                "git_revision": metadata["git_revision"],
                "git_dirty": metadata["git_dirty"],
                "cell_id": cell_id,
                "main_file": main_path.name,
                "main_sha256": metadata["source_sha256"],
                "impedance_file": metadata["impedance_file"],
                "impedance_file_present": metadata["impedance_file_present"],
                "impedance_sha256": metadata["impedance_sha256"],
                "row_count": metadata["input_row_count"],
                "gap_threshold_s": metadata["gap_threshold_s"],
                "gap_threshold_source": metadata["gap_threshold_source"],
                "block_count": metadata["block_count"],
                "time_duplicate_count": counts["time_duplicate"],
                "time_reversal_count": counts["time_reversal"],
                "current_time_non_finite_count": counts[
                    "current_time_non_finite"
                ],
                "previous_time_non_finite_count": counts[
                    "previous_time_non_finite"
                ],
                "gap_exceeded_count": counts["gap_exceeded"],
                **delta_stats,
            }
        )
        del result

    return pd.DataFrame(records, columns=RAW_QC_SUMMARY_COLUMNS)


def write_raw_qc_summary(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a validated per-cell raw QC summary CSV."""
    missing = sorted(set(RAW_QC_SUMMARY_COLUMNS) - set(summary.columns))
    if missing:
        raise ValueError(f"Raw QC summary missing column(s): {missing}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.loc[:, RAW_QC_SUMMARY_COLUMNS].to_csv(output_path, index=False)
    return output_path
