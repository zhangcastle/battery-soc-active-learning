"""Shared utilities for manifest-driven ``.mat`` dataset adapters.

This module holds field-mapping constants, QC detection, DataFrame construction,
and a lightweight :class:`ManifestMatAdapter` base that wires together
``manifest_loader`` and ``mat_loader`` without dataset-specific SOC heuristics.

Dataset adapters (Panasonic, LG HG2, …) subclass :class:`ManifestMatAdapter`
and override only the small hook methods for SOC metadata and optional extras.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from battery_benchmark.data.adapter_qc import AdapterSchemaError, QCResult
from battery_benchmark.data.manifest_loader import (
    SHA256MismatchError,
    ManifestEntry,
    entry_manifest_splits,
    load_manifest,
    select_manifest_entries,
)
from battery_benchmark.data.mat_loader import MatLoadResult, load_mat_file
from battery_benchmark.data.sequence_record import SequenceRecord

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field mapping (Panasonic / LG HG2 share the same meas struct at 25 °C)
# ---------------------------------------------------------------------------

FIELD_MAP: dict[str, str] = {
    "Time": "time_s",
    "TimeStamp": "timestamp",
    "Voltage": "voltage_v",
    "Current": "current_a",
    "Battery_Temp_degC": "temperature_c",
    "Ah": "cycler_ah",
}

REQUIRED_MEAS_FIELDS: tuple[str, ...] = (
    "Time",
    "Voltage",
    "Current",
    "Battery_Temp_degC",
    "Ah",
)

OPTIONAL_FIELD_MAP: dict[str, str] = {
    "Wh": "wh",
    "Power": "power_w",
    "Chamber_Temp_degC": "chamber_temp_c",
}

NUMERIC_DST_COLS: tuple[str, ...] = (
    "time_s", "voltage_v", "current_a", "temperature_c", "cycler_ah"
)

# QC thresholds (shared across adapters — do not change per dataset)
_AH_RESET_FRACTION = 0.05
_AH_RESET_MIN_MAX = 0.05
_TEMP_DEVIATION_WARN = 10.0


# ---------------------------------------------------------------------------
# Summary dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AdapterSummary:
    """Lightweight statistics over a ``load_split`` result."""
    split: str
    expected_manifest_entries: int
    loaded_records: int
    skipped_manifest_failures: int
    pass_count: int
    warn_count: int
    fail_count: int
    total_rows: int
    duplicate_timestamp_count: int
    ah_reset_count: int
    source_files: list[str]


@dataclass
class AdapterLoadStats:
    """Bookkeeping for one :meth:`ManifestMatAdapter.load_split` call."""
    split: str
    expected_manifest_entries: int
    loaded_records: int
    skipped_manifest_failures: int


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def require_paper_profile(entry: ManifestEntry, dataset_label: str) -> None:
    """Ensure selected manifest rows carry an explicit ``paper_profile``."""
    if entry.is_selected and not entry.paper_profile.strip():
        raise AdapterSchemaError(
            f"Selected manifest entry {entry.filename!r} has empty paper_profile. "
            f"{dataset_label} paper_profile must be defined in the manifest; "
            "the adapter does not infer it from profile."
        )


def parse_manifest_soc_init(entry: ManifestEntry) -> float | None:
    """Parse explicit ``soc_init`` from the manifest row with strict validation.

    Empty string returns ``None``.  Non-empty values must parse to a finite
    float in ``[0.0, 1.0]``; otherwise :class:`AdapterSchemaError` is raised.
    """
    text = entry.soc_init_str.strip()
    if not text:
        return None

    try:
        value = float(text)
    except ValueError:
        raise AdapterSchemaError(
            f"Invalid soc_init for {entry.filename!r}: raw_value={text!r}. "
            "Must be a finite number in [0.0, 1.0] or empty."
        ) from None

    if not math.isfinite(value):
        raise AdapterSchemaError(
            f"Invalid soc_init for {entry.filename!r}: raw_value={text!r}. "
            "Must be a finite number in [0.0, 1.0] or empty."
        )

    if not 0.0 <= value <= 1.0:
        raise AdapterSchemaError(
            f"Invalid soc_init for {entry.filename!r}: raw_value={text!r}. "
            "Must be a finite number in [0.0, 1.0] or empty."
        )

    return value


def validate_soc_metadata(
    entry: ManifestEntry,
    requested_split: str,
    soc_init: float | None,
    soc_anchor_source: str,
) -> None:
    """Validate SOC initialization metadata for selected model-input drive-cycles.

    Records with ``record_role == "model_input"`` must carry a finite
    ``soc_init`` and a non-empty ``soc_anchor_source`` that is not
    ``"not_applicable"``.  ``soc_anchor`` and ``excluded`` records are not
    subject to drive-cycle SOC requirements.
    """
    if entry.record_role != "model_input":
        return

    missing: list[str] = []
    if soc_init is None:
        missing.append("soc_init")
    anchor = soc_anchor_source.strip()
    if not anchor:
        missing.append("soc_anchor_source")
    elif anchor == "not_applicable":
        missing.append("soc_anchor_source (must not be 'not_applicable')")

    if missing:
        raise AdapterSchemaError(
            f"SOC metadata incomplete for model_input drive-cycle "
            f"{entry.filename!r} (requested_split={requested_split!r}): "
            f"missing or invalid: {', '.join(missing)}"
        )


def resolve_raw_path(raw_root: Path, entry: ManifestEntry) -> Path:
    """Resolve ``entry.relative_path`` against *raw_root*."""
    path = raw_root / entry.relative_path
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path}\n"
            f"  relative_path in manifest: {entry.relative_path}\n"
            f"  raw_root: {raw_root}"
        )
    return path


def make_sequence_id(prefix: str, entry: ManifestEntry) -> str:
    """Build a stable sequence ID from *prefix* and the filename stem."""
    return f"{prefix}_{Path(entry.filename).stem}"


def verify_sha256(
    entry: ManifestEntry,
    abs_path: Path,
    actual_sha256: str,
    qc: QCResult,
) -> None:
    """Compare manifest SHA-256 to the loaded file hash."""
    if entry.sha256 == "":
        qc.add_warning(
            "SHA-256 not recorded in manifest; file identity unverified. "
            f"Actual SHA: {actual_sha256[:16]}…"
        )
    elif actual_sha256 != entry.sha256:
        raise SHA256MismatchError(abs_path, entry.sha256, actual_sha256)


def detect_adapter_qc(
    data_df: pd.DataFrame,
    entry: ManifestEntry,
    qc: QCResult,
) -> None:
    """Populate *qc* from *data_df* using shared adapter rules."""
    n = len(data_df)

    if n == 0:
        qc.add_failure("Record has 0 rows.")
        return

    if "time_s" in data_df.columns:
        nan_time = int(data_df["time_s"].isna().sum())
        if nan_time > 0:
            qc.add_failure(
                f"time_s contains {nan_time} NaN value(s); "
                "time is essential and cannot contain NaN."
            )

    total_nan = 0
    for col in NUMERIC_DST_COLS:
        if col == "time_s" or col not in data_df.columns:
            continue
        col_nan = int(data_df[col].isna().sum())
        if col_nan > 0:
            total_nan += col_nan
            qc.add_warning(f"{col}: {col_nan} NaN value(s).")
    qc.nan_count = (
        int(data_df[list(NUMERIC_DST_COLS)].isnull().sum().sum())
        if all(c in data_df.columns for c in NUMERIC_DST_COLS)
        else total_nan
    )

    total_inf = 0
    for col in NUMERIC_DST_COLS:
        if col not in data_df.columns:
            continue
        arr = pd.to_numeric(data_df[col], errors="coerce").values
        col_inf = int(np.isinf(arr).sum())
        if col_inf > 0:
            total_inf += col_inf
            qc.add_failure(f"{col}: {col_inf} Inf value(s).")
    qc.inf_count = total_inf

    if "time_s" in data_df.columns and n > 1:
        time_arr = pd.to_numeric(data_df["time_s"], errors="coerce").values
        dt = np.diff(time_arr)
        neg = int((dt < 0).sum())
        zero = int((dt == 0).sum())
        qc.negative_time_step_count = neg
        qc.duplicate_timestamp_count = zero
        if neg > 0:
            qc.add_failure(
                f"{neg} strictly-decreasing time step(s) detected (dt < 0)."
            )
        if zero > 0:
            qc.add_warning(
                f"{zero} duplicate consecutive timestamp(s) (dt == 0; cycler artifact)."
            )

    if "cycler_ah" in data_df.columns:
        ah_arr = pd.to_numeric(data_df["cycler_ah"], errors="coerce").values
        ah_abs_max = float(np.nanmax(np.abs(ah_arr))) if len(ah_arr) > 0 else 0.0
        ah_abs_end = abs(float(ah_arr[-1])) if len(ah_arr) > 0 else 0.0
        if (
            ah_abs_max > _AH_RESET_MIN_MAX
            and ah_abs_end < _AH_RESET_FRACTION * ah_abs_max
        ):
            qc.ah_reset_detected = True
            qc.add_warning(
                f"Ah counter reset detected: "
                f"cycler_ah[-1]={ah_arr[-1]:.4f} Ah but abs_max={ah_abs_max:.3f} Ah."
            )

    if "temperature_c" in data_df.columns:
        temp_arr = pd.to_numeric(data_df["temperature_c"], errors="coerce").values
        temp_mean = float(np.nanmean(temp_arr))
        if abs(temp_mean - entry.temperature_c) > _TEMP_DEVIATION_WARN:
            qc.add_warning(
                f"Measured temperature mean {temp_mean:.1f} °C deviates more than "
                f"{_TEMP_DEVIATION_WARN:.0f} °C from nominal {entry.temperature_c:.0f} °C."
            )

    if entry.qc_status == "WARN":
        qc.add_note(f"Manifest QC note: {entry.qc_notes}")


def measurements_to_dataframe(
    meas: dict[str, np.ndarray],
    qc: QCResult,
) -> pd.DataFrame:
    """Map meas arrays to a unified DataFrame, recording QC failures."""
    missing_fields = [f for f in REQUIRED_MEAS_FIELDS if f not in meas]
    for field in missing_fields:
        qc.add_failure(
            f"Required MATLAB measurement field '{field}' is absent from meas struct."
        )

    available = {f: meas[f] for f in REQUIRED_MEAS_FIELDS if f in meas}
    lengths = {f: len(arr) for f, arr in available.items()}
    if lengths:
        n = max(lengths.values())
        inconsistent = {f: length for f, length in lengths.items() if length != n}
        if inconsistent:
            detail = ", ".join(f"{f}={length}" for f, length in inconsistent.items())
            qc.add_failure(
                f"Measurement arrays have inconsistent lengths (expected {n}): {detail}."
            )
    else:
        n = 0

    data_dict: dict[str, Any] = {}
    for src_field, dst_col in FIELD_MAP.items():
        if src_field in meas:
            arr = meas[src_field]
            if dst_col == "timestamp":
                data_dict[dst_col] = arr
            else:
                data_dict[dst_col] = pd.to_numeric(
                    pd.Series(arr), errors="coerce"
                ).values
        else:
            data_dict[dst_col] = (
                np.full(n, np.nan) if n > 0 else np.array([], dtype=float)
            )

    for src_field, dst_col in OPTIONAL_FIELD_MAP.items():
        if src_field in meas:
            data_dict[dst_col] = pd.to_numeric(
                pd.Series(meas[src_field]), errors="coerce"
            ).values

    if n > 0:
        for col, arr in list(data_dict.items()):
            if not hasattr(arr, "__len__"):
                continue
            arr_len = len(arr)
            if arr_len == n:
                continue
            if hasattr(arr, "dtype") and arr.dtype.kind == "O":
                padded: np.ndarray = np.empty(n, dtype=object)
                padded[:] = None
                copy_len = min(arr_len, n)
                padded[:copy_len] = arr[:copy_len]
            else:
                padded = np.full(n, np.nan, dtype=float)
                copy_len = min(arr_len, n)
                padded[:copy_len] = arr[:copy_len]
            data_dict[col] = padded

    return pd.DataFrame(data_dict)


def build_adapter_summary(
    records: list[SequenceRecord],
    *,
    split: str,
    stats: AdapterLoadStats | None,
) -> AdapterSummary:
    """Aggregate QC and load statistics for one split.

    When *records* is non-empty, every record must have
    ``metadata["split"] == split``; otherwise :class:`AdapterSchemaError`
    is raised.
    """
    if records:
        observed = {r.metadata.get("split") for r in records}
        if observed != {split}:
            raise AdapterSchemaError(
                f"Summary split {split!r} does not match record splits "
                f"{sorted(observed)}."
            )

    expected = stats.expected_manifest_entries if stats else len(records)
    skipped = stats.skipped_manifest_failures if stats else 0
    return AdapterSummary(
        split=split,
        expected_manifest_entries=expected,
        loaded_records=len(records),
        skipped_manifest_failures=skipped,
        pass_count=sum(1 for r in records if r.qc.status == "PASS"),
        warn_count=sum(1 for r in records if r.qc.status == "WARN"),
        fail_count=sum(1 for r in records if r.qc.status == "FAIL"),
        total_rows=sum(r.n_rows for r in records),
        duplicate_timestamp_count=sum(
            r.qc.duplicate_timestamp_count for r in records
        ),
        ah_reset_count=sum(1 for r in records if r.qc.ah_reset_detected),
        source_files=[r.metadata["source_file"] for r in records],
    )


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class ManifestMatAdapter:
    """Manifest CSV + ``.mat`` loader shared by Vidal 2019 dataset adapters.

    Subclasses set class attributes and override hook methods for SOC metadata.

    **Training input contract:** only records with
    ``metadata["record_role"] == "model_input"`` may enter the SOC model
    sequence builder.  ``soc_anchor`` records supply full-charge evidence;
    ``excluded`` records are audit-only.
    """

    DATASET_NAME: str = ""
    SEQUENCE_ID_PREFIX: str = ""
    DEFAULT_PAPER_BATTERY_NAME: str = ""
    DEFAULT_DATASET_IDENTITY_STATUS: str = "CONFIRMED"

    def __init__(
        self,
        raw_root: str | Path,
        manifest_path: str | Path,
        capacity_hint_ah: float,
    ) -> None:
        self._raw_root = Path(raw_root).resolve()
        self._manifest_path = Path(manifest_path)
        self._capacity_hint_ah = capacity_hint_ah
        self._entries: list[ManifestEntry] | None = None
        self._load_stats_by_split: dict[str, AdapterLoadStats] = {}

    @property
    def entries(self) -> list[ManifestEntry]:
        if self._entries is None:
            self._entries = load_manifest(self._manifest_path)
        return self._entries

    # ---- hooks for dataset-specific metadata ----

    def get_soc_init(self, entry: ManifestEntry) -> float | None:
        raise NotImplementedError

    def get_soc_anchor_source(self, entry: ManifestEntry) -> str:
        raise NotImplementedError

    def get_metadata_extras(
        self,
        entry: ManifestEntry,
        mat_result: MatLoadResult,
    ) -> dict[str, Any]:
        return {}

    # ---- core API ----

    def load_entry(
        self,
        entry: ManifestEntry,
        *,
        requested_split: str,
    ) -> SequenceRecord:
        require_paper_profile(entry, self.DATASET_NAME)
        abs_path = resolve_raw_path(self._raw_root, entry)
        qc = QCResult()

        mat_result = load_mat_file(abs_path)
        verify_sha256(entry, abs_path, mat_result.sha256, qc)

        for warning in mat_result.warnings:
            if "Optional field" in warning:
                qc.add_note(warning)
            else:
                qc.add_warning(f"mat_loader: {warning}")

        data_df = measurements_to_dataframe(mat_result.measurements, qc)
        detect_adapter_qc(data_df, entry, qc)

        soc_init = self.get_soc_init(entry)
        soc_anchor_source = self.get_soc_anchor_source(entry)
        validate_soc_metadata(
            entry, requested_split, soc_init, soc_anchor_source
        )

        metadata: dict[str, Any] = {
            "dataset": self.DATASET_NAME,
            "battery": entry.battery,
            "source_file": entry.relative_path,
            "source_sha256": mat_result.sha256,
            "sequence_id": make_sequence_id(self.SEQUENCE_ID_PREFIX, entry),
            "profile": entry.profile,
            "paper_profile": entry.paper_profile,
            "file_type": entry.file_type,
            "record_role": entry.record_role,
            "split": requested_split,
            "temperature_nominal_c": entry.temperature_c,
            "soc_init": soc_init,
            "soc_anchor_source": soc_anchor_source,
            "capacity_hint_ah": self._capacity_hint_ah,
            "manifest_splits": entry_manifest_splits(entry),
            "paper_battery_name": entry.paper_battery_name or self.DEFAULT_PAPER_BATTERY_NAME,
            "dataset_identity_status": (
                entry.dataset_identity_status or self.DEFAULT_DATASET_IDENTITY_STATUS
            ),
            "pairing_status": entry.pairing_status,
            "confidence": entry.confidence,
            "evidence": entry.evidence,
            "notes": entry.notes,
            "manifest_qc_status": entry.qc_status,
            "manifest_qc_notes": entry.qc_notes,
            "current_sign_convention": "as_recorded",
            "loader_backend": mat_result.loader_backend,
        }
        metadata.update(self.get_metadata_extras(entry, mat_result))

        return SequenceRecord(data=data_df, metadata=metadata, qc=qc)

    def load_split(self, split: str) -> list[SequenceRecord]:
        selected = select_manifest_entries(self.entries, split)
        records: list[SequenceRecord] = []
        skipped = 0

        for entry in selected:
            if entry.qc_status == "FAIL":
                log.warning(
                    "Skipping %s: manifest qc_status=FAIL (%s)",
                    entry.filename,
                    entry.qc_notes,
                )
                skipped += 1
                continue
            records.append(self.load_entry(entry, requested_split=split))

        self._load_stats_by_split[split] = AdapterLoadStats(
            split=split,
            expected_manifest_entries=len(selected),
            loaded_records=len(records),
            skipped_manifest_failures=skipped,
        )
        return records

    def summarize(
        self,
        records: list[SequenceRecord],
        split: str = "",
    ) -> AdapterSummary:
        if not split:
            raise ValueError(
                "summarize() requires an explicit split label so load stats "
                "are looked up per split."
            )
        stats = self._load_stats_by_split.get(split)
        return build_adapter_summary(records, split=split, stats=stats)
