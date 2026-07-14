"""Panasonic NCR18650PF dataset adapter.

Source: McMaster University Battery Dataset
URL: https://data.mendeley.com/datasets/wykht8y7tg/1

Two classes are provided:

``PanasonicDataset``
    Legacy ``BaseBatteryDataset`` subclass registered under
    ``"panasonic_18650pf"``.  Still raises ``NotImplementedError`` for all
    methods — kept for registry compatibility while the framework migrates to
    the SequenceRecord API.

``PanasonicAdapter``
    New-style adapter that reads the Vidal 2019 25 °C manifest CSV and returns
    a ``list[SequenceRecord]``.  This is the class to use for current work.

Current implementation status
------------------------------
- Manifest loading: complete.
- Field mapping (mat → DataFrame): complete.
- Adapter-level QC detection: complete.
- SequenceRecord construction: complete.
- SOC calculation: not implemented.
- Data cleaning (deduplication, Ah reset, resampling): not implemented.
- Normalisation / windowing: not implemented.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from battery_benchmark.data.adapter_qc import AdapterSchemaError, QCResult
from battery_benchmark.data.base_dataset import BaseBatteryDataset
from battery_benchmark.data.manifest_loader import (
    SHA256MismatchError,
    ManifestEntry,
    entry_manifest_splits,
    load_manifest,
    select_manifest_entries,
)
from battery_benchmark.data.mat_loader import load_mat_file
from battery_benchmark.data.registry import register_dataset
from battery_benchmark.data.sequence_record import SequenceRecord

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy placeholder (registry compatibility)
# ---------------------------------------------------------------------------

@register_dataset("panasonic_18650pf")
class PanasonicDataset(BaseBatteryDataset):
    """Legacy adapter stub — kept for registry compatibility.

    All methods raise ``NotImplementedError``.  Use :class:`PanasonicAdapter`
    for the new SequenceRecord-based API.
    """

    def discover_files(self, raw_dir: str | Path) -> list[Path]:
        raise NotImplementedError(
            "PanasonicDataset is a placeholder. Use PanasonicAdapter instead."
        )

    def build_manifest(self, raw_dir: str | Path) -> pd.DataFrame:
        raise NotImplementedError("PanasonicDataset is a placeholder.")

    def load_cycle(self, file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError("PanasonicDataset is a placeholder.")

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("PanasonicDataset is a placeholder.")

    def validate_cycle(self, df: pd.DataFrame) -> bool:
        raise NotImplementedError("PanasonicDataset is a placeholder.")

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "panasonic_18650pf",
            "status": "placeholder",
            "nominal_capacity_ah": 2.9,
            "note": "Use PanasonicAdapter for the SequenceRecord API.",
        }


# ---------------------------------------------------------------------------
# Field mapping constants
# ---------------------------------------------------------------------------

#: MATLAB meas-struct field → SequenceRecord.data column name.
FIELD_MAP: dict[str, str] = {
    "Time": "time_s",
    "TimeStamp": "timestamp",
    "Voltage": "voltage_v",
    "Current": "current_a",
    "Battery_Temp_degC": "temperature_c",
    "Ah": "cycler_ah",
}

#: MATLAB fields that MUST be present for a valid record.
REQUIRED_MEAS_FIELDS: tuple[str, ...] = (
    "Time",
    "Voltage",
    "Current",
    "Battery_Temp_degC",
    "Ah",
)

#: Optional numeric fields to preserve if present (renamed).
OPTIONAL_FIELD_MAP: dict[str, str] = {
    "Wh": "wh",
    "Power": "power_w",
    "Chamber_Temp_degC": "chamber_temp_c",
}

#: Numeric dtype fields used for NaN / Inf detection.
_NUMERIC_DST_COLS: tuple[str, ...] = (
    "time_s", "voltage_v", "current_a", "temperature_c", "cycler_ah"
)

# ---------------------------------------------------------------------------
# QC thresholds  (all documented in docstrings and tests)
# ---------------------------------------------------------------------------

#: Ah reset threshold: end < this fraction of abs-max → likely reset.
_AH_RESET_FRACTION = 0.05
#: Minimum abs Ah for reset check to apply (avoids false alarm on near-zero traces).
_AH_RESET_MIN_MAX = 0.05
#: Temperature deviation (°C) from nominal that triggers a WARN.
_TEMP_DEVIATION_WARN = 10.0


# ---------------------------------------------------------------------------
# Private QC helpers
# ---------------------------------------------------------------------------

def _detect_qc(
    measurements: dict[str, np.ndarray],
    data_df: pd.DataFrame,
    entry: ManifestEntry,
    qc: QCResult,
) -> None:
    """Populate *qc* by inspecting *measurements* and *data_df* in-place.

    QC rules
    --------
    FAIL conditions:
        - Any required meas field is missing (detected before calling this).
        - Core arrays have mismatched lengths.
        - n_rows == 0.
        - NaN in ``time_s`` (Time column).
        - Inf in any numeric column.
        - dt < 0 (strictly decreasing timestamp).

    WARN conditions:
        - NaN in any non-Time numeric column (count recorded in notes).
        - dt == 0 (duplicate consecutive timestamp — cycler artifact).
        - Ah counter reset detected (end ≈ 0 but abs-max >> 0).
        - temperature_c mean deviates > 10 °C from nominal.
        - manifest qc_status == WARN (propagated to notes).

    FAIL via manifest:
        - manifest qc_status == FAIL → loading is skipped upstream.
    """
    n = len(data_df)

    # 3. n_rows == 0
    if n == 0:
        qc.add_failure("Record has 0 rows.")
        return  # nothing else to check

    # 4. NaN in time_s → FAIL; NaN in other numeric cols → WARN
    if "time_s" in data_df.columns:
        nan_time = int(data_df["time_s"].isna().sum())
        if nan_time > 0:
            qc.add_failure(
                f"time_s contains {nan_time} NaN value(s); "
                "time is essential and cannot contain NaN."
            )

    total_nan = 0
    for col in _NUMERIC_DST_COLS:
        if col == "time_s":
            continue
        if col not in data_df.columns:
            continue
        col_nan = int(data_df[col].isna().sum())
        if col_nan > 0:
            total_nan += col_nan
            qc.add_warning(f"{col}: {col_nan} NaN value(s).")
    qc.nan_count = int(data_df[list(_NUMERIC_DST_COLS)].isnull().sum().sum()) if all(
        c in data_df.columns for c in _NUMERIC_DST_COLS
    ) else total_nan

    # 5. Inf in any numeric column → FAIL
    total_inf = 0
    for col in _NUMERIC_DST_COLS:
        if col not in data_df.columns:
            continue
        arr = pd.to_numeric(data_df[col], errors="coerce").values
        col_inf = int(np.isinf(arr).sum())
        if col_inf > 0:
            total_inf += col_inf
            qc.add_failure(f"{col}: {col_inf} Inf value(s).")
    qc.inf_count = total_inf

    # 6 & 7. Time monotonicity
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

    # 8. Ah reset detection
    # Rule: abs(last_Ah) < AH_RESET_FRACTION × abs_max_Ah AND abs_max > threshold
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

    # 12. Temperature check
    if "temperature_c" in data_df.columns and entry.temperature_c > 0:
        temp_arr = pd.to_numeric(data_df["temperature_c"], errors="coerce").values
        temp_mean = float(np.nanmean(temp_arr))
        if abs(temp_mean - entry.temperature_c) > _TEMP_DEVIATION_WARN:
            qc.add_warning(
                f"Measured temperature mean {temp_mean:.1f} °C deviates more than "
                f"{_TEMP_DEVIATION_WARN:.0f} °C from nominal {entry.temperature_c:.0f} °C."
            )

    # 11. Propagate manifest QC WARN notes
    if entry.qc_status == "WARN":
        qc.add_note(f"Manifest QC note: {entry.qc_notes}")


# ---------------------------------------------------------------------------
# soc_anchor_source / soc_init derivation from Panasonic manifest
# ---------------------------------------------------------------------------

def _derive_soc_anchor_source(entry: ManifestEntry) -> str:
    """Infer soc_anchor_source from manifest evidence and pairing_status.

    Priority:
    1. Explicit ``soc_anchor_source`` field in manifest (Samsung manifests).
    2. "full_charge_protocol" in ``soc_anchor_evidence``.
    3. ``pairing_status == "confirmed"`` → ``"preceding_full_charge"``.
    4. ``pairing_status == "unresolved"`` → ``"unresolved"``.
    5. charge / characterisation files → ``"not_applicable"``.
    """
    if entry.soc_anchor_source:
        return entry.soc_anchor_source
    ev = entry.soc_anchor_evidence.lower()
    if "full_charge_protocol" in ev:
        return "full_charge_protocol"
    if entry.pairing_status == "confirmed":
        return "preceding_full_charge"
    if entry.pairing_status == "unresolved":
        return "unresolved"
    if entry.file_type in ("charge", "characterization"):
        return "not_applicable"
    return "not_applicable"


def _derive_soc_init(entry: ManifestEntry) -> float | None:
    """Infer soc_init from manifest evidence.

    Returns 1.0 for confirmed and unresolved drive-cycle pairings
    (the paper uses a full-charge protocol before every drive cycle).
    Returns None for charge and characterisation files.
    """
    ev = entry.soc_anchor_evidence
    if "SOC_init=1.0" in ev or entry.pairing_status in ("confirmed", "unresolved"):
        return 1.0
    return None


# ---------------------------------------------------------------------------
# Summary dataclass
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
class _AdapterLoadStats:
    """Internal bookkeeping from the most recent :meth:`load_split` call."""
    split: str
    expected_manifest_entries: int
    loaded_records: int
    skipped_manifest_failures: int


# ---------------------------------------------------------------------------
# PanasonicAdapter
# ---------------------------------------------------------------------------

class PanasonicAdapter:
    """Vidal 2019 adapter for the Panasonic NCR18650PF 25 °C dataset.

    Reads the manifest CSV produced by ``scripts/inspect_panasonic_25c_split.py``,
    selects the requested split, loads each ``.mat`` file via
    :func:`~battery_benchmark.data.mat_loader.load_mat_file`, maps raw MATLAB
    fields to the unified DataFrame schema, performs adapter-level QC, and
    returns a :class:`~battery_benchmark.data.sequence_record.SequenceRecord`
    per file.

    Parameters
    ----------
    raw_root:
        Repository root directory.  ``ManifestEntry.relative_path`` values
        (which start with ``data/raw/…``) are resolved relative to this root.
    manifest_path:
        Path to ``data/manifests/panasonic_vidal2019_25c_manifest.csv``.
    capacity_hint_ah:
        Nominal cell capacity hint passed to every ``SequenceRecord``
        (default 2.9 Ah for Panasonic NCR18650PF).

    Examples
    --------
    >>> from pathlib import Path
    >>> from battery_benchmark.data.panasonic import PanasonicAdapter
    >>> adapter = PanasonicAdapter(
    ...     raw_root=Path("."),
    ...     manifest_path="data/manifests/panasonic_vidal2019_25c_manifest.csv",
    ... )
    >>> records = adapter.load_split("full_train")
    >>> print(len(records), "records loaded")
    """

    DATASET_NAME: str = "panasonic_18650pf"

    def __init__(
        self,
        raw_root: str | Path,
        manifest_path: str | Path,
        capacity_hint_ah: float = 2.9,
    ) -> None:
        self._raw_root = Path(raw_root).resolve()
        self._manifest_path = Path(manifest_path)
        self._capacity_hint_ah = capacity_hint_ah
        self._entries: list[ManifestEntry] | None = None  # loaded lazily
        self._last_load_stats: _AdapterLoadStats | None = None

    # ------------------------------------------------------------------
    # Lazy manifest access
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[ManifestEntry]:
        """All manifest entries (loaded once, cached)."""
        if self._entries is None:
            self._entries = load_manifest(self._manifest_path)
        return self._entries

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_path(self, entry: ManifestEntry) -> Path:
        """Return the absolute path for *entry* using its relative_path.

        Raises
        ------
        FileNotFoundError
            If the resolved path does not exist.
        """
        p = self._raw_root / entry.relative_path
        if not p.exists():
            raise FileNotFoundError(
                f"Raw file not found: {p}\n"
                f"  relative_path in manifest: {entry.relative_path}\n"
                f"  raw_root: {self._raw_root}"
            )
        return p

    # ------------------------------------------------------------------
    # sequence_id generation
    # ------------------------------------------------------------------

    @staticmethod
    def _make_sequence_id(entry: ManifestEntry) -> str:
        """Generate a stable, unique sequence ID from the manifest entry.

        Format: ``panasonic_18650pf_25c_{filename_stem}``

        The filename already contains a timestamp prefix (e.g.
        ``03-18-17_02.17 25degC_Cycle_1_Pan18650PF``) so collisions
        between files are not possible.
        """
        stem = Path(entry.filename).stem
        return f"panasonic_18650pf_25c_{stem}"

    # ------------------------------------------------------------------
    # Core loading method
    # ------------------------------------------------------------------

    @staticmethod
    def _require_paper_profile(entry: ManifestEntry) -> None:
        """Ensure selected manifest rows carry an explicit paper_profile."""
        if entry.is_selected and not entry.paper_profile.strip():
            raise AdapterSchemaError(
                f"Selected manifest entry {entry.filename!r} has empty paper_profile. "
                "Panasonic paper_profile must be defined in the manifest; "
                "the adapter does not infer it from profile."
            )

    def load_entry(
        self,
        entry: ManifestEntry,
        *,
        requested_split: str,
    ) -> SequenceRecord:
        """Load a single manifest entry and return a :class:`SequenceRecord`.

        Parameters
        ----------
        entry:
            A :class:`~battery_benchmark.data.manifest_loader.ManifestEntry`
            from this dataset's manifest.
        requested_split:
            Split label from the caller's :meth:`load_split` request.  Stored
            verbatim in ``metadata["split"]`` regardless of manifest membership.

        Returns
        -------
        SequenceRecord
            May have ``qc.status == "FAIL"`` if the file fails QC checks.

        Raises
        ------
        FileNotFoundError
            If the raw ``.mat`` file is not on disk.
        SHA256MismatchError
            If the manifest records a SHA-256 that does not match the actual
            file content (indicates unexpected file modification).
        AdapterSchemaError
            If a selected entry is missing ``paper_profile``.
        """
        self._require_paper_profile(entry)
        abs_path = self._resolve_path(entry)
        qc = QCResult()

        # ---- Load raw .mat ----
        mat_result = load_mat_file(abs_path)
        actual_sha256 = mat_result.sha256

        # ---- SHA-256 check ----
        if entry.sha256 == "":
            qc.add_warning(
                "SHA-256 not recorded in manifest; file identity unverified. "
                f"Actual SHA: {actual_sha256[:16]}…"
            )
        elif actual_sha256 != entry.sha256:
            raise SHA256MismatchError(abs_path, entry.sha256, actual_sha256)

        # ---- Propagate loader warnings ----
        for w in mat_result.warnings:
            if "Optional field" in w:
                qc.add_note(w)  # optional fields are informational
            else:
                qc.add_warning(f"mat_loader: {w}")

        # ---- Check required meas fields ----
        meas = mat_result.measurements
        missing_fields = [f for f in REQUIRED_MEAS_FIELDS if f not in meas]
        for f in missing_fields:
            qc.add_failure(
                f"Required MATLAB measurement field '{f}' is absent from meas struct."
            )

        # ---- Determine common array length ----
        available = {f: meas[f] for f in REQUIRED_MEAS_FIELDS if f in meas}
        lengths = {f: len(arr) for f, arr in available.items()}
        if lengths:
            n = max(lengths.values())
            # Array length consistency check (rule 2)
            inconsistent = {f: l for f, l in lengths.items() if l != n}
            if inconsistent:
                detail = ", ".join(f"{f}={l}" for f, l in inconsistent.items())
                qc.add_failure(
                    f"Measurement arrays have inconsistent lengths (expected {n}): {detail}."
                )
        else:
            n = 0

        # ---- Build data dict (fill missing with NaN to allow SequenceRecord construction) ----
        data_dict: dict[str, Any] = {}

        for src_field, dst_col in FIELD_MAP.items():
            if src_field in meas:
                arr = meas[src_field]
                if dst_col == "timestamp":
                    # Preserve as object (may be strings or None)
                    data_dict[dst_col] = arr
                else:
                    data_dict[dst_col] = pd.to_numeric(
                        pd.Series(arr), errors="coerce"
                    ).values
            else:
                # Fill missing required columns with NaN
                data_dict[dst_col] = np.full(n, np.nan) if n > 0 else np.array([], dtype=float)

        # Optional fields (Time-linked numeric extras)
        for src_field, dst_col in OPTIONAL_FIELD_MAP.items():
            if src_field in meas:
                data_dict[dst_col] = pd.to_numeric(
                    pd.Series(meas[src_field]), errors="coerce"
                ).values

        # ---- Normalise array lengths (pad shorter arrays to n with NaN) ----
        # This ensures pd.DataFrame creation succeeds even when some arrays were
        # shorter due to mismatch; the FAIL QC entry above already records it.
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

        data_df = pd.DataFrame(data_dict)

        # ---- Run QC detection ----
        _detect_qc(meas, data_df, entry, qc)

        # ---- Build metadata ----
        soc_anchor_src = _derive_soc_anchor_source(entry)
        soc_init = _derive_soc_init(entry)
        sequence_id = self._make_sequence_id(entry)

        metadata: dict[str, Any] = {
            # ---- Required SequenceRecord fields ----
            "dataset": self.DATASET_NAME,
            "battery": entry.battery,
            "source_file": entry.relative_path,
            "source_sha256": actual_sha256,
            "sequence_id": sequence_id,
            "profile": entry.profile,
            "paper_profile": entry.paper_profile,
            "file_type": entry.file_type,
            "split": requested_split,
            "temperature_nominal_c": entry.temperature_c,
            "soc_init": soc_init,
            "soc_anchor_source": soc_anchor_src,
            "capacity_hint_ah": self._capacity_hint_ah,
            # ---- Provenance extras ----
            "manifest_splits": entry_manifest_splits(entry),
            "paper_battery_name": entry.paper_battery_name or "Panasonic_NCR18650PF",
            "dataset_identity_status": entry.dataset_identity_status or "CONFIRMED",
            "pairing_status": entry.pairing_status,
            "confidence": entry.confidence,
            "evidence": entry.evidence,
            "notes": entry.notes,
            "manifest_qc_status": entry.qc_status,
            "manifest_qc_notes": entry.qc_notes,
            "current_sign_convention": "as_recorded",
            "loader_backend": mat_result.loader_backend,
        }

        return SequenceRecord(data=data_df, metadata=metadata, qc=qc)

    # ------------------------------------------------------------------
    # Batch loading
    # ------------------------------------------------------------------

    def load_split(self, split: str) -> list[SequenceRecord]:
        """Load all manifest entries for *split* and return SequenceRecords.

        Parameters
        ----------
        split:
            One of ``"full_train"``, ``"partial_train"``, ``"test"``,
            ``"exclude"``.

        Returns
        -------
        list[SequenceRecord]
            One record per selected file, in manifest order.  Records with
            ``qc.status == "FAIL"`` are included so callers can inspect them;
            entries whose manifest ``qc_status == "FAIL"`` are skipped.

        Notes
        -----
        - The raw files are never modified.
        - Files from other splits are not loaded.
        - No concatenation of files occurs here.
        """
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
            rec = self.load_entry(entry, requested_split=split)
            records.append(rec)

        self._last_load_stats = _AdapterLoadStats(
            split=split,
            expected_manifest_entries=len(selected),
            loaded_records=len(records),
            skipped_manifest_failures=skipped,
        )
        return records

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summarize(
        self,
        records: list[SequenceRecord],
        split: str = "",
    ) -> AdapterSummary:
        """Return lightweight aggregate statistics over *records*.

        Parameters
        ----------
        records:
            Result of :meth:`load_split`.
        split:
            Split label string.  When omitted, uses the most recent
            :meth:`load_split` split if available.
        """
        stats = self._last_load_stats
        requested_split = split or (stats.split if stats else "")
        expected = stats.expected_manifest_entries if stats else len(records)
        skipped = stats.skipped_manifest_failures if stats else 0
        loaded = len(records)

        return AdapterSummary(
            split=requested_split,
            expected_manifest_entries=expected,
            loaded_records=loaded,
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
