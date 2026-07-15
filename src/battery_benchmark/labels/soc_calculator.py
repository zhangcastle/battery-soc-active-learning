"""Vidal-compatible SOC reconstruction from per-record cycler Ah."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from battery_benchmark.data.sequence_record import SequenceRecord
from battery_benchmark.labels.exceptions import (
    InvalidRecordRoleForSOCError,
    InvalidSOCInputError,
    UnsupportedSOCFileTypeError,
)
from battery_benchmark.labels.label_qc import (
    DENOMINATOR_EPS,
    SOCLabelQCResult,
    run_label_qc,
)

SUPPORTED_VIDAL_DATASETS: frozenset[str] = frozenset({
    "panasonic_18650pf",
    "lg_hg2",
    "samsung_30t",
})

SUPPORTED_VIDAL_FILE_TYPES: frozenset[str] = frozenset({"drive_cycle", "charge"})

LABEL_SOURCE: str = "vidal_compatible_cycler_ah"
CALCULATOR_NAME: str = "vidal_per_record_ah_normalization"
SOC_FORMULA_VERSION: str = "v1"


def compute_current_integrated_ah(time_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    """Trapezoidal current integration with duplicate dt=0 contributing zero."""
    n = len(time_s)
    out = np.zeros(n, dtype=float)
    if n <= 1:
        return out
    dt = np.diff(time_s.astype(float))
    for i in range(1, n):
        out[i] = out[i - 1] + 0.5 * (current_a[i - 1] + current_a[i]) * dt[i - 1] / 3600.0
    return out


def compute_vidal_drive_soc(ah: np.ndarray) -> tuple[float, np.ndarray]:
    """Open-reference drive-cycle SOC: ``(ah + |min(ah)|) / |min(ah)|``."""
    finite = ah[np.isfinite(ah)]
    if finite.size == 0:
        raise InvalidSOCInputError("cycler_ah has no finite values.")
    denominator = abs(float(np.min(finite)))
    if not math.isfinite(denominator) or denominator <= DENOMINATOR_EPS:
        raise InvalidSOCInputError(
            f"Drive-cycle denominator_ah must be finite and > {DENOMINATOR_EPS}; "
            f"got {denominator!r}."
        )
    soc_raw = (ah + denominator) / denominator
    return denominator, soc_raw


def compute_vidal_charge_soc(ah: np.ndarray) -> tuple[float, np.ndarray]:
    """Open-reference charge SOC: ``ah / |max(ah)|``."""
    finite = ah[np.isfinite(ah)]
    if finite.size == 0:
        raise InvalidSOCInputError("cycler_ah has no finite values.")
    denominator = abs(float(np.max(finite)))
    if not math.isfinite(denominator) or denominator <= DENOMINATOR_EPS:
        raise InvalidSOCInputError(
            f"Charge denominator_ah must be finite and > {DENOMINATOR_EPS}; "
            f"got {denominator!r}."
        )
    soc_raw = ah / denominator
    return denominator, soc_raw


class SOCCalculator:
    """Generate Vidal-compatible SOC labels from adapter ``SequenceRecord`` objects."""

    def calculate_vidal(self, record: SequenceRecord) -> SequenceRecord:
        """Apply open-reference per-record Ah normalization (Vidal-compatible).

        Returns a new ``SequenceRecord``; the input record is not modified.
        """
        if not isinstance(record, SequenceRecord):
            raise InvalidSOCInputError(
                f"Expected SequenceRecord, got {type(record).__name__}."
            )

        meta = record.metadata
        record_role = meta.get("record_role", "")
        if record_role != "model_input":
            raise InvalidRecordRoleForSOCError(
                f"record_role must be 'model_input' for Vidal SOC labelling; "
                f"got {record_role!r} for {meta.get('sequence_id', '?')!r}."
            )

        file_type = meta.get("file_type", "")
        if file_type not in SUPPORTED_VIDAL_FILE_TYPES:
            raise UnsupportedSOCFileTypeError(
                f"Unsupported file_type {file_type!r} for calculate_vidal()."
            )

        dataset = meta.get("dataset", "")
        if dataset not in SUPPORTED_VIDAL_DATASETS:
            raise UnsupportedSOCFileTypeError(
                f"Unsupported dataset {dataset!r} for calculate_vidal()."
            )

        if record.qc.status == "FAIL":
            raise InvalidSOCInputError(
                f"Adapter QC status is FAIL for {meta.get('sequence_id', '?')!r}."
            )

        data = record.data
        if len(data) == 0:
            raise InvalidSOCInputError("Record has 0 rows.")

        for col in ("cycler_ah", "current_a", "time_s"):
            if col not in data.columns:
                raise InvalidSOCInputError(f"Missing required column {col!r}.")

        ah = pd.to_numeric(data["cycler_ah"], errors="coerce").to_numpy(dtype=float)
        current = pd.to_numeric(data["current_a"], errors="coerce").to_numpy(dtype=float)
        time_s = pd.to_numeric(data["time_s"], errors="coerce").to_numpy(dtype=float)

        if np.all(np.isnan(ah)):
            raise InvalidSOCInputError("cycler_ah is entirely NaN.")
        if np.isinf(ah).any():
            raise InvalidSOCInputError("cycler_ah contains Inf.")
        if np.isinf(current).any():
            raise InvalidSOCInputError("current_a contains Inf.")
        if np.isnan(time_s).any() or np.isinf(time_s).any():
            raise InvalidSOCInputError("time_s contains NaN or Inf.")
        if len(time_s) > 1 and np.any(np.diff(time_s) < 0):
            raise InvalidSOCInputError("time_s contains strictly decreasing steps (dt < 0).")

        if file_type == "drive_cycle":
            denominator, soc_raw = compute_vidal_drive_soc(ah)
            formula_mode = "drive_discharge"
        else:
            denominator, soc_raw = compute_vidal_charge_soc(ah)
            formula_mode = "charge"

        current_integrated = compute_current_integrated_ah(time_s, current)
        ah_start = float(ah[0]) if len(ah) else 0.0
        cycler_ah_delta = ah - ah_start
        ah_reference_difference = cycler_ah_delta - current_integrated

        out_df = data.copy()
        out_df["soc_raw"] = soc_raw
        out_df["soc"] = soc_raw
        out_df["cycler_ah_delta"] = cycler_ah_delta
        out_df["current_integrated_ah"] = current_integrated
        out_df["ah_reference_difference"] = ah_reference_difference

        label_qc = run_label_qc(
            out_df,
            file_type=file_type,
            adapter_qc=record.qc,
            denominator_ah=denominator,
            formula_mode=formula_mode,
        )

        out_meta: dict[str, Any] = deepcopy(meta)
        out_meta.update({
            "label_available": True,
            "label_source": LABEL_SOURCE,
            "soc_calculator": CALCULATOR_NAME,
            "soc_formula_mode": formula_mode,
            "soc_formula_version": SOC_FORMULA_VERSION,
            "soc_denominator_ah": float(denominator),
            "soc_clipped": False,
            "soc_column": "soc",
            "soc_raw_column": "soc_raw",
            "label_qc_status": label_qc.status,
        })

        labelled = SequenceRecord(data=out_df, metadata=out_meta, qc=record.qc)
        return labelled

    def calculate_cmu(
        self,
        record: SequenceRecord,
        **kwargs: Any,
    ) -> SequenceRecord:
        """Placeholder for CMU eVTOL SOC — not implemented in this phase."""
        raise NotImplementedError(
            "CMU SOC requires dynamic capacity, RPT interpolation, and "
            "mission/charge continuity logic."
        )
