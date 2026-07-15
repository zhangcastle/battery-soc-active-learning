"""Quality control for Vidal-compatible SOC labels."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from battery_benchmark.data.adapter_qc import QCResult

# ---------------------------------------------------------------------------
# Thresholds (centralised)
# ---------------------------------------------------------------------------

DENOMINATOR_EPS: float = 1e-9
AH_START_NONZERO_WARN: float = 0.05
AH_DIRECTION_MIN_MAGNITUDE: float = 0.05
MAX_ABS_AH_REF_DIFF_WARN: float = 0.5
SOC_OUT_OF_BOUNDS_WARN_COUNT: int = 1
DRIVE_SOC_RISING_STEP_WARN_FRACTION: float = 0.05
CHARGE_SOC_FALLING_STEP_WARN_FRACTION: float = 0.05
DRIVE_SOC_START_WARN_HIGH: float = 1.05
DRIVE_SOC_END_WARN_LOW: float = -0.05
CHARGE_SOC_START_WARN_LOW: float = -0.05
CHARGE_SOC_END_WARN_HIGH: float = 1.05


@dataclass
class SOCLabelQCResult:
    """QC outcome for one labelled ``SequenceRecord``."""

    status: str = "PASS"
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    n_rows: int = 0
    soc_min: float | None = None
    soc_max: float | None = None
    soc_start: float | None = None
    soc_end: float | None = None
    soc_out_of_bounds_count: int = 0

    ah_start: float | None = None
    ah_end: float | None = None
    ah_min: float | None = None
    ah_max: float | None = None
    denominator_ah: float | None = None

    cycler_ah_delta_end: float | None = None
    current_integrated_ah_end: float | None = None
    max_abs_ah_reference_difference: float | None = None
    mean_abs_ah_reference_difference: float | None = None
    ah_direction_consistent: bool | None = None

    duplicate_timestamp_count: int = 0
    negative_time_step_count: int = 0
    ah_reset_detected: bool = False

    def add_failure(self, message: str) -> None:
        self.failures.append(message)
        self.status = "FAIL"

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        if self.status == "PASS":
            self.status = "WARN"

    def add_note(self, message: str) -> None:
        self.notes.append(message)


def _finite_stats(arr: np.ndarray) -> tuple[float | None, float | None, float | None, float | None]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None, None, None
    return float(finite[0]), float(finite[-1]), float(np.min(finite)), float(np.max(finite))


def run_label_qc(
    data: pd.DataFrame,
    *,
    file_type: str,
    adapter_qc: QCResult,
    denominator_ah: float,
    formula_mode: str,
) -> SOCLabelQCResult:
    """Evaluate labelled data and populate :class:`SOCLabelQCResult`."""
    qc = SOCLabelQCResult(n_rows=len(data))
    qc.duplicate_timestamp_count = adapter_qc.duplicate_timestamp_count
    qc.negative_time_step_count = adapter_qc.negative_time_step_count
    qc.ah_reset_detected = adapter_qc.ah_reset_detected

    if adapter_qc.status == "FAIL":
        qc.add_failure("Adapter QC status is FAIL.")

    if not math.isfinite(denominator_ah) or denominator_ah <= DENOMINATOR_EPS:
        qc.add_failure(
            f"denominator_ah must be finite and > {DENOMINATOR_EPS}; got {denominator_ah!r}."
        )
        return qc

    qc.denominator_ah = float(denominator_ah)

    ah = pd.to_numeric(data["cycler_ah"], errors="coerce").to_numpy(dtype=float)
    soc_raw = pd.to_numeric(data["soc_raw"], errors="coerce").to_numpy(dtype=float)
    ah_delta = pd.to_numeric(data["cycler_ah_delta"], errors="coerce").to_numpy(dtype=float)
    int_ah = pd.to_numeric(data["current_integrated_ah"], errors="coerce").to_numpy(dtype=float)
    ref_diff = pd.to_numeric(data["ah_reference_difference"], errors="coerce").to_numpy(dtype=float)

    if np.all(np.isnan(ah)):
        qc.add_failure("cycler_ah is entirely NaN.")
        return qc
    if np.all(np.isnan(soc_raw)):
        qc.add_failure("soc_raw is entirely NaN.")
        return qc

    qc.ah_start, qc.ah_end, qc.ah_min, qc.ah_max = _finite_stats(ah)
    qc.soc_start, qc.soc_end, qc.soc_min, qc.soc_max = _finite_stats(soc_raw)
    qc.cycler_ah_delta_end = float(ah_delta[-1]) if len(ah_delta) else None
    qc.current_integrated_ah_end = float(int_ah[-1]) if len(int_ah) else None

    finite_ref = ref_diff[np.isfinite(ref_diff)]
    if finite_ref.size:
        qc.max_abs_ah_reference_difference = float(np.max(np.abs(finite_ref)))
        qc.mean_abs_ah_reference_difference = float(np.mean(np.abs(finite_ref)))

    finite_soc = soc_raw[np.isfinite(soc_raw)]
    if finite_soc.size:
        oob = int(np.sum((finite_soc < 0.0) | (finite_soc > 1.0)))
        qc.soc_out_of_bounds_count = oob
        if oob >= SOC_OUT_OF_BOUNDS_WARN_COUNT:
            qc.add_warning(f"{oob} SOC value(s) outside [0, 1].")

    if qc.ah_start is not None and abs(qc.ah_start) > AH_START_NONZERO_WARN:
        qc.add_warning(f"Non-zero Ah start: {qc.ah_start:.4f} Ah.")

    if qc.duplicate_timestamp_count > 0:
        qc.add_warning(
            f"{qc.duplicate_timestamp_count} duplicate timestamp(s) (dt == 0)."
        )

    if qc.ah_reset_detected:
        qc.add_warning("Adapter detected Ah counter reset.")

    if adapter_qc.nan_count > 0:
        qc.add_warning(f"Adapter reported {adapter_qc.nan_count} NaN value(s) in numeric columns.")

    # Direction consistency
    delta_end = qc.cycler_ah_delta_end
    int_end = qc.current_integrated_ah_end
    if delta_end is not None and int_end is not None:
        if (
            abs(delta_end) > AH_DIRECTION_MIN_MAGNITUDE
            and abs(int_end) > AH_DIRECTION_MIN_MAGNITUDE
            and np.sign(delta_end) != np.sign(int_end)
        ):
            qc.ah_direction_consistent = False
            qc.add_failure(
                "cycler Ah delta and current integration have opposite total direction."
            )
        else:
            qc.ah_direction_consistent = True

    if file_type == "drive_cycle":
        if delta_end is not None and delta_end > AH_DIRECTION_MIN_MAGNITUDE:
            qc.add_failure("Drive-cycle cycler Ah shows net positive growth.")
        if qc.soc_start is not None and qc.soc_start > DRIVE_SOC_START_WARN_HIGH:
            qc.add_warning(f"Drive-cycle SOC start {qc.soc_start:.4f} > 1.")
        if qc.soc_end is not None and qc.soc_end < DRIVE_SOC_END_WARN_LOW:
            qc.add_warning(f"Drive-cycle SOC end {qc.soc_end:.4f} < 0.")
        if len(finite_soc) > 1:
            rising = int(np.sum(np.diff(finite_soc) > 0))
            if rising / max(len(finite_soc) - 1, 1) > DRIVE_SOC_RISING_STEP_WARN_FRACTION:
                qc.add_warning(
                    f"Drive-cycle SOC has {rising} rising step(s) "
                    f"(>{DRIVE_SOC_RISING_STEP_WARN_FRACTION:.0%} of steps)."
                )
    elif file_type == "charge":
        if delta_end is not None and delta_end < -AH_DIRECTION_MIN_MAGNITUDE:
            qc.add_failure("Charge cycler Ah shows net negative growth.")
        if qc.soc_start is not None and qc.soc_start < CHARGE_SOC_START_WARN_LOW:
            qc.add_warning(f"Charge SOC start {qc.soc_start:.4f} < 0.")
        if qc.soc_end is not None and qc.soc_end > CHARGE_SOC_END_WARN_HIGH:
            qc.add_warning(f"Charge SOC end {qc.soc_end:.4f} > 1.")
        if len(finite_soc) > 1:
            falling = int(np.sum(np.diff(finite_soc) < 0))
            if falling / max(len(finite_soc) - 1, 1) > CHARGE_SOC_FALLING_STEP_WARN_FRACTION:
                qc.add_warning(
                    f"Charge SOC has {falling} falling step(s) "
                    f"(>{CHARGE_SOC_FALLING_STEP_WARN_FRACTION:.0%} of steps)."
                )

    if (
        qc.max_abs_ah_reference_difference is not None
        and qc.max_abs_ah_reference_difference > MAX_ABS_AH_REF_DIFF_WARN
    ):
        qc.add_warning(
            f"Large cycler vs current-integration difference: "
            f"max_abs={qc.max_abs_ah_reference_difference:.4f} Ah."
        )

    qc.add_note(f"soc_formula_mode={formula_mode}")
    return qc
