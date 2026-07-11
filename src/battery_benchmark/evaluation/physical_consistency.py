"""Physical-consistency metrics for unlabelled target-domain predictions.

These metrics assess plausibility of SOC predictions when ground-truth labels
are unavailable.  They are NOT accuracy metrics.

Do NOT present these as substitutes for MAE/RMSE.  Report them separately
as supplementary diagnostic information.
"""

from __future__ import annotations

import numpy as np


def soc_boundary_violation_rate(y_pred: np.ndarray) -> float:
    """Fraction of predictions outside [0, 1]."""
    y = np.asarray(y_pred).ravel()
    return float(np.mean((y < 0.0) | (y > 1.0)))


def monotonicity_violation_rate(
    y_pred: np.ndarray, expected_direction: str = "decreasing"
) -> float:
    """Fraction of consecutive steps violating expected SOC direction.

    Parameters
    ----------
    y_pred:
        SOC predictions for a discharge (or charge) cycle, in time order.
    expected_direction:
        ``"decreasing"`` for discharge, ``"increasing"`` for charge.
    """
    y = np.asarray(y_pred).ravel()
    if len(y) < 2:
        return 0.0
    delta = np.diff(y)
    if expected_direction == "decreasing":
        violations = delta > 0
    else:
        violations = delta < 0
    return float(np.mean(violations))


def prediction_jump_rate(y_pred: np.ndarray, threshold: float = 0.1) -> float:
    """Fraction of consecutive steps with SOC jump larger than *threshold*."""
    y = np.asarray(y_pred).ravel()
    if len(y) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(y)) > threshold))


def coulomb_consistency_error(
    y_pred: np.ndarray,
    current: np.ndarray,
    dt: np.ndarray,
    capacity_ah: float,
) -> float:
    """RMS error between coulomb-counting delta-SOC and predicted delta-SOC.

    Parameters
    ----------
    y_pred:
        Predicted SOC sequence.
    current:
        Current values (A; negative = discharge).
    dt:
        Time step durations (seconds).
    capacity_ah:
        Nominal capacity in Ah.
    """
    cc_delta = np.diff(current * dt / 3600.0 / capacity_ah)
    pred_delta = np.diff(np.asarray(y_pred).ravel())
    if len(cc_delta) != len(pred_delta):
        return float("nan")
    return float(np.sqrt(np.mean((pred_delta - cc_delta) ** 2)))
