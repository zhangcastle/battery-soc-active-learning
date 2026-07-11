"""Supervised and operational evaluation metrics for SOC estimation.

Supervised metrics require ground-truth SOC labels.
Physical-consistency metrics can be computed without labels.

IMPORTANT: Physical-consistency metrics (boundary violation rate,
monotonicity violations, etc.) are NOT substitutes for SOC accuracy.
They measure internal plausibility, not prediction accuracy.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum Absolute Error."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        return float("nan")
    return float(np.max(np.abs(y_true[mask] - y_pred[mask])))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error (positive = over-prediction)."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        return float("nan")
    return float(np.mean(y_pred[mask] - y_true[mask]))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        return float("nan")
    yt, yp = y_true[mask], y_pred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """Compute all supervised SOC metrics.

    Parameters
    ----------
    y_true:
        Ground-truth SOC values (flat array).
    y_pred:
        Predicted SOC values (same shape).
    prefix:
        Optional string prefix for metric keys (e.g. ``"val_"``).

    Returns
    -------
    dict[str, float]
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    p = prefix
    return {
        f"{p}mae": mae(y_true, y_pred),
        f"{p}rmse": rmse(y_true, y_pred),
        f"{p}max_error": max_error(y_true, y_pred),
        f"{p}bias": bias(y_true, y_pred),
        f"{p}r2": r_squared(y_true, y_pred),
    }


def compute_operational_metrics(
    model: Any,
    n_samples: int = 100,
) -> dict[str, Any]:
    """Compute parameter count and estimated inference characteristics."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"n_parameters": n_params}
