"""Per-profile and per-cycle metric aggregation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from battery_benchmark.evaluation.metrics import compute_metrics


def compute_per_profile_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    profile_labels: np.ndarray,
) -> pd.DataFrame:
    """Compute metrics broken down by drive-cycle profile.

    Returns a DataFrame with one row per profile.
    """
    rows = []
    profiles = np.unique(profile_labels)
    for profile in profiles:
        mask = profile_labels == profile
        if not mask.any():
            continue
        m = compute_metrics(y_true[mask], y_pred[mask])
        m["profile"] = profile
        rows.append(m)
    return pd.DataFrame(rows).set_index("profile")


def compute_per_cell_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cell_labels: np.ndarray,
) -> pd.DataFrame:
    """Compute metrics broken down by cell_id."""
    rows = []
    for cell in np.unique(cell_labels):
        mask = cell_labels == cell
        m = compute_metrics(y_true[mask], y_pred[mask])
        m["cell_id"] = cell
        rows.append(m)
    return pd.DataFrame(rows).set_index("cell_id")
