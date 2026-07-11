"""Heuristic label quality scoring for coulomb-counted SOC references."""

from __future__ import annotations

import numpy as np
import pandas as pd

from battery_benchmark.data.schema import COLUMNS


def estimate_label_quality(
    df: pd.DataFrame,
    max_drift: float = 0.1,
) -> float:
    """Estimate quality of the SOC label in [0, 1].

    Heuristics:
    - SOC drift across a theoretically closed cycle (full discharge back to
      similar voltage) should be near zero.
    - Large drift indicates capacity uncertainty or rest imbalances.

    Parameters
    ----------
    df:
        Cycle DataFrame with COLUMNS.SOC column.
    max_drift:
        SOC drift above which quality is penalised.

    Returns
    -------
    float
        Quality score in [0, 1].  1.0 = high confidence.
    """
    if COLUMNS.SOC not in df.columns:
        return 0.0

    soc = df[COLUMNS.SOC].dropna().to_numpy()
    if len(soc) < 2:
        return 0.0

    soc_range = float(soc.max() - soc.min())
    if soc_range < 1e-6:
        return 0.0

    # Penalise if SOC jumps unexpectedly
    delta_soc = np.abs(np.diff(soc))
    max_jump = float(delta_soc.max())
    jump_penalty = min(max_jump / 0.5, 1.0)  # jumps > 0.5 get max penalty

    quality = max(0.0, 1.0 - jump_penalty)
    return round(quality, 4)
