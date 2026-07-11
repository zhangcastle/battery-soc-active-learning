"""Coulomb counting SOC estimation.

Conventions:
- Current > 0 → charging (SOC increases)
- Current < 0 → discharging (SOC decreases)
- dt in seconds
- Capacity in Ah
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from battery_benchmark.data.schema import COLUMNS


def coulomb_count(
    df: pd.DataFrame,
    capacity_ah: float,
    soc_initial: float = 1.0,
    efficiency: float = 1.0,
    clamp: bool = True,
) -> np.ndarray:
    """Compute SOC via coulomb counting for one cycle.

    Parameters
    ----------
    df:
        Cycle DataFrame with ``COLUMNS.TIMESTAMP`` and ``COLUMNS.CURRENT``.
    capacity_ah:
        Nominal (or estimated) capacity in Ah.
    soc_initial:
        SOC at the start of the cycle (default 1.0 = fully charged).
    efficiency:
        Coulombic efficiency (≤ 1.0).  Applied to charge current only.
    clamp:
        If True, clamp SOC to [0, 1].

    Returns
    -------
    np.ndarray
        SOC array of shape ``(len(df),)``.
    """
    if COLUMNS.TIMESTAMP not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column.")
    if COLUMNS.CURRENT not in df.columns:
        raise ValueError("DataFrame must have a 'current' column.")

    t = df[COLUMNS.TIMESTAMP].to_numpy(dtype=float)
    I = df[COLUMNS.CURRENT].to_numpy(dtype=float)

    dt = np.diff(t, prepend=t[0])  # seconds; first dt = 0
    dt[0] = t[1] - t[0] if len(t) > 1 else 0.0

    # Apply charging efficiency
    I_eff = np.where(I > 0, I * efficiency, I)

    # Cumulative charge in Ah
    delta_soc = (I_eff * dt) / 3600.0 / capacity_ah

    soc = soc_initial + np.cumsum(delta_soc)

    if clamp:
        soc = np.clip(soc, 0.0, 1.0)

    return soc.astype(np.float32)


def add_coulomb_soc(
    df: pd.DataFrame,
    capacity_ah: float,
    soc_initial: float = 1.0,
    efficiency: float = 1.0,
    clamp: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Add or overwrite the SOC column in *df* using coulomb counting."""
    if COLUMNS.SOC in df.columns and not overwrite:
        raise ValueError(
            "DataFrame already has an SOC column. "
            "Pass overwrite=True to replace it."
        )
    df = df.copy()
    df[COLUMNS.SOC] = coulomb_count(
        df, capacity_ah=capacity_ah, soc_initial=soc_initial,
        efficiency=efficiency, clamp=clamp,
    )
    df[COLUMNS.LABEL_AVAILABLE] = True
    df[COLUMNS.LABEL_SOURCE] = "coulomb_counted_fixed_capacity"
    return df
