"""Signal filtering utilities for battery time-series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def apply_savgol(
    df: pd.DataFrame,
    cols: list[str],
    window_length: int = 11,
    polyorder: int = 2,
) -> pd.DataFrame:
    """Apply Savitzky-Golay filter to specified columns."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        arr = df[col].to_numpy(dtype=float)
        if len(arr) > window_length:
            df[col] = savgol_filter(arr, window_length=window_length, polyorder=polyorder)
    return df


def clip_physical_bounds(
    df: pd.DataFrame,
    voltage_range: tuple[float, float] = (2.5, 4.3),
    current_range: tuple[float, float] = (-10.0, 10.0),
    temperature_range: tuple[float, float] = (-20.0, 80.0),
    soc_range: tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    """Clip measurements to physically plausible bounds."""
    from battery_benchmark.data.schema import COLUMNS

    df = df.copy()
    bounds = {
        COLUMNS.VOLTAGE: voltage_range,
        COLUMNS.CURRENT: current_range,
        COLUMNS.TEMPERATURE: temperature_range,
        COLUMNS.SOC: soc_range,
    }
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df
