"""Cycle and mission segmentation from continuous battery data."""

from __future__ import annotations

import pandas as pd
import numpy as np

from battery_benchmark.data.schema import COLUMNS


def segment_by_cycle_id(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a combined DataFrame into per-cycle DataFrames by CYCLE_ID."""
    if COLUMNS.CYCLE_ID not in df.columns:
        raise ValueError(f"DataFrame must have column '{COLUMNS.CYCLE_ID}'.")
    return [group.reset_index(drop=True) for _, group in df.groupby(COLUMNS.CYCLE_ID)]


def segment_by_sign_change(
    df: pd.DataFrame,
    current_col: str = COLUMNS.CURRENT,
    min_samples: int = 10,
) -> list[pd.DataFrame]:
    """Segment into charge/discharge/rest phases by current sign.

    Parameters
    ----------
    df:
        Continuous recording.
    current_col:
        Column containing current values.
    min_samples:
        Minimum number of samples per segment to be kept.

    Returns
    -------
    list[pd.DataFrame]
        One DataFrame per charge/discharge/rest segment.
    """
    signs = np.sign(df[current_col].to_numpy())
    segments: list[pd.DataFrame] = []
    start = 0
    current_sign = signs[0]

    for i in range(1, len(signs)):
        if signs[i] != current_sign:
            seg = df.iloc[start:i].reset_index(drop=True)
            if len(seg) >= min_samples:
                segments.append(seg)
            start = i
            current_sign = signs[i]

    # Last segment
    seg = df.iloc[start:].reset_index(drop=True)
    if len(seg) >= min_samples:
        segments.append(seg)

    return segments
