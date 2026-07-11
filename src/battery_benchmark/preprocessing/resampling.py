"""Time-series resampling to a uniform sampling rate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from battery_benchmark.data.schema import COLUMNS


def resample_cycle(
    df: pd.DataFrame,
    target_hz: float,
    method: str = "linear",
) -> pd.DataFrame:
    """Resample a cycle DataFrame to *target_hz* Hz.

    Parameters
    ----------
    df:
        Cycle DataFrame with a ``COLUMNS.TIMESTAMP`` column (seconds).
    target_hz:
        Target sampling rate in Hz.
    method:
        Interpolation method (``"linear"``, ``"nearest"``, etc.).

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame.
    """
    if COLUMNS.TIMESTAMP not in df.columns:
        raise ValueError("Cycle DataFrame must have a 'timestamp' column for resampling.")

    t_start = df[COLUMNS.TIMESTAMP].iloc[0]
    t_end = df[COLUMNS.TIMESTAMP].iloc[-1]
    t_new = np.arange(t_start, t_end, 1.0 / target_hz)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_indexed = df.set_index(COLUMNS.TIMESTAMP)

    df_reindexed = df_indexed.reindex(
        df_indexed.index.union(t_new)
    ).sort_index()

    df_interp = df_reindexed[numeric_cols].interpolate(method=method)
    df_out = df_interp.loc[t_new].reset_index()
    df_out = df_out.rename(columns={"index": COLUMNS.TIMESTAMP})

    # Copy non-numeric columns (fill forward)
    for col in df.columns:
        if col not in numeric_cols and col != COLUMNS.TIMESTAMP:
            df_out[col] = df[col].iloc[0]

    return df_out
