"""Dynamic capacity estimation from discharge cycles.

Used to generate proxy SOC references for the CMU eVTOL dataset where
fixed nominal capacity is unreliable due to ageing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from battery_benchmark.data.schema import COLUMNS


def estimate_capacity_from_discharge(
    df: pd.DataFrame,
    soc_start: float = 1.0,
    soc_end: float = 0.0,
) -> float:
    """Estimate actual capacity (Ah) from a discharge cycle.

    Integrates negative current over time to infer the delivered capacity.

    Parameters
    ----------
    df:
        Discharge cycle DataFrame.
    soc_start:
        Assumed SOC at the beginning of the discharge (default 1.0).
    soc_end:
        Assumed SOC at the end of the discharge (default 0.0).

    Returns
    -------
    float
        Estimated capacity in Ah.
    """
    if COLUMNS.TIMESTAMP not in df.columns or COLUMNS.CURRENT not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' and 'current' columns.")

    t = df[COLUMNS.TIMESTAMP].to_numpy(dtype=float)
    I = df[COLUMNS.CURRENT].to_numpy(dtype=float)

    dt = np.diff(t, prepend=t[0])
    dt[0] = t[1] - t[0] if len(t) > 1 else 0.0

    discharge_current = np.where(I < 0, -I, 0.0)
    delivered_ah = float(np.trapz(discharge_current, t) / 3600.0)

    soc_range = abs(soc_start - soc_end)
    if soc_range == 0 or delivered_ah == 0:
        return float("nan")

    return delivered_ah / soc_range
