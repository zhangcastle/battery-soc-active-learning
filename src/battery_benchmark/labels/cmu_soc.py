"""Development-only constant-capacity SOC reference for CMU candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd


LABEL_SOURCE = "cmu_coulomb_counted_constant_capacity_reference_v0"


def integrate_cmu_candidate_soc(
    rows: pd.DataFrame,
    *,
    capacity_ah: float = 3.0,
    soc_initial: float = 1.0,
) -> pd.DataFrame:
    """Integrate one candidate independently using trapezoidal current integration."""
    if not np.isfinite(capacity_ah) or capacity_ah <= 0:
        raise ValueError("capacity_ah must be finite and positive")
    if rows.empty:
        raise ValueError("candidate rows must not be empty")

    time_s = rows["time_s"].to_numpy(dtype=float)
    if not np.isfinite(time_s).all():
        raise ValueError("candidate time_s must be finite")
    if len(time_s) > 1 and not (np.diff(time_s) > 0).all():
        raise ValueError("candidate time_s must be strictly increasing")

    current_a = rows["I_mA"].to_numpy(dtype=float) / 1000.0
    if not np.isfinite(current_a).all():
        raise ValueError("candidate I_mA must be finite")
    delta_ah = np.zeros(len(rows), dtype=float)
    if len(rows) > 1:
        delta_ah[1:] = (
            0.5 * (current_a[:-1] + current_a[1:]) * np.diff(time_s) / 3600.0
        )
    soc = soc_initial + np.cumsum(delta_ah) / capacity_ah

    result = rows.copy()
    result["capacity_mode"] = "constant"
    result["capacity_ah"] = float(capacity_ah)
    result["soc_initial"] = float(soc_initial)
    result["soc_raw"] = soc
    result["soc_reference_v0"] = soc
    result["label_source"] = LABEL_SOURCE
    result["label_is_ground_truth"] = False
    result["formal_benchmark_eligible"] = False
    result["qc_flags"] = np.where((soc < 0) | (soc > 1), "soc_out_of_bounds", "")
    return result
