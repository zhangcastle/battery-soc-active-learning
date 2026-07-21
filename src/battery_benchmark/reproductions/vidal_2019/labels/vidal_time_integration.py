"""Non-uniform trapezoidal coulomb integration for Vidal sequence SOC v2."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


DEFAULT_MAX_DT_S = 120.0


@dataclass
class IntegrationResult:
    """Output of trapezoidal current integration."""

    effective_delta_ah: np.ndarray
    cumulative_charge_ah: np.ndarray
    cumulative_discharge_ah: np.ndarray
    cumulative_signed_ah: np.ndarray
    dt_s: np.ndarray
    issues: list[str] = field(default_factory=list)
    duplicate_time_count: int = 0
    negative_dt_count: int = 0
    large_gap_count: int = 0
    nan_time_count: int = 0
    nan_current_count: int = 0


def trapezoidal_coulomb_integration(
    time_s: np.ndarray,
    current_a: np.ndarray,
    *,
    positive_is_charge: bool = True,
    max_dt_s: float = DEFAULT_MAX_DT_S,
) -> IntegrationResult:
    """Integrate current over real ``time_s`` using trapezoidal rule.

    ``delta_ah_i = 0.5 * (I_i + I_{i-1}) * dt_i / 3600``

    Signed Ah follows adapter convention: positive current increases SOC when
    ``positive_is_charge`` is True.
    """
    t = np.asarray(time_s, dtype=float)
    i_arr = np.asarray(current_a, dtype=float)
    n = len(t)
    issues: list[str] = []

    if n == 0:
        empty = np.array([], dtype=float)
        return IntegrationResult(
            effective_delta_ah=empty,
            cumulative_charge_ah=empty,
            cumulative_discharge_ah=empty,
            cumulative_signed_ah=empty,
            dt_s=empty,
            issues=["empty_series"],
        )

    nan_time = int(np.isnan(t).sum())
    nan_current = int(np.isnan(i_arr).sum())
    if nan_time:
        issues.append(f"nan_time_count={nan_time}")
    if nan_current:
        issues.append(f"nan_current_count={nan_current}")

    dt = np.zeros(n, dtype=float)
    if n > 1:
        dt[1:] = t[1:] - t[:-1]

    dup = int(np.sum(dt[1:] == 0))
    neg = int(np.sum(dt[1:] < 0))
    large = int(np.sum(dt[1:] > max_dt_s))
    if dup:
        issues.append(f"duplicate_dt_count={dup}")
    if neg:
        issues.append(f"negative_dt_count={neg}")
    if large:
        issues.append(f"large_dt_gap_count={large}")

    safe_dt = dt.copy()
    safe_dt[safe_dt < 0] = 0.0

    i_prev = np.roll(i_arr, 1)
    i_prev[0] = i_arr[0]
    delta_ah = 0.5 * (i_arr + i_prev) * safe_dt / 3600.0

    if not positive_is_charge:
        delta_ah = -delta_ah

    charge_delta = np.where(delta_ah > 0, delta_ah, 0.0)
    discharge_delta = np.where(delta_ah < 0, -delta_ah, 0.0)
    cum_signed = np.cumsum(delta_ah)
    cum_charge = np.cumsum(charge_delta)
    cum_discharge = np.cumsum(discharge_delta)

    return IntegrationResult(
        effective_delta_ah=delta_ah,
        cumulative_charge_ah=cum_charge,
        cumulative_discharge_ah=cum_discharge,
        cumulative_signed_ah=cum_signed,
        dt_s=safe_dt,
        issues=issues,
        duplicate_time_count=dup,
        negative_dt_count=neg,
        large_gap_count=large,
        nan_time_count=nan_time,
        nan_current_count=nan_current,
    )


def apply_soc_clipping(
    soc: np.ndarray,
    *,
    low: float = 0.0,
    high: float = 1.0,
) -> tuple[np.ndarray, int, float]:
    """Clip SOC as final safety guard; return clip count and max excess."""
    clipped = np.clip(soc, low, high)
    below = soc < low
    above = soc > high
    count = int(below.sum() + above.sum())
    excess = 0.0
    if count:
        below_ex = float(np.max(low - soc[below])) if below.any() else 0.0
        above_ex = float(np.max(soc[above] - high)) if above.any() else 0.0
        excess = max(below_ex, above_ex)
    return clipped.astype(float), count, excess
