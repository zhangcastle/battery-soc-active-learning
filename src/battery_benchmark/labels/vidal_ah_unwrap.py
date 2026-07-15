"""Cycler Ah reset unwrap for Vidal sequence SOC v2."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


RESET_DROP_FRACTION = 0.5
NOISE_TOL_AH = 0.01


@dataclass
class AhUnwrapResult:
    cycler_ah_raw: np.ndarray
    cycler_ah_effective: np.ndarray
    reset_indices: list[int] = field(default_factory=list)
    unwrap_offsets: list[float] = field(default_factory=list)
    status: str = "PASS"
    reasons: list[str] = field(default_factory=list)


def detect_ah_resets(
    raw_ah: np.ndarray,
    *,
    adapter_reset_detected: bool = False,
) -> list[int]:
    """Detect indices where cycler Ah counter reset."""
    ah = np.asarray(raw_ah, dtype=float)
    resets: list[int] = []
    if len(ah) < 2:
        return resets

    peak = float(np.nanmax(ah[:1])) if len(ah) else 0.0
    for idx in range(1, len(ah)):
        prev = ah[idx - 1]
        cur = ah[idx]
        if np.isnan(prev) or np.isnan(cur):
            continue
        peak = max(peak, prev)
        drop = prev - cur
        if peak > NOISE_TOL_AH and drop > RESET_DROP_FRACTION * peak:
            if cur <= peak * RESET_DROP_FRACTION:
                resets.append(idx)
                peak = cur
        else:
            peak = max(peak, cur)

    if adapter_reset_detected and not resets and len(ah) >= 2:
        end_drop = ah[-2] - ah[-1]
        if end_drop > NOISE_TOL_AH and ah[-1] < ah[-2] * RESET_DROP_FRACTION:
            resets.append(len(ah) - 1)

    return resets


def unwrap_cycler_ah(
    raw_ah: np.ndarray,
    *,
    adapter_reset_detected: bool = False,
) -> AhUnwrapResult:
    """Build monotonic ``cycler_ah_effective`` preserving raw values."""
    raw = np.asarray(raw_ah, dtype=float).copy()
    resets = detect_ah_resets(raw, adapter_reset_detected=adapter_reset_detected)
    effective = np.empty_like(raw)
    offsets: list[float] = []
    cumulative_offset = 0.0

    reset_set = set(resets)
    previous_effective = float(raw[0]) if len(raw) else 0.0
    for idx, value in enumerate(raw):
        if idx == 0:
            effective[idx] = value
            continue
        if idx in reset_set:
            increment = previous_effective - value
            if increment > NOISE_TOL_AH:
                cumulative_offset += increment
                offsets.append(increment)
        effective[idx] = value + cumulative_offset
        previous_effective = effective[idx]

    reasons: list[str] = []
    status = "PASS"
    if resets:
        reasons.append(f"reset_indices={resets}")
        diffs = np.diff(effective)
        if np.any(diffs < -NOISE_TOL_AH):
            status = "FAIL"
            reasons.append("effective_ah_not_monotonic_after_unwrap")
    elif adapter_reset_detected:
        reasons.append("adapter_flagged_reset_but_no_index_detected")

    return AhUnwrapResult(
        cycler_ah_raw=raw,
        cycler_ah_effective=effective,
        reset_indices=resets,
        unwrap_offsets=offsets,
        status=status,
        reasons=reasons,
    )
