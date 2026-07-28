"""Record-safe 40-step windows for Liu 2025."""

from __future__ import annotations

import numpy as np

from battery_benchmark.reproductions.vidal_2019.sequence_v2 import VidalV2Record


def build_liu_windows(
    records: list[VidalV2Record],
    *,
    length: int = 40,
    require_labels: bool,
    feature_order: tuple[str, str, str] = ("voltage", "current", "temperature"),
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build stride-one windows without crossing record boundaries."""
    feature_windows: list[np.ndarray] = []
    targets: list[float] = []
    for record in records:
        values = record.data[list(feature_order)].to_numpy(np.float32)
        for start in range(len(values) - length + 1):
            feature_windows.append(values[start : start + length])
            if require_labels:
                targets.append(float(record.data["soc"].iloc[start + length - 1]))
    if not feature_windows:
        raise ValueError("No complete windows were produced")
    x = np.stack(feature_windows)
    if not require_labels:
        return x, None
    y = np.asarray(targets, dtype=np.float32)
    if not np.isfinite(y).all():
        raise ValueError("Labelled split contains non-finite SOC")
    return x, y
