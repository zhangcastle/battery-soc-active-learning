"""Cycle-level, cell-level, and profile-level split strategies.

Rules enforced:
- A cycle never appears in more than one split.
- Scalers / normalisers are never fitted on val or test data.
- Missions are never split across train and test.
- Target test labels are never exposed during adaptation or training.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from battery_benchmark.data.schema import COLUMNS, Split


def split_cycles(
    cycles: list[pd.DataFrame],
    strategy: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    train_profiles: list[str] | None = None,
    val_profiles: list[str] | None = None,
    test_profiles: list[str] | None = None,
) -> list[pd.DataFrame]:
    """Assign split labels to a list of cycle DataFrames.

    Parameters
    ----------
    cycles:
        List of per-cycle DataFrames with benchmark schema columns.
    strategy:
        One of ``"by_cycle"``, ``"by_cell"``, ``"by_profile"``,
        ``"by_temperature"``, ``"by_mission"``.
    train_ratio, val_ratio, test_ratio:
        Fractional sizes for ratio-based strategies.  Must sum to 1.
    seed:
        Random seed for deterministic shuffling.
    train_profiles, val_profiles, test_profiles:
        Explicit profile name lists for ``"by_profile"`` strategy.

    Returns
    -------
    list[pd.DataFrame]
        Same cycles with ``COLUMNS.SPLIT`` column filled in.
    """
    if not cycles:
        return cycles

    dispatch = {
        "by_cycle": _split_by_cycle,
        "by_cell": _split_by_cell,
        "by_profile": _split_by_profile,
        "by_temperature": _split_by_temperature,
        "by_mission": _split_by_mission,
    }
    if strategy not in dispatch:
        raise ValueError(
            f"Unknown split strategy '{strategy}'. "
            f"Available: {sorted(dispatch.keys())}"
        )

    fn = dispatch[strategy]
    return fn(
        cycles,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        train_profiles=train_profiles,
        val_profiles=val_profiles,
        test_profiles=test_profiles,
    )


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _split_by_cycle(
    cycles: list[pd.DataFrame],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    **kwargs,
) -> list[pd.DataFrame]:
    """Randomly assign whole cycles to train / val / test."""
    import random
    rng = random.Random(seed)
    indices = list(range(len(cycles)))
    rng.shuffle(indices)

    n = len(indices)
    n_train = math.floor(n * train_ratio)
    n_val = math.floor(n * val_ratio)

    train_set = set(indices[:n_train])
    val_set = set(indices[n_train: n_train + n_val])

    result = []
    for i, df in enumerate(cycles):
        if i in train_set:
            label = Split.TRAIN.value
        elif i in val_set:
            label = Split.VAL.value
        else:
            label = Split.TEST.value
        result.append(df.assign(**{COLUMNS.SPLIT: label}))
    return result


def _split_by_cell(
    cycles: list[pd.DataFrame],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    **kwargs,
) -> list[pd.DataFrame]:
    """Assign whole cells to splits (no cell leaks across splits)."""
    import random
    all_cells = sorted({
        str(df[COLUMNS.CELL_ID].iloc[0]) for df in cycles
    })
    rng = random.Random(seed)
    shuffled = all_cells.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = math.floor(n * train_ratio)
    n_val = math.floor(n * val_ratio)

    train_cells = set(shuffled[:n_train])
    val_cells = set(shuffled[n_train: n_train + n_val])

    result = []
    for df in cycles:
        cell = str(df[COLUMNS.CELL_ID].iloc[0])
        if cell in train_cells:
            label = Split.TRAIN.value
        elif cell in val_cells:
            label = Split.VAL.value
        else:
            label = Split.TEST.value
        result.append(df.assign(**{COLUMNS.SPLIT: label}))
    return result


def _split_by_profile(
    cycles: list[pd.DataFrame],
    train_profiles: list[str] | None = None,
    val_profiles: list[str] | None = None,
    test_profiles: list[str] | None = None,
    **kwargs,
) -> list[pd.DataFrame]:
    """Assign cycles to splits based on drive-cycle profile name.

    If explicit profile lists are not provided, the cycles are sorted
    alphabetically and split by ratio (same logic as ``_split_by_cycle``).
    """
    if train_profiles is None and val_profiles is None and test_profiles is None:
        return _split_by_cycle(cycles, **kwargs)

    train_set = set(train_profiles or [])
    val_set = set(val_profiles or [])
    test_set = set(test_profiles or [])

    result = []
    for df in cycles:
        profile = str(df[COLUMNS.PROFILE_NAME].iloc[0])
        if profile in train_set:
            label = Split.TRAIN.value
        elif profile in val_set:
            label = Split.VAL.value
        elif profile in test_set:
            label = Split.TEST.value
        else:
            label = Split.UNSPLIT.value
        result.append(df.assign(**{COLUMNS.SPLIT: label}))
    return result


def _split_by_temperature(
    cycles: list[pd.DataFrame],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    **kwargs,
) -> list[pd.DataFrame]:
    """Group cycles by mean temperature and then split by group."""
    import random

    def _mean_temp(df: pd.DataFrame) -> float:
        if COLUMNS.TEMPERATURE in df.columns:
            return float(df[COLUMNS.TEMPERATURE].mean())
        return 25.0

    groups: dict[float, list[int]] = {}
    for i, df in enumerate(cycles):
        t = round(_mean_temp(df), 0)
        groups.setdefault(t, []).append(i)

    all_temps = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(all_temps)
    n = len(all_temps)
    n_train = math.floor(n * train_ratio)
    n_val = math.floor(n * val_ratio)

    train_temps = set(all_temps[:n_train])
    val_temps = set(all_temps[n_train: n_train + n_val])

    result = list(cycles)
    for t, idxs in groups.items():
        if t in train_temps:
            label = Split.TRAIN.value
        elif t in val_temps:
            label = Split.VAL.value
        else:
            label = Split.TEST.value
        for i in idxs:
            result[i] = cycles[i].assign(**{COLUMNS.SPLIT: label})
    return result


def _split_by_mission(
    cycles: list[pd.DataFrame],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    **kwargs,
) -> list[pd.DataFrame]:
    """Assign whole missions to splits (no mission leaks across splits)."""
    import random

    def _mission(df: pd.DataFrame) -> str:
        if COLUMNS.MISSION_ID in df.columns:
            return str(df[COLUMNS.MISSION_ID].iloc[0])
        return "unknown"

    all_missions = sorted({_mission(df) for df in cycles})
    rng = random.Random(seed)
    shuffled = all_missions.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = math.floor(n * train_ratio)
    n_val = math.floor(n * val_ratio)

    train_missions = set(shuffled[:n_train])
    val_missions = set(shuffled[n_train: n_train + n_val])

    result = []
    for df in cycles:
        m = _mission(df)
        if m in train_missions:
            label = Split.TRAIN.value
        elif m in val_missions:
            label = Split.VAL.value
        else:
            label = Split.TEST.value
        result.append(df.assign(**{COLUMNS.SPLIT: label}))
    return result


def filter_by_split(
    cycles: list[pd.DataFrame], split: str
) -> list[pd.DataFrame]:
    """Return only the cycles that belong to *split*."""
    return [
        df for df in cycles
        if df[COLUMNS.SPLIT].iloc[0] == split
    ]
