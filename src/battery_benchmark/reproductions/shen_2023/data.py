"""Panasonic temperature protocol for the Shen et al. TATN reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from battery_benchmark.data.mat_loader import load_mat_file

TEMPERATURES = (25, 10, 0, -10, -20)
PROFILES = (
    "Cycle_1",
    "Cycle_2",
    "Cycle_3",
    "Cycle_4",
    "NN",
    "US06",
    "HWFET",
    "LA92",
    "UDDS",
)
TEMP_DIRS = {25: "25degC", 10: "10degC", 0: "0degC", -10: "-10degC", -20: "-20degC"}
WINDOW = 90
SAMPLE_INTERVAL = 10
TENSOR_LAYOUTS = ("paper", "released_repo")


@dataclass(frozen=True)
class ShenCycle:
    profile: str
    temperature: int
    features: np.ndarray
    soc: np.ndarray
    paths: tuple[Path, ...]


def _profile_paths(raw_root: Path, temperature: int, profile: str) -> tuple[Path, ...]:
    root = raw_root / TEMP_DIRS[temperature]
    candidates = sorted(root.rglob("*.mat"))
    if profile == "HWFET" and temperature == 25:
        paths = tuple(p for p in candidates if "_HWFTa_" in p.name or "_HWFTb_" in p.name)
    else:
        token = f"_{profile}_"
        paths = tuple(
            p
            for p in candidates
            if token in p.name
            and sum(f"_{name}_" in p.name for name in ("NN", "US06", "HWFET", "UDDS", "LA92"))
            <= 1
        )
    if len(paths) != (2 if profile == "HWFET" and temperature == 25 else 1):
        raise FileNotFoundError(
            f"Expected {'two' if profile == 'HWFET' and temperature == 25 else 'one'} "
            f"{temperature}C {profile} file(s) below {root}, found {[p.name for p in paths]}"
        )
    return paths


def _load_profile(paths: tuple[Path, ...]) -> tuple[np.ndarray, np.ndarray]:
    feature_parts, ah_parts = [], []
    for path in paths:
        measurements = load_mat_file(path).measurements
        required = ("Current", "Voltage", "Battery_Temp_degC", "Ah")
        missing = set(required) - set(measurements)
        if missing:
            raise ValueError(f"{path.name} missing fields: {sorted(missing)}")
        size = min(len(measurements[name]) for name in required)
        indices = np.arange(0, size, SAMPLE_INTERVAL)
        feature_parts.append(
            np.column_stack(
                [
                    np.asarray(measurements["Current"][:size], dtype=np.float64)[indices],
                    np.asarray(measurements["Voltage"][:size], dtype=np.float64)[indices],
                    np.asarray(measurements["Battery_Temp_degC"][:size], dtype=np.float64)[indices],
                ]
            )
        )
        ah_parts.append(np.asarray(measurements["Ah"][:size], dtype=np.float64)[indices])
    return np.concatenate(feature_parts), np.concatenate(ah_parts)


def load_temperature_cycles_and_range(
    raw_root: str | Path, temperature: int
) -> tuple[dict[str, ShenCycle], np.ndarray, np.ndarray]:
    """Load all profiles and return paper-normalized cycles plus feature limits.

    The paper explicitly fits min-max ranges on all nine profiles separately at
    each temperature. This is intentionally paper-faithful, not benchmark-fair.
    """

    if temperature not in TEMP_DIRS:
        raise ValueError(f"Unsupported temperature {temperature}; choose from {TEMPERATURES}")
    raw_root = Path(raw_root)
    raw = {
        profile: (*_load_profile(paths := _profile_paths(raw_root, temperature, profile)), paths)
        for profile in PROFILES
    }
    all_x = np.concatenate([raw[p][0] for p in PROFILES])
    all_ah = np.concatenate([raw[p][1] for p in PROFILES])
    x_min, x_max = all_x.min(axis=0), all_x.max(axis=0)
    ah_min, ah_max = all_ah.min(), all_ah.max()
    x_scale = np.where(x_max == x_min, 1.0, x_max - x_min)
    ah_scale = ah_max - ah_min or 1.0
    cycles = {
        profile: ShenCycle(
            profile=profile,
            temperature=temperature,
            features=((raw[profile][0] - x_min) / x_scale).astype(np.float32),
            soc=((raw[profile][1] - ah_min) / ah_scale).astype(np.float32),
            paths=raw[profile][2],
        )
        for profile in PROFILES
    }
    return cycles, x_min, x_max


def load_temperature_cycles(raw_root: str | Path, temperature: int) -> dict[str, ShenCycle]:
    """Load and paper-normalize all nine profiles at one temperature."""

    return load_temperature_cycles_and_range(raw_root, temperature)[0]


def _feature_windows(features: np.ndarray, tensor_layout: str) -> np.ndarray:
    if tensor_layout not in TENSOR_LAYOUTS:
        raise ValueError(f"tensor_layout must be one of {TENSOR_LAYOUTS}")
    usable = len(features) // WINDOW * WINDOW
    if tensor_layout == "paper":
        return features[:usable].reshape(-1, WINDOW, 3)
    return features[:usable].reshape(-1, 3, WINDOW).transpose(0, 2, 1)


def windows(
    cycles: list[ShenCycle], tensor_layout: str = "paper"
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for cycle in cycles:
        usable = len(cycle.soc) // WINDOW * WINDOW
        if usable:
            xs.append(_feature_windows(cycle.features[:usable], tensor_layout))
            ys.append(cycle.soc[:usable].reshape(-1, WINDOW))
    if not xs:
        return np.empty((0, WINDOW, 3), np.float32), np.empty((0, WINDOW), np.float32)
    return np.concatenate(xs), np.concatenate(ys)


def selected_windows(
    cycle: ShenCycle,
    pseudo_soc: np.ndarray,
    *,
    threshold: float = 0.08,
    lookahead: int = 200,
    tensor_layout: str = "paper",
) -> tuple[np.ndarray, np.ndarray]:
    """Select stable pseudo-label segments using Eqs. 11-12 and repo defaults."""

    n = min(len(cycle.soc), len(pseudo_soc))
    if n <= lookahead:
        return windows([])
    offsets = (lookahead // 3, 2 * lookahead // 3, lookahead)
    score = np.abs(
        sum(
            pseudo_soc[: n - lookahead] - pseudo_soc[offset : offset + n - lookahead]
            for offset in offsets
        )
        / 2
    )
    keep = np.ones(n - lookahead, dtype=bool)
    for index in np.flatnonzero(score > threshold):
        keep[max(0, index - lookahead // 10) : min(len(keep), index + lookahead)] = False

    # Preserve chronology and never create artificial windows across rejected gaps.
    boundaries = np.flatnonzero(np.diff(np.r_[False, keep, False]))
    xs, ys = [], []
    for start, stop in boundaries.reshape(-1, 2):
        jumps = np.flatnonzero(np.abs(np.diff(pseudo_soc[start:stop])) > threshold) + start + 1
        for left, right in zip(np.r_[start, jumps], np.r_[jumps, stop]):
            usable = (right - left) // WINDOW * WINDOW
            if usable:
                xs.append(_feature_windows(cycle.features[left : left + usable], tensor_layout))
                ys.append(pseudo_soc[left : left + usable].reshape(-1, WINDOW))
    if not xs:
        return windows([])
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)
