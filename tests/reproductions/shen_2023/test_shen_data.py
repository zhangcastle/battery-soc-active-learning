"""Shared Shen (2023) data plumbing used by the TATN CMU benchmark: window
truncation and the pseudo-label stability selection (threshold 0.08,
lookahead 200). CMU-agnostic and label-free on the selection side."""

from __future__ import annotations

import numpy as np

from battery_benchmark.reproductions.shen_2023.data import (
    WINDOW,
    ShenCycle,
    selected_windows,
    windows,
)


def _cycle(n: int) -> ShenCycle:
    rng = np.random.default_rng(0)
    return ShenCycle(
        profile="m", temperature=25,
        features=rng.normal(size=(n, 3)).astype(np.float32),
        soc=np.linspace(0.9, 0.1, n).astype(np.float32), paths=(),
    )


def test_window_length_is_ninety_and_truncates_partial_window():
    assert WINDOW == 90
    x, y = windows([_cycle(90 * 3 + 17)])
    assert len(x) == len(y) == 3
    assert y.shape[1] == 90


def test_selected_windows_empty_when_cycle_not_longer_than_lookahead():
    cycle = _cycle(200)
    x, y = selected_windows(cycle, np.full(200, 0.5, np.float32))
    assert len(x) == 0 and len(y) == 0


def test_selected_windows_accepts_stable_and_rejects_jump_segments():
    cycle = _cycle(900)
    stable = np.full(900, 0.5, np.float32)
    x, y = selected_windows(cycle, stable)
    assert len(x) > 0 and np.allclose(y, 0.5)
    jumpy = stable.copy()
    jumpy[450:] += 0.5  # a step larger than the 0.08 threshold
    x2, _ = selected_windows(cycle, jumpy)
    assert 0 <= len(x2) < len(x)
