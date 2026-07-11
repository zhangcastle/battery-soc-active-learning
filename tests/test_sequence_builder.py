"""Tests for SequenceBuilder."""

import numpy as np
import pandas as pd
import pytest

from battery_benchmark.data.schema import COLUMNS, Split
from battery_benchmark.preprocessing.sequence_builder import SequenceBuilder


def _make_cycle(n: int = 100, cell_id: str = "c1", cycle_id: int = 0) -> pd.DataFrame:
    return pd.DataFrame({
        COLUMNS.TIMESTAMP: np.arange(n, dtype=float),
        COLUMNS.VOLTAGE: np.linspace(4.2, 3.0, n),
        COLUMNS.CURRENT: np.full(n, -1.5),
        COLUMNS.TEMPERATURE: np.full(n, 25.0),
        COLUMNS.SOC: np.linspace(1.0, 0.0, n),
        COLUMNS.CELL_ID: cell_id,
        COLUMNS.CYCLE_ID: cycle_id,
        COLUMNS.DOMAIN_ID: "test",
        COLUMNS.PROFILE_NAME: "discharge",
        COLUMNS.SPLIT: Split.TRAIN.value,
    })


class TestSequenceBuilderFixed:

    def test_output_shapes(self):
        builder = SequenceBuilder(seq_len=20, mode="fixed_non_overlapping")
        cycle = _make_cycle(100)
        X, y, meta = builder.build([cycle])
        assert X.ndim == 3
        assert X.shape[1] == 20
        assert X.shape[2] == 3
        assert y.shape == (X.shape[0], 20)

    def test_number_of_windows(self):
        builder = SequenceBuilder(seq_len=25, mode="fixed_non_overlapping")
        X, y, _ = builder.build([_make_cycle(100)])
        assert X.shape[0] == 4  # 100 / 25 = 4

    def test_multiple_cycles(self):
        builder = SequenceBuilder(seq_len=20, mode="fixed_non_overlapping")
        cycles = [_make_cycle(100, cycle_id=i) for i in range(3)]
        X, y, meta = builder.build(cycles)
        assert X.shape[0] == 15  # 3 cycles × 5 windows each

    def test_metadata_has_cell_id(self):
        builder = SequenceBuilder(seq_len=50, mode="fixed_non_overlapping")
        X, y, meta = builder.build([_make_cycle(100)])
        assert meta[0][COLUMNS.CELL_ID] == "c1"

    def test_empty_cycles_returns_empty(self):
        builder = SequenceBuilder(seq_len=20, mode="fixed_non_overlapping")
        X, y, meta = builder.build([])
        assert X.shape == (0, 20, 3)
        assert y.shape == (0, 20)

    def test_cycle_shorter_than_seq_len_skipped(self):
        builder = SequenceBuilder(seq_len=50, mode="fixed_non_overlapping")
        X, y, _ = builder.build([_make_cycle(30)])
        assert X.shape[0] == 0

    def test_dtype_float32(self):
        builder = SequenceBuilder(seq_len=20, mode="fixed_non_overlapping")
        X, y, _ = builder.build([_make_cycle(100)])
        assert X.dtype == np.float32
        assert y.dtype == np.float32


class TestSequenceBuilderSliding:

    def test_sliding_window_count(self):
        builder = SequenceBuilder(seq_len=20, mode="sliding_window", stride=10)
        X, y, _ = builder.build([_make_cycle(100)])
        # windows: start at 0, 10, 20, ..., 80 → 9 windows
        assert X.shape[0] == 9


class TestSequenceBuilderFullCycle:

    def test_full_cycle_single_window(self):
        builder = SequenceBuilder(seq_len=100, mode="full_cycle")
        X, y, _ = builder.build([_make_cycle(100)])
        assert X.shape[0] == 1
        assert X.shape[1] == 100

    def test_full_cycle_truncation(self):
        builder = SequenceBuilder(seq_len=50, mode="full_cycle")
        X, y, _ = builder.build([_make_cycle(100)])
        assert X.shape[1] == 50


def test_missing_feature_column_raises():
    builder = SequenceBuilder(seq_len=10, feature_cols=["voltage", "current", "nonexistent"])
    with pytest.raises(ValueError, match="nonexistent"):
        builder.build([_make_cycle(50)])
