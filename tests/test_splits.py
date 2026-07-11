"""Tests for split strategies."""

import numpy as np
import pandas as pd
import pytest

from battery_benchmark.data.schema import COLUMNS, Split
from battery_benchmark.data.splits import split_cycles, filter_by_split


def _make_cycle(cycle_id: int, cell_id: str = "c1", profile: str = "US06") -> pd.DataFrame:
    n = 20
    return pd.DataFrame({
        COLUMNS.TIMESTAMP: np.arange(n, dtype=float),
        COLUMNS.VOLTAGE: np.linspace(4.2, 3.0, n),
        COLUMNS.CURRENT: np.full(n, -1.5),
        COLUMNS.TEMPERATURE: np.full(n, 25.0),
        COLUMNS.SOC: np.linspace(1.0, 0.0, n),
        COLUMNS.CELL_ID: cell_id,
        COLUMNS.CYCLE_ID: cycle_id,
        COLUMNS.DOMAIN_ID: "test",
        COLUMNS.PROFILE_NAME: profile,
        COLUMNS.SPLIT: Split.UNSPLIT.value,
    })


def _make_multi_cell_cycles(n_cells: int = 3, cycles_per_cell: int = 5) -> list:
    cycles = []
    for c in range(n_cells):
        for i in range(cycles_per_cell):
            cycles.append(_make_cycle(cycle_id=i, cell_id=f"cell_{c:02d}"))
    return cycles


class TestByCycleSplit:

    def test_split_labels_assigned(self):
        cycles = [_make_cycle(i) for i in range(20)]
        labelled = split_cycles(cycles, "by_cycle", train_ratio=0.7, val_ratio=0.15, seed=0)
        for df in labelled:
            assert df[COLUMNS.SPLIT].iloc[0] in ("train", "val", "test")

    def test_no_cycle_in_two_splits(self):
        cycles = [_make_cycle(i) for i in range(20)]
        labelled = split_cycles(cycles, "by_cycle", seed=42)
        # Each cycle appears exactly once
        assert len(labelled) == 20

    def test_approximate_ratios(self):
        n = 100
        cycles = [_make_cycle(i) for i in range(n)]
        labelled = split_cycles(cycles, "by_cycle", train_ratio=0.7, val_ratio=0.15, seed=0)
        n_train = sum(1 for df in labelled if df[COLUMNS.SPLIT].iloc[0] == "train")
        n_val = sum(1 for df in labelled if df[COLUMNS.SPLIT].iloc[0] == "val")
        n_test = sum(1 for df in labelled if df[COLUMNS.SPLIT].iloc[0] == "test")
        assert 60 <= n_train <= 75
        assert 10 <= n_val <= 20
        assert 10 <= n_test <= 20

    def test_deterministic_with_seed(self):
        cycles = [_make_cycle(i) for i in range(30)]
        a = [df[COLUMNS.SPLIT].iloc[0] for df in split_cycles(cycles, "by_cycle", seed=99)]
        b = [df[COLUMNS.SPLIT].iloc[0] for df in split_cycles(cycles, "by_cycle", seed=99)]
        assert a == b


class TestByCellSplit:

    def test_no_cell_in_two_splits(self):
        cycles = _make_multi_cell_cycles(n_cells=9, cycles_per_cell=4)
        labelled = split_cycles(cycles, "by_cell", seed=0)
        # For each cell, all its cycles must be in the same split
        from collections import defaultdict
        cell_splits = defaultdict(set)
        for df in labelled:
            cell = df[COLUMNS.CELL_ID].iloc[0]
            cell_splits[cell].add(df[COLUMNS.SPLIT].iloc[0])
        for cell, splits in cell_splits.items():
            assert len(splits) == 1, f"Cell {cell} appears in multiple splits: {splits}"


class TestByProfileSplit:

    def test_explicit_profiles(self):
        profiles = ["US06", "UDDS", "LA92", "NN"]
        cycles = [_make_cycle(i, profile=profiles[i % 4]) for i in range(20)]
        labelled = split_cycles(
            cycles, "by_profile",
            train_profiles=["US06", "UDDS"],
            val_profiles=["LA92"],
            test_profiles=["NN"],
        )
        for df in labelled:
            p = df[COLUMNS.PROFILE_NAME].iloc[0]
            s = df[COLUMNS.SPLIT].iloc[0]
            if p in ("US06", "UDDS"):
                assert s == "train"
            elif p == "LA92":
                assert s == "val"
            elif p == "NN":
                assert s == "test"


class TestFilterBySplit:

    def test_filter_returns_correct_split(self):
        cycles = [_make_cycle(i) for i in range(20)]
        labelled = split_cycles(cycles, "by_cycle", seed=0)
        train = filter_by_split(labelled, "train")
        for df in train:
            assert df[COLUMNS.SPLIT].iloc[0] == "train"

    def test_filter_empty_returns_empty(self):
        assert filter_by_split([], "train") == []


def test_unknown_strategy_raises():
    cycles = [_make_cycle(i) for i in range(5)]
    with pytest.raises(ValueError, match="Unknown split strategy"):
        split_cycles(cycles, "by_magic")
