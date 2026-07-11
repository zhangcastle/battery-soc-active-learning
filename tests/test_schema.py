"""Tests for data schema validation."""

import numpy as np
import pandas as pd
import pytest

from battery_benchmark.data.schema import (
    COLUMNS,
    OPTIONAL_COLS,
    REQUIRED_COLS,
    LabelSource,
    Split,
    validate_cycle,
)


def _make_minimal_df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        COLUMNS.TIMESTAMP: np.arange(n, dtype=float),
        COLUMNS.VOLTAGE: np.linspace(4.2, 3.0, n),
        COLUMNS.CURRENT: np.full(n, -1.5),
        COLUMNS.TEMPERATURE: np.full(n, 25.0),
        COLUMNS.CELL_ID: "cell_001",
        COLUMNS.CYCLE_ID: 0,
        COLUMNS.DOMAIN_ID: "test",
        COLUMNS.PROFILE_NAME: "test_profile",
        COLUMNS.SPLIT: Split.TRAIN.value,
    })


def test_required_cols_present():
    assert len(REQUIRED_COLS) > 0
    assert COLUMNS.VOLTAGE in REQUIRED_COLS
    assert COLUMNS.CURRENT in REQUIRED_COLS
    assert COLUMNS.TEMPERATURE in REQUIRED_COLS


def test_validate_cycle_passes():
    df = _make_minimal_df()
    result = validate_cycle(df)
    assert isinstance(result, pd.DataFrame)
    # Optional columns should be added as NaN
    for col in OPTIONAL_COLS:
        assert col in result.columns


def test_validate_cycle_fails_missing_required():
    df = _make_minimal_df().drop(columns=[COLUMNS.VOLTAGE])
    with pytest.raises(ValueError, match="voltage"):
        validate_cycle(df, raise_on_missing=True)


def test_validate_cycle_warns_missing_required():
    df = _make_minimal_df().drop(columns=[COLUMNS.VOLTAGE])
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_cycle(df, raise_on_missing=False)
        assert len(w) == 1


def test_label_source_enum():
    assert LabelSource.COULOMB_COUNTED_FIXED_CAPACITY.value == "coulomb_counted_fixed_capacity"
    assert LabelSource.UNAVAILABLE.value == "unavailable"


def test_split_enum():
    assert Split.TRAIN.value == "train"
    assert Split.TEST.value == "test"
