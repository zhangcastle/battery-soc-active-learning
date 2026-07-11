"""Tests for coulomb counting SOC computation."""

import numpy as np
import pandas as pd
import pytest

from battery_benchmark.data.schema import COLUMNS
from battery_benchmark.labels.coulomb_counting import add_coulomb_soc, coulomb_count


def _discharge_cycle(capacity_ah: float = 2.5, n: int = 100) -> pd.DataFrame:
    """Simulate a full discharge from SOC=1 to SOC=0."""
    t = np.linspace(0, 3600.0, n)  # 1 hour
    I = np.full(n, -capacity_ah)   # discharge at 1C rate
    return pd.DataFrame({
        COLUMNS.TIMESTAMP: t,
        COLUMNS.CURRENT: I,
        COLUMNS.VOLTAGE: np.linspace(4.2, 3.0, n),
        COLUMNS.TEMPERATURE: np.full(n, 25.0),
    })


def test_coulomb_count_decreases_on_discharge():
    df = _discharge_cycle()
    soc = coulomb_count(df, capacity_ah=2.5, soc_initial=1.0)
    assert soc[0] == pytest.approx(1.0, abs=0.02)
    assert soc[-1] < 0.1  # should be near 0 after full discharge


def test_coulomb_count_clamped():
    df = _discharge_cycle()
    soc = coulomb_count(df, capacity_ah=2.5, soc_initial=1.0, clamp=True)
    assert soc.min() >= 0.0
    assert soc.max() <= 1.0


def test_coulomb_count_unclamped_can_go_negative():
    df = _discharge_cycle()
    soc = coulomb_count(df, capacity_ah=0.5, soc_initial=1.0, clamp=False)
    # With small capacity, SOC goes below 0 quickly
    assert soc.min() < 0.0


def test_add_coulomb_soc_adds_column():
    df = _discharge_cycle()
    df_out = add_coulomb_soc(df, capacity_ah=2.5)
    assert COLUMNS.SOC in df_out.columns
    assert COLUMNS.LABEL_AVAILABLE in df_out.columns
    assert df_out[COLUMNS.LABEL_AVAILABLE].all()


def test_add_coulomb_soc_raises_if_exists():
    df = _discharge_cycle()
    df[COLUMNS.SOC] = 0.5
    with pytest.raises(ValueError, match="overwrite"):
        add_coulomb_soc(df, capacity_ah=2.5)


def test_add_coulomb_soc_overwrite():
    df = _discharge_cycle()
    df[COLUMNS.SOC] = 0.5
    df_out = add_coulomb_soc(df, capacity_ah=2.5, overwrite=True)
    assert df_out[COLUMNS.SOC].iloc[0] != pytest.approx(0.5)


def test_missing_timestamp_raises():
    df = pd.DataFrame({COLUMNS.CURRENT: [-1.0, -1.0]})
    with pytest.raises(ValueError, match="timestamp"):
        coulomb_count(df, capacity_ah=2.5)


def test_missing_current_raises():
    df = pd.DataFrame({COLUMNS.TIMESTAMP: [0.0, 1.0]})
    with pytest.raises(ValueError, match="current"):
        coulomb_count(df, capacity_ah=2.5)
