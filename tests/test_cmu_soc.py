from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_benchmark.labels.cmu_soc import integrate_cmu_candidate_soc


def test_trapezoidal_soc_units_bounds_and_candidate_reset() -> None:
    rows = pd.DataFrame({"time_s": [0.0, 3600.0, 7200.0], "I_mA": [0.0, -1000.0, -1000.0]})
    result = integrate_cmu_candidate_soc(rows, capacity_ah=2.0)
    np.testing.assert_allclose(result["soc_raw"], [1.0, 0.75, 0.25])
    assert result["soc_reference_v0"].equals(result["soc_raw"])
    assert result["soc_raw"].iloc[0] == 1.0
    assert integrate_cmu_candidate_soc(rows.iloc[:1], capacity_ah=2.0)["soc_raw"].iloc[0] == 1.0

    reset = integrate_cmu_candidate_soc(rows.iloc[1:].reset_index(drop=True), capacity_ah=2.0)
    assert reset["soc_raw"].iloc[0] == 1.0
    out = integrate_cmu_candidate_soc(
        pd.DataFrame({"time_s": [0.0, 7200.0], "I_mA": [-2000.0, -2000.0]}),
        capacity_ah=1.0,
    )
    assert out["soc_raw"].iloc[-1] == -3.0
    assert out["qc_flags"].iloc[-1] == "soc_out_of_bounds"


@pytest.mark.parametrize("capacity", [0.0, -1.0, np.nan])
def test_invalid_capacity_fails(capacity: float) -> None:
    with pytest.raises(ValueError, match="capacity_ah"):
        integrate_cmu_candidate_soc(pd.DataFrame({"time_s": [0.0], "I_mA": [0.0]}), capacity_ah=capacity)


@pytest.mark.parametrize("times", [[0.0, 0.0], [1.0, 0.0], [0.0, np.nan]])
def test_invalid_candidate_time_fails(times: list[float]) -> None:
    with pytest.raises(ValueError, match="time_s"):
        integrate_cmu_candidate_soc(pd.DataFrame({"time_s": times, "I_mA": [-1000.0, -1000.0]}))
