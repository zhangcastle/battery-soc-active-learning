import pytest

from battery_benchmark.experiments.vidal_case2 import compare_to_case1, load_target_case1_baseline
from vidal_case1_test_utils import make_case1_run


def test_baseline_uses_target_case1_not_source(tmp_path):
    make_case1_run(tmp_path, "samsung_30t", formal=True)
    baseline = load_target_case1_baseline(tmp_path, "samsung_30t")
    comparison = compare_to_case1(
        {"rmse_percent": 2.4, "mae_percent": 1.5, "max_error_percent": 7.0},
        baseline,
    )
    assert comparison["target_case1_run_path"].endswith("samsung_30t/seed_42/run_001")
    assert comparison["rmse_delta_percent_points"] == pytest.approx(-0.6)
    assert comparison["rmse_relative_change_percent"] == pytest.approx(-20.0)
