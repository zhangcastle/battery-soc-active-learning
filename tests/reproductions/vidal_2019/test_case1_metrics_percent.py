import numpy as np

from battery_benchmark.evaluation.metrics import compute_metrics


def test_fraction_to_percent_conversion():
    metric = compute_metrics(np.array([0.0, 1.0]), np.array([0.1, 0.9]))
    assert np.isclose(metric["mae"] * 100, 10.0)
    assert np.isclose(metric["rmse"] * 100, 10.0)
    assert np.isclose(metric["max_error"] * 100, 10.0)
