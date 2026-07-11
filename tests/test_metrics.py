"""Tests for evaluation metrics."""

import math

import numpy as np
import pytest

from battery_benchmark.evaluation.metrics import (
    bias,
    compute_metrics,
    mae,
    max_error,
    r_squared,
    rmse,
)


def test_mae_perfect():
    y = np.array([0.1, 0.5, 0.9])
    assert mae(y, y) == pytest.approx(0.0)


def test_mae_known():
    y_true = np.array([0.0, 0.5, 1.0])
    y_pred = np.array([0.1, 0.5, 0.9])
    assert mae(y_true, y_pred) == pytest.approx(np.mean([0.1, 0.0, 0.1]))


def test_rmse_known():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 1.0])
    assert rmse(y_true, y_pred) == pytest.approx(1.0)


def test_max_error_known():
    y_true = np.array([0.0, 0.5])
    y_pred = np.array([0.2, 0.5])
    assert max_error(y_true, y_pred) == pytest.approx(0.2)


def test_bias_sign():
    y_true = np.array([0.5, 0.5])
    y_pred = np.array([0.6, 0.6])
    assert bias(y_true, y_pred) > 0  # over-prediction


def test_r_squared_perfect():
    y = np.array([0.1, 0.5, 0.9])
    assert r_squared(y, y) == pytest.approx(1.0)


def test_r_squared_zero():
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.5, 0.5])  # constant prediction = mean
    assert r_squared(y_true, y_pred) == pytest.approx(0.0)


def test_compute_metrics_keys():
    y = np.array([0.1, 0.5, 0.9])
    metrics = compute_metrics(y, y)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "max_error" in metrics
    assert "bias" in metrics
    assert "r2" in metrics


def test_compute_metrics_with_prefix():
    y = np.ones(5) * 0.5
    metrics = compute_metrics(y, y, prefix="test_")
    assert "test_mae" in metrics


def test_metrics_with_nan():
    y_true = np.array([0.1, np.nan, 0.9])
    y_pred = np.array([0.1, 0.5, 0.9])
    m = mae(y_true, y_pred)
    assert not math.isnan(m)
    assert m == pytest.approx(0.0)  # only valid pairs used


def test_metrics_all_nan():
    y_true = np.array([np.nan, np.nan])
    y_pred = np.array([0.5, 0.5])
    assert math.isnan(mae(y_true, y_pred))
