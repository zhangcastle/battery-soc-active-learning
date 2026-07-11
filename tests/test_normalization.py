"""Tests for MinMaxNormalizer and StandardNormalizer."""

import numpy as np
import pytest

from battery_benchmark.preprocessing.normalization import MinMaxNormalizer, StandardNormalizer


def _make_data(shape=(20, 10, 3), seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 10, size=shape).astype(np.float32)


class TestMinMaxNormalizer:

    def test_fit_transform_range(self):
        X = _make_data()
        norm = MinMaxNormalizer()
        X_norm = norm.fit_transform(X)
        assert X_norm.shape == X.shape
        assert float(X_norm.min()) >= -1e-6
        assert float(X_norm.max()) <= 1.0 + 1e-6

    def test_inverse_transform_roundtrip(self):
        X = _make_data()
        norm = MinMaxNormalizer()
        X_norm = norm.fit_transform(X)
        X_rec = norm.inverse_transform(X_norm)
        np.testing.assert_allclose(X, X_rec, atol=1e-5)

    def test_not_fitted_raises(self):
        norm = MinMaxNormalizer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            norm.transform(np.zeros((5, 3)))

    def test_state_dict_roundtrip(self):
        X = _make_data()
        norm = MinMaxNormalizer()
        norm.fit(X)
        state = norm.state_dict()
        norm2 = MinMaxNormalizer()
        norm2.load_state_dict(state)
        np.testing.assert_allclose(norm.transform(X), norm2.transform(X))

    def test_2d_input(self):
        X = _make_data(shape=(100, 3))
        norm = MinMaxNormalizer()
        X_norm = norm.fit_transform(X)
        assert X_norm.shape == (100, 3)

    def test_constant_feature_no_divide_by_zero(self):
        X = np.ones((10, 5, 3), dtype=np.float32)
        norm = MinMaxNormalizer()
        X_norm = norm.fit_transform(X)
        assert not np.isnan(X_norm).any()


class TestStandardNormalizer:

    def test_fit_transform_statistics(self):
        rng = np.random.default_rng(42)
        X = rng.normal(5.0, 2.0, size=(200, 3)).astype(np.float32)
        norm = StandardNormalizer()
        X_norm = norm.fit_transform(X)
        assert abs(X_norm.mean()) < 0.1
        assert abs(X_norm.std() - 1.0) < 0.1

    def test_inverse_transform_roundtrip(self):
        X = _make_data(shape=(50, 3))
        norm = StandardNormalizer()
        X_norm = norm.fit_transform(X)
        X_rec = norm.inverse_transform(X_norm)
        np.testing.assert_allclose(X, X_rec, atol=1e-4)

    def test_state_dict_roundtrip(self):
        X = _make_data(shape=(30, 3))
        norm = StandardNormalizer()
        norm.fit(X)
        state = norm.state_dict()
        norm2 = StandardNormalizer()
        norm2.load_state_dict(state)
        np.testing.assert_allclose(norm.transform(X), norm2.transform(X))
