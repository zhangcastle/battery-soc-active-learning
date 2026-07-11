"""Feature normalisation for battery time-series.

Rules:
- Normalisers are always fitted on training data only.
- Normaliser state is serialisable for checkpoint storage.
- Normalisers operate on arrays of shape (N, L, F) or (N, F).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseNormalizer(ABC):
    """Abstract base for feature normalisers."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseNormalizer":
        """Fit statistics from *X* (training data only)."""

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalisation to *X*."""

    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Undo normalisation."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return serialisable state."""

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state from a dict."""

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit then transform in one call."""
        return self.fit(X).transform(X)

    def _flatten(self, X: np.ndarray) -> tuple[np.ndarray, tuple]:
        """Flatten 3-D (N, L, F) → 2-D (N*L, F) for statistics."""
        shape = X.shape
        if X.ndim == 3:
            return X.reshape(-1, shape[-1]), shape
        if X.ndim == 2:
            return X, shape
        raise ValueError(f"Expected 2-D or 3-D array, got shape {shape}")

    def _restore(self, X_flat: np.ndarray, original_shape: tuple) -> np.ndarray:
        if len(original_shape) == 3:
            return X_flat.reshape(original_shape)
        return X_flat


class MinMaxNormalizer(BaseNormalizer):
    """Per-feature min-max scaling to [0, 1].

    Parameters
    ----------
    feature_range:
        Output range ``(min, max)``.  Default ``(0.0, 1.0)``.
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        self.feature_range = feature_range
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxNormalizer":
        X_flat, _ = self._flatten(X)
        self._min = X_flat.min(axis=0)
        self._max = X_flat.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._min is None:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")
        X_flat, shape = self._flatten(X)
        lo, hi = self.feature_range
        scale = self._max - self._min
        # Avoid division by zero for constant features
        scale = np.where(scale == 0, 1.0, scale)
        X_norm = (X_flat - self._min) / scale * (hi - lo) + lo
        return self._restore(X_norm, shape)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self._min is None:
            raise RuntimeError("Normalizer has not been fitted.")
        X_flat, shape = self._flatten(X)
        lo, hi = self.feature_range
        scale = self._max - self._min
        scale = np.where(scale == 0, 1.0, scale)
        X_orig = (X_flat - lo) / (hi - lo) * scale + self._min
        return self._restore(X_orig, shape)

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "minmax",
            "feature_range": self.feature_range,
            "min": self._min.tolist() if self._min is not None else None,
            "max": self._max.tolist() if self._max is not None else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.feature_range = tuple(state["feature_range"])
        self._min = np.array(state["min"]) if state["min"] is not None else None
        self._max = np.array(state["max"]) if state["max"] is not None else None


class StandardNormalizer(BaseNormalizer):
    """Per-feature standardisation to zero mean and unit variance."""

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardNormalizer":
        X_flat, _ = self._flatten(X)
        self._mean = X_flat.mean(axis=0)
        self._std = X_flat.std(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("Normalizer has not been fitted.")
        X_flat, shape = self._flatten(X)
        std = np.where(self._std == 0, 1.0, self._std)
        return self._restore((X_flat - self._mean) / std, shape)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("Normalizer has not been fitted.")
        X_flat, shape = self._flatten(X)
        std = np.where(self._std == 0, 1.0, self._std)
        return self._restore(X_flat * std + self._mean, shape)

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "standard",
            "mean": self._mean.tolist() if self._mean is not None else None,
            "std": self._std.tolist() if self._std is not None else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._mean = np.array(state["mean"]) if state["mean"] is not None else None
        self._std = np.array(state["std"]) if state["std"] is not None else None
