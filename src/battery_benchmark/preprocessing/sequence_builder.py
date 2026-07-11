"""Sequence builder: convert lists of cycle DataFrames into (X, y) arrays.

Supported modes:

- ``full_cycle``: one window per cycle covering all samples
- ``fixed_non_overlapping``: non-overlapping windows of fixed length
- ``sliding_window``: overlapping windows (stride < seq_len)
- ``mission_phase``: one window per phase within a mission (future)

Each window is accompanied by metadata preserving cycle identity and split
provenance so that evaluation can be attributed to specific cycles/cells.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from battery_benchmark.data.schema import COLUMNS


class SequenceBuilder:
    """Converts cycle DataFrames into fixed-length (X, y) numpy arrays.

    Parameters
    ----------
    seq_len:
        Length of each output sequence.
    mode:
        Windowing mode.
    stride:
        Step size for ``sliding_window`` mode.  Defaults to ``seq_len``
        (non-overlapping) when None.
    feature_cols:
        Column names to use as input features.
    target_col:
        Column name for the regression target (SOC).
    """

    SUPPORTED_MODES = ("full_cycle", "fixed_non_overlapping", "sliding_window", "mission_phase")

    def __init__(
        self,
        seq_len: int = 50,
        mode: str = "fixed_non_overlapping",
        stride: int | None = None,
        feature_cols: list[str] | None = None,
        target_col: str = COLUMNS.SOC,
    ) -> None:
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unknown mode '{mode}'. Supported: {self.SUPPORTED_MODES}"
            )
        self.seq_len = seq_len
        self.mode = mode
        self.stride = stride if stride is not None else seq_len
        self.feature_cols = feature_cols or [
            COLUMNS.VOLTAGE,
            COLUMNS.CURRENT,
            COLUMNS.TEMPERATURE,
        ]
        self.target_col = target_col

    def build(
        self, cycles: list[pd.DataFrame]
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Build sequences from a list of cycle DataFrames.

        Parameters
        ----------
        cycles:
            List of DataFrames with benchmark schema columns.

        Returns
        -------
        X: np.ndarray of shape (N, seq_len, n_features)
        y: np.ndarray of shape (N, seq_len)
        metadata: list of N dicts with window provenance
        """
        all_X: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        all_meta: list[dict[str, Any]] = []

        for df in cycles:
            if len(df) == 0:
                continue
            X_windows, y_windows, meta_windows = self._build_cycle(df)
            all_X.extend(X_windows)
            all_y.extend(y_windows)
            all_meta.extend(meta_windows)

        if not all_X:
            n_feat = len(self.feature_cols)
            return (
                np.empty((0, self.seq_len, n_feat), dtype=np.float32),
                np.empty((0, self.seq_len), dtype=np.float32),
                [],
            )

        return (
            np.stack(all_X).astype(np.float32),
            np.stack(all_y).astype(np.float32),
            all_meta,
        )

    def _build_cycle(
        self, df: pd.DataFrame
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
        """Extract windows from one cycle."""
        if self.mode == "full_cycle":
            return self._full_cycle(df)
        if self.mode in ("fixed_non_overlapping", "sliding_window"):
            return self._windowed(df)
        if self.mode == "mission_phase":
            return self._mission_phase(df)
        raise ValueError(f"Unhandled mode: {self.mode}")

    def _full_cycle(
        self, df: pd.DataFrame
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
        """One window covering the entire cycle (padded/truncated to seq_len)."""
        X_arr = self._extract_features(df)
        y_arr = self._extract_target(df)
        n = len(X_arr)

        if n < self.seq_len:
            pad = self.seq_len - n
            X_arr = np.pad(X_arr, ((0, pad), (0, 0)), mode="edge")
            y_arr = np.pad(y_arr, (0, pad), mode="edge")
        else:
            X_arr = X_arr[: self.seq_len]
            y_arr = y_arr[: self.seq_len]

        meta = self._make_meta(df, start=0, end=n)
        return [X_arr], [y_arr], [meta]

    def _windowed(
        self, df: pd.DataFrame
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
        """Sliding or non-overlapping windows."""
        X_arr = self._extract_features(df)
        y_arr = self._extract_target(df)
        n = len(X_arr)

        xs, ys, metas = [], [], []
        start = 0
        while start + self.seq_len <= n:
            end = start + self.seq_len
            xs.append(X_arr[start:end])
            ys.append(y_arr[start:end])
            metas.append(self._make_meta(df, start=start, end=end))
            start += self.stride

        return xs, ys, metas

    def _mission_phase(
        self, df: pd.DataFrame
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
        """One window per mission phase (split by PHASE_NAME column)."""
        if COLUMNS.PHASE_NAME not in df.columns:
            return self._windowed(df)

        xs, ys, metas = [], [], []
        for phase, group in df.groupby(COLUMNS.PHASE_NAME, sort=False):
            sub_xs, sub_ys, sub_metas = self._windowed(group.reset_index(drop=True))
            xs.extend(sub_xs)
            ys.extend(sub_ys)
            metas.extend(sub_metas)
        return xs, ys, metas

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Feature columns {missing} not found in DataFrame. "
                f"Available: {list(df.columns)}"
            )
        return df[self.feature_cols].to_numpy(dtype=np.float32)

    def _extract_target(self, df: pd.DataFrame) -> np.ndarray:
        if self.target_col not in df.columns or df[self.target_col].isna().all():
            return np.full(len(df), float("nan"), dtype=np.float32)
        return df[self.target_col].to_numpy(dtype=np.float32)

    def _make_meta(self, df: pd.DataFrame, start: int, end: int) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "start_idx": start,
            "end_idx": end,
        }
        for col in (COLUMNS.CELL_ID, COLUMNS.CYCLE_ID, COLUMNS.DOMAIN_ID,
                    COLUMNS.SPLIT, COLUMNS.LABEL_AVAILABLE, COLUMNS.MISSION_ID):
            if col in df.columns:
                meta[col] = df[col].iloc[0]
        return meta
