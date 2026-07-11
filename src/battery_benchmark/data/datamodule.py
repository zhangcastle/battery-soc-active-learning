"""DataModule: convenience wrapper that assembles train/val/test DataLoaders."""

from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from battery_benchmark.data.cycle_dataset import CycleDataset
from battery_benchmark.data.schema import COLUMNS, Split
from battery_benchmark.data.splits import filter_by_split
from battery_benchmark.preprocessing.normalization import MinMaxNormalizer, StandardNormalizer
from battery_benchmark.preprocessing.sequence_builder import SequenceBuilder


class BatteryDataModule:
    """Assembles DataLoaders from a list of labelled or unlabelled cycles.

    Parameters
    ----------
    cycles:
        All cycles with split labels already assigned (``COLUMNS.SPLIT``).
    cfg:
        Experiment config dict.  Must contain ``preprocessing``, ``sequence``,
        ``batch_size``, and ``label_availability`` sections.
    """

    def __init__(self, cycles: list, cfg: dict[str, Any]) -> None:
        self.cycles = cycles
        self.cfg = cfg
        self.normalizer = None
        self._train_ds: CycleDataset | None = None
        self._val_ds: CycleDataset | None = None
        self._test_ds: CycleDataset | None = None

    def setup(self) -> None:
        """Build sequences, fit normaliser on train set, create Datasets."""
        pre_cfg = self.cfg.get("preprocessing", {})
        seq_cfg = self.cfg.get("sequence", {})
        feature_cols = pre_cfg.get("feature_cols", ["voltage", "current", "temperature"])
        target_col = COLUMNS.SOC

        builder = SequenceBuilder(
            seq_len=seq_cfg.get("seq_len", 50),
            mode=seq_cfg.get("mode", "fixed_non_overlapping"),
            stride=seq_cfg.get("stride", None),
            feature_cols=feature_cols,
            target_col=target_col,
        )

        train_cycles = filter_by_split(self.cycles, Split.TRAIN.value)
        val_cycles = filter_by_split(self.cycles, Split.VAL.value)
        test_cycles = filter_by_split(self.cycles, Split.TEST.value)

        X_train, y_train, meta_train = builder.build(train_cycles)
        X_val, y_val, meta_val = builder.build(val_cycles)
        X_test, y_test, meta_test = builder.build(test_cycles)

        if pre_cfg.get("normalize", True):
            norm_name = pre_cfg.get("normalizer", "minmax")
            self.normalizer = (
                MinMaxNormalizer() if norm_name == "minmax" else StandardNormalizer()
            )
            # Fit only on training data
            self.normalizer.fit(X_train)
            X_train = self.normalizer.transform(X_train)
            if X_val.size > 0:
                X_val = self.normalizer.transform(X_val)
            if X_test.size > 0:
                X_test = self.normalizer.transform(X_test)

        self._train_ds = CycleDataset(X_train, y_train, meta_train)
        self._val_ds = CycleDataset(X_val, y_val if y_val.size > 0 else None, meta_val)
        self._test_ds = CycleDataset(X_test, y_test if y_test.size > 0 else None, meta_test)

    def train_dataloader(self, batch_size: int | None = None) -> DataLoader:
        bs = batch_size or self.cfg.get("batch_size", 32)
        return DataLoader(self._train_ds, batch_size=bs, shuffle=True, drop_last=False)

    def val_dataloader(self, batch_size: int | None = None) -> DataLoader:
        bs = batch_size or self.cfg.get("batch_size", 32)
        return DataLoader(self._val_ds, batch_size=bs, shuffle=False)

    def test_dataloader(self, batch_size: int | None = None) -> DataLoader:
        bs = batch_size or self.cfg.get("batch_size", 32)
        return DataLoader(self._test_ds, batch_size=bs, shuffle=False)

    @property
    def n_features(self) -> int:
        if self._train_ds is None:
            raise RuntimeError("Call setup() first.")
        return self._train_ds.n_features

    @property
    def n_train(self) -> int:
        return len(self._train_ds) if self._train_ds else 0

    @property
    def n_val(self) -> int:
        return len(self._val_ds) if self._val_ds else 0

    @property
    def n_test(self) -> int:
        return len(self._test_ds) if self._test_ds else 0
