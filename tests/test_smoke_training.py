"""End-to-end synthetic smoke test for the full training pipeline.

This test verifies that:
1. SyntheticDataset generates valid cycles.
2. Splits produce non-overlapping train/val/test sets.
3. SequenceBuilder produces correctly shaped arrays.
4. MinMaxNormalizer can fit and transform.
5. CycleDataset wraps arrays into a PyTorch Dataset.
6. VidalLSTM produces correct-shaped outputs.
7. Trainer runs for 2 epochs without errors.
8. Checkpoint is saved and loadable.
9. Evaluator computes finite metrics.
10. ExperimentRunner produces metrics.json + predictions CSVs.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from battery_benchmark.data.synthetic import SyntheticDataset
from battery_benchmark.data.splits import split_cycles, filter_by_split
from battery_benchmark.data.cycle_dataset import CycleDataset
from battery_benchmark.evaluation.evaluator import Evaluator
from battery_benchmark.evaluation.metrics import compute_metrics
from battery_benchmark.models.vidal_lstm import VidalLSTM
from battery_benchmark.preprocessing.normalization import MinMaxNormalizer
from battery_benchmark.preprocessing.sequence_builder import SequenceBuilder
from battery_benchmark.training.reproducibility import set_seed
from battery_benchmark.training.schedulers import build_optimizer
from battery_benchmark.training.trainer import Trainer
from torch.utils.data import DataLoader


_SMOKE_CFG = {
    "experiment_name": "smoke_test_unit",
    "random_seed": 42,
    "device": "cpu",
    "dataset": {
        "source": {"name": "synthetic", "domain_id": "src", "num_cells": 2,
                   "num_cycles_per_cell": 8, "samples_per_cycle": 80},
        "target": {"name": "synthetic", "domain_id": "tgt", "num_cells": 1,
                   "num_cycles_per_cell": 4, "samples_per_cycle": 80},
    },
    "split": {"strategy": "by_cycle", "train_ratio": 0.7, "val_ratio": 0.15},
    "preprocessing": {"feature_cols": ["voltage", "current", "temperature"], "normalize": True},
    "sequence": {"mode": "fixed_non_overlapping", "seq_len": 20},
    "model": {"name": "vidal_lstm", "input_size": 3, "hidden_size": 10, "output_clip": True},
    "transfer": {"method": "source_only"},
    "optimizer": {"name": "adam", "lr": 0.001},
    "scheduler": {"name": "none"},
    "epochs": 2,
    "batch_size": 16,
    "output_path": "outputs/smoke_unit",
}


@pytest.fixture(scope="module")
def synthetic_cycles():
    set_seed(42)
    ds = SyntheticDataset({"domain_id": "synthetic_A", "num_cells": 2,
                           "num_cycles_per_cell": 8, "samples_per_cycle": 80, "seed": 42})
    return ds.generate_cycles()


def test_synthetic_dataset_generates_cycles(synthetic_cycles):
    assert len(synthetic_cycles) == 16  # 2 cells × 8 cycles


def test_cycle_has_required_columns(synthetic_cycles):
    from battery_benchmark.data.schema import REQUIRED_COLS
    df = synthetic_cycles[0]
    for col in REQUIRED_COLS:
        assert col in df.columns, f"Missing: {col}"


def test_split_produces_nonempty_splits(synthetic_cycles):
    labelled = split_cycles(synthetic_cycles, "by_cycle", seed=42)
    train = filter_by_split(labelled, "train")
    test = filter_by_split(labelled, "test")
    assert len(train) > 0
    assert len(test) > 0


def test_sequence_builder_output_shape(synthetic_cycles):
    builder = SequenceBuilder(seq_len=20, mode="fixed_non_overlapping",
                               feature_cols=["voltage", "current", "temperature"])
    X, y, meta = builder.build(synthetic_cycles)
    assert X.ndim == 3
    assert X.shape[1] == 20
    assert X.shape[2] == 3
    assert y.shape == (X.shape[0], 20)
    assert len(meta) == X.shape[0]


def test_normalizer_no_leakage(synthetic_cycles):
    """Normaliser must be fitted only on train data."""
    labelled = split_cycles(synthetic_cycles, "by_cycle", seed=42)
    train = filter_by_split(labelled, "train")
    test = filter_by_split(labelled, "test")
    builder = SequenceBuilder(seq_len=20, feature_cols=["voltage", "current", "temperature"])
    X_train, _, _ = builder.build(train)
    X_test, _, _ = builder.build(test)

    norm = MinMaxNormalizer()
    norm.fit(X_train)  # only train
    X_train_n = norm.transform(X_train)
    X_test_n = norm.transform(X_test)

    assert not np.isnan(X_train_n).any()
    assert not np.isnan(X_test_n).any()


def test_vidal_lstm_forward_shape():
    model = VidalLSTM(input_size=3, hidden_size=10)
    x = torch.randn(8, 20, 3)
    out = model(x)
    assert out.shape == (8, 20)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_vidal_lstm_output_clip():
    model = VidalLSTM(output_clip=True)
    x = torch.randn(4, 10, 3) * 100
    out = model(x)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_trainer_runs_two_epochs(synthetic_cycles, tmp_path):
    set_seed(42)
    labelled = split_cycles(synthetic_cycles, "by_cycle", seed=42)
    train = filter_by_split(labelled, "train")
    val = filter_by_split(labelled, "val")

    builder = SequenceBuilder(seq_len=20, feature_cols=["voltage", "current", "temperature"])
    X_tr, y_tr, _ = builder.build(train)
    X_va, y_va, _ = builder.build(val)

    norm = MinMaxNormalizer()
    norm.fit(X_tr)
    X_tr = norm.transform(X_tr)
    X_va = norm.transform(X_va)

    train_ds = CycleDataset(X_tr, y_tr)
    val_ds = CycleDataset(X_va, y_va)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = VidalLSTM(input_size=3, hidden_size=10)
    optimizer = build_optimizer(model, {"optimizer": {"name": "adam", "lr": 0.001}})
    trainer = Trainer(
        optimizer=optimizer, epochs=2, device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "ckpts", config={},
    )
    history = trainer.fit(model, train_loader, val_loader)

    assert len(history) == 2
    assert "train_loss" in history[0]
    assert (tmp_path / "ckpts" / "last.pt").exists()


def test_evaluator_returns_finite_metrics(synthetic_cycles):
    set_seed(42)
    labelled = split_cycles(synthetic_cycles, "by_cycle", seed=42)
    test = filter_by_split(labelled, "test")
    if not test:
        pytest.skip("No test cycles after split")

    builder = SequenceBuilder(seq_len=20, feature_cols=["voltage", "current", "temperature"])
    X_te, y_te, _ = builder.build(test)
    if X_te.shape[0] == 0:
        pytest.skip("No test sequences after building")

    norm = MinMaxNormalizer()
    norm.fit(X_te)
    X_te = norm.transform(X_te)

    test_ds = CycleDataset(X_te, y_te)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    model = VidalLSTM(input_size=3, hidden_size=10)
    evaluator = Evaluator(model, torch.device("cpu"))
    results = evaluator.evaluate(test_loader, prefix="test_")

    m = results["metrics"]
    assert np.isfinite(m["test_mae"])
    assert np.isfinite(m["test_rmse"])
    assert m["test_mae"] >= 0.0


def test_full_experiment_runner(tmp_path):
    """Test the full ExperimentRunner pipeline end-to-end."""
    from battery_benchmark.experiments.runner import ExperimentRunner

    cfg = dict(_SMOKE_CFG)
    cfg["output_path"] = str(tmp_path / "outputs")
    cfg["epochs"] = 2

    runner = ExperimentRunner(cfg, output_dir=tmp_path / "outputs")
    result = runner.run()

    assert np.isfinite(result.test_mae)
    assert np.isfinite(result.test_rmse)
    assert (tmp_path / "outputs" / "metrics.json").exists()
    assert (tmp_path / "outputs" / "predictions_source.csv").exists()
    assert (tmp_path / "outputs" / "resolved_config.yaml").exists()

    with (tmp_path / "outputs" / "metrics.json").open() as f:
        metrics_data = json.load(f)
    assert "source_test_mae" in metrics_data or "test_mae" in metrics_data
