"""Tests for checkpoint save/load and training resume."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from battery_benchmark.training.checkpointing import load_checkpoint, save_checkpoint


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)


def _make_model_and_optimizer():
    model = _TinyModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer


def test_save_and_load_model_weights():
    model, optimizer = _make_model_and_optimizer()
    original_weights = {k: v.clone() for k, v in model.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "ckpt.pt"
        save_checkpoint(ckpt_path, model, optimizer, epoch=5, best_metric=0.01, config={})

        model2, optimizer2 = _make_model_and_optimizer()
        ckpt = load_checkpoint(ckpt_path, model2, optimizer2)

        for k in original_weights:
            assert torch.allclose(original_weights[k], model2.state_dict()[k])

        assert ckpt["epoch"] == 5
        assert ckpt["best_metric"] == pytest.approx(0.01)


def test_checkpoint_contains_config():
    model, optimizer = _make_model_and_optimizer()
    cfg = {"experiment_name": "test", "epochs": 10}

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "ckpt.pt"
        save_checkpoint(ckpt_path, model, optimizer, epoch=0, best_metric=0.5, config=cfg)
        ckpt = load_checkpoint(ckpt_path, model)
        assert ckpt["config"]["experiment_name"] == "test"


def test_checkpoint_with_normalizer():
    from battery_benchmark.preprocessing.normalization import MinMaxNormalizer
    import numpy as np

    model, optimizer = _make_model_and_optimizer()
    normalizer = MinMaxNormalizer()
    normalizer.fit(np.random.rand(10, 3))

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "ckpt.pt"
        save_checkpoint(
            ckpt_path, model, optimizer, epoch=1, best_metric=0.1,
            config={}, normalizer=normalizer,
        )
        norm2 = MinMaxNormalizer()
        load_checkpoint(ckpt_path, model, normalizer=norm2)
        assert norm2._min is not None


def test_load_nonexistent_raises():
    model, _ = _make_model_and_optimizer()
    with pytest.raises(FileNotFoundError):
        load_checkpoint("/nonexistent/path.pt", model)


def test_extra_payload_preserved():
    model, optimizer = _make_model_and_optimizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "ckpt.pt"
        save_checkpoint(
            ckpt_path, model, optimizer, epoch=2, best_metric=0.2,
            config={}, extra={"custom_key": "custom_value"},
        )
        ckpt = load_checkpoint(ckpt_path, model)
        assert ckpt["custom_key"] == "custom_value"
