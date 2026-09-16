from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/benchmarks/run_bian_cmu.py"
spec = importlib.util.spec_from_file_location("run_bian_cmu", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _synthetic_adapt_inputs(
    *, source_n: int = 6, target_n: int = 5, seq_len: int = 4, seed: int = 0
):
    """Small windowed arrays matching DTNN's expected (n, seq_len, 2) / (n,
    seq_len) shapes -- no real CMU/Panasonic data needed for this smoke
    test, only _adapt()'s internal bookkeeping is under test."""

    rng = np.random.default_rng(seed)
    source_x = rng.normal(size=(source_n, seq_len, 2)).astype(np.float32)
    source_y = rng.uniform(0, 1, size=(source_n, seq_len)).astype(np.float32)
    target_x = rng.normal(size=(target_n, seq_len, 2)).astype(np.float32)
    target_y = rng.uniform(0, 1, size=(target_n, seq_len)).astype(np.float32)
    return source_x, source_y, target_x, target_y


def _adapt_args(*, transfer_epochs: int, transfer_batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        transfer_epochs=transfer_epochs,
        transfer_batch_size=transfer_batch_size,
        transfer_lr=1e-3,
        target_weight=5.0,
        mmd_weight=0.1,
    )


def test_adapt_history_has_one_row_per_epoch_with_expected_columns() -> None:
    torch.manual_seed(0)
    model = module.DTNN(2, 8, 4, 8)
    source_x, source_y, target_x, target_y = _synthetic_adapt_inputs()
    args = _adapt_args(transfer_epochs=3, transfer_batch_size=2)

    final, history = module._adapt(
        model, source_x, source_y, target_x, target_y, args, torch.device("cpu")
    )

    assert len(history) == args.transfer_epochs
    assert [row["epoch"] for row in history] == [1, 2, 3]
    expected_columns = {
        "epoch", "source_mse", "target_mse", "adaptive_mmd", "total_loss",
        "layer_mmd_cnn2", "layer_mmd_bilstm2", "layer_mmd_fc",
        "layer_weight_cnn2", "layer_weight_bilstm2", "layer_weight_fc",
        "learning_rate", "optimizer_steps",
    }
    for row in history:
        assert expected_columns <= set(row.keys())
        assert row["learning_rate"] == args.transfer_lr
    # final (the last-optimizer-step-only dict used for training_final_batch)
    # must be untouched: same shape/keys as before this change.
    assert set(final) == {
        "source_mse", "target_mse", "adaptive_mmd", "layer_mmd", "layer_weights", "optimizer_steps"
    }
    assert final["optimizer_steps"] == sum(row["optimizer_steps"] for row in history)


def test_adapt_epoch_averages_are_not_just_the_last_batch() -> None:
    """Core correctness check: every per-epoch value in history must equal
    the mean over ALL optimizer steps run in that epoch, not the value from
    the final batch alone. Verified by independently recording every
    per-step source/target MSE and MMD value via monkeypatched criterion /
    adaptive_multiscale_mmd wrappers, then recomputing the epoch means
    ourselves and comparing against what _adapt() returned."""

    torch.manual_seed(0)
    model = module.DTNN(2, 8, 4, 8)
    source_x, source_y, target_x, target_y = _synthetic_adapt_inputs(
        source_n=7, target_n=5, seq_len=4
    )
    transfer_batch_size = 2  # target_n=5 -> steps_per_epoch = ceil(5/2) = 3 (>1, so
    # a "last batch only" bug and a true average necessarily disagree)
    args = _adapt_args(transfer_epochs=3, transfer_batch_size=transfer_batch_size)
    steps_per_epoch = -(-len(target_x) // transfer_batch_size)
    assert steps_per_epoch > 1

    mse_calls: list[float] = []
    mmd_calls: list[tuple[float, list[float], list[float]]] = []
    real_mmd_fn = module.adaptive_multiscale_mmd

    class _RecordingMSELoss(nn.Module):
        def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            value = ((prediction - target) ** 2).mean()
            mse_calls.append(float(value.detach()))
            return value

    def _recording_mmd(source_features, target_features):
        mmd, layer_mmd, weights = real_mmd_fn(source_features, target_features)
        mmd_calls.append(
            (
                float(mmd.detach()),
                [float(v) for v in layer_mmd.detach().cpu()],
                [float(v) for v in weights.cpu()],
            )
        )
        return mmd, layer_mmd, weights

    original_mseloss = module.nn.MSELoss
    original_mmd_fn = module.adaptive_multiscale_mmd
    module.nn.MSELoss = lambda *a, **kw: _RecordingMSELoss()
    module.adaptive_multiscale_mmd = _recording_mmd
    try:
        final, history = module._adapt(
            model, source_x, source_y, target_x, target_y, args, torch.device("cpu")
        )
    finally:
        module.nn.MSELoss = original_mseloss
        module.adaptive_multiscale_mmd = original_mmd_fn

    assert len(history) == args.transfer_epochs
    assert len(mmd_calls) == args.transfer_epochs * steps_per_epoch
    # source_loss then target_loss, in that order, once per optimizer step.
    assert len(mse_calls) == 2 * args.transfer_epochs * steps_per_epoch
    source_calls = mse_calls[0::2]
    target_calls = mse_calls[1::2]

    for epoch_index, row in enumerate(history):
        start, stop = epoch_index * steps_per_epoch, (epoch_index + 1) * steps_per_epoch
        epoch_source = source_calls[start:stop]
        epoch_target = target_calls[start:stop]
        epoch_mmd = mmd_calls[start:stop]

        expected_source_mse = float(np.mean(epoch_source))
        expected_target_mse = float(np.mean(epoch_target))
        expected_adaptive_mmd = float(np.mean([c[0] for c in epoch_mmd]))
        expected_layer_mmd = np.mean([c[1] for c in epoch_mmd], axis=0)
        expected_layer_weight = np.mean([c[2] for c in epoch_mmd], axis=0)
        expected_total_loss = float(
            np.mean(
                [
                    s + args.target_weight * t + args.mmd_weight * m[0]
                    for s, t, m in zip(epoch_source, epoch_target, epoch_mmd)
                ]
            )
        )

        assert row["optimizer_steps"] == steps_per_epoch
        assert np.isclose(row["source_mse"], expected_source_mse)
        assert np.isclose(row["target_mse"], expected_target_mse)
        assert np.isclose(row["adaptive_mmd"], expected_adaptive_mmd)
        assert np.isclose(row["total_loss"], expected_total_loss)
        for name, expected in zip(module.LAYER_NAMES, expected_layer_mmd):
            assert np.isclose(row[f"layer_mmd_{name}"], expected)
        for name, expected in zip(module.LAYER_NAMES, expected_layer_weight):
            assert np.isclose(row[f"layer_weight_{name}"], expected)

        # The specific failure mode being guarded against: the epoch average
        # must actually differ from a "last batch only" readout whenever the
        # per-step values aren't all identical (true here, since MMD/MSE
        # vary step to step as the model updates).
        last_step_source_mse = epoch_source[-1]
        if not np.isclose(last_step_source_mse, expected_source_mse):
            assert row["source_mse"] != last_step_source_mse


def test_adapt_history_does_not_change_final_result_or_optimizer_step_count() -> None:
    """Adding the per-epoch diagnostic must not alter the existing
    ``final`` (last-optimizer-step) return value's contents, nor the total
    optimizer_steps count -- both computed exactly as before."""

    torch.manual_seed(0)
    model = module.DTNN(2, 8, 4, 8)
    source_x, source_y, target_x, target_y = _synthetic_adapt_inputs()
    args = _adapt_args(transfer_epochs=2, transfer_batch_size=2)

    final, history = module._adapt(
        model, source_x, source_y, target_x, target_y, args, torch.device("cpu")
    )

    steps_per_epoch = history[0]["optimizer_steps"]
    assert final["optimizer_steps"] == args.transfer_epochs * steps_per_epoch
    assert isinstance(final["layer_mmd"], list) and len(final["layer_mmd"]) == 3
    assert isinstance(final["layer_weights"], list) and len(final["layer_weights"]) == 3
