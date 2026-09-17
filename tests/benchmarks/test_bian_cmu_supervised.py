from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
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


def _adapt_args(
    *, transfer_epochs: int, transfer_batch_size: int, batch_size: int = 4,
    validation_interval: int = 10,
) -> SimpleNamespace:
    return SimpleNamespace(
        transfer_epochs=transfer_epochs,
        transfer_batch_size=transfer_batch_size,
        transfer_lr=1e-3,
        target_weight=5.0,
        mmd_weight=0.1,
        batch_size=batch_size,
        validation_interval=validation_interval,
    )


def _synthetic_validation_inputs(*, n: int = 6, seq_len: int = 4, seed: int = 5):
    rng = np.random.default_rng(seed)
    validation_x = rng.normal(size=(n, seq_len, 2)).astype(np.float32)
    validation_y = rng.uniform(0, 1, size=(n, seq_len)).astype(np.float32)
    return validation_x, validation_y


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


def test_adapt_history_includes_positive_runtime_seconds() -> None:
    """runtime_seconds on each training_history.csv row must be present and
    a real positive wall-clock measurement (not a placeholder/zero) -- the
    epoch's actual adaptation wall-clock time, timed with a single _Timer
    (one sync+start before the epoch's target_loader loop, one sync+stop
    right after), which may include the lightweight diagnostic
    float-extraction/accumulation interleaved with the training steps.
    Verified indirectly by confirming it scales up when more optimizer
    steps run in the epoch (a fixed/placeholder value would not)."""

    torch.manual_seed(0)
    model_small = module.DTNN(2, 8, 4, 8)
    source_x, source_y, target_x_small, target_y_small = _synthetic_adapt_inputs(
        source_n=6, target_n=4, seq_len=4
    )
    args_small = _adapt_args(transfer_epochs=2, transfer_batch_size=4)  # 1 step/epoch
    _final_small, history_small = module._adapt(
        model_small, source_x, source_y, target_x_small, target_y_small, args_small,
        torch.device("cpu"),
    )
    for row in history_small:
        assert "runtime_seconds" in row
        assert isinstance(row["runtime_seconds"], float)
        assert row["runtime_seconds"] > 0

    torch.manual_seed(0)
    model_large = module.DTNN(2, 8, 4, 8)
    _source_x2, _source_y2, target_x_large, target_y_large = _synthetic_adapt_inputs(
        source_n=6, target_n=40, seq_len=4, seed=1
    )
    args_large = _adapt_args(transfer_epochs=2, transfer_batch_size=4)  # 10 steps/epoch
    _final_large, history_large = module._adapt(
        model_large, source_x, source_y, target_x_large, target_y_large, args_large,
        torch.device("cpu"),
    )
    for row in history_large:
        assert row["runtime_seconds"] > 0

    # 10x the optimizer steps per epoch should take meaningfully longer than
    # 1x -- catches a bug where runtime_seconds is timed around the wrong
    # (per-step-only, not accumulated) scope, or is a constant/no-op.
    mean_small = float(np.mean([row["runtime_seconds"] for row in history_small]))
    mean_large = float(np.mean([row["runtime_seconds"] for row in history_large]))
    assert mean_large > mean_small


def test_timer_synchronizes_exactly_once_on_enter_and_once_on_exit_for_cuda() -> None:
    """_Timer itself must call torch.cuda.synchronize() exactly once in
    __enter__ and exactly once in __exit__ when device.type == 'cuda' (one
    sync pair per `with _Timer(...)` block), and never on a CPU device."""

    sync_calls = []
    original_synchronize = torch.cuda.synchronize
    torch.cuda.synchronize = lambda *a, **kw: sync_calls.append(1)
    try:
        cuda_like_device = SimpleNamespace(type="cuda")
        with module._Timer(cuda_like_device):
            pass
        assert len(sync_calls) == 2  # one on __enter__, one on __exit__

        sync_calls.clear()
        with module._Timer(torch.device("cpu")):
            pass
        assert len(sync_calls) == 0
    finally:
        torch.cuda.synchronize = original_synchronize


def test_adapt_opens_exactly_one_timer_per_epoch_not_per_step() -> None:
    """The fix under test: _adapt must construct exactly one _Timer per
    epoch (wrapping the whole target_loader loop), not one per optimizer
    step. Verified by monkeypatching module._Timer with a subclass that
    counts its own instantiations while still behaving identically, then
    checking the count equals transfer_epochs -- NOT
    transfer_epochs * steps_per_epoch, which is what a per-step timer would
    produce."""

    instantiation_count = 0
    real_timer = module._Timer

    class _CountingTimer(real_timer):
        def __init__(self, device):
            nonlocal instantiation_count
            instantiation_count += 1
            super().__init__(device)

    torch.manual_seed(0)
    model = module.DTNN(2, 8, 4, 8)
    source_x, source_y, target_x, target_y = _synthetic_adapt_inputs(
        source_n=6, target_n=9, seq_len=4  # batch_size=2 -> 5 steps/epoch
    )
    args = _adapt_args(transfer_epochs=3, transfer_batch_size=2)
    steps_per_epoch = -(-len(target_x) // args.transfer_batch_size)
    assert steps_per_epoch > 1  # otherwise this test can't distinguish the two schemes

    module._Timer = _CountingTimer
    try:
        _final, history = module._adapt(
            model, source_x, source_y, target_x, target_y, args, torch.device("cpu")
        )
    finally:
        module._Timer = real_timer

    assert len(history) == args.transfer_epochs
    assert instantiation_count == args.transfer_epochs
    assert instantiation_count != args.transfer_epochs * steps_per_epoch


def test_train_source_history_matches_the_printed_epoch_mse_and_has_runtime() -> None:
    """_train_source's returned per-epoch history must carry the exact same
    source_mse the function already computed for its print statement (a
    len(xb)-weighted mean over the epoch's batches, not just the last
    batch), plus a constant learning_rate (no scheduler here) and a
    positive per-epoch runtime_seconds -- all without changing what
    _train_source actually trains."""

    torch.manual_seed(0)
    model = module.DTNN(2, 8, 4, 8)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(9, 4, 2)).astype(np.float32)
    y = rng.uniform(0, 1, size=(9, 4)).astype(np.float32)
    epochs, batch_size, lr = 3, 4, 5e-3

    mse_calls: list[float] = []
    original_mseloss = module.nn.MSELoss

    class _RecordingMSELoss(nn.Module):
        def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            value = ((prediction - target) ** 2).mean()
            mse_calls.append(float(value.detach()))
            return value

    module.nn.MSELoss = lambda *a, **kw: _RecordingMSELoss()
    try:
        history = module._train_source(model, x, y, epochs, batch_size, lr, torch.device("cpu"))
    finally:
        module.nn.MSELoss = original_mseloss

    steps_per_epoch = -(-len(x) // batch_size)  # ceil
    assert len(history) == epochs
    assert len(mse_calls) == epochs * steps_per_epoch

    # Reproduce the exact len(xb)-weighted-mean formula _train_source uses,
    # from independently-recorded per-batch losses, using the loader's own
    # batch sizes (last batch of 9 items at batch_size=4 is smaller: 4,4,1).
    remaining = len(x)
    batch_sizes = []
    for _ in range(steps_per_epoch):
        batch_sizes.append(min(batch_size, remaining))
        remaining -= batch_sizes[-1]

    for epoch_index, row in enumerate(history):
        assert row["epoch"] == epoch_index + 1
        assert row["learning_rate"] == lr
        assert row["runtime_seconds"] > 0
        start, stop = epoch_index * steps_per_epoch, (epoch_index + 1) * steps_per_epoch
        epoch_losses = mse_calls[start:stop]
        expected_source_mse = sum(
            loss * size for loss, size in zip(epoch_losses, batch_sizes)
        ) / len(x)
        assert np.isclose(row["source_mse"], expected_source_mse)


def test_validation_evaluated_on_epoch_1_last_epoch_and_every_interval() -> None:
    """validation_evaluated must be True exactly on epoch 1, the final
    epoch, and every `validation_interval`-th epoch, and False everywhere
    else; validation_mae/validation_rmse must be finite [0,1]-range numbers
    when evaluated and NaN when not."""

    torch.manual_seed(0)
    model = module.DTNN(2, 8, 4, 8)
    source_x, source_y, target_x, target_y = _synthetic_adapt_inputs()
    validation_x, validation_y = _synthetic_validation_inputs()
    args = _adapt_args(transfer_epochs=7, transfer_batch_size=2, validation_interval=3)

    _final, history = module._adapt(
        model, source_x, source_y, target_x, target_y, args, torch.device("cpu"),
        validation_x=validation_x, validation_y=validation_y,
    )

    # epochs 1..7, interval 3, always epoch 1 and epoch 7 (last):
    # evaluated on {1, 3, 6, 7} (3 and 6 are multiples of 3).
    expected_evaluated = {1, 3, 6, 7}
    for row in history:
        assert row["validation_evaluated"] == (row["epoch"] in expected_evaluated)
        if row["validation_evaluated"]:
            assert 0.0 <= row["validation_mae"] <= 1.0
            assert 0.0 <= row["validation_rmse"] <= 1.0
        else:
            assert math.isnan(row["validation_mae"])
            assert math.isnan(row["validation_rmse"])


def test_validation_interval_zero_disables_validation_entirely() -> None:
    """--validation-interval 0 must turn validation recording off on every
    epoch, including epoch 1 and the last epoch, even though valid
    validation_x/validation_y were supplied."""

    torch.manual_seed(0)
    model = module.DTNN(2, 8, 4, 8)
    source_x, source_y, target_x, target_y = _synthetic_adapt_inputs()
    validation_x, validation_y = _synthetic_validation_inputs()
    args = _adapt_args(transfer_epochs=3, transfer_batch_size=2, validation_interval=0)

    _final, history = module._adapt(
        model, source_x, source_y, target_x, target_y, args, torch.device("cpu"),
        validation_x=validation_x, validation_y=validation_y,
    )

    for row in history:
        assert row["validation_evaluated"] is False
        assert math.isnan(row["validation_mae"])
        assert math.isnan(row["validation_rmse"])


def test_validation_diagnostic_metrics_are_fractions_not_percent() -> None:
    """_validation_diagnostic_metrics must return MAE/RMSE as raw [0,1]
    fractions (matching compute_metrics' own output), never the *100
    percent scaling _evaluate() applies for its overall_mae_percent."""

    torch.manual_seed(0)
    model = module.DTNN(2, 8, 4, 8)
    validation_x, validation_y = _synthetic_validation_inputs(n=8)

    mae, rmse = module._validation_diagnostic_metrics(
        model, validation_x, validation_y, batch_size=4, device=torch.device("cpu")
    )

    model.eval()
    with torch.no_grad():
        prediction = torch.cat(
            [
                model(xb.to(torch.device("cpu"))).cpu()
                for xb, _ in module._loader(validation_x, validation_y, 4, False)
            ]
        ).numpy()
    prediction = np.clip(prediction, 0.0, 1.0)
    expected = module.compute_metrics(validation_y, prediction)

    assert np.isclose(mae, expected["mae"])
    assert np.isclose(rmse, expected["rmse"])
    # SOC values live in [0,1], so a fraction-scaled MAE/RMSE must also be
    # (well) under 1 -- catches an accidental *100 regression immediately.
    assert mae < 1.0
    assert rmse < 1.0


def test_validation_toggle_does_not_change_final_model_or_predictions() -> None:
    """Core requirement: turning the validation diagnostic on vs. off must
    produce a byte-for-byte identical final model and identical
    predictions on held-out data. Verified two ways -- passing
    validation_x/validation_y vs. not, and validation_interval>0 vs. 0 --
    since either one alone disables the diagnostic."""

    def _run(*, with_validation: bool, validation_interval: int) -> torch.nn.Module:
        torch.manual_seed(123)
        model = module.DTNN(2, 8, 4, 8)
        source_x, source_y, target_x, target_y = _synthetic_adapt_inputs(seed=2)
        validation_x, validation_y = (
            _synthetic_validation_inputs() if with_validation else (None, None)
        )
        args = _adapt_args(
            transfer_epochs=4, transfer_batch_size=2, validation_interval=validation_interval
        )
        module._adapt(
            model, source_x, source_y, target_x, target_y, args, torch.device("cpu"),
            validation_x=validation_x, validation_y=validation_y,
        )
        return model

    model_with_validation = _run(with_validation=True, validation_interval=1)
    model_without_validation_data = _run(with_validation=False, validation_interval=1)
    model_with_interval_zero = _run(with_validation=True, validation_interval=0)

    probe_x = _synthetic_validation_inputs(n=5, seed=77)[0]
    probe = torch.from_numpy(probe_x)

    state_with = model_with_validation.state_dict()
    for other in (model_without_validation_data, model_with_interval_zero):
        state_other = other.state_dict()
        assert set(state_with) == set(state_other)
        for key in state_with:
            assert torch.equal(state_with[key], state_other[key]), f"parameter {key} diverged"

    model_with_validation.eval()
    model_without_validation_data.eval()
    model_with_interval_zero.eval()
    with torch.no_grad():
        prediction_with = model_with_validation(probe)
        prediction_without_data = model_without_validation_data(probe)
        prediction_interval_zero = model_with_interval_zero(probe)
    assert torch.equal(prediction_with, prediction_without_data)
    assert torch.equal(prediction_with, prediction_interval_zero)


def _synthetic_evaluate_inputs(*, seq_len: int = 4, seed: int = 9):
    """Windows spread across missions with UNEQUAL window counts per
    mission (2 then 1 then 3), so per-mission window_index bookkeeping in
    _evaluate()'s prediction records is actually exercised rather than
    trivially uniform."""

    rng = np.random.default_rng(seed)
    n = 2 + 1 + 3
    x = rng.normal(size=(n, seq_len, 2)).astype(np.float32)
    y = rng.uniform(0, 1, size=(n, seq_len)).astype(np.float32)
    missions = [("mission-a", 2), ("mission-b", 1), ("mission-c", 3)]
    return x, y, missions


def test_evaluate_return_predictions_false_matches_previous_single_value_return() -> None:
    """Backward compatibility: every existing call site calls _evaluate()
    without return_predictions and expects a single dict back, exactly as
    before this change. Also checked byte-for-byte against the `overall`
    half of the return_predictions=True call, so the metric computation
    itself is provably identical either way."""

    torch.manual_seed(0)
    model = module.DTNN(2, 8, 4, 8)
    x, y, missions = _synthetic_evaluate_inputs()
    device = torch.device("cpu")

    overall_only = module._evaluate(model, x, y, missions, batch_size=4, device=device)
    assert isinstance(overall_only, dict)

    overall_with_records, records = module._evaluate(
        model, x, y, missions, batch_size=4, device=device, return_predictions=True
    )
    assert overall_only == overall_with_records
    assert isinstance(records, list) and records


def test_evaluate_predictions_have_expected_shape_and_fields() -> None:
    """One prediction record per (window, timestep), with mission_cycle_id/
    window_index/timestep_index reconstructing exactly the windows/missions
    layout _evaluate() was given -- window_index restarts at 0 for each
    mission, timestep_index runs 0..seq_len-1 within each window."""

    torch.manual_seed(1)
    model = module.DTNN(2, 8, 4, 8)
    seq_len = 4
    x, y, missions = _synthetic_evaluate_inputs(seq_len=seq_len)
    device = torch.device("cpu")

    _, records = module._evaluate(
        model, x, y, missions, batch_size=4, device=device, return_predictions=True
    )

    assert len(records) == len(x) * seq_len
    expected_fields = {
        "mission_cycle_id", "window_index", "timestep_index",
        "soc_reference", "soc_prediction", "signed_error", "absolute_error",
    }
    for record in records:
        assert expected_fields <= set(record.keys())
        assert math.isclose(
            record["signed_error"], record["soc_prediction"] - record["soc_reference"]
        )
        assert math.isclose(record["absolute_error"], abs(record["signed_error"]))

    # Rebuild (mission_cycle_id -> set of window_index -> set of
    # timestep_index) from the records and compare against what the input
    # `missions` list says it should be.
    seen: dict[str, dict[int, set[int]]] = {}
    for record in records:
        by_window = seen.setdefault(record["mission_cycle_id"], {})
        by_window.setdefault(record["window_index"], set()).add(record["timestep_index"])
    for mission, count in missions:
        assert set(seen[mission]) == set(range(count))
        for timesteps in seen[mission].values():
            assert timesteps == set(range(seq_len))


def test_evaluate_predictions_recompute_mae_rmse_matches_evaluate_result() -> None:
    """Required determinism-of-reporting check: independently recomputing
    MAE/RMSE from the saved per-timestep prediction records (soc_reference/
    soc_prediction only, exactly as they'd be read back from
    test_predictions.parquet) must reproduce _evaluate()'s own
    overall_mae_percent/overall_rmse_percent -- proving the saved records
    are a faithful, lossless record of what was actually evaluated."""

    torch.manual_seed(2)
    model = module.DTNN(2, 8, 4, 8)
    x, y, missions = _synthetic_evaluate_inputs(seed=17)
    device = torch.device("cpu")

    overall, records = module._evaluate(
        model, x, y, missions, batch_size=4, device=device, return_predictions=True
    )

    reference = np.array([r["soc_reference"] for r in records])
    prediction = np.array([r["soc_prediction"] for r in records])
    recomputed = module.compute_metrics(reference, prediction)

    assert np.isclose(recomputed["mae"] * 100, overall["overall_mae_percent"])
    assert np.isclose(recomputed["rmse"] * 100, overall["overall_rmse_percent"])


def _sample_prediction_chunk(*, repeat, seed, cell_id, method, n_records=3):
    """A small, schema-shaped chunk of prediction rows -- the same shape
    _write_prediction_chunk receives from main() for one repeat x cell x
    method, but built directly here so these tests exercise the streaming
    writer without needing real CMU/Panasonic data or running main()."""

    return [
        {
            "framework": "bian_2021",
            "method": method,
            "repeat": repeat,
            "seed": seed,
            "cell_id": cell_id,
            "condition": "baseline_protocol",
            "split": "test",
            "mission_cycle_id": f"{cell_id}-mission-{i % 2}",
            "window_index": i,
            "timestep_index": 0,
            "mission_sequence_order": i % 2,
            "elapsed_candidate_s": 1.0 * i,
            "original_time_s": 100.0 + i,
            "source_row_right": 1000 + i,
            "development_phase_role": "high_1" if i % 2 == 0 else None,
            "voltage_v_raw": 3.5 + 0.01 * i,
            "current_a_raw": -1.0 - 0.01 * i,
            "temperature_c_raw": 25.0 + 0.1 * i,
            "capacity_ah": 2.5 if i % 3 else None,
            "capacity_source": "rpt" if i % 3 else None,
            "source_rpt_id": f"rpt-{i}" if i % 3 else None,
            "soc_reference": 0.1 * i,
            "soc_prediction": 0.1 * i + 0.01,
            "signed_error": 0.01,
            "absolute_error": 0.01,
            "ns_raw": i % 8,
            "mission_phase": module.NS_MISSION_PHASE_MAP.get(i % 8),
        }
        for i in range(n_records)
    ]


def test_write_prediction_chunk_multi_chunk_roundtrip_preserves_schema_and_rows(
    tmp_path,
) -> None:
    """Core requirement: writing several repeat x cell x method chunks
    through ParquetWriter/_write_prediction_chunk (never accumulating them
    in one big list first), then closing the writer and reading the file
    back, must preserve every row, both methods, and the exact fixed
    TEST_PREDICTIONS_SCHEMA column order/types -- regardless of how many
    separate write_table() calls it took to get there."""

    tmp_output = tmp_path / "test_predictions.parquet.incomplete"
    writer = pq.ParquetWriter(tmp_output, module.TEST_PREDICTIONS_SCHEMA)
    chunks = [
        _sample_prediction_chunk(repeat=1, seed=1, cell_id="VAH01", method="Source only", n_records=2),
        _sample_prediction_chunk(repeat=1, seed=1, cell_id="VAH01", method="Bian MDA", n_records=2),
        _sample_prediction_chunk(repeat=1, seed=1, cell_id="VAH17", method="Source only", n_records=3),
        _sample_prediction_chunk(repeat=1, seed=1, cell_id="VAH17", method="Bian MDA", n_records=3),
        _sample_prediction_chunk(repeat=2, seed=2, cell_id="VAH01", method="Source only", n_records=1),
        _sample_prediction_chunk(repeat=2, seed=2, cell_id="VAH01", method="Bian MDA", n_records=1),
    ]
    for chunk in chunks:
        module._write_prediction_chunk(writer, chunk)
    writer.close()

    final_path = tmp_path / "test_predictions.parquet"
    tmp_output.replace(final_path)

    read_back = pq.read_table(final_path)
    assert read_back.schema.equals(module.TEST_PREDICTIONS_SCHEMA)

    frame = read_back.to_pandas()
    expected_total = sum(len(chunk) for chunk in chunks)
    assert len(frame) == expected_total

    assert set(frame["method"]) == {"Source only", "Bian MDA"}
    assert set(frame["cell_id"]) == {"VAH01", "VAH17"}
    assert set(frame["repeat"]) == {1, 2}

    # Every individual row from every chunk must be present, unchanged, in
    # the file read back -- not just matching totals/sets.
    expected_rows = pd.DataFrame([row for chunk in chunks for row in chunk])
    expected_rows = expected_rows[list(frame.columns)]
    frame_sorted = frame.sort_values(list(frame.columns)).reset_index(drop=True)
    expected_sorted = expected_rows.sort_values(list(frame.columns)).reset_index(drop=True)
    pd.testing.assert_frame_equal(frame_sorted, expected_sorted, check_dtype=False)


def test_write_prediction_chunk_empty_chunk_is_a_no_op(tmp_path) -> None:
    """An empty per-repeat-x-cell-x-method chunk (e.g. a cell/method that
    happened to produce zero prediction records) must not write a
    malformed or zero-row record batch -- it's simply skipped."""

    tmp_output = tmp_path / "test_predictions.parquet.incomplete"
    writer = pq.ParquetWriter(tmp_output, module.TEST_PREDICTIONS_SCHEMA)
    module._write_prediction_chunk(writer, [])
    module._write_prediction_chunk(
        writer, _sample_prediction_chunk(repeat=1, seed=1, cell_id="VAH01", method="Source only")
    )
    writer.close()

    frame = pq.read_table(tmp_output).to_pandas()
    assert len(frame) == 3


def test_streaming_writer_success_path_renames_to_final_name(tmp_path) -> None:
    """Mirrors main()'s own try/finally around test_predictions_writer:
    on a clean run (no exception), the writer is closed and the
    ".incomplete" temp file is atomically replaced onto the final
    "test_predictions.parquet" name -- and only that final name exists
    afterward."""

    tmp_output = tmp_path / "test_predictions.parquet.incomplete"
    final_output = tmp_path / "test_predictions.parquet"
    writer = pq.ParquetWriter(tmp_output, module.TEST_PREDICTIONS_SCHEMA)
    completed = False
    try:
        for repeat in (1, 2):
            for method in ("Source only", "Bian MDA"):
                module._write_prediction_chunk(
                    writer,
                    _sample_prediction_chunk(
                        repeat=repeat, seed=repeat, cell_id="VAH01", method=method
                    ),
                )
        completed = True
    finally:
        writer.close()
        if completed:
            tmp_output.replace(final_output)

    assert final_output.exists()
    assert not tmp_output.exists()
    frame = pq.read_table(final_output).to_pandas()
    assert len(frame) == 3 * 2 * 2


def test_streaming_writer_failure_path_leaves_incomplete_file_and_raises(tmp_path) -> None:
    """Core requirement: if something raises partway through writing chunks
    (simulating a mid-run crash in main()), the writer must still be
    closed (so the temp file is readable, not a truncated/corrupt parquet
    file), the temp file must be left at its ".incomplete" name, NO file
    must exist at the final "test_predictions.parquet" name (never a
    partial file masquerading as complete), and the original exception
    must still propagate rather than being swallowed."""

    tmp_output = tmp_path / "test_predictions.parquet.incomplete"
    final_output = tmp_path / "test_predictions.parquet"
    writer = pq.ParquetWriter(tmp_output, module.TEST_PREDICTIONS_SCHEMA)
    completed = False

    class _BoomError(RuntimeError):
        pass

    def _run():
        nonlocal completed
        try:
            module._write_prediction_chunk(
                writer,
                _sample_prediction_chunk(repeat=1, seed=1, cell_id="VAH01", method="Source only"),
            )
            raise _BoomError("simulated mid-run failure after one chunk")
        finally:
            writer.close()
            if completed:
                tmp_output.replace(final_output)

    with pytest.raises(_BoomError):
        _run()

    assert not completed
    assert not final_output.exists()
    assert tmp_output.exists()
    # The temp file must still be a valid, readable parquet file (the
    # writer was properly closed despite the exception), containing
    # exactly the one chunk written before the failure.
    frame = pq.read_table(tmp_output).to_pandas()
    assert len(frame) == 3


def test_run_metadata_records_validation_interval_and_label_usage_flags() -> None:
    """run.json's metadata must record args.validation_interval and the
    three target_validation_labels_used_for_* flags with the exact values
    the user asked for: diagnostics=True, training=False,
    checkpoint_selection=False. This is pure metadata/bookkeeping -- no
    computation -- so it's checked directly against _build_run_metadata's
    return value rather than by running the full pipeline."""

    args = SimpleNamespace(
        validation_interval=7,
        panasonic_data_root=Path("/data/panasonic"),
        seed=1,
        repeats=2,
    )
    runs = [{"repeat": 1, "cell_id": "VAH01"}]

    metadata = module._build_run_metadata(args, runs)

    assert metadata["validation_interval"] == 7
    assert metadata["target_validation_labels_used_for_diagnostics"] is True
    assert metadata["target_validation_labels_used_for_training"] is False
    assert metadata["target_validation_labels_used_for_checkpoint_selection"] is False
    # Untouched existing flags/fields must still be present and correct.
    assert metadata["target_soc_labels_used_for_adaptation"] is True
    assert metadata["target_soc_labels_used_for_checkpoint_selection"] is False
    assert metadata["label_is_ground_truth"] is False
    assert metadata["formal_benchmark_eligible"] is False
    assert metadata["runs"] is runs
    # Path-typed argument values are still stringified for JSON-safety.
    assert metadata["arguments"]["panasonic_data_root"] == "/data/panasonic"
    assert metadata["arguments"]["validation_interval"] == 7


def test_run_metadata_is_json_serializable() -> None:
    """run.json is written with json.dumps(metadata) -- guard against a
    future field sneaking in a non-JSON-serializable value (e.g. a raw
    Path or numpy scalar)."""

    args = SimpleNamespace(validation_interval=10, output_root=Path("/tmp/out"))
    metadata = module._build_run_metadata(args, runs=[{"repeat": 1}])

    json.dumps(metadata)  # must not raise


def test_run_json_metadata_uses_validation_missions_diagnostic_not_unused() -> None:
    """Static regression check on the script source: the per-run dict
    built inside main() (the ``runs.append({...})`` literal) must key the
    validation-split mission count as ``validation_missions_diagnostic``
    -- reflecting that it's now used for the periodic validation
    diagnostic -- and must never reintroduce the old
    ``validation_missions_unused`` key as an actual dict key (a mention in
    a comment, such as the one documenting the rename, is fine)."""

    source = SCRIPT.read_text()
    tree = ast.parse(source)

    dict_keys_containing_validation_missions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and "validation_missions" in key.value
                ):
                    dict_keys_containing_validation_missions.add(key.value)

    assert dict_keys_containing_validation_missions == {"validation_missions_diagnostic"}


def _fake_target_record(
    *, mission, sequence_order, soc, current=None, voltage=None,
    elapsed_candidate_s=None, original_time_s=None, source_row_right=None,
):
    """A minimal stand-in for a VidalV2Record (only ``.data`` and
    ``.metadata`` are read by _target_parts): ``soc`` fully determines the
    mission's length and its initial-SOC/30-70 split behavior, while the
    other per-timestep columns are set to easily distinguishable sequences
    so a test can independently recompute, from the frame itself, exactly
    which raw index each retained timestep must have come from."""

    n = len(soc)
    frame = pd.DataFrame(
        {
            "current": (np.arange(n, dtype=np.float64) if current is None else current),
            "voltage": (np.arange(n, dtype=np.float64) + 100.0 if voltage is None else voltage),
            "soc": np.asarray(soc, dtype=np.float64),
            "elapsed_candidate_s": (
                np.arange(n, dtype=np.float64) if elapsed_candidate_s is None
                else elapsed_candidate_s
            ),
            "original_time_s": (
                np.arange(n, dtype=np.float64) + 10_000.0 if original_time_s is None
                else original_time_s
            ),
            "source_row_right": (
                np.arange(n, dtype=np.int64) + 500 if source_row_right is None
                else source_row_right
            ),
        }
    )
    return SimpleNamespace(
        data=frame,
        metadata={"mission_cycle_id": mission, "sequence_order": sequence_order},
    )


def test_target_parts_metadata_matches_raw_frame_at_the_exact_kept_indices() -> None:
    """Core alignment requirement, at the _target_parts level (before any
    windowing/reshape): for the "test" portion, the metadata slice for a
    mission must equal the mission's raw elapsed_candidate_s/
    original_time_s/source_row_right/sequence_order at EXACTLY the same
    raw-frame indices that ended up in x/y -- reconstructed independently
    here from starts[0] and the fraction boundary, not by re-running
    _target_parts' own logic."""

    initial_soc = 0.80
    fraction = 0.30
    # 10 timesteps, SOC decreasing from 0.95 to 0.05: first index with
    # soc <= 0.80 is index 2 (soc=0.75), so starts[0] == 2.
    soc = np.linspace(0.95, 0.05, 10)
    record = _fake_target_record(mission="M1", sequence_order=3, soc=soc)

    parts = module._target_parts(
        [record], {"M1"}, initial_soc, fraction, "test", include_metadata=True
    )
    assert len(parts) == 1
    x, y, mission, metadata = parts[0]
    assert mission == "M1"

    starts0 = int(np.flatnonzero(soc <= initial_soc)[0])
    truncated_len = len(soc) - starts0
    boundary = int(truncated_len * fraction)
    kept = slice(starts0 + boundary, len(soc))  # portion="test" -> [boundary:] of the truncated array

    frame = record.data
    assert np.array_equal(y, frame["soc"].to_numpy()[kept])
    assert np.array_equal(
        metadata["elapsed_candidate_s"], frame["elapsed_candidate_s"].to_numpy()[kept]
    )
    assert np.array_equal(metadata["original_time_s"], frame["original_time_s"].to_numpy()[kept])
    assert np.array_equal(metadata["source_row_right"], frame["source_row_right"].to_numpy()[kept])
    assert np.all(metadata["mission_sequence_order"] == 3)
    assert len(x) == len(y) == len(metadata["elapsed_candidate_s"])


def test_target_data_metadata_survives_truncation_with_correct_dropped_tail() -> None:
    """Core alignment requirement under truncation: when a mission's
    selected timesteps aren't an exact multiple of the window length,
    _windows drops the same trailing timesteps from x/y AND from every
    metadata column -- never leaving metadata one element longer/shorter,
    or shifted, relative to x/y."""

    initial_soc = 0.90
    fraction = 0.0  # portion="test" keeps everything from starts[0] onward
    # 13 timesteps all <= initial_soc, so starts[0] == 0 and the "test"
    # selection keeps all 13 -- not a multiple of sequence_length=3, so
    # _windows must drop the trailing 1 timestep (13 -> usable=12).
    soc = np.linspace(0.85, 0.05, 13)
    record = _fake_target_record(mission="M2", sequence_order=7, soc=soc)
    length = 3

    parts = module._target_parts(
        [record], {"M2"}, initial_soc, fraction, "test", include_metadata=True
    )
    x, y, missions, metadata = module._target_data(
        parts,
        length,
        feature_min=np.array([0.0, 0.0]),
        feature_max=np.array([20.0, 120.0]),
        include_metadata=True,
    )

    assert missions == [("M2", 4)]  # usable=12 -> 4 windows of length 3
    assert x.shape == (4, 3, 2)
    assert y.shape == (4, 3)
    for name in module.METADATA_COLUMNS:
        assert metadata[name].shape == (4, 3)

    frame = record.data
    kept = slice(0, 12)  # first 12 of the 13 selected timesteps; last one dropped
    assert np.array_equal(y.reshape(-1), frame["soc"].to_numpy()[kept].astype(np.float32))
    assert np.array_equal(
        metadata["elapsed_candidate_s"].reshape(-1),
        frame["elapsed_candidate_s"].to_numpy()[kept],
    )
    assert np.array_equal(
        metadata["source_row_right"].reshape(-1), frame["source_row_right"].to_numpy()[kept]
    )
    assert np.all(metadata["mission_sequence_order"] == 7)


def test_target_data_metadata_stays_aligned_across_multiple_missions_of_different_lengths() -> (
    None
):
    """Two missions with different lengths and different amounts of
    truncation, concatenated by _windows into one array (as main() does
    for the full test split): every window/timestep's metadata must still
    match its own mission's raw frame at the correct offset -- proving
    concatenation across missions doesn't misalign metadata relative to
    x/y for any mission after the first."""

    length = 4
    initial_soc = 1.0  # keep every timestep (starts[0] == 0 always)
    fraction = 0.0  # keep every timestep from starts[0] onward too

    soc_a = np.linspace(0.9, 0.1, 9)  # 9 timesteps -> usable=8 -> 2 windows, 1 dropped
    soc_b = np.linspace(0.8, 0.2, 12)  # 12 timesteps -> usable=12 -> 3 windows, none dropped
    record_a = _fake_target_record(
        mission="MA", sequence_order=0, soc=soc_a,
        source_row_right=np.arange(9, dtype=np.int64) + 2_000,
    )
    record_b = _fake_target_record(
        mission="MB", sequence_order=1, soc=soc_b,
        source_row_right=np.arange(12, dtype=np.int64) + 5_000,
    )

    parts = module._target_parts(
        [record_a, record_b], {"MA", "MB"}, initial_soc, fraction, "test",
        include_metadata=True,
    )
    x, y, missions, metadata = module._target_data(
        parts, length, feature_min=np.array([0.0, 0.0]), feature_max=np.array([20.0, 120.0]),
        include_metadata=True,
    )

    assert missions == [("MA", 2), ("MB", 3)]
    assert x.shape[0] == 5 and y.shape == (5, 4)

    # Reconstruct, per mission, the expected raw-frame indices kept after
    # truncation, and check every retained window/timestep against them.
    expected_frames = {"MA": (record_a.data, 8), "MB": (record_b.data, 12)}
    start = 0
    for mission, count in missions:
        frame, usable = expected_frames[mission]
        expected_source_row = frame["source_row_right"].to_numpy()[:usable].reshape(count, length)
        expected_soc = frame["soc"].to_numpy()[:usable].reshape(count, length).astype(np.float32)
        stop = start + count
        assert np.array_equal(metadata["source_row_right"][start:stop], expected_source_row)
        assert np.array_equal(y[start:stop], expected_soc)
        start = stop


def test_evaluate_prediction_records_carry_aligned_metadata_across_missions() -> None:
    """End-to-end alignment check through _evaluate() itself: for windows
    drawn from missions of different lengths/window counts, every
    prediction record's mission_sequence_order/elapsed_candidate_s/
    original_time_s/source_row_right must equal metadata[global_window_
    index, timestep_index] -- the SAME indices used for that record's
    soc_reference -- and the plain (metadata=None) call path must still
    omit those four fields entirely."""

    torch.manual_seed(3)
    model = module.DTNN(2, 8, 4, 8)
    seq_len = 4
    x, y, missions = _synthetic_evaluate_inputs(seq_len=seq_len, seed=21)
    device = torch.device("cpu")

    rng = np.random.default_rng(99)
    metadata = {
        "mission_sequence_order": rng.integers(0, 5, size=y.shape).astype(np.int64),
        "elapsed_candidate_s": rng.uniform(0, 1000, size=y.shape),
        "original_time_s": rng.uniform(10_000, 20_000, size=y.shape),
        "source_row_right": rng.integers(0, 100_000, size=y.shape).astype(np.int64),
    }

    _, records_with_metadata = module._evaluate(
        model, x, y, missions, batch_size=4, device=device,
        return_predictions=True, metadata=metadata,
    )
    _, records_without_metadata = module._evaluate(
        model, x, y, missions, batch_size=4, device=device,
        return_predictions=True, metadata=None,
    )

    for record in records_without_metadata:
        assert not ({"mission_sequence_order", "elapsed_candidate_s",
                     "original_time_s", "source_row_right"} & set(record.keys()))

    start = 0
    for mission, count in missions:
        for local_window_index in range(count):
            global_window_index = start + local_window_index
            for timestep_index in range(seq_len):
                record = next(
                    r for r in records_with_metadata
                    if r["mission_cycle_id"] == mission
                    and r["window_index"] == local_window_index
                    and r["timestep_index"] == timestep_index
                )
                assert record["mission_sequence_order"] == int(
                    metadata["mission_sequence_order"][global_window_index, timestep_index]
                )
                assert math.isclose(
                    record["elapsed_candidate_s"],
                    float(metadata["elapsed_candidate_s"][global_window_index, timestep_index]),
                )
                assert math.isclose(
                    record["original_time_s"],
                    float(metadata["original_time_s"][global_window_index, timestep_index]),
                )
                assert record["source_row_right"] == int(
                    metadata["source_row_right"][global_window_index, timestep_index]
                )
                # Same index also used for soc_reference -- confirms the
                # metadata is aligned to the SAME timestep, not just
                # internally self-consistent.
                assert math.isclose(
                    record["soc_reference"],
                    float(y[global_window_index, timestep_index]),
                )
        start += count


def test_join_diagnostic_fields_by_source_row_aligns_exact_values() -> None:
    """Core alignment requirement for the diagnostic join: every position
    in `source_row_right` must get exactly the raw_diagnostic row whose
    source_row matches -- checked here with a REPEATED source_row inside
    `source_row_right` (two different timesteps mapping to the same raw
    row, as can genuinely happen when several resampled timesteps fall
    between the same pair of original rows), to prove the vectorized
    merge isn't just a coincidental 1:1 index correspondence."""

    raw_diagnostic = pd.DataFrame(
        {
            "source_row": [100, 101, 102, 103],
            "development_phase_role": ["high_1", "lower", "high_2", "lower"],
            "voltage_v_raw": [3.60, 3.55, 3.50, 3.45],
            "current_a_raw": [-1.0, -1.1, -1.2, -1.3],
            "temperature_c_raw": [24.0, 24.5, 25.0, 25.5],
            "capacity_ah": [2.50, 2.49, 2.48, 2.47],
            "capacity_source": ["rpt", "rpt", "rpt", "rpt"],
            "source_rpt_id": ["R1", "R1", "R2", "R2"],
            "ns_raw": [4, 5, 6, 5],
            "mission_phase": ["takeoff", "cruise", "landing", "cruise"],
        }
    )
    # 2 windows x 3 timesteps; source_row 101 is repeated within one
    # window and source_row 103 is repeated within the other.
    source_row_right = np.array([[100, 101, 101], [102, 103, 103]], dtype=np.int64)

    result = module._join_diagnostic_fields_by_source_row(source_row_right, raw_diagnostic)

    assert result["voltage_v_raw"].shape == source_row_right.shape
    expected_voltage = np.array([[3.60, 3.55, 3.55], [3.50, 3.45, 3.45]])
    assert np.allclose(result["voltage_v_raw"].astype(float), expected_voltage)
    assert result["development_phase_role"].tolist() == [
        ["high_1", "lower", "lower"],
        ["high_2", "lower", "lower"],
    ]
    assert result["source_rpt_id"].tolist() == [["R1", "R1", "R1"], ["R2", "R2", "R2"]]
    assert result["capacity_source"].tolist() == [["rpt", "rpt", "rpt"], ["rpt", "rpt", "rpt"]]


def test_join_diagnostic_fields_by_source_row_missing_values_are_none_not_nan_string() -> None:
    """Two kinds of missing values must both round-trip as real Python
    None, never the literal string "nan": (a) a matched source_row whose
    individual field (capacity_ah/capacity_source/source_rpt_id) is
    genuinely null in the raw parquet, and (b) a source_row_right with NO
    matching row in raw_diagnostic at all (defensive -- should not happen
    in practice, but must not silently fabricate a value)."""

    raw_diagnostic = pd.DataFrame(
        {
            "source_row": [200, 201],
            "development_phase_role": ["high_1", "lower"],
            "voltage_v_raw": [3.6, 3.5],
            "current_a_raw": [-1.0, -1.1],
            "temperature_c_raw": [24.0, 24.5],
            "capacity_ah": [None, 2.5],
            "capacity_source": [None, "rpt"],
            "source_rpt_id": [None, "R1"],
            "ns_raw": [4, 5],
            "mission_phase": ["takeoff", "cruise"],
        }
    )
    source_row_right = np.array([[200, 201, 999]], dtype=np.int64)  # 999 matches nothing

    result = module._join_diagnostic_fields_by_source_row(source_row_right, raw_diagnostic)

    # (a) matched row, individually-null fields.
    assert result["capacity_ah"][0, 0] is None
    assert result["capacity_source"][0, 0] is None
    assert result["source_rpt_id"][0, 0] is None
    assert result["development_phase_role"][0, 0] == "high_1"  # non-null field still comes through
    # (b) unmatched source_row -- every diagnostic field is null.
    for name in module.DIAGNOSTIC_METADATA_COLUMNS:
        assert result[name][0, 2] is None
    # Matched row with everything present.
    assert result["capacity_ah"][0, 1] == pytest.approx(2.5)
    assert result["source_rpt_id"][0, 1] == "R1"

    # Never the literal string "nan" anywhere.
    for name in module.DIAGNOSTIC_METADATA_COLUMNS:
        for value in result[name].reshape(-1):
            assert value != "nan"


def test_load_raw_diagnostic_fields_rejects_duplicate_source_row(tmp_path) -> None:
    """Defensive guard: a raw parquet with a duplicated source_row would
    silently multiply rows through the join (a plain merge duplicates left
    rows for each right-side match) -- must fail loudly instead."""

    path = tmp_path / "duplicate_source_row.parquet"
    pd.DataFrame(
        {
            "source_row": [1, 1],
            "development_phase_role": ["high_1", "high_1"],
            "voltage_v": [3.6, 3.6],
            "current_a": [-1.0, -1.0],
            "Temperature__C": [24.0, 24.0],
            "capacity_ah": [2.5, 2.5],
            "capacity_source": ["rpt", "rpt"],
            "source_rpt_id": ["R1", "R1"],
            "Ns": [4, 4],
        }
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="source_row"):
        module._load_raw_diagnostic_fields(path)


def test_run_bian_cmu_reuses_canonical_ns_mission_phase_map_by_identity() -> None:
    """This benchmark must directly reuse
    battery_benchmark.datasets.cmu_evtol.ns_phase_labeling_v1's
    NS_MISSION_PHASE_MAP/MISSION_FLIGHT_PHASE_ROLES -- not a copy, not a
    re-typed local dict with different spellings. Checked by object
    identity (``is``), which a copy (even a value-equal one) would fail."""

    from battery_benchmark.datasets.cmu_evtol import ns_phase_labeling_v1

    assert module.NS_MISSION_PHASE_MAP is ns_phase_labeling_v1.NS_MISSION_PHASE_MAP
    assert module.MISSION_FLIGHT_PHASE_ROLES is ns_phase_labeling_v1.MISSION_FLIGHT_PHASE_ROLES
    # DTNN's own previous, differently-spelled local copy must be gone.
    assert not hasattr(module, "BIAN_NS_MISSION_PHASE_MAP")
    assert not hasattr(module, "BIAN_FLIGHT_MISSION_PHASES")


def test_load_raw_diagnostic_fields_derives_mission_phase_from_ns_raw_and_keeps_ns_raw(
    tmp_path,
) -> None:
    """Requirement #1: ``ns_raw`` must survive unchanged (the raw Ns value,
    not just the semantic label), and ``mission_phase`` must be derived
    from it via the canonical NS_MISSION_PHASE_MAP -- every one of the 7
    mapped Ns values (0,1,3,4,5,6,7) round-trips to its documented label,
    and an unmapped Ns value (2 or 8, both real tester-native values with
    no entry in NS_MISSION_PHASE_MAP) becomes a real ``None``, never a
    fabricated label or a raised error."""

    path = tmp_path / "ns_mission_phase.parquet"
    ns_values = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    pd.DataFrame(
        {
            "source_row": list(range(len(ns_values))),
            # development_phase_role deliberately empty ("x", not one of
            # take_off/cruise/landing) so the new consistency check has
            # nothing to cross-validate against in this fixture -- that
            # cross-check is exercised separately below.
            "development_phase_role": ["x"] * len(ns_values),
            "voltage_v": [3.6] * len(ns_values),
            "current_a": [-1.0] * len(ns_values),
            "Temperature__C": [24.0] * len(ns_values),
            "capacity_ah": [2.5] * len(ns_values),
            "capacity_source": ["rpt"] * len(ns_values),
            "source_rpt_id": ["R1"] * len(ns_values),
            "Ns": ns_values,
        }
    ).to_parquet(path, index=False)

    frame = module._load_raw_diagnostic_fields(path)

    assert frame["ns_raw"].tolist() == ns_values
    expected_phase = [
        "cc_charge", "cv_charge", None, "rest_2",
        "take_off", "cruise", "landing", "rest_1", None,
    ]
    actual_phase = [
        None if pd.isna(value) else value for value in frame["mission_phase"].tolist()
    ]
    assert actual_phase == expected_phase
    # MISSION_FLIGHT_PHASE_ROLES names exactly the three flight-phase labels
    # the mapping produces for Ns 4/5/6.
    assert module.MISSION_FLIGHT_PHASE_ROLES == ("take_off", "cruise", "landing")
    for ns_value, phase in zip((4, 5, 6), module.MISSION_FLIGHT_PHASE_ROLES):
        assert module.NS_MISSION_PHASE_MAP[ns_value] == phase


def test_load_raw_diagnostic_fields_ns_4_5_6_match_take_off_cruise_landing(tmp_path) -> None:
    """Requirement #3/#5: for Ns=4/5/6 rows, mission_phase must equal
    take_off/cruise/landing exactly, AND must agree with
    development_phase_role when it is one of those three values -- the
    pre-run consistency check must pass silently on genuinely consistent
    data (the failure paths are covered by the two tests below)."""

    path = tmp_path / "consistent.parquet"
    ns_values = [4, 5, 6, 4, 5, 6]
    roles = ["take_off", "cruise", "landing", "", "", ""]  # some rows outside a detected candidate
    pd.DataFrame(
        {
            "source_row": list(range(len(ns_values))),
            "development_phase_role": roles,
            "voltage_v": [3.6] * len(ns_values),
            "current_a": [-1.0] * len(ns_values),
            "Temperature__C": [24.0] * len(ns_values),
            "capacity_ah": [2.5] * len(ns_values),
            "capacity_source": ["rpt"] * len(ns_values),
            "source_rpt_id": ["R1"] * len(ns_values),
            "Ns": ns_values,
        }
    ).to_parquet(path, index=False)

    frame = module._load_raw_diagnostic_fields(path)  # must not raise
    assert frame["mission_phase"].tolist() == ["take_off", "cruise", "landing"] * 2


def test_ns_mission_phase_consistency_check_rejects_role_disagreement(tmp_path) -> None:
    """Requirement #5: if development_phase_role says "cruise" for a row
    whose Ns is 4 (so mission_phase is derived as take_off), the pre-run
    consistency check must raise immediately, not silently write a
    mismatched row into test_predictions.parquet."""

    path = tmp_path / "inconsistent.parquet"
    pd.DataFrame(
        {
            "source_row": [0],
            "development_phase_role": ["cruise"],  # disagrees with Ns=4 -> take_off
            "voltage_v": [3.6],
            "current_a": [-1.0],
            "Temperature__C": [24.0],
            "capacity_ah": [2.5],
            "capacity_source": ["rpt"],
            "source_rpt_id": ["R1"],
            "Ns": [4],
        }
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="disagrees"):
        module._load_raw_diagnostic_fields(path)


def test_ns_mission_phase_consistency_check_rejects_corrupted_map(tmp_path) -> None:
    """Requirement #5 (the tautological half): if NS_MISSION_PHASE_MAP were
    ever edited so Ns=4 no longer maps to "take_off", the pre-run check
    must catch it rather than silently mislabeling every row. Simulated
    here by monkeypatching the map's already-`.map()`-applied output
    column directly is not representative, so instead this calls the
    checker function directly with a deliberately corrupted frame."""

    frame = pd.DataFrame(
        {
            "ns_raw": [4],
            "mission_phase": ["cruise"],  # should be "take_off" for ns_raw=4
            "development_phase_role": [""],
        }
    )
    with pytest.raises(ValueError, match="mission_phase"):
        module._check_ns_mission_phase_consistency(frame, Path("fake.parquet"))


def test_ns_raw_and_mission_phase_are_diagnostic_only_and_do_not_change_predictions() -> None:
    """Core requirement #7 for this addition specifically: joining ns_raw/
    mission_phase into test_metadata (exactly as main() now does, via
    _load_raw_diagnostic_fields + _join_diagnostic_fields_by_source_row)
    and passing that richer metadata into _evaluate() must not change a
    single soc_prediction/soc_reference value, nor the overall metrics --
    verified by running _evaluate() twice on the identical model/x/y/
    missions, once with metadata carrying ns_raw/mission_phase and once
    with metadata=None, and requiring every prediction's soc fields (and
    the whole `overall` dict) to match exactly."""

    torch.manual_seed(7)
    model = module.DTNN(2, 8, 4, 8)
    seq_len = 4
    x, y, missions = _synthetic_evaluate_inputs(seq_len=seq_len, seed=31)
    device = torch.device("cpu")

    rng = np.random.default_rng(11)
    metadata = {
        "mission_sequence_order": rng.integers(0, 5, size=y.shape).astype(np.int64),
        "elapsed_candidate_s": rng.uniform(0, 1000, size=y.shape),
        "original_time_s": rng.uniform(10_000, 20_000, size=y.shape),
        "source_row_right": rng.integers(0, 100_000, size=y.shape).astype(np.int64),
        "ns_raw": rng.integers(0, 8, size=y.shape).astype(np.int64),
        "mission_phase": np.full(y.shape, "cruise", dtype=object),
    }

    overall_with_ns, records_with_ns = module._evaluate(
        model, x, y, missions, batch_size=4, device=device,
        return_predictions=True, metadata=metadata,
    )
    overall_without, records_without = module._evaluate(
        model, x, y, missions, batch_size=4, device=device,
        return_predictions=True, metadata=None,
    )

    assert overall_with_ns == overall_without
    assert len(records_with_ns) == len(records_without)
    for with_ns, without in zip(records_with_ns, records_without):
        assert with_ns["soc_reference"] == without["soc_reference"]
        assert with_ns["soc_prediction"] == without["soc_prediction"]
        assert with_ns["signed_error"] == without["signed_error"]
        assert with_ns["absolute_error"] == without["absolute_error"]
        assert "ns_raw" in with_ns and "mission_phase" in with_ns
        assert "ns_raw" not in without and "mission_phase" not in without


def test_build_run_metadata_records_evaluation_unit_metric_units_and_provenance() -> None:
    """Requirements #4/#5/#6: run.json must document evaluation_unit
    ("all_timesteps", vs. Vidal's last_step_per_window), a metric_units
    dict distinguishing the legacy percent-scale metrics from the
    fraction-scale machine-readable diagnostics, git commit/dirty
    provenance, and the target_soc_labels_used_for_training flag -- all
    pure metadata, added without touching any existing key (checked
    against the same fixed values test_run_metadata_records_validation_
    interval_and_label_usage_flags already asserts)."""

    args = SimpleNamespace(
        validation_interval=7,
        panasonic_data_root=Path("/data/panasonic"),
        seed=1,
        repeats=2,
    )
    runs = [{"repeat": 1, "cell_id": "VAH01"}]

    metadata = module._build_run_metadata(args, runs)

    assert metadata["evaluation_unit"] == "all_timesteps"
    assert "last_step_per_window" in metadata["evaluation_unit_note"]
    assert metadata["metric_units"]["overall_mae_percent"].startswith("percent")
    assert metadata["metric_units"]["soc_reference"].startswith("fraction")
    assert metadata["metric_units"]["validation_mae"].startswith("fraction")
    assert metadata["target_soc_labels_used_for_training"] is True
    assert metadata["target_soc_labels_used_for_adaptation"] is True
    assert isinstance(metadata["git_commit"], str) and metadata["git_commit"]
    assert isinstance(metadata["git_dirty"], bool)
    # args here has no cmu_manifest attribute (matching the pre-existing
    # SimpleNamespace fixtures in this file) -- must degrade to None, never
    # raise AttributeError/FileNotFoundError.
    assert metadata["cmu_manifest_sha256"] is None
    assert "validation_protocol" in metadata
    json.dumps(metadata)  # still fully JSON-serializable


def test_build_run_metadata_computes_cmu_manifest_sha256_when_present(tmp_path) -> None:
    """When args.cmu_manifest points at a real, existing file, run.json
    must record its sha256 -- the same hash function already used
    elsewhere in this benchmark (sha256_file), for the CMU manifest the
    same way Vidal's run_metadata.json hashes its own manifest_path."""

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("cell_id\nVAH01\n", encoding="utf-8")
    args = SimpleNamespace(validation_interval=1, cmu_manifest=manifest_path)

    metadata = module._build_run_metadata(args, runs=[])

    assert metadata["cmu_manifest_sha256"] == module.sha256_file(manifest_path)


def test_diagnostic_join_end_to_end_through_evaluate_with_missing_and_unmatched_rows() -> None:
    """Full pipeline: fake target records -> _target_parts/_target_data
    (include_metadata=True, giving test_metadata['source_row_right']) ->
    _join_diagnostic_fields_by_source_row -> merged into test_metadata ->
    _evaluate(..., metadata=..., return_predictions=True). Two missions of
    different lengths (one with a dropped trailing window), a genuinely
    missing capacity_ah/capacity_source/source_rpt_id on one matched row,
    and one source_row with NO row at all in raw_diagnostic (defensive) --
    every prediction record's diagnostic fields must match raw_diagnostic
    at that record's own source_row_right, verified against an
    independently-built expectation, not the production join re-run."""

    initial_soc = 1.0
    fraction = 0.0
    length = 3

    soc_a = np.linspace(0.9, 0.1, 9)  # usable=9 -> 3 whole windows, nothing dropped
    soc_b = np.linspace(0.8, 0.2, 7)  # usable=6 -> 2 windows, last timestep dropped
    source_row_a = np.arange(9, dtype=np.int64) + 1000
    source_row_b = np.arange(7, dtype=np.int64) + 5000
    record_a = _fake_target_record(
        mission="MA", sequence_order=0, soc=soc_a, source_row_right=source_row_a
    )
    record_b = _fake_target_record(
        mission="MB", sequence_order=1, soc=soc_b, source_row_right=source_row_b
    )

    parts = module._target_parts(
        [record_a, record_b], {"MA", "MB"}, initial_soc, fraction, "test", include_metadata=True
    )
    test_x, test_y, missions, test_metadata = module._target_data(
        parts, length, feature_min=np.array([0.0, 0.0]), feature_max=np.array([20.0, 120.0]),
        include_metadata=True,
    )

    all_rows = np.concatenate([source_row_a, source_row_b])
    missing_row = int(all_rows[3])  # dropped from raw_diagnostic entirely -> unmatched
    present_rows = [int(r) for r in all_rows if r != missing_row]
    null_field_row = present_rows[0]  # matched, but with null capacity_ah/source/rpt_id
    raw_diagnostic = pd.DataFrame(
        {
            "source_row": present_rows,
            "development_phase_role": ["high_1"] * len(present_rows),
            "voltage_v_raw": [3.5 + 0.001 * r for r in present_rows],
            "current_a_raw": [-1.0 - 0.001 * r for r in present_rows],
            "temperature_c_raw": [25.0 + 0.001 * r for r in present_rows],
            "capacity_ah": [None if r == null_field_row else 2.5 for r in present_rows],
            "capacity_source": [None if r == null_field_row else "rpt" for r in present_rows],
            "source_rpt_id": [None if r == null_field_row else f"R{r}" for r in present_rows],
            "ns_raw": [4 + (r % 3) for r in present_rows],
            "mission_phase": [
                module.NS_MISSION_PHASE_MAP.get(4 + (r % 3)) for r in present_rows
            ],
        }
    )

    diagnostic_fields = module._join_diagnostic_fields_by_source_row(
        test_metadata["source_row_right"], raw_diagnostic
    )
    test_metadata = {**test_metadata, **diagnostic_fields}

    torch.manual_seed(4)
    model = module.DTNN(2, 8, 4, 8)
    _, records = module._evaluate(
        model, test_x, test_y, missions, batch_size=4, device=torch.device("cpu"),
        return_predictions=True, metadata=test_metadata,
    )

    raw_lookup = raw_diagnostic.set_index("source_row")
    checked_missing_field_row = False
    checked_unmatched_row = False
    for record in records:
        row = record["source_row_right"]
        for name in module.DIAGNOSTIC_METADATA_COLUMNS:
            assert record[name] != "nan"
        if row == missing_row:
            checked_unmatched_row = True
            for name in module.DIAGNOSTIC_METADATA_COLUMNS:
                assert record[name] is None
            continue
        expected = raw_lookup.loc[row]
        if row == null_field_row:
            checked_missing_field_row = True
            assert record["capacity_ah"] is None
            assert record["capacity_source"] is None
            assert record["source_rpt_id"] is None
        else:
            assert record["capacity_ah"] == pytest.approx(float(expected["capacity_ah"]))
            assert record["capacity_source"] == expected["capacity_source"]
            assert record["source_rpt_id"] == expected["source_rpt_id"]
        assert record["development_phase_role"] == expected["development_phase_role"]
        assert record["voltage_v_raw"] == pytest.approx(float(expected["voltage_v_raw"]))
        assert record["current_a_raw"] == pytest.approx(float(expected["current_a_raw"]))
        assert record["temperature_c_raw"] == pytest.approx(float(expected["temperature_c_raw"]))

    assert checked_missing_field_row
    assert checked_unmatched_row


def test_multi_chunk_write_preserves_diagnostic_fields_and_nulls(tmp_path) -> None:
    """Requirement #4/#5 together: the diagnostic columns are part of the
    same fixed TEST_PREDICTIONS_SCHEMA and go through the same streaming
    ParquetWriter chunk-at-a-time mechanism as every other column -- write
    several repeat x cell x method chunks (some rows with real nulls in
    the diagnostic fields, matching what _sample_prediction_chunk already
    produces), close, and read back: every value must survive, and a null
    must come back as a real pandas/pyarrow null, never the string
    "nan"."""

    tmp_output = tmp_path / "test_predictions.parquet.incomplete"
    writer = pq.ParquetWriter(tmp_output, module.TEST_PREDICTIONS_SCHEMA)
    chunks = [
        _sample_prediction_chunk(repeat=1, seed=1, cell_id="VAH01", method="Source only", n_records=4),
        _sample_prediction_chunk(repeat=1, seed=1, cell_id="VAH01", method="Bian MDA", n_records=4),
        _sample_prediction_chunk(repeat=2, seed=2, cell_id="VAH17", method="Source only", n_records=5),
    ]
    for chunk in chunks:
        module._write_prediction_chunk(writer, chunk)
    writer.close()
    final_path = tmp_path / "test_predictions.parquet"
    tmp_output.replace(final_path)

    frame = pq.read_table(final_path).to_pandas()
    expected = pd.DataFrame([row for chunk in chunks for row in chunk])

    assert len(frame) == len(expected)
    diagnostic_columns = (
        "development_phase_role", "voltage_v_raw", "current_a_raw", "temperature_c_raw",
        "capacity_ah", "capacity_source", "source_rpt_id",
    )
    for column in diagnostic_columns:
        assert column in frame.columns
        for value in frame[column]:
            assert value != "nan"
    # _sample_prediction_chunk sets development_phase_role/capacity_ah/
    # capacity_source/source_rpt_id to None for some rows (i % 2 == 0 or
    # i % 3 == 0) -- at least one such row must exist in this chunk set,
    # and it must read back as a real null, never the text "nan".
    for column in ("development_phase_role", "capacity_ah", "capacity_source", "source_rpt_id"):
        assert frame[column].isna().any()

    # And every non-null value round-trips exactly.
    frame_sorted = frame.sort_values(list(frame.columns)).reset_index(drop=True)
    expected_sorted = expected[list(frame.columns)].sort_values(list(frame.columns)).reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(frame_sorted, expected_sorted, check_dtype=False)


# ---------------------------------------------------------------------------
# Full-mission diagnostic (RQ3/RQ4) -- entirely additive, diagnostic-only.
# These tests exist to prove the addition cannot touch the primary
# benchmark: _target_parts/_target_data/_adapt/_evaluate themselves are
# never edited by it (see the module source), so the tests below focus on
# (a) the new full-mission data construction reusing the same
# truncation/reshape as the primary split, (b) the diagnostic evaluate
# helper's RNG isolation, (c) that inserting the diagnostic between two
# cells cannot change the second cell's trained weights or predictions,
# and (d) the phase summaries are exactly recomputable from the parquet.
# ---------------------------------------------------------------------------


def test_target_parts_full_mission_keeps_entire_mission_no_truncation_no_split() -> None:
    """Requirement #2: no initial_soc truncation, no adapt/test split --
    every raw timestep of a selected mission is kept, and a mission not in
    mission_ids is skipped entirely (matching _target_parts' own
    filtering)."""
    soc = np.linspace(0.95, 0.05, 10)
    record = _fake_target_record(mission="M1", sequence_order=2, soc=soc)
    other = _fake_target_record(mission="OTHER", sequence_order=9, soc=soc)

    parts = module._target_parts_full_mission([record, other], {"M1"})

    assert len(parts) == 1
    x, y, mission, metadata = parts[0]
    assert mission == "M1"
    frame = record.data
    assert np.array_equal(y, frame["soc"].to_numpy())
    assert np.array_equal(x, frame.loc[:, module.FEATURES].to_numpy())
    assert np.array_equal(metadata["elapsed_candidate_s"], frame["elapsed_candidate_s"].to_numpy())
    assert np.array_equal(metadata["original_time_s"], frame["original_time_s"].to_numpy())
    assert np.array_equal(metadata["source_row_right"], frame["source_row_right"].to_numpy())
    assert np.all(metadata["mission_sequence_order"] == 2)
    assert len(x) == len(soc)  # nothing truncated -- the WHOLE mission


def test_target_data_full_mission_reuses_same_windows_truncation_as_primary_split() -> None:
    """Requirement #2: normalization, window length, and the trailing-
    truncation rule all come from the completely unmodified
    _windows()/_target_data() -- verified here the same way the primary
    split's own metadata-alignment tests already do (see
    test_target_data_metadata_survives_truncation_with_correct_dropped_tail),
    just fed from _target_parts_full_mission instead of _target_parts."""
    soc = np.linspace(0.9, 0.1, 13)  # 13 timesteps, length=3 -> usable=12, 1 dropped
    record = _fake_target_record(mission="M2", sequence_order=4, soc=soc)
    length = 3

    parts = module._target_parts_full_mission([record], {"M2"})
    x, y, missions, metadata = module._target_data(
        parts, length,
        feature_min=np.array([0.0, 0.0]), feature_max=np.array([20.0, 120.0]),
        include_metadata=True,
    )

    assert missions == [("M2", 4)]
    assert x.shape == (4, 3, 2)
    assert y.shape == (4, 3)
    frame = record.data
    kept = slice(0, 12)
    assert np.array_equal(y.reshape(-1), frame["soc"].to_numpy()[kept].astype(np.float32))
    assert np.array_equal(
        metadata["source_row_right"].reshape(-1), frame["source_row_right"].to_numpy()[kept]
    )


def test_full_mission_diagnostic_join_has_nonzero_samples_for_all_three_phases(tmp_path) -> None:
    """Requirement #8: full-mission output must have non-zero take_off/
    cruise/landing samples -- built end-to-end from a synthetic mission
    covering Ns 4/5/6/7 (cycled), through _target_parts_full_mission ->
    _target_data -> _join_diagnostic_fields_by_source_row, exactly the
    pipeline main()'s full-mission diagnostic block uses."""
    n = 24
    source_row_right = np.arange(n, dtype=np.int64) + 1000
    soc = np.linspace(0.95, 0.05, n)
    record = _fake_target_record(
        mission="M1", sequence_order=0, soc=soc, source_row_right=source_row_right,
    )
    ns_values = [4, 5, 6, 7] * (n // 4)

    raw_path = tmp_path / "raw.parquet"
    pd.DataFrame(
        {
            "source_row": source_row_right,
            "development_phase_role": [""] * n,
            "voltage_v": [3.6] * n,
            "current_a": [-1.0] * n,
            "Temperature__C": [24.0] * n,
            "capacity_ah": [2.5] * n,
            "capacity_source": ["rpt"] * n,
            "source_rpt_id": ["R1"] * n,
            "Ns": ns_values,
        }
    ).to_parquet(raw_path, index=False)
    raw_diagnostic = module._load_raw_diagnostic_fields(raw_path)

    parts = module._target_parts_full_mission([record], {"M1"})
    x, y, missions, metadata = module._target_data(
        parts, 4, feature_min=np.array([0.0, 0.0]), feature_max=np.array([20.0, 120.0]),
        include_metadata=True,
    )
    metadata = {
        **metadata,
        **module._join_diagnostic_fields_by_source_row(
            metadata["source_row_right"], raw_diagnostic
        ),
    }

    phases, counts = np.unique(metadata["mission_phase"], return_counts=True)
    phase_counts = dict(zip(phases, counts))
    for phase in module.FULL_MISSION_SUMMARY_PHASES:
        assert phase_counts.get(phase, 0) > 0


def test_run_full_mission_diagnostic_evaluate_does_not_perturb_global_rng() -> None:
    """Requirement #8: the diagnostic inference call must not change
    torch's global RNG state -- checked directly around the helper main()
    actually calls for the full-mission diagnostic."""
    torch.manual_seed(0)
    model = module.DTNN(2, 8, 4, 8)
    x = np.random.default_rng(1).normal(size=(3, 4, 2)).astype(np.float32)
    y = np.random.default_rng(2).uniform(0, 1, size=(3, 4)).astype(np.float32)
    missions = [("m1", 3)]

    rng_state_before = torch.get_rng_state()
    overall, predictions = module._run_full_mission_diagnostic_evaluate(
        model, x, y, missions, 2, torch.device("cpu"), None,
    )
    rng_state_after = torch.get_rng_state()

    assert torch.equal(rng_state_before, rng_state_after)
    assert isinstance(overall, dict) and predictions


def _run_two_cell_sequence(*, with_full_mission_diagnostic: bool):
    """Simulates main()'s per-repeat, per-cell sequence for exactly two
    cells sharing one global RNG stream (as main() itself does): a fresh
    model per cell, loaded from the same frozen source_state, adapted, and
    evaluated -- optionally with the full-mission diagnostic block's two
    extra evaluate calls inserted after cell1 finishes, exactly where
    main() runs them (after the primary evaluate/adapt for that cell,
    before moving to the next cell)."""
    import copy as _copy

    torch.manual_seed(123)
    widths = (8, 4, 8)
    source_x, source_y, target_x, target_y = _synthetic_adapt_inputs(seed=1)
    source_model = module.DTNN(2, *widths)
    source_state = _copy.deepcopy(source_model.state_dict())
    args = _adapt_args(transfer_epochs=2, transfer_batch_size=2, batch_size=2)
    device = torch.device("cpu")
    missions = [("m1", len(target_x))]

    results = {}
    for cell in ("cell1", "cell2"):
        model = module.DTNN(2, *widths)
        model.load_state_dict(source_state)
        module._adapt(model, source_x, source_y, target_x, target_y, args, device)
        overall, predictions = module._evaluate(
            model, target_x, target_y, missions, args.batch_size, device,
            return_predictions=True,
        )
        results[cell] = (
            {name: value.clone() for name, value in model.state_dict().items()},
            overall,
            [p["soc_prediction"] for p in predictions],
        )
        if cell == "cell1" and with_full_mission_diagnostic:
            # Mirrors main()'s own full-mission diagnostic block exactly:
            # the fresh model's construction (weight init draws from the
            # RNG too) is inside the SAME _preserve_torch_rng_state guard
            # as the evaluate calls -- not just each evaluate call
            # individually.
            with module._preserve_torch_rng_state():
                diagnostic_source_model = module.DTNN(2, *widths)
                diagnostic_source_model.load_state_dict(source_state)
                module._run_full_mission_diagnostic_evaluate(
                    diagnostic_source_model, target_x, target_y, missions,
                    args.batch_size, device, None,
                )
                module._run_full_mission_diagnostic_evaluate(
                    model, target_x, target_y, missions, args.batch_size, device, None,
                )
    return results


def test_full_mission_diagnostic_toggle_does_not_change_next_cell_state_dict_or_predictions() -> (
    None
):
    """Requirement #8's core safety property: turning the full-mission
    diagnostic on/off must not change the SECOND cell's final state_dict
    or primary predictions -- the diagnostic's two extra evaluate() calls
    happen strictly after cell1's own primary evaluate/adapt, and must
    leave torch's global RNG exactly as it would have been without them
    (see _run_full_mission_diagnostic_evaluate), so cell2 (which shares
    the same global RNG stream) trains and evaluates identically either
    way."""
    without_diagnostic = _run_two_cell_sequence(with_full_mission_diagnostic=False)
    with_diagnostic = _run_two_cell_sequence(with_full_mission_diagnostic=True)

    for cell in ("cell1", "cell2"):
        base_state, base_overall, base_predictions = without_diagnostic[cell]
        diag_state, diag_overall, diag_predictions = with_diagnostic[cell]
        for name in base_state:
            assert torch.equal(base_state[name], diag_state[name])
        assert base_overall == diag_overall
        assert base_predictions == pytest.approx(diag_predictions)


def test_phase_metrics_full_mission_diagnostic_is_recomputable_from_parquet(tmp_path) -> None:
    """Requirement #5/#8: phase_metrics_full_mission_diagnostic.csv's
    per-(repeat, seed, cell_id, method, mission_phase) n_points/n_missions/
    mae_percent/rmse_percent/mean_signed_error_percent must be exactly
    recomputable from test_predictions_full_mission_diagnostic.parquet's
    own soc_reference/soc_prediction/mission_phase -- checked here against
    an INDEPENDENTLY, manually computed expectation, and confirms
    non-flight phases (e.g. cc_charge) are excluded from the summary."""
    path = tmp_path / "full_mission.parquet"
    take_off_rows = [
        {"repeat": 1, "seed": 10, "cell_id": "VAH01", "method": "Source only",
         "mission_phase": "take_off", "mission_cycle_id": "m1",
         "soc_reference": 0.90, "soc_prediction": 0.86},
        {"repeat": 1, "seed": 10, "cell_id": "VAH01", "method": "Source only",
         "mission_phase": "take_off", "mission_cycle_id": "m1",
         "soc_reference": 0.80, "soc_prediction": 0.79},
        {"repeat": 1, "seed": 10, "cell_id": "VAH01", "method": "Source only",
         "mission_phase": "take_off", "mission_cycle_id": "m2",
         "soc_reference": 0.50, "soc_prediction": 0.60},
    ]
    other_rows = [
        {"repeat": 1, "seed": 10, "cell_id": "VAH01", "method": "Source only",
         "mission_phase": "cruise", "mission_cycle_id": "m1",
         "soc_reference": 0.40, "soc_prediction": 0.42},
        {"repeat": 1, "seed": 10, "cell_id": "VAH01", "method": "Source only",
         "mission_phase": "cc_charge", "mission_cycle_id": "m1",
         "soc_reference": 0.10, "soc_prediction": 0.90},
        {"repeat": 1, "seed": 10, "cell_id": "VAH01", "method": "Bian MDA",
         "mission_phase": "take_off", "mission_cycle_id": "m1",
         "soc_reference": 0.90, "soc_prediction": 0.91},
    ]
    pd.DataFrame(take_off_rows + other_rows).to_parquet(path, index=False)

    summary = module._compute_phase_metrics_full_mission_diagnostic(path)

    # Non-flight phase (cc_charge) must not appear at all.
    assert set(summary["mission_phase"]) == {"take_off", "cruise"}

    take_off_source = summary.loc[
        (summary["method"] == "Source only") & (summary["mission_phase"] == "take_off")
    ].iloc[0]
    manual = pd.DataFrame(take_off_rows)
    manual_signed = manual["soc_prediction"] - manual["soc_reference"]
    assert take_off_source["n_points"] == 3
    assert take_off_source["n_missions"] == 2
    assert take_off_source["mae_percent"] == pytest.approx(manual_signed.abs().mean() * 100)
    assert take_off_source["rmse_percent"] == pytest.approx(
        float(np.sqrt((manual_signed**2).mean()) * 100)
    )
    assert take_off_source["mean_signed_error_percent"] == pytest.approx(
        manual_signed.mean() * 100
    )


def test_phase_transfer_effects_full_mission_diagnostic_matches_manual_deltas() -> None:
    """Requirement #6: delta_mae_percent/delta_rmse_percent =
    adapted - source, plus a boolean negative-transfer flag for each,
    computed per (repeat, seed, cell_id, mission_phase)."""
    phase_metrics = pd.DataFrame(
        [
            {"repeat": 1, "seed": 10, "cell_id": "VAH01", "method": "Source only",
             "mission_phase": "take_off", "n_points": 3, "n_missions": 2,
             "mae_percent": 5.0, "rmse_percent": 6.0, "mean_signed_error_percent": -1.0},
            {"repeat": 1, "seed": 10, "cell_id": "VAH01", "method": "Bian MDA",
             "mission_phase": "take_off", "n_points": 3, "n_missions": 2,
             "mae_percent": 7.0, "rmse_percent": 5.5, "mean_signed_error_percent": 0.5},
        ]
    )

    effects = module._compute_phase_transfer_effects_full_mission_diagnostic(phase_metrics)

    assert len(effects) == 1
    row = effects.iloc[0]
    assert row["delta_mae_percent"] == pytest.approx(2.0)
    assert row["delta_rmse_percent"] == pytest.approx(-0.5)
    assert bool(row["negative_transfer_mae"]) is True
    assert bool(row["negative_transfer_rmse"]) is False


def test_build_run_metadata_records_primary_scope_and_full_mission_diagnostic_role() -> None:
    """Requirement #7: run.json must state that the primary evaluation
    scope is unchanged, and that the full-mission diagnostic never
    participates in training/early stopping/checkpoint selection and never
    replaces the primary benchmark metrics -- pure metadata, degrading to
    enabled=False for a bare SimpleNamespace args without the attribute
    (matching this file's other _build_run_metadata fixtures)."""
    args = SimpleNamespace(validation_interval=5)

    metadata = module._build_run_metadata(args, runs=[])

    assert metadata["primary_evaluation_scope"] == "initial_soc_70_then_test_70_percent"
    diagnostic = metadata["full_mission_diagnostic"]
    assert diagnostic["enabled"] is False
    assert diagnostic["used_for_training"] is False
    assert diagnostic["used_for_early_stopping"] is False
    assert diagnostic["used_for_checkpoint_selection"] is False
    assert diagnostic["replaces_primary_benchmark_metrics"] is False
    json.dumps(metadata)

    args_enabled = SimpleNamespace(validation_interval=5, full_mission_diagnostic=True)
    metadata_enabled = module._build_run_metadata(args_enabled, runs=[])
    assert metadata_enabled["full_mission_diagnostic"]["enabled"] is True


def test_write_prediction_chunk_accepts_full_mission_diagnostic_schema(tmp_path) -> None:
    """The generalized _write_prediction_chunk(schema=...) must round-trip
    the extra evaluation_scope column for FULL_MISSION_DIAGNOSTIC_SCHEMA,
    while every existing (schema-less) call site is completely
    unaffected."""
    path = tmp_path / "full_mission.parquet"
    writer = pq.ParquetWriter(path, module.FULL_MISSION_DIAGNOSTIC_SCHEMA)
    base_row = _sample_prediction_chunk(
        repeat=1, seed=10, cell_id="VAH01", method="Source only", n_records=2
    )[0]
    row = {**base_row, "evaluation_scope": "full_mission_diagnostic"}
    module._write_prediction_chunk(writer, [row], schema=module.FULL_MISSION_DIAGNOSTIC_SCHEMA)
    writer.close()

    frame = pq.read_table(path).to_pandas()
    assert frame.loc[0, "evaluation_scope"] == "full_mission_diagnostic"
    assert list(frame.columns) == [field.name for field in module.FULL_MISSION_DIAGNOSTIC_SCHEMA]
