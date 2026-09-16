from __future__ import annotations

import importlib.util
import math
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
