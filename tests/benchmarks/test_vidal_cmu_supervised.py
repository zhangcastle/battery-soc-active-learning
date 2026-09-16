from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/benchmarks/run_cmu_transfer.py"
spec = importlib.util.spec_from_file_location("run_cmu_transfer", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_setting_selection_uses_fixed_splits_without_leakage() -> None:
    rows = []
    for rank in range(1, 11):
        split = "train" if rank <= 6 else "validation" if rank <= 8 else "test"
        rows.append(
            {
                "cell_id": "VAH01",
                "mission_cycle_id": f"m{rank}",
                "split": split,
                "budget_1pct": rank == 1,
                "budget_5pct": rank == 1,
                "budget_10pct": rank == 1,
                "vidal_partial": rank <= 3,
                "vidal_full": rank <= 6,
            }
        )
    manifest = pd.DataFrame(rows)

    partial, validation, test = module.select_mission_ids(
        manifest, "VAH01", "vidal_partial"
    )
    full, validation_again, test_again = module.select_mission_ids(
        manifest, "VAH01", "vidal_full"
    )
    assert partial == {"m1", "m2", "m3"}
    assert full == {"m1", "m2", "m3", "m4", "m5", "m6"}
    assert validation == validation_again == {"m7", "m8"}
    assert test == test_again == {"m9", "m10"}
    assert not partial & validation
    assert not full & test


def test_zero_shot_selects_no_target_training_labels() -> None:
    manifest = pd.DataFrame(
        [
            {
                "cell_id": "VAH01",
                "mission_cycle_id": "train",
                "split": "train",
                "budget_1pct": True,
                "budget_5pct": True,
                "budget_10pct": True,
                "vidal_partial": True,
                "vidal_full": True,
            },
            {
                "cell_id": "VAH01",
                "mission_cycle_id": "validation",
                "split": "validation",
                "budget_1pct": False,
                "budget_5pct": False,
                "budget_10pct": False,
                "vidal_partial": False,
                "vidal_full": False,
            },
            {
                "cell_id": "VAH01",
                "mission_cycle_id": "test",
                "split": "test",
                "budget_1pct": False,
                "budget_5pct": False,
                "budget_10pct": False,
                "vidal_partial": False,
                "vidal_full": False,
            },
        ]
    )
    train, validation, test = module.select_mission_ids(
        manifest, "VAH01", "zero_shot"
    )
    assert train == set()
    assert validation == {"validation"}
    assert test == {"test"}


def _synthetic_windows(n: int, seq_len: int = 50, seed: int = 0):
    """Minimal Case1Windows-compatible object for a short, data-free smoke
    test of _train()'s per-epoch validation diagnostic -- no real CMU
    parquet/manifest/checkpoint fixtures needed. metadata is unused by
    _train()/_validation_diagnostic_metrics(), only by the full
    evaluate_target_last() path, so it is left empty here."""

    from battery_benchmark.reproductions.vidal_2019.windows import Case1Windows

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, seq_len, 3)).astype(np.float32)
    y = rng.uniform(0, 1, size=(n, seq_len)).astype(np.float32)
    mask = np.ones((n, seq_len), dtype=bool)
    return Case1Windows(x, y, mask, [{} for _ in range(n)], 0)


def _run_train_smoke(record_validation_history: bool, validation_windows):
    train_windows = _synthetic_windows(20, seed=1)
    torch.manual_seed(42)
    np.random.seed(42)
    model = module.VidalLSTM()
    cfg = module.RunConfig(
        setting="vidal_full",
        epochs=3,
        batch_size=8,
        record_validation_history=record_validation_history,
    )
    history, _optimizer, _scheduler = module._train(
        model, train_windows, cfg, torch.device("cpu"),
        validation_windows=validation_windows,
    )
    return model, history


def test_validation_history_diagnostic_columns_toggle() -> None:
    validation_windows = _synthetic_windows(8, seed=2)

    _model_on, history_on = _run_train_smoke(True, validation_windows)
    _model_off, history_off = _run_train_smoke(False, validation_windows)

    assert len(history_on) == len(history_off) == 3
    assert {"validation_mae", "validation_rmse"} <= set(history_on.columns)
    assert not ({"validation_mae", "validation_rmse"} & set(history_off.columns))
    assert history_on["validation_mae"].notna().all()
    assert history_on["validation_rmse"].notna().all()


def test_validation_history_toggle_does_not_change_training() -> None:
    """The diagnostic must be read-only: enabling/disabling it must not
    change the training trajectory, the final model parameters, or the
    model's subsequent predictions (515-epoch full runs are not exercised
    here -- 3 epochs on synthetic data is enough to prove the toggle has no
    side effect on training itself)."""

    validation_windows = _synthetic_windows(8, seed=2)

    model_on, history_on = _run_train_smoke(True, validation_windows)
    model_off, history_off = _run_train_smoke(False, validation_windows)

    assert np.allclose(
        history_on["train_mse"].to_numpy(), history_off["train_mse"].to_numpy()
    )
    assert np.allclose(
        history_on["learning_rate"].to_numpy(), history_off["learning_rate"].to_numpy()
    )

    state_on, state_off = model_on.state_dict(), model_off.state_dict()
    assert set(state_on) == set(state_off)
    for key in state_on:
        assert torch.equal(state_on[key], state_off[key]), f"parameter {key} diverged"

    probe = torch.from_numpy(_synthetic_windows(5, seed=99).X)
    model_on.eval()
    model_off.eval()
    with torch.no_grad():
        predictions_on = model_on(probe)
        predictions_off = model_off(probe)
    assert torch.equal(predictions_on, predictions_off)


def test_validation_history_diagnostic_skipped_without_validation_windows() -> None:
    """record_validation_history=True with no validation_windows passed in
    (e.g. the zero_shot setting, which never calls _train at all -- but any
    other caller that omits validation_windows too) must not crash and must
    simply omit the diagnostic columns."""

    _model, history = _run_train_smoke(True, None)
    assert not ({"validation_mae", "validation_rmse"} & set(history.columns))
