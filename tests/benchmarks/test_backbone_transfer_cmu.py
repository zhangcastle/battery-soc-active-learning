"""Diagnostic-only additions to run_backbone_transfer_cmu.py.

Covers metadata alignment through _windows()'s non-overlapping truncation,
the shared source_row_right join, Ns -> mission_phase mapping, the
diagnostic-only validation probe never perturbing training RNG/weights, and
recomputing error from test_predictions matching _evaluate's own metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from battery_benchmark.benchmarks.cmu_diagnostics import load_raw_diagnostic_fields
from scripts.benchmarks.run_backbone_transfer_cmu import (
    _build_prediction_rows,
    _evaluate,
    _loader,
    _predict_all,
    _target_windows,
    _train,
    _validation_diagnostic_metrics,
    _windows,
)


class _LinearProbe(nn.Module):
    """Tiny deterministic model: per-timestep linear map, no randomness in
    forward -- used so this test suite doesn't depend on any real backbone
    architecture."""

    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def _synthetic_records(mission, sequence_order, n, start_row):
    frame = pd.DataFrame(
        {
            "current": np.linspace(-1, 1, n, dtype=np.float64),
            "voltage": np.linspace(3.0, 4.2, n, dtype=np.float64),
            "temperature": np.linspace(20, 30, n, dtype=np.float64),
            "soc": np.linspace(1.0, 0.0, n, dtype=np.float64),
            "elapsed_candidate_s": np.arange(n, dtype=np.float64),
            "original_time_s": np.arange(n, dtype=np.float64) + 2000.0,
            "source_row_right": np.arange(start_row, start_row + n, dtype=np.int64),
        }
    )

    class Record:
        metadata = {"mission_cycle_id": mission, "sequence_order": sequence_order}
        data = frame

    return Record()


def test_windows_include_metadata_matches_unmetadata_output():
    x = np.random.default_rng(0).normal(size=(25, 3))
    y = np.random.default_rng(1).normal(size=25)
    parts_plain = [(x, y, "m1")]
    x_ref, y_ref, ids_ref = _windows(parts_plain, 10)

    metadata = {
        "mission_sequence_order": np.zeros(25, dtype=np.int64),
        "elapsed_candidate_s": np.arange(25, dtype=np.float64),
        "original_time_s": np.arange(25, dtype=np.float64),
        "source_row_right": np.arange(25, dtype=np.int64),
    }
    parts_meta = [(x, y, "m1", metadata)]
    x_meta, y_meta, ids_meta, metadata_out = _windows(parts_meta, 10, include_metadata=True)

    assert np.array_equal(x_ref, x_meta)
    assert np.array_equal(y_ref, y_meta)
    assert ids_ref == ids_meta
    assert metadata_out["source_row_right"].shape == (2, 10)
    assert np.array_equal(metadata_out["source_row_right"][0], np.arange(10))
    assert np.array_equal(metadata_out["source_row_right"][1], np.arange(10, 20))


def test_target_windows_with_metadata_survives_multi_mission_and_truncation():
    records = [
        _synthetic_records("short", 0, 12, start_row=0),
        _synthetic_records("long", 1, 27, start_row=1000),
    ]
    feature_min = np.array([-1.0, 3.0, 20.0])
    feature_max = np.array([1.0, 4.2, 30.0])
    x, y, missions, metadata = _target_windows(
        records, {"short", "long"}, feature_min, feature_max, 10, include_metadata=True
    )
    assert missions == [("short", 1), ("long", 2)]
    assert metadata["source_row_right"].shape == (3, 10)
    # "short" mission's single window covers rows [0, 10)
    assert np.array_equal(metadata["source_row_right"][0], np.arange(10))
    # "long" mission's first window covers rows [1000, 1010)
    assert np.array_equal(metadata["source_row_right"][1], np.arange(1000, 1010))
    assert (metadata["mission_sequence_order"][0] == 0).all()
    assert (metadata["mission_sequence_order"][1] == 1).all()


def test_predict_all_matches_evaluate_error():
    torch.manual_seed(0)
    model = _LinearProbe(3)
    x = np.random.default_rng(2).normal(size=(4, 10, 3)).astype(np.float32)
    y = np.random.default_rng(3).uniform(0, 1, size=(4, 10)).astype(np.float32)
    missions = [("m1", 2), ("m2", 2)]
    device = torch.device("cpu")

    metrics = _evaluate(model, x, y, missions, batch_size=4, device=device)
    prediction = _predict_all(model, x, batch_size=4, device=device)
    clipped = np.clip(prediction, 0.0, 1.0)

    recomputed_mae = float(np.mean(np.abs(clipped.ravel() - y.ravel())) * 100)
    recomputed_rmse = float(np.sqrt(np.mean((clipped.ravel() - y.ravel()) ** 2)) * 100)
    assert np.isclose(recomputed_mae, metrics["overall_mae_percent"])
    assert np.isclose(recomputed_rmse, metrics["overall_rmse_percent"])


def test_validation_diagnostic_does_not_perturb_training_rng_or_weights():
    torch.manual_seed(0)
    model = _LinearProbe(3)
    before_state = {k: v.clone() for k, v in model.state_dict().items()}
    before_mode = model.training

    x = np.random.default_rng(4).normal(size=(6, 10, 3)).astype(np.float32)
    y = np.random.default_rng(5).uniform(0, 1, size=(6, 10)).astype(np.float32)

    # Snapshot the default generator's state, run the diagnostic, and
    # confirm it is unchanged: the validation loader must use its own
    # local generator, never torch.default_generator.
    rng_state_before = torch.get_rng_state()
    mae, rmse = _validation_diagnostic_metrics(model, x, y, batch_size=6, device=torch.device("cpu"))
    rng_state_after = torch.get_rng_state()

    assert torch.equal(rng_state_before, rng_state_after)
    assert model.training == before_mode
    after_state = model.state_dict()
    for key in before_state:
        assert torch.equal(before_state[key], after_state[key])
    assert isinstance(mae, float) and isinstance(rmse, float)


def test_train_with_validation_data_reports_diagnostics_without_leaking_into_training():
    torch.manual_seed(0)
    model = _LinearProbe(3)
    x = np.random.default_rng(6).normal(size=(4, 10, 3)).astype(np.float32)
    y = np.random.default_rng(7).uniform(0, 1, size=(4, 10)).astype(np.float32)
    val_x = np.random.default_rng(8).normal(size=(2, 10, 3)).astype(np.float32)
    val_y = np.random.default_rng(9).uniform(0, 1, size=(2, 10)).astype(np.float32)

    history_with_val = _train(
        model, x, y, epochs=2, batch_size=4, lr=1e-3, device=torch.device("cpu"),
        label="test", validation_data=(val_x, val_y),
    )
    assert all(row["validation_evaluated"] for row in history_with_val)
    assert all(row["validation_mae"] is not None for row in history_with_val)

    torch.manual_seed(0)
    model2 = _LinearProbe(3)
    history_no_val = _train(
        model2, x, y, epochs=2, batch_size=4, lr=1e-3, device=torch.device("cpu"), label="test",
    )
    assert all(row["validation_evaluated"] is False for row in history_no_val)
    assert all(row["validation_mae"] is None for row in history_no_val)
    # Training curves (train_mse) must be identical whether or not the
    # diagnostic validation probe ran -- it must never affect training.
    assert [row["train_mse"] for row in history_with_val] == [row["train_mse"] for row in history_no_val]


def _write_raw_diagnostic_parquet(path, n, ns_values, start_row=0):
    frame = pd.DataFrame(
        {
            "source_row": np.arange(start_row, start_row + n, dtype=np.int64),
            "voltage_v": np.linspace(3.0, 4.2, n),
            "current_a": np.linspace(-5.0, 5.0, n),
            "Temperature__C": np.linspace(20.0, 30.0, n),
            "capacity_ah": np.full(n, 2.5),
            "capacity_source": ["rpt"] * n,
            "source_rpt_id": [f"rpt_{i}" for i in range(n)],
            "Ns": ns_values,
        }
    )
    frame.to_parquet(path)


def test_build_prediction_rows_joins_ns_and_all_three_models_share_schema(tmp_path):
    path = tmp_path / "raw.parquet"
    _write_raw_diagnostic_parquet(path, 4, ns_values=[4, 5, 6, 7])
    raw_diagnostic = load_raw_diagnostic_fields(path)

    metadata = {
        "mission_sequence_order": np.array([0, 0, 0, 0]),
        "elapsed_candidate_s": np.arange(4, dtype=np.float64),
        "original_time_s": np.arange(4, dtype=np.float64),
        "source_row_right": np.array([0, 1, 2, 3], dtype=np.int64),
    }
    for model_name in ("cnn_lstm", "tcn", "transformer_encoder"):
        rows = _build_prediction_rows(
            model_name=model_name,
            method="fine_tuned",
            budget="5pct",
            seed=0,
            cell_id="VAH01",
            condition="baseline protocol",
            soc_reference=np.array([0.9, 0.8, 0.7, 0.6]),
            soc_prediction=np.array([0.91, 0.79, 0.72, 0.55]),
            metadata=metadata,
            missions=[("m1", 4)],
            sequence_length=4,
            raw_diagnostic=raw_diagnostic,
        )
        assert [row["mission_phase"] for row in rows] == ["takeoff", "cruise", "landing", "post_flight_rest"]
        assert rows[0]["model"] == model_name
        # Identical schema keys regardless of model, per Section V.4.
        assert set(rows[0].keys()) == {
            "model", "method", "budget", "seed", "cell_id", "condition", "split",
            "mission_cycle_id", "window_index", "timestep_index",
            "mission_sequence_order", "elapsed_candidate_s", "original_time_s",
            "source_row_right", "soc_reference", "soc_prediction", "signed_error",
            "absolute_error", "voltage_v_raw", "current_a_raw", "temperature_c_raw",
            "capacity_ah", "capacity_source", "source_rpt_id", "ns_raw", "mission_phase",
        }
