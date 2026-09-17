"""Diagnostic-only additions to run_sftnn_transfer_cmu.py.

Covers metadata alignment through windows()'s non-overlapping WINDOW-length
truncation, the exact source_row_right join (shared with DARM/DTNN via
battery_benchmark.benchmarks.cmu_diagnostics), Ns -> mission_phase mapping,
and recomputing error from test_predictions matching _evaluate's own metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from battery_benchmark.benchmarks.cmu_diagnostics import load_raw_diagnostic_fields
from battery_benchmark.reproductions.shen_2023.data import ShenCycle
from battery_benchmark.reproductions.shen_2023.model import MEDNet
from scripts.benchmarks.run_sftnn_transfer_cmu import (
    WINDOW,
    _build_prediction_rows,
    _evaluate,
    _flatten_dict_history,
    _flatten_scalar_history,
    _mission_metadata,
    _predict_all_with_metadata,
    _windows_metadata,
)


class _Record:
    def __init__(self, mission, sequence_order, n, start_row):
        self.metadata = {"mission_cycle_id": mission, "sequence_order": sequence_order}
        self.data = pd.DataFrame(
            {
                "elapsed_candidate_s": np.arange(n, dtype=np.float64),
                "original_time_s": np.arange(n, dtype=np.float64) + 500.0,
                "source_row_right": np.arange(start_row, start_row + n, dtype=np.int64),
            }
        )


def test_windows_metadata_matches_windows_truncation():
    values = np.arange(2 * WINDOW + 5, dtype=np.int64)
    reshaped = _windows_metadata(values)
    assert reshaped.shape == (2, WINDOW)
    assert np.array_equal(reshaped[0], values[:WINDOW])
    assert np.array_equal(reshaped[1], values[WINDOW : 2 * WINDOW])


def test_mission_metadata_reads_same_frame_as_mission_cycles():
    records = [_Record("m1", 0, WINDOW + 3, start_row=0), _Record("m2", 1, WINDOW * 2, start_row=1000)]
    metadata = _mission_metadata(records, {"m1", "m2"})
    assert set(metadata) == {"m1", "m2"}
    assert len(metadata["m1"]["source_row_right"]) == WINDOW + 3
    assert metadata["m2"]["source_row_right"][0] == 1000
    assert (metadata["m1"]["mission_sequence_order"] == 0).all()
    assert (metadata["m2"]["mission_sequence_order"] == 1).all()


def test_predict_all_with_metadata_matches_evaluate_metrics():
    torch.manual_seed(0)
    model = MEDNet()
    n = WINDOW * 2
    rng = np.random.default_rng(0)
    cycle = ShenCycle(
        profile="m1",
        temperature=None,
        features=rng.normal(size=(n, 3)).astype(np.float32),
        soc=rng.uniform(0, 1, size=n).astype(np.float32),
        paths=(),
    )
    metadata = {
        "m1": {
            "mission_sequence_order": np.zeros(n, dtype=np.int64),
            "elapsed_candidate_s": np.arange(n, dtype=np.float64),
            "original_time_s": np.arange(n, dtype=np.float64),
            "source_row_right": np.arange(n, dtype=np.int64),
        }
    }
    device = torch.device("cpu")
    metrics = _evaluate(model, [cycle], device, branch="source")
    prediction, reference, mission_ids, window_idx, timestep_idx, pred_metadata = (
        _predict_all_with_metadata(model, [cycle], metadata, device, branch="source")
    )
    assert len(prediction) == n
    assert set(mission_ids) == {"m1"}
    assert window_idx.max() == 1
    assert timestep_idx.max() == WINDOW - 1

    clipped = np.clip(prediction, 0.0, 1.0)
    recomputed_mae = float(np.mean(np.abs(clipped - reference)) * 100)
    recomputed_rmse = float(np.sqrt(np.mean((clipped - reference) ** 2)) * 100)
    assert np.isclose(recomputed_mae, metrics["overall_mae_percent"])
    assert np.isclose(recomputed_rmse, metrics["overall_rmse_percent"])


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


def test_build_prediction_rows_joins_ns_and_never_stringifies_nan(tmp_path):
    path = tmp_path / "raw.parquet"
    _write_raw_diagnostic_parquet(path, 4, ns_values=[4, 5, 6, 99])
    raw_diagnostic = load_raw_diagnostic_fields(path)

    rows = _build_prediction_rows(
        framework="sftnn_med",
        method="med",
        seed=0,
        cell_id="VAH01",
        condition="baseline protocol",
        soc_reference=np.array([0.9, 0.8, 0.7, 0.6]),
        soc_prediction=np.array([0.95, 0.75, 1.2, -0.1]),  # deliberately out-of-range
        mission_ids=np.array(["m1"] * 4, dtype=object),
        window_indices=np.array([0, 0, 1, 1]),
        timestep_indices=np.array([0, 1, 0, 1]),
        metadata={
            "mission_sequence_order": np.zeros(4, dtype=np.int64),
            "elapsed_candidate_s": np.arange(4, dtype=np.float64),
            "original_time_s": np.arange(4, dtype=np.float64),
            "source_row_right": np.array([0, 1, 2, 3], dtype=np.int64),
        },
        raw_diagnostic=raw_diagnostic,
    )
    assert [row["mission_phase"] for row in rows] == ["takeoff", "cruise", "landing", None]
    assert rows[-1]["ns_raw"] == 99
    # soc_prediction is clipped to [0, 1] to match _evaluate's own clip.
    assert rows[2]["soc_prediction"] == pytest.approx(1.0)
    assert rows[3]["soc_prediction"] == pytest.approx(0.0)
    for row in rows:
        for value in row.values():
            assert value != "nan"


def test_flatten_scalar_history_and_dict_history_never_invent_fields():
    scalar_rows = _flatten_scalar_history([1.0, 0.5, 0.25], framework="sftnn_med", method="med_source_pretrain", seed=1, learning_rate=1e-3)
    assert [row["epoch"] for row in scalar_rows] == [1, 2, 3]
    assert scalar_rows[0]["train_loss"] == 1.0
    assert scalar_rows[0]["validation_evaluated"] is False

    dict_rows = _flatten_dict_history(
        [{"soc": 0.1, "dis": 0.02, "total": 0.12, "learning_rate": 1e-4}],
        framework="sftnn_med", method="med", seed=1, cell_id="VAH01",
    )
    assert dict_rows[0]["soc"] == 0.1
    assert dict_rows[0]["dis"] == 0.02
    assert "estimate" not in dict_rows[0]  # DANN-only field must not leak into MED rows

    dann_rows = _flatten_dict_history(
        [{"estimate": 0.3, "adversarial": 0.4, "mmd": 0.05, "total": 0.75, "learning_rate": 1e-4}],
        framework="sftnn_med", method="dann_mmd", seed=1, cell_id="VAH01",
    )
    assert dann_rows[0]["estimate"] == 0.3
    assert "soc" not in dann_rows[0]  # MED-only field must not leak into DANN+MMD rows
