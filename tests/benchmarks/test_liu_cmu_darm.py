"""Diagnostic-only additions to run_liu_cmu_darm.py.

Covers Section X's test requirements as they apply to DARM: metadata
alignment through the (stride-one, last-step-labelled) window construction,
the exact source_row_right join, Ns -> mission_phase mapping, unknown-Ns ->
null, prediction-parquet schema, multi-cell writes not overwriting each
other, and recomputing error from test_predictions matching _evaluate's own
metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch

from battery_benchmark.reproductions.liu_2025.darm import DARM
from battery_benchmark.reproductions.liu_2025.windows import build_liu_windows
from battery_benchmark.reproductions.vidal_2019.sequence_v2 import VidalV2Record
from scripts.benchmarks.run_liu_cmu_darm import (
    WINDOW_LENGTH,
    _build_prediction_rows,
    _evaluate,
    _flatten_training_history,
    _labelled_windows_with_metadata,
    _mission_id_array,
    _predict_all,
)
from battery_benchmark.benchmarks.cmu_diagnostics import load_raw_diagnostic_fields


def _synthetic_record(mission: str, sequence_order: int, n: int, *, start_row: int) -> VidalV2Record:
    rng = np.random.default_rng(abs(hash(mission)) % (2**32))
    frame = pd.DataFrame(
        {
            "current": rng.normal(size=n).astype(np.float32),
            "voltage": rng.normal(size=n).astype(np.float32),
            "temperature": rng.normal(size=n).astype(np.float32),
            "soc": np.linspace(1.0, 0.0, n, dtype=np.float64),
            "elapsed_candidate_s": np.arange(n, dtype=np.float64),
            "original_time_s": np.arange(n, dtype=np.float64) + 1000.0,
            "source_row_right": np.arange(start_row, start_row + n, dtype=np.int64),
        }
    )
    return VidalV2Record(
        data=frame,
        metadata={"mission_cycle_id": mission, "sequence_order": sequence_order},
    )


def test_labelled_windows_with_metadata_matches_build_liu_windows():
    records = [
        _synthetic_record("m1", 0, WINDOW_LENGTH + 5, start_row=0),
        _synthetic_record("m2", 1, WINDOW_LENGTH + 2, start_row=1000),
    ]
    x_ref, y_ref = build_liu_windows(records, require_labels=True, feature_order=("current", "voltage", "temperature"))
    x_diag, y_diag, missions, metadata = _labelled_windows_with_metadata(records)

    assert np.array_equal(x_ref, x_diag)
    assert np.array_equal(y_ref, y_diag)
    assert sum(count for _, count in missions) == len(y_diag)
    # Every metadata array must be index-for-index aligned with y_diag.
    for name in ("mission_sequence_order", "elapsed_candidate_s", "original_time_s", "source_row_right"):
        assert len(metadata[name]) == len(y_diag)

    # The metadata for the first window of m1 must describe row index
    # WINDOW_LENGTH - 1 (the window's last, labelled row) -- not row 0.
    assert metadata["source_row_right"][0] == WINDOW_LENGTH - 1
    assert metadata["mission_sequence_order"][0] == 0
    # m2's windows start after m1's -- first m2 window's last row is at
    # source_row_right == 1000 + WINDOW_LENGTH - 1.
    m1_windows = WINDOW_LENGTH + 5 - WINDOW_LENGTH + 1
    assert metadata["source_row_right"][m1_windows] == 1000 + WINDOW_LENGTH - 1
    assert metadata["mission_sequence_order"][m1_windows] == 1


def test_labelled_windows_with_metadata_survives_multi_mission_and_truncation():
    # Deliberately different lengths so windows-per-mission truncation
    # differs; alignment must still hold per-mission.
    records = [
        _synthetic_record("short", 0, WINDOW_LENGTH, start_row=0),
        _synthetic_record("long", 1, WINDOW_LENGTH + 10, start_row=5000),
    ]
    _, y_diag, missions, metadata = _labelled_windows_with_metadata(records)
    assert missions == [("short", 1), ("long", 11)]
    mission_ids = _mission_id_array(missions)
    assert list(mission_ids[:1]) == ["short"]
    assert list(mission_ids[1:]) == ["long"] * 11


def test_predict_all_matches_evaluate_error(tmp_path):
    model = DARM()
    for parameter in model.parameters():
        parameter.data.zero_()
    x = np.zeros((4, WINDOW_LENGTH, 3), dtype=np.float32)
    y = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    missions = [("a", 2), ("b", 2)]

    metrics = _evaluate(model, x, y, missions, batch_size=4, device=torch.device("cpu"))
    prediction = _predict_all(model, x, batch_size=4, device=torch.device("cpu"))

    recomputed_mae = float(np.mean(np.abs(prediction - y)) * 100)
    recomputed_rmse = float(np.sqrt(np.mean((prediction - y) ** 2)) * 100)
    assert np.isclose(recomputed_mae, metrics["overall_mae_percent"])
    assert np.isclose(recomputed_rmse, metrics["overall_rmse_percent"])


def _write_raw_diagnostic_parquet(path, n: int, *, ns_values, start_row: int = 0):
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


def test_join_and_ns_mission_phase_mapping_unknown_ns_is_null(tmp_path):
    path = tmp_path / "raw.parquet"
    ns_values = [4, 5, 6, 99]  # 99 is deliberately unmapped
    _write_raw_diagnostic_parquet(path, 4, ns_values=ns_values)
    raw_diagnostic = load_raw_diagnostic_fields(path)

    source_row_right = np.array([0, 1, 2, 3], dtype=np.int64)
    rows = _build_prediction_rows(
        framework="liu_2025_darm",
        method="Source only",
        seed=0,
        cell_id="VAH01",
        condition="baseline protocol",
        soc_reference=np.array([0.9, 0.8, 0.7, 0.6]),
        soc_prediction=np.array([0.91, 0.79, 0.72, 0.55]),
        metadata={
            "mission_sequence_order": np.array([0, 0, 0, 0]),
            "elapsed_candidate_s": np.array([0.0, 1.0, 2.0, 3.0]),
            "original_time_s": np.array([0.0, 1.0, 2.0, 3.0]),
            "source_row_right": source_row_right,
        },
        mission_ids=np.array(["m1", "m1", "m1", "m1"], dtype=object),
        raw_diagnostic=raw_diagnostic,
    )
    assert [row["mission_phase"] for row in rows] == ["takeoff", "cruise", "landing", None]
    assert rows[-1]["ns_raw"] == 99
    assert rows[0]["voltage_v_raw"] == pytest.approx(3.0)
    assert np.isclose(rows[0]["signed_error"], 0.91 - 0.9)
    assert np.isclose(rows[0]["absolute_error"], abs(0.91 - 0.9))


def test_join_defends_against_unmatched_source_row(tmp_path):
    path = tmp_path / "raw.parquet"
    _write_raw_diagnostic_parquet(path, 2, ns_values=[4, 5])
    raw_diagnostic = load_raw_diagnostic_fields(path)

    rows = _build_prediction_rows(
        framework="liu_2025_darm",
        method="DARM",
        seed=0,
        cell_id="VAH01",
        condition="baseline protocol",
        soc_reference=np.array([0.5]),
        soc_prediction=np.array([0.5]),
        metadata={
            "mission_sequence_order": np.array([0]),
            "elapsed_candidate_s": np.array([0.0]),
            "original_time_s": np.array([0.0]),
            "source_row_right": np.array([999]),  # not present in raw_diagnostic
        },
        mission_ids=np.array(["m1"], dtype=object),
        raw_diagnostic=raw_diagnostic,
    )
    assert rows[0]["ns_raw"] is None
    assert rows[0]["mission_phase"] is None
    assert rows[0]["voltage_v_raw"] is None


def test_join_rejects_duplicate_source_row(tmp_path):
    path = tmp_path / "raw.parquet"
    _write_raw_diagnostic_parquet(path, 2, ns_values=[4, 4], start_row=0)
    # Force a duplicate by writing source_row=[0, 0].
    frame = pd.read_parquet(path)
    frame["source_row"] = [0, 0]
    frame.to_parquet(path)
    with pytest.raises(ValueError, match="source_row must be unique"):
        load_raw_diagnostic_fields(path)


def test_missing_provenance_is_real_null_never_the_string_nan(tmp_path):
    path = tmp_path / "raw.parquet"
    frame = pd.DataFrame(
        {
            "source_row": [0],
            "voltage_v": [3.7],
            "current_a": [1.0],
            "Temperature__C": [25.0],
            "capacity_ah": [np.nan],
            "capacity_source": [None],
            "source_rpt_id": [None],
            "Ns": [4],
        }
    )
    frame.to_parquet(path)
    raw_diagnostic = load_raw_diagnostic_fields(path)

    rows = _build_prediction_rows(
        framework="liu_2025_darm",
        method="Source only",
        seed=0,
        cell_id="VAH01",
        condition="baseline protocol",
        soc_reference=np.array([0.5]),
        soc_prediction=np.array([0.5]),
        metadata={
            "mission_sequence_order": np.array([0]),
            "elapsed_candidate_s": np.array([0.0]),
            "original_time_s": np.array([0.0]),
            "source_row_right": np.array([0]),
        },
        mission_ids=np.array(["m1"], dtype=object),
        raw_diagnostic=raw_diagnostic,
    )
    assert rows[0]["capacity_ah"] is None
    assert rows[0]["capacity_source"] is None
    assert rows[0]["source_rpt_id"] is None
    for value in rows[0].values():
        assert value != "nan"


def test_flatten_training_history_reports_real_darm_diagnostics_and_no_validation_leak():
    epoch_diagnostics = [
        {
            "epoch": 1,
            "training": {
                "adversarial": 0.6,
                "reconstruction": 0.05,
                "mmd": 0.02,
                "total": 1.1,
                "discriminator": 0.6,
                "adversarial_weight": 1.0,
                "source_domain_accuracy": 0.9,
                "target_domain_accuracy": 0.4,
                "overall_domain_accuracy": 0.65,
                "mean_p_source_given_source": 0.8,
                "mean_p_source_given_target": 0.6,
                "target_feature_rms_drift": 0.01,
                "target_bn_running_stat_drift": {"layer1": {"running_mean_max_abs_change": 0.001}},
                "gradient_l2": {"adversarial": 1.2, "total": 2.0},
                "gradient_cosine": {"adversarial_vs_total": 0.5},
            },
        }
    ]
    rows = _flatten_training_history(
        epoch_diagnostics,
        framework="liu_2025_darm",
        method="DARM",
        seed=42,
        cell_id="VAH01",
        learning_rate=0.0001,
        target_bn_mode="frozen_stats",
        adversarial_update="grl_joint",
        adversarial_schedule="constant",
        grl_strength=0.1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["epoch"] == 1
    assert row["total_loss"] == 1.1
    assert row["adversarial_loss"] == 0.6
    assert row["reconstruction_loss"] == 0.05
    assert row["mmd_loss"] == 0.02
    assert row["grl_strength"] == 0.1
    assert row["target_bn_mode"] == "frozen_stats"
    # DARM's adapt() is called with diagnostic_loaders={} in
    # run_liu_cmu_darm.py -- no target labels are used for any per-epoch
    # validation, so this must be reported truthfully, never forced to
    # look like Vidal/DTNN's diagnostic-validation convention.
    assert row["validation_evaluated"] is False
    assert row["validation_mae"] is None
    assert row["validation_rmse"] is None
    assert "layer1" in row["target_bn_running_stat_drift_json"]
