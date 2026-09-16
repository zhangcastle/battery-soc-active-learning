from __future__ import annotations

import importlib.util
import math
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


def _synthetic_prediction_windows():
    """Two synthetic windows with distinct, known metadata values for
    evaluate_target_last()'s new prediction-record fields -- these are read
    straight off build_cmu_vidal_windows' existing per-window metadata, so a
    short synthetic case (no real CMU parquet/manifest) is enough to prove
    they're wired through correctly."""

    from battery_benchmark.reproductions.vidal_2019.windows import Case1Windows

    seq_len = 50
    n = 2
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, seq_len, 3)).astype(np.float32)
    y = rng.uniform(0, 1, size=(n, seq_len)).astype(np.float32)
    mask = np.ones((n, seq_len), dtype=bool)
    metadata = [
        {
            "cell_id": "VAH01",
            "mission_cycle_id": "m1",
            "source_row_right": 1000,
            "sequence_order": 3,
            "elapsed_candidate_s_end": 12.5,
            "original_time_s_end": 100.0,
        },
        {
            "cell_id": "VAH01",
            "mission_cycle_id": "m2",
            "source_row_right": 2000,
            "sequence_order": 7,
            "elapsed_candidate_s_end": 45.0,
            "original_time_s_end": 200.0,
        },
    ]
    return Case1Windows(x, y, mask, metadata, 0)


def test_evaluate_target_last_prediction_fields() -> None:
    """New prediction-record fields (signed_error, mission_sequence_order,
    elapsed_candidate_s_end, original_time_s_end) must be present, taken
    verbatim from window metadata, and signed_error must equal
    soc_prediction - soc_reference row-by-row. Metric calculation itself
    (overall_mae/rmse etc.) is untouched -- not re-tested here, only the new
    columns on the returned predictions frame."""

    windows = _synthetic_prediction_windows()
    torch.manual_seed(0)
    model = module.VidalLSTM()
    frame, _result = module.evaluate_target_last(
        model, windows, phase_by_source_row={}, batch_size=8, device=torch.device("cpu")
    )

    expected_columns = {
        "signed_error",
        "mission_sequence_order",
        "elapsed_candidate_s_end",
        "original_time_s_end",
    }
    assert expected_columns <= set(frame.columns)
    assert len(frame) == 2

    assert np.allclose(
        frame["signed_error"].to_numpy(),
        (frame["soc_prediction"] - frame["soc_reference"]).to_numpy(),
    )

    by_mission = frame.set_index("mission_cycle_id")
    assert by_mission.loc["m1", "mission_sequence_order"] == 3
    assert by_mission.loc["m1", "elapsed_candidate_s_end"] == 12.5
    assert by_mission.loc["m1", "original_time_s_end"] == 100.0
    assert by_mission.loc["m2", "mission_sequence_order"] == 7
    assert by_mission.loc["m2", "elapsed_candidate_s_end"] == 45.0
    assert by_mission.loc["m2", "original_time_s_end"] == 200.0


def test_evaluate_target_last_raw_end_conditions_come_from_raw_lookup() -> None:
    """voltage_v_end/current_a_end/temperature_c_end must be the exact raw
    (unnormalized) values passed in via raw_features_by_source_row, joined
    on source_row_right -- deliberately chosen far outside the synthetic
    normalized x range (which is ~N(0,1)) so a bug that instead derived
    these from x (e.g. inverse-scaling the model input) would be caught by
    the magnitude/value mismatch, not just a missing-column check."""

    windows = _synthetic_prediction_windows()  # source_row_right: 1000, 2000
    raw_features_by_source_row = {
        1000: (3.7, -12.4, 26.1),   # (voltage_v, current_a, Temperature__C)
        2000: (4.05, 8.9, 31.7),
        999999: (0.0, 0.0, 0.0),  # unrelated row -- must not be picked up
    }
    torch.manual_seed(0)
    model = module.VidalLSTM()
    frame, _result = module.evaluate_target_last(
        model, windows, phase_by_source_row={}, batch_size=8, device=torch.device("cpu"),
        raw_features_by_source_row=raw_features_by_source_row,
    )

    assert {"voltage_v_end", "current_a_end", "temperature_c_end"} <= set(frame.columns)
    by_mission = frame.set_index("mission_cycle_id")
    assert by_mission.loc["m1", "voltage_v_end"] == 3.7
    assert by_mission.loc["m1", "current_a_end"] == -12.4
    assert by_mission.loc["m1", "temperature_c_end"] == 26.1
    assert by_mission.loc["m2", "voltage_v_end"] == 4.05
    assert by_mission.loc["m2", "current_a_end"] == 8.9
    assert by_mission.loc["m2", "temperature_c_end"] == 31.7

    # x is standard-normal synthetic noise (see _synthetic_prediction_windows);
    # the raw values above are physically-scaled and outside that range, so
    # this also rules out the fields being some transform of the model input.
    assert abs(by_mission.loc["m1", "voltage_v_end"]) > 3
    assert abs(by_mission.loc["m2", "current_a_end"]) > 3


def test_evaluate_target_last_raw_end_conditions_default_to_nan_when_unavailable() -> None:
    """Callers that don't pass raw_features_by_source_row (e.g. any external
    code still using the old evaluate_target_last signature) must not crash;
    the new columns are present but NaN rather than silently fabricated."""

    windows = _synthetic_prediction_windows()
    torch.manual_seed(0)
    model = module.VidalLSTM()
    frame, _result = module.evaluate_target_last(
        model, windows, phase_by_source_row={}, batch_size=8, device=torch.device("cpu"),
    )
    assert math.isnan(frame["voltage_v_end"].iloc[0])
    assert math.isnan(frame["current_a_end"].iloc[0])
    assert math.isnan(frame["temperature_c_end"].iloc[0])


def test_evaluate_target_last_provenance_passthrough() -> None:
    """capacity_ah/capacity_source/source_rpt_id must be the exact raw
    provenance values passed in via raw_provenance_by_source_row, joined on
    source_row_right -- same convention as the voltage/current/temperature
    raw lookup. Not used by training or metric calculation."""

    windows = _synthetic_prediction_windows()  # source_row_right: 1000, 2000
    raw_provenance_by_source_row = {
        1000: (4.85, "rpt_measured", "cmu_v3_rpt_capacity_v1:VAH01:b0:s787:e19120"),
        2000: (4.79, "rpt_measured", "cmu_v3_rpt_capacity_v1:VAH01:b1:s900:e20500"),
    }
    torch.manual_seed(0)
    model = module.VidalLSTM()
    frame, _result = module.evaluate_target_last(
        model, windows, phase_by_source_row={}, batch_size=8, device=torch.device("cpu"),
        raw_provenance_by_source_row=raw_provenance_by_source_row,
    )

    assert {"capacity_ah", "capacity_source", "source_rpt_id"} <= set(frame.columns)
    by_mission = frame.set_index("mission_cycle_id")
    assert by_mission.loc["m1", "capacity_ah"] == 4.85
    assert by_mission.loc["m1", "capacity_source"] == "rpt_measured"
    assert by_mission.loc["m1", "source_rpt_id"] == "cmu_v3_rpt_capacity_v1:VAH01:b0:s787:e19120"
    assert by_mission.loc["m2", "capacity_ah"] == 4.79
    assert by_mission.loc["m2", "source_rpt_id"] == "cmu_v3_rpt_capacity_v1:VAH01:b1:s900:e20500"


def test_evaluate_target_last_missing_source_rpt_id_stays_none_not_nan_string() -> None:
    """A genuinely missing source_rpt_id (and capacity_source) must survive
    as a real null (None / NaN in the DataFrame), never the literal string
    'nan' -- this is the exact failure mode of naively .astype(str)'ing a
    column that contains missing values, which the phase_by_source_row
    lookup above (development_phase_role) already does; the new provenance
    lookup deliberately does not follow that pattern."""

    windows = _synthetic_prediction_windows()  # source_row_right: 1000, 2000
    raw_provenance_by_source_row = {
        1000: (4.85, "rpt_measured", "cmu_v3_rpt_capacity_v1:VAH01:b0:s787:e19120"),
        2000: (None, None, None),  # e.g. a phase with no RPT reference at all
    }
    torch.manual_seed(0)
    model = module.VidalLSTM()
    frame, _result = module.evaluate_target_last(
        model, windows, phase_by_source_row={}, batch_size=8, device=torch.device("cpu"),
        raw_provenance_by_source_row=raw_provenance_by_source_row,
    )

    by_mission = frame.set_index("mission_cycle_id")
    missing_source_rpt_id = by_mission.loc["m2", "source_rpt_id"]
    missing_capacity_source = by_mission.loc["m2", "capacity_source"]
    missing_capacity_ah = by_mission.loc["m2", "capacity_ah"]

    assert missing_source_rpt_id != "nan"
    assert missing_capacity_source != "nan"
    assert pd.isna(missing_source_rpt_id) or missing_source_rpt_id is None
    assert pd.isna(missing_capacity_source) or missing_capacity_source is None
    assert pd.isna(missing_capacity_ah)

    # Untouched row (m1) still round-trips correctly alongside the missing one.
    assert by_mission.loc["m1", "source_rpt_id"] == "cmu_v3_rpt_capacity_v1:VAH01:b0:s787:e19120"


def test_evaluate_target_last_provenance_defaults_to_none_when_unavailable() -> None:
    """Callers that don't pass raw_provenance_by_source_row must not crash;
    the new columns are present but None/NaN, never fabricated or 'nan'."""

    windows = _synthetic_prediction_windows()
    torch.manual_seed(0)
    model = module.VidalLSTM()
    frame, _result = module.evaluate_target_last(
        model, windows, phase_by_source_row={}, batch_size=8, device=torch.device("cpu"),
    )
    assert pd.isna(frame["capacity_ah"].iloc[0])
    assert pd.isna(frame["capacity_source"].iloc[0]) or frame["capacity_source"].iloc[0] is None
    assert pd.isna(frame["source_rpt_id"].iloc[0]) or frame["source_rpt_id"].iloc[0] is None
    assert frame["source_rpt_id"].iloc[0] != "nan"
