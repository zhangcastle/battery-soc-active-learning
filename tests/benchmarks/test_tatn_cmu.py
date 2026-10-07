"""Data-version/output-isolation/resume-safety AND training-monitoring
diagnostics for run_tatn_cmu.py.

Part 1 (data version, output isolation, resume safety): the default
CMU/manifest paths, the isolated default output directory, and the
resume-safety check that must reject a stale results.json before ever
reusing it.

Part 2 (training monitoring): TATN's shen_tatn.train.train_source/adapt
per-epoch diagnostics (epoch/learning_rate/runtime_seconds, plus adapt's
estimate_loss/adversarial_loss/mmd_loss/total_loss) and the read-only
validation probe (validation_mae/validation_rmse/validation_evaluated/
validation_runtime_seconds) must never perturb the global RNG, the final
state_dict, test predictions, or the original training-loss history --
proven directly against shen_tatn.train, not just wired into the script.

Neither part touches TATN's model, pseudo-label selection, loss formulas,
optimizers, batch order, early stopping, checkpoint selection, or primary
metrics; no predictions/phase analysis is added here.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch

from battery_benchmark.reproductions.shen_2023.data import PROFILES, ShenCycle
from battery_benchmark.reproductions.shen_tatn.train import (
    adapt,
    predict as shen_predict,
    train_source,
)
from scripts.benchmarks.run_tatn_cmu import (
    REPO_ROOT,
    RESUME_CHECK_FIELDS,
    TEST_PREDICTIONS_SCHEMA,
    _build_prediction_rows,
    _check_resume_tatn_result,
    _compute_cell_transfer_effects,
    _compute_phase_metrics,
    _compute_phase_transfer_effects,
    _evaluate,
    _flatten_prediction_metadata,
    _mission_metadata,
    _to_shen_cycles,
    _windows_metadata,
    _write_test_predictions_atomic,
    parse_args,
)

WINDOW = 90


def _make_cycle(profile: str, n_windows: int, seed: int) -> ShenCycle:
    rng = np.random.default_rng(seed)
    n = n_windows * WINDOW
    features = rng.normal(size=(n, 3)).astype(np.float32)
    soc = np.clip(rng.uniform(0.1, 0.9, size=n).astype(np.float32), 0.0, 1.0)
    return ShenCycle(profile=profile, temperature=25, features=features, soc=soc, paths=())


def test_cmu_tatn_uses_source_feature_range_without_target_fitting():
    frame = pd.DataFrame(
        {
            "current": np.zeros(90),
            "voltage": np.full(90, 3.0),
            "temperature": np.full(90, 30.0),
            "soc": np.linspace(1.0, 0.0, 90),
        }
    )
    record = SimpleNamespace(
        data=frame,
        metadata={"mission_cycle_id": "mission_1", "source_path": "cell.parquet"},
    )

    cycle = _to_shen_cycles(
        [record],
        np.array([-1.0, 2.0, 20.0]),
        np.array([1.0, 4.0, 25.0]),
        require_labels=True,
    )[0]

    np.testing.assert_allclose(cycle.features[0], [0.5, 0.5, 2.0])
    np.testing.assert_allclose(cycle.soc, frame["soc"])


def _parse_args(extra: list[str] | None = None):
    argv = [
        "run_tatn_cmu.py",
        "--panasonic-data-root", "/tmp/panasonic",
        "--source-kind", "trained",
    ]
    if extra:
        argv += extra
    import sys

    old_argv = sys.argv
    sys.argv = argv
    try:
        return parse_args()
    finally:
        sys.argv = old_argv


def test_cmu_base_root_default_is_ns_relabel():
    args = _parse_args()
    assert args.cmu_base_root == (
        REPO_ROOT / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1_ns_relabel"
    )


def test_cmu_condition_root_default_is_ns_relabel_extension4():
    args = _parse_args()
    assert args.cmu_condition_root == (
        REPO_ROOT
        / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1_extension4_ns_relabel"
    )


def test_cmu_manifest_default_is_ns_relabel():
    args = _parse_args()
    assert args.cmu_manifest == (
        REPO_ROOT
        / "data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1_ns_relabel.csv"
    )


def test_output_root_default_is_isolated_tatn_v2_unsupervised_formal():
    args = _parse_args()
    assert args.output_root == (
        REPO_ROOT / "outputs/benchmarks/panasonic_to_cmu_evtol/tatn_ns_relabel_v2_unsupervised_formal"
    )


def test_cmu_paths_and_output_root_remain_overridable():
    """Adding defaults must not remove the ability to override any of these
    on the command line (e.g. for a non-default data version or a scratch
    output directory)."""
    args = _parse_args(
        [
            "--cmu-base-root", "/tmp/base",
            "--cmu-condition-root", "/tmp/condition",
            "--cmu-manifest", "/tmp/manifest.csv",
            "--output-root", "/tmp/out",
        ]
    )
    from pathlib import Path

    assert args.cmu_base_root == Path("/tmp/base")
    assert args.cmu_condition_root == Path("/tmp/condition")
    assert args.cmu_manifest == Path("/tmp/manifest.csv")
    assert args.output_root == Path("/tmp/out")


def _complete_expected() -> dict[str, object]:
    return {
        "output_schema_version": "tatn_cmu_v2_diagnostic",
        "repeat": 1,
        "base_seed": 100,
        "cmu_input_sha256": "cmu_hash",
        "manifest_sha256": "manifest_hash",
        "source_checkpoint_sha256": "checkpoint_hash",
        "source_kind": "trained",
        "scenario": "unsupervised",
        "source_epochs": 300,
        "transfer_epochs": 150,
        "batch_size": 64,
        "seed": 100,
        "tensor_layout": "paper",
        "tatn_learning_rate": 1e-3,
        "tatn_lambdas": [1.0, 1.0, 1.0],
        "source_early_stopping_patience": 30,
        "source_min_epochs": 60,
        "transfer_early_stopping_patience": 15,
        "transfer_min_epochs": 30,
        "validation_interval": 10,
    }


def _write_sidecars(root, *, skip: tuple[str, ...] = ()):
    if "target.pt" not in skip:
        torch.save({"model_state_dict": {}}, root / "target.pt")
    if "training_history.csv" not in skip:
        (root / "training_history.csv").write_text("epoch\n1\n", encoding="utf-8")
    if "test_predictions.parquet" not in skip:
        pd.DataFrame({"a": [1]}).to_parquet(root / "test_predictions.parquet")


def _write_result(path, expected, *, drop: str | None = None, override: dict | None = None):
    payload = dict(expected)
    if drop is not None:
        payload.pop(drop, None)
    if override:
        payload.update(override)
    payload["metrics"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_check_resume_tatn_result_accepts_matching_configuration(tmp_path):
    """A complete, matching result must be accepted (must not raise)."""
    expected = _complete_expected()
    result_path = tmp_path / "results.json"
    _write_result(result_path, expected)
    _write_sidecars(tmp_path)

    _check_resume_tatn_result(result_path=result_path, cell_output=tmp_path, expected=expected)


@pytest.mark.parametrize("field", RESUME_CHECK_FIELDS)
def test_check_resume_tatn_result_rejects_missing_field(tmp_path, field):
    """Every field listed in RESUME_CHECK_FIELDS must independently be
    required -- an old result predating any one of them must never be
    silently treated as resumable."""
    expected = _complete_expected()
    result_path = tmp_path / "results.json"
    _write_result(result_path, expected, drop=field)
    _write_sidecars(tmp_path)

    with pytest.raises(ValueError, match=field):
        _check_resume_tatn_result(result_path=result_path, cell_output=tmp_path, expected=expected)


@pytest.mark.parametrize(
    ("field", "stale_value"),
    [
        ("cmu_input_sha256", "old_cmu_hash"),
        ("manifest_sha256", "old_manifest_hash"),
        ("source_checkpoint_sha256", "old_checkpoint_hash"),
        ("source_kind", "released"),
        ("scenario", "semisupervised"),
        ("source_epochs", 100),
        ("transfer_epochs", 50),
        ("batch_size", 32),
        ("seed", 7),
        ("repeat", 2),
        ("base_seed", 7),
        ("output_schema_version", "tatn_cmu_v1_legacy"),
        ("tensor_layout", "released_repo"),
        ("tatn_learning_rate", 1e-2),
        ("tatn_lambdas", [1.0, 0.5, 1.0]),
        ("source_early_stopping_patience", 5),
        ("source_min_epochs", 10),
        ("transfer_early_stopping_patience", 5),
        ("transfer_min_epochs", 10),
        ("validation_interval", 1),
    ],
)
def test_check_resume_tatn_result_rejects_stale_field(tmp_path, field, stale_value):
    """A stale value for ANY tracked field (a different CMU data version,
    manifest, source checkpoint, scenario, or hyperparameter) must raise
    immediately, never silently reused."""
    expected = _complete_expected()
    result_path = tmp_path / "results.json"
    _write_result(result_path, expected, override={field: stale_value})
    _write_sidecars(tmp_path)

    with pytest.raises(ValueError, match=field):
        _check_resume_tatn_result(result_path=result_path, cell_output=tmp_path, expected=expected)


def test_check_resume_tatn_result_rejects_missing_target_checkpoint(tmp_path):
    """A fully-matching result JSON is still not resumable if its target.pt
    checkpoint sidecar is missing -- e.g. an interrupted prior run."""
    expected = _complete_expected()
    result_path = tmp_path / "results.json"
    _write_result(result_path, expected)
    _write_sidecars(tmp_path, skip=("target.pt",))

    with pytest.raises(ValueError, match="target.pt"):
        _check_resume_tatn_result(result_path=result_path, cell_output=tmp_path, expected=expected)


def test_check_resume_tatn_result_rejects_missing_training_history(tmp_path):
    """A fully-matching result JSON with target.pt present is still not
    resumable if its training_history.csv sidecar is missing -- e.g. a run
    from before this diagnostic was added, or an interrupted write."""
    expected = _complete_expected()
    result_path = tmp_path / "results.json"
    _write_result(result_path, expected)
    _write_sidecars(tmp_path, skip=("training_history.csv",))

    with pytest.raises(ValueError, match="training_history.csv"):
        _check_resume_tatn_result(result_path=result_path, cell_output=tmp_path, expected=expected)


def test_check_resume_tatn_result_rejects_missing_test_predictions(tmp_path):
    """A fully-matching result JSON with target.pt/training_history.csv
    present is still not resumable if its test_predictions.parquet sidecar
    is missing -- e.g. an interrupted write that never reached the atomic
    rename onto the final name."""
    expected = _complete_expected()
    result_path = tmp_path / "results.json"
    _write_result(result_path, expected)
    _write_sidecars(tmp_path, skip=("test_predictions.parquet",))
    # Deliberately no tmp_path / "test_predictions.parquet" (and
    # a leftover ".incomplete" file, if any, must never satisfy this check).
    (tmp_path / "test_predictions.parquet.incomplete").write_bytes(b"")

    with pytest.raises(ValueError, match="test_predictions.parquet"):
        _check_resume_tatn_result(result_path=result_path, cell_output=tmp_path, expected=expected)


def test_train_source_record_epoch_seconds_backward_compatible():
    """Requirement #1: the default call (no flag) keeps the exact original
    2-tuple return and per-epoch training trace; record_epoch_seconds=True
    ADDS a third element -- a positive, per-epoch diagnostics list of the
    same length as history -- without changing the trace itself."""
    device = torch.device("cpu")
    source_cycles = {p: _make_cycle(p, 2, i) for i, p in enumerate(PROFILES)}

    model, history = train_source(
        source_cycles, epochs=2, batch_size=32, learning_rate=1e-3, device=device, seed=0
    )
    assert isinstance(history, list) and isinstance(history[0], float)

    model2, history2, epoch_diagnostics = train_source(
        source_cycles, epochs=2, batch_size=32, learning_rate=1e-3, device=device, seed=0,
        record_epoch_seconds=True,
    )
    assert history2 == history
    assert len(epoch_diagnostics) == len(history) == 2
    for i, row in enumerate(epoch_diagnostics):
        assert row["epoch"] == i + 1
        assert row["runtime_seconds"] >= 0
        assert row["loss"] == history[i]
        assert "learning_rate" in row


def test_adapt_validation_diagnostic_is_bit_identical_no_op():
    """Requirement #4: toggling the diagnostic-only validation probe (and
    record_epoch_seconds) on/off for adapt() must leave the global RNG, the
    final target_extractor state_dict (parameter-by-parameter), the test
    predictions, and the original training-loss history COMPLETELY
    unchanged -- it is read-only and must never feed back into training,
    pseudo-labeling, the scheduler, early stopping, checkpoint, or model
    selection."""
    device = torch.device("cpu")
    source_cycles = {p: _make_cycle(p, 2, i) for i, p in enumerate(PROFILES)}
    source_model, _ = train_source(
        source_cycles, epochs=1, batch_size=32, learning_rate=1e-3, device=device, seed=0
    )
    target_cycles = {p: _make_cycle(p, 2, 100 + i) for i, p in enumerate(PROFILES)}
    validation_cycles = [_make_cycle("NN", 2, 200)]

    common_kwargs = dict(
        test_profile="UDDS", scenario="semisupervised", labeled_profile="Cycle_1",
        epochs=3, batch_size=32, learning_rate=1e-3, lambdas=(1.0, 0.1, 0.1),
        device=device, seed=7,
    )

    torch.manual_seed(123)
    adapted_off, history_off, _ = adapt(
        source_model, source_cycles, target_cycles, **common_kwargs
    )
    rng_after_off = torch.get_rng_state().clone()

    torch.manual_seed(123)
    adapted_on, history_on, _ = adapt(
        source_model, source_cycles, target_cycles, **common_kwargs,
        record_epoch_seconds=True, validation_cycles=validation_cycles, validation_interval=1,
    )
    rng_after_on = torch.get_rng_state().clone()

    assert torch.equal(rng_after_off, rng_after_on), "validation probe perturbed the global RNG"
    for p_off, p_on in zip(
        adapted_off.target_extractor.state_dict().values(),
        adapted_on.target_extractor.state_dict().values(),
    ):
        assert torch.equal(p_off, p_on), "validation probe changed the final state_dict"

    _, est_off = shen_predict(adapted_off, list(target_cycles.values()), device)
    _, est_on = shen_predict(adapted_on, list(target_cycles.values()), device)
    assert np.array_equal(est_off, est_on), "validation probe changed test predictions"

    assert len(history_off) == len(history_on)
    for row_off, row_on in zip(history_off, history_on):
        for key in ("estimate", "adversarial", "mmd", "total"):
            assert row_off[key] == row_on[key], (
                "validation probe changed the original training loss history"
            )

    for row in history_on:
        assert row["runtime_seconds"] is not None and row["runtime_seconds"] >= 0
        assert row["validation_evaluated"] is True
        assert row["validation_mae"] is not None
        assert row["validation_rmse"] is not None
        assert row["validation_runtime_seconds"] is not None
    for row in history_off:
        assert row["runtime_seconds"] is None
        assert row["validation_evaluated"] is False
        assert row["validation_mae"] is None
        assert row["validation_rmse"] is None
        assert row["validation_runtime_seconds"] is None


def test_adapt_validation_schedule_covers_epoch_one_interval_and_last_epoch():
    """Requirement #3: the validation probe must fire on epoch 1, every
    validation_interval-th epoch, AND the actual last epoch -- even when
    that count doesn't land on a multiple of the interval."""
    device = torch.device("cpu")
    source_cycles = {p: _make_cycle(p, 2, i) for i, p in enumerate(PROFILES)}
    source_model, _ = train_source(
        source_cycles, epochs=1, batch_size=32, learning_rate=1e-3, device=device, seed=0
    )
    target_cycles = {p: _make_cycle(p, 2, 100 + i) for i, p in enumerate(PROFILES)}
    validation_cycles = [_make_cycle("NN", 2, 200)]

    _, history, _ = adapt(
        source_model, source_cycles, target_cycles,
        test_profile="UDDS", scenario="semisupervised", labeled_profile="Cycle_1",
        epochs=7, batch_size=32, learning_rate=1e-3, lambdas=(1.0, 0.1, 0.1),
        device=device, seed=7,
        validation_cycles=validation_cycles, validation_interval=3,
    )

    evaluated_epochs = [row["epoch"] for row in history if row["validation_evaluated"]]
    assert evaluated_epochs == [1, 3, 6, 7], evaluated_epochs


def test_adapt_validation_never_evaluated_without_validation_cycles():
    """No validation_cycles supplied (the resume-compatible default) must
    never evaluate the probe, regardless of record_epoch_seconds."""
    device = torch.device("cpu")
    source_cycles = {p: _make_cycle(p, 2, i) for i, p in enumerate(PROFILES)}
    source_model, _ = train_source(
        source_cycles, epochs=1, batch_size=32, learning_rate=1e-3, device=device, seed=0
    )
    target_cycles = {p: _make_cycle(p, 2, 100 + i) for i, p in enumerate(PROFILES)}

    _, history, _ = adapt(
        source_model, source_cycles, target_cycles,
        test_profile="UDDS", scenario="semisupervised", labeled_profile="Cycle_1",
        epochs=3, batch_size=32, learning_rate=1e-3, lambdas=(1.0, 0.1, 0.1),
        device=device, seed=7, record_epoch_seconds=True,
    )
    assert all(row["validation_evaluated"] is False for row in history)
    assert all(row["validation_mae"] is None for row in history)


def test_run_tatn_cmu_has_validation_interval_cli_arg_default_ten():
    args = _parse_args()
    assert args.validation_interval == 10


def test_windows_metadata_matches_predict_truncation():
    """Requirement #2: _windows_metadata must truncate/reshape with the
    IDENTICAL usable = len // WINDOW * WINDOW logic windows() uses inside
    predict() -- never a different truncation for metadata than for
    features/SOC."""
    values = np.arange(205, dtype=np.float64)
    windowed = _windows_metadata(values, length=90)
    assert windowed.shape == (2, 90)
    np.testing.assert_array_equal(windowed.ravel(), np.arange(180, dtype=np.float64))


def test_flatten_prediction_metadata_alignment_and_truncation():
    """Requirement #2: window_index starts at 0 per mission, timestep_index
    runs 0-89, and the tail beyond usable = len // 90 * 90 is dropped in
    lockstep for every metadata field -- never independently."""
    cycle = ShenCycle(
        profile="mission_1", temperature=25,
        features=np.zeros((205, 3), dtype=np.float32),
        soc=np.zeros(205, dtype=np.float32),
        paths=(),
    )
    metadata_by_mission = {
        "mission_1": {
            "mission_sequence_order": np.full(205, 7, dtype=np.int64),
            "elapsed_candidate_s": np.arange(205, dtype=np.float64),
            "original_time_s": np.arange(205, dtype=np.float64) + 1000.0,
            "source_row_right": np.arange(1000, 1205, dtype=np.int64),
        }
    }
    mission_ids, window_idx, timestep_idx, metadata = _flatten_prediction_metadata(
        [cycle], metadata_by_mission
    )
    assert len(mission_ids) == 180
    assert set(mission_ids) == {"mission_1"}
    np.testing.assert_array_equal(window_idx, np.repeat([0, 1], 90))
    np.testing.assert_array_equal(timestep_idx, np.tile(np.arange(90), 2))
    np.testing.assert_array_equal(metadata["elapsed_candidate_s"], np.arange(180, dtype=np.float64))
    np.testing.assert_array_equal(
        metadata["source_row_right"], np.arange(1000, 1180, dtype=np.int64)
    )
    assert np.all(metadata["mission_sequence_order"] == 7)
    # The dropped tail (rows 180-204) must never appear.
    assert 1180 not in metadata["source_row_right"]


def test_mission_metadata_reads_same_frame_as_to_shen_cycles():
    """Requirement #2: _mission_metadata must read the SAME (untruncated)
    record.data frame _to_shen_cycles reads FEATURES/soc from, in full --
    truncation only happens later, in _flatten_prediction_metadata."""
    n = 97
    frame = pd.DataFrame(
        {
            "current": np.zeros(n), "voltage": np.full(n, 3.0), "temperature": np.full(n, 30.0),
            "soc": np.linspace(1.0, 0.0, n),
            "elapsed_candidate_s": np.arange(n, dtype=np.float64),
            "original_time_s": np.arange(n, dtype=np.float64) + 500.0,
            "source_row_right": np.arange(n, dtype=np.int64),
        }
    )
    record = SimpleNamespace(
        data=frame,
        metadata={"mission_cycle_id": "mission_x", "sequence_order": 3, "source_path": "cell.parquet"},
    )
    cycle = _to_shen_cycles(
        [record], np.array([-1.0, 2.0, 20.0]), np.array([1.0, 4.0, 25.0]), require_labels=True
    )[0]
    metadata = _mission_metadata([record], {"mission_x"})

    assert len(cycle.soc) == n
    assert len(metadata["mission_x"]["elapsed_candidate_s"]) == n
    assert np.all(metadata["mission_x"]["mission_sequence_order"] == 3)


def test_build_prediction_rows_joins_raw_fields_and_never_stringifies_nan():
    """Requirements #3/#4: joins by source_row_right via cmu_diagnostics'
    canonical helpers only, keeps the joined fields named *_raw
    (no local phase map), and a missing join match
    becomes a real None -- never the string 'nan'."""
    raw_diagnostic = pd.DataFrame(
        {
            "source_row": [10, 20, 30, 40],
            "voltage_v_raw": [3.1, 3.2, 3.3, 3.4],
            "current_a_raw": [-1.0, -1.1, -1.2, -1.3],
            "temperature_c_raw": [24.0, 25.0, 26.0, 27.0],
            "capacity_ah": [2.0, 2.0, 2.0, 2.0],
            "capacity_source": ["measured"] * 4,
            "source_rpt_id": ["rpt1"] * 4,
            "ns_raw": [4, 5, 6, 99],
            "mission_phase": ["take_off", "cruise", "landing", None],
        }
    )
    soc_reference = np.array([0.1, 0.2, 0.3, 0.4])
    soc_prediction = np.array([0.15, 0.25, 0.28, 0.5])
    metadata = {
        "mission_sequence_order": np.array([0, 0, 0, 0], dtype=np.int64),
        "elapsed_candidate_s": np.array([0.0, 1.0, 2.0, 3.0]),
        "original_time_s": np.array([100.0, 101.0, 102.0, 103.0]),
        # 50 has no match in raw_diagnostic -- must yield real None, never "nan".
        "source_row_right": np.array([10, 20, 30, 50], dtype=np.int64),
    }
    rows = _build_prediction_rows(
        framework="shen_tatn", method="TATN", source_kind="trained", scenario="unsupervised",
        repeat=2, seed=100, cell_id="VAH01", condition="baseline protocol",
        soc_reference=soc_reference, soc_prediction=soc_prediction,
        mission_ids=np.array(["m1", "m1", "m1", "m1"]),
        window_indices=np.array([0, 0, 1, 1]), timestep_indices=np.array([0, 1, 0, 1]),
        metadata=metadata, raw_diagnostic=raw_diagnostic,
    )
    assert [r["mission_phase"] for r in rows] == ["take_off", "cruise", "landing", None]
    assert [r["ns_raw"] for r in rows] == [4, 5, 6, None]
    assert rows[0]["voltage_v_raw"] == pytest.approx(3.1)
    assert rows[0]["current_a_raw"] == pytest.approx(-1.0)
    assert rows[0]["temperature_c_raw"] == pytest.approx(24.0)
    # source_row_right=50 has no match -- every joined field is a real None.
    for column in ("voltage_v_raw", "current_a_raw", "temperature_c_raw", "capacity_ah",
                    "capacity_source", "source_rpt_id", "ns_raw", "mission_phase"):
        assert rows[3][column] is None
        assert rows[3][column] != "nan"
    for row, ref, pred in zip(rows, soc_reference, soc_prediction):
        assert row["soc_reference"] == pytest.approx(ref)
        assert row["soc_prediction"] == pytest.approx(pred)
        assert row["signed_error"] == pytest.approx(pred - ref)
        assert row["absolute_error"] == pytest.approx(abs(pred - ref))


def test_evaluate_return_predictions_matches_default_metrics_and_is_a_no_op():
    """Requirements #4/#8: return_predictions=True must return EXACTLY the
    same metrics dict as the default call (never a different prediction or
    metric), must not mutate the model's state_dict, must add no extra RNG
    consumption of its own, and the returned truth/estimate must recompute
    the same MAE/RMSE the metrics dict reports."""
    device = torch.device("cpu")
    source_cycles = {p: _make_cycle(p, 2, i) for i, p in enumerate(PROFILES)}
    model, _ = train_source(
        source_cycles, epochs=1, batch_size=32, learning_rate=1e-3, device=device, seed=0
    )
    test_cycles = [_make_cycle("HWFET", 2, 90), _make_cycle("LA92", 3, 91)]
    state_before = {k: v.clone() for k, v in model.state_dict().items()}

    torch.manual_seed(999)
    metrics_only = _evaluate(model, test_cycles, device, "paper")
    rng_after_off = torch.get_rng_state().clone()

    torch.manual_seed(999)
    metrics_with_predictions, truth, estimate = _evaluate(
        model, test_cycles, device, "paper", return_predictions=True
    )
    rng_after_on = torch.get_rng_state().clone()

    assert metrics_only == metrics_with_predictions
    # Both branches call predict() the exact same way -- from the same seed,
    # they must land on the exact same RNG state, so return_predictions=True
    # adds no extra RNG consumption of its own.
    assert torch.equal(rng_after_off, rng_after_on)
    for key, value in model.state_dict().items():
        assert torch.equal(state_before[key], value)

    recomputed_rmse = float(np.sqrt(np.mean((estimate - truth) ** 2)) * 100)
    recomputed_mae = float(np.mean(np.abs(estimate - truth)) * 100)
    assert recomputed_rmse == pytest.approx(metrics_only["overall_rmse_percent"])
    assert recomputed_mae == pytest.approx(metrics_only["overall_mae_percent"])
    assert len(truth) == metrics_only["test_points"]


def test_recompute_mae_rmse_from_prediction_rows_matches_evaluate():
    """Requirement #8: MAE/RMSE recomputed from the prediction rows (as a
    reader of test_predictions.parquet would) must match _evaluate's own
    primary metrics exactly -- proving the parquet rows carry the SAME
    truth/prediction values, never a separately-recomputed approximation."""
    device = torch.device("cpu")
    source_cycles = {p: _make_cycle(p, 2, i) for i, p in enumerate(PROFILES)}
    model, _ = train_source(
        source_cycles, epochs=1, batch_size=32, learning_rate=1e-3, device=device, seed=0
    )
    test_cycles = [_make_cycle("HWFET", 2, 90), _make_cycle("LA92", 3, 91)]
    metrics, truth, estimate = _evaluate(model, test_cycles, device, "paper", return_predictions=True)

    metadata_by_mission, offset = {}, 0
    for cycle in test_cycles:
        n = len(cycle.soc)
        metadata_by_mission[cycle.profile] = {
            "mission_sequence_order": np.zeros(n, dtype=np.int64),
            "elapsed_candidate_s": np.arange(n, dtype=np.float64),
            "original_time_s": np.arange(n, dtype=np.float64),
            "source_row_right": np.arange(offset, offset + n, dtype=np.int64),
        }
        offset += n
    mission_ids, window_idx, timestep_idx, pred_metadata = _flatten_prediction_metadata(
        test_cycles, metadata_by_mission
    )
    raw_diagnostic = pd.DataFrame(
        {
            "source_row": np.arange(offset, dtype=np.int64),
            "voltage_v_raw": 3.7, "current_a_raw": -1.0, "temperature_c_raw": 25.0,
            "capacity_ah": 2.0, "capacity_source": "measured", "source_rpt_id": "rpt1",
            "ns_raw": 4, "mission_phase": "take_off",
        }
    )
    rows = _build_prediction_rows(
        framework="shen_tatn", method="TATN", source_kind="trained", scenario="unsupervised",
        seed=100, cell_id="VAH01", condition="baseline protocol",
        soc_reference=truth, soc_prediction=estimate,
        mission_ids=mission_ids, window_indices=window_idx, timestep_indices=timestep_idx,
        metadata=pred_metadata, raw_diagnostic=raw_diagnostic,
    )
    frame = pd.DataFrame(rows)
    assert len(frame) == len(truth) == metrics["test_points"]
    recomputed_rmse = float(np.sqrt(np.mean(frame["signed_error"] ** 2)) * 100)
    recomputed_mae = float(frame["absolute_error"].mean() * 100)
    assert recomputed_rmse == pytest.approx(metrics["overall_rmse_percent"])
    assert recomputed_mae == pytest.approx(metrics["overall_mae_percent"])
    # window_index starts at 0 for every mission; every timestep 0-89 present.
    for mission in frame["mission_cycle_id"].unique():
        mission_rows = frame.loc[frame["mission_cycle_id"] == mission]
        assert mission_rows["window_index"].min() == 0
        assert set(mission_rows.loc[mission_rows["window_index"] == 0, "timestep_index"]) == set(range(90))


def _schema_row(**overrides) -> dict:
    row = {
        "framework": "shen_tatn", "method": "Source only", "source_kind": "trained",
        "scenario": "unsupervised", "repeat": 1, "seed": 100, "cell_id": "VAH01", "condition": "baseline protocol",
        "split": "test", "mission_cycle_id": "m1", "mission_sequence_order": 0,
        "window_index": 0, "timestep_index": 0, "elapsed_candidate_s": 0.0, "original_time_s": 0.0,
        "source_row_right": 0, "ns_raw": 4, "mission_phase": "take_off",
        "voltage_v_raw": 3.7, "current_a_raw": -1.0, "temperature_c_raw": 25.0,
        "capacity_ah": 2.0, "capacity_source": "measured", "source_rpt_id": "rpt1",
        "soc_reference": 0.5, "soc_prediction": 0.5, "signed_error": 0.0, "absolute_error": 0.0,
    }
    row.update(overrides)
    return row


def test_write_test_predictions_atomic_success_writes_final_no_incomplete_left(tmp_path):
    """Requirement #6: on success, the final test_predictions.parquet has
    every streamed chunk's rows, and no '.incomplete' file is left behind."""
    final_path = tmp_path / "test_predictions.parquet"
    rows_by_method = [[_schema_row(method="Source only")], [_schema_row(method="TATN")]]
    _write_test_predictions_atomic(rows_by_method, TEST_PREDICTIONS_SCHEMA, final_path)

    assert final_path.is_file()
    assert not (tmp_path / "test_predictions.parquet.incomplete").exists()
    table = pq.read_table(final_path)
    assert table.num_rows == 2
    assert sorted(table.column("method").to_pylist()) == ["Source only", "TATN"]


def test_write_test_predictions_atomic_failure_leaves_incomplete_not_final(tmp_path):
    """Requirement #6: a write that fails partway must never produce a
    falsely-complete test_predictions.parquet -- the failure propagates,
    the final file is never created, and the '.incomplete' temp file (with
    a valid parquet footer from whatever chunks succeeded before the
    failure) is left on disk as an explicit partial-write marker."""
    final_path = tmp_path / "test_predictions.parquet"
    good_row = _schema_row(method="Source only")
    bad_row = _schema_row(method="TATN", window_index="not_an_int")

    with pytest.raises(Exception):
        _write_test_predictions_atomic([[good_row], [bad_row]], TEST_PREDICTIONS_SCHEMA, final_path)

    assert not final_path.exists()
    incomplete_path = tmp_path / "test_predictions.parquet.incomplete"
    assert incomplete_path.exists()
    # The first (good) chunk was already flushed before the failure, so the
    # .incomplete file has a valid, readable parquet footer.
    assert pq.read_table(incomplete_path).num_rows == 1


def test_compute_phase_metrics_restricted_to_three_phases_and_preserves_source_kind_scenario(tmp_path):
    """Requirement #5: phase_metrics.csv is restricted to take_off/cruise/
    landing and grouping must keep source_kind/scenario separate -- a
    trained-checkpoint row and a released-checkpoint row for the same cell/
    condition/phase must never be averaged together."""
    frame = pd.DataFrame(
        [
            {"source_kind": "trained", "scenario": "unsupervised", "cell_id": "VAH01",
             "condition": "baseline protocol", "method": "Source only", "mission_phase": "take_off",
             "mission_cycle_id": "m1", "soc_reference": 0.5, "soc_prediction": 0.6},
            {"source_kind": "trained", "scenario": "unsupervised", "cell_id": "VAH01",
             "condition": "baseline protocol", "method": "Source only", "mission_phase": "cc_charge",
             "mission_cycle_id": "m1", "soc_reference": 0.5, "soc_prediction": 0.9},
            {"source_kind": "released", "scenario": "unsupervised", "cell_id": "VAH01",
             "condition": "baseline protocol", "method": "Source only", "mission_phase": "take_off",
             "mission_cycle_id": "m1", "soc_reference": 0.5, "soc_prediction": 0.4},
        ]
    )
    predictions_path = tmp_path / "test_predictions.parquet"
    frame.to_parquet(predictions_path)

    result = _compute_phase_metrics(predictions_path)

    assert set(result["mission_phase"]) == {"take_off"}
    trained = result.loc[result["source_kind"] == "trained"].iloc[0]
    released = result.loc[result["source_kind"] == "released"].iloc[0]
    assert trained["mae_percent"] == pytest.approx(10.0)
    assert released["mae_percent"] == pytest.approx(10.0)
    assert trained["rmse_percent"] == pytest.approx(10.0)


def test_compute_phase_transfer_effects_delta_and_negative_transfer():
    """Requirement #5: delta_*_percent = TATN - Source only, and
    negative_transfer_* flags a positive delta (TATN worse)."""
    phase_metrics = pd.DataFrame(
        [
            {"source_kind": "trained", "scenario": "unsupervised", "cell_id": "VAH01",
             "condition": "baseline protocol", "method": "Source only", "mission_phase": "cruise",
             "mae_percent": 5.0, "rmse_percent": 6.0},
            {"source_kind": "trained", "scenario": "unsupervised", "cell_id": "VAH01",
             "condition": "baseline protocol", "method": "TATN", "mission_phase": "cruise",
             "mae_percent": 8.0, "rmse_percent": 4.0},
        ]
    )
    effects = _compute_phase_transfer_effects(phase_metrics)
    assert len(effects) == 1
    row = effects.iloc[0]
    assert row["method"] == "TATN"
    assert row["reference_method"] == "Source only"
    assert row["delta_mae_percent"] == pytest.approx(3.0)
    assert row["delta_rmse_percent"] == pytest.approx(-2.0)
    assert bool(row["negative_transfer_mae"]) is True
    assert bool(row["negative_transfer_rmse"]) is False


def test_compute_cell_transfer_effects_delta_and_negative_transfer(tmp_path):
    """Requirement #5: the overall (non-phase) counterpart, read back from
    condition_metrics.csv, same delta/negative_transfer convention,
    grouping preserving source_kind/scenario/cell_id/condition."""
    frame = pd.DataFrame(
        [
            {"source_kind": "trained", "scenario": "unsupervised", "cell_id": "VAH01",
             "condition": "baseline protocol", "method": "Source only",
             "overall_mae_percent": 5.0, "overall_rmse_percent": 6.0},
            {"source_kind": "trained", "scenario": "unsupervised", "cell_id": "VAH01",
             "condition": "baseline protocol", "method": "TATN",
             "overall_mae_percent": 3.0, "overall_rmse_percent": 9.0},
        ]
    )
    condition_metrics_path = tmp_path / "condition_metrics.csv"
    frame.to_csv(condition_metrics_path, index=False)

    effects = _compute_cell_transfer_effects(condition_metrics_path)
    assert len(effects) == 1
    row = effects.iloc[0]
    assert row["delta_mae_percent"] == pytest.approx(-2.0)
    assert row["delta_rmse_percent"] == pytest.approx(3.0)
    assert bool(row["negative_transfer_mae"]) is False
    assert bool(row["negative_transfer_rmse"]) is True


