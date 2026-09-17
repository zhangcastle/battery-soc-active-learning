"""Diagnostic-only additions to run_sftnn_transfer_cmu.py.

Covers metadata alignment through windows()'s non-overlapping WINDOW-length
truncation, the exact source_row_right join (shared with DARM/DTNN via
battery_benchmark.benchmarks.cmu_diagnostics), Ns -> mission_phase mapping,
and recomputing error from test_predictions matching _evaluate's own metric.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from torch.utils.data import TensorDataset

from battery_benchmark.benchmarks.cmu_diagnostics import load_raw_diagnostic_fields
from battery_benchmark.reproductions.shen_2023.data import PROFILES, ShenCycle, windows
from battery_benchmark.reproductions.shen_2023.model import MEDNet
from battery_benchmark.reproductions.shen_2023.train import (
    adapt_dann_mmd,
    med_adapt,
    pretrain_med,
    predict as shen_predict,
    train_source,
)
from scripts.benchmarks.run_sftnn_transfer_cmu import (
    PHASE_SUMMARY_PHASES,
    REPO_ROOT,
    WINDOW,
    _budget_mission_ids,
    _build_prediction_rows,
    _check_resume_cmu_input_sha256,
    _check_resume_result_sidecars,
    _compute_phase_metrics,
    _compute_phase_transfer_effects,
    _evaluate,
    _flatten_dict_history,
    _flatten_scalar_history,
    _mission_metadata,
    _predict_all_with_metadata,
    _windows_metadata,
    parse_args,
)


def test_parse_args_cmu_defaults_use_ns_relabel_dataset_version(monkeypatch) -> None:
    """This SFTNN/MED benchmark's default CMU inputs must point at the
    relabeled dataset build/manifest, not the pre-relabel version -- data
    version only, no model/pseudo-labeling/training/hyperparameter change
    involved."""
    monkeypatch.setattr(sys, "argv", ["run_sftnn_transfer_cmu.py"])
    args = parse_args()
    assert args.cmu_base_root.name == "dynamic_capacity_soc_development_v1_ns_relabel"
    assert args.cmu_condition_root.name == "dynamic_capacity_soc_development_v1_extension4_ns_relabel"
    assert args.cmu_manifest.name == "cmu_evtol_supervised_budget_manifest_v0_1_ns_relabel.csv"


def test_run_sftnn_transfer_cmu_script_has_no_old_cmu_paths() -> None:
    """Guards against a stale/old CMU path silently surviving somewhere else
    in the actual running script (not just the argparse defaults checked
    above)."""
    script_path = REPO_ROOT / "scripts/benchmarks/run_sftnn_transfer_cmu.py"
    text = script_path.read_text(encoding="utf-8")
    assert "dynamic_capacity_soc_development_v1_ns_relabel" in text
    assert "dynamic_capacity_soc_development_v1_extension4_ns_relabel" in text
    assert "cmu_evtol_supervised_budget_manifest_v0_1_ns_relabel.csv" in text
    # The old (pre-relabel) path strings must not appear anywhere as a live
    # value -- checked as the exact old literal terminated by the quote
    # character actually used in this file, so it can't accidentally match
    # a substring of the new, longer "_ns_relabel" name.
    assert 'dynamic_capacity_soc_development_v1"' not in text
    assert 'dynamic_capacity_soc_development_v1_extension4"' not in text
    assert "cmu_evtol_supervised_budget_manifest_v0_1.csv" not in text


def test_parse_args_output_root_is_isolated_ns_relabel_v2(monkeypatch) -> None:
    """The SFTNN/MED benchmark's default --output-root must be the NEW
    "_v2" directory (never the pre-relabel '.../sftnn_med_transfer', and
    never the pre-diagnostics '..._ns_relabel_v1', which predates
    per-epoch runtime/validation/phase-level diagnostics) so a --resume
    against it can never pick up a result produced before these
    diagnostics existed. v1 itself is never read, written, or touched by
    this default."""
    monkeypatch.setattr(sys, "argv", ["run_sftnn_transfer_cmu.py"])
    args = parse_args()
    assert args.output_root.name == "sftnn_med_transfer_ns_relabel_v2"


def test_run_sftnn_transfer_cmu_script_never_reads_or_writes_v1_output_root() -> None:
    """The literal v1 output-root directory name must not appear anywhere
    in the running script as a live default/path -- only as the
    isolation-rationale comment explaining why v2 exists."""
    script_path = REPO_ROOT / "scripts/benchmarks/run_sftnn_transfer_cmu.py"
    text = script_path.read_text(encoding="utf-8")
    assert 'sftnn_med_transfer_ns_relabel_v2"' in text
    assert 'default=REPO_ROOT / "outputs/benchmarks/panasonic_to_cmu_evtol/sftnn_med_transfer_ns_relabel_v1"' not in text


def test_check_resume_cmu_input_sha256_allows_matching_hash(tmp_path) -> None:
    """Requirement #4: resuming is allowed when the existing result's
    cmu_input_sha256 matches the freshly-verified current hash."""
    result_path = tmp_path / "med.json"
    result_path.write_text('{"cmu_input_sha256": "abc123"}', encoding="utf-8")
    _check_resume_cmu_input_sha256(result_path, "abc123")  # must not raise


def test_check_resume_cmu_input_sha256_rejects_mismatch(tmp_path) -> None:
    """Requirement #4: a hash mismatch (result from a different CMU data
    version) must raise immediately, never silently skip/overwrite/delete."""
    result_path = tmp_path / "med.json"
    result_path.write_text('{"cmu_input_sha256": "old_hash"}', encoding="utf-8")
    with pytest.raises(ValueError, match="cmu_input_sha256"):
        _check_resume_cmu_input_sha256(result_path, "new_hash")


def test_check_resume_cmu_input_sha256_rejects_missing_field(tmp_path) -> None:
    """Requirement #4: an existing result with NO cmu_input_sha256 recorded
    at all must also raise -- never treated as safely resumable."""
    result_path = tmp_path / "med.json"
    result_path.write_text('{"overall_mae_percent": 5.0}', encoding="utf-8")
    with pytest.raises(ValueError, match="no cmu_input_sha256"):
        _check_resume_cmu_input_sha256(result_path, "new_hash")


def test_run_sftnn_transfer_cmu_calls_resume_hash_check_before_skipping() -> None:
    """The resume-safety check must actually be wired into main()'s
    per-(cell, method) resume-skip branch, not just exist as a standalone,
    unused function."""
    script_path = REPO_ROOT / "scripts/benchmarks/run_sftnn_transfer_cmu.py"
    text = script_path.read_text(encoding="utf-8")
    assert "_check_resume_cmu_input_sha256(result_path, actual)" in text


def test_sftnn_notebook_result_root_is_isolated_ns_relabel_v2() -> None:
    """The notebook's RESULT_ROOT assignment must point at the NEW,
    isolated 'sftnn_med_transfer_ns_relabel_v2' directory (never v1, which
    predates per-epoch runtime/validation/phase-level diagnostics) -- an
    OLD run's cell output elsewhere in this notebook may still print the
    pre-relabel or v1 path (historical evidence, never edited), so this
    checks the actual RESULT_ROOT source assignment specifically, not just
    any occurrence of the string."""
    notebook_path = REPO_ROOT / "notebooks/benchmarks/35_sftnn_med_transfer_panasonic_to_cmu.ipynb"
    text = notebook_path.read_text(encoding="utf-8")
    assert (
        "'benchmarks/panasonic_to_cmu_evtol/sftnn_med_transfer_ns_relabel_v2'"
    ) in text
    # The exact old (pre-relabel) source literal, terminated by the closing
    # quote, must be gone -- a historical printed OUTPUT elsewhere in the
    # notebook has no such trailing quote, so it can't false-positive here.
    assert "'benchmarks/panasonic_to_cmu_evtol/sftnn_med_transfer'" not in text


def test_sftnn_notebook_uses_ns_relabel_cmu_paths() -> None:
    """Requirement #1: the SFTNN/MED notebook's own CMU path constants must
    be updated in lockstep with the script's argparse defaults."""
    notebook_path = REPO_ROOT / "notebooks/benchmarks/35_sftnn_med_transfer_panasonic_to_cmu.ipynb"
    text = notebook_path.read_text(encoding="utf-8")
    assert "dynamic_capacity_soc_development_v1_ns_relabel'" in text
    assert "dynamic_capacity_soc_development_v1_extension4_ns_relabel'" in text
    assert "cmu_evtol_supervised_budget_manifest_v0_1_ns_relabel.csv'" in text
    # Old (pre-relabel) path strings must not remain anywhere in the
    # notebook, checked as the exact old literal terminated by the closing
    # quote so it can't match a substring of the new, longer name.
    assert "dynamic_capacity_soc_development_v1'" not in text
    assert "dynamic_capacity_soc_development_v1_extension4'" not in text
    assert "cmu_evtol_supervised_budget_manifest_v0_1.csv'" not in text


def test_run_metadata_records_cmu_data_version_and_phase_definition() -> None:
    """Requirement #2: run_metadata.json must explicitly record which CMU
    dataset build/manifest this run read, the manifest's hash (so a
    silently-swapped manifest with the same filename is detectable), and the
    canonical Ns phase definition -- diagnostic/provenance fields only,
    verified here by inspecting the source that assembles ``metadata``
    rather than running the full benchmark."""
    script_path = REPO_ROOT / "scripts/benchmarks/run_sftnn_transfer_cmu.py"
    text = script_path.read_text(encoding="utf-8")
    assert '"cmu_base_root": str(args.cmu_base_root)' in text
    assert '"cmu_condition_root": str(args.cmu_condition_root)' in text
    assert '"cmu_manifest_path": str(args.cmu_manifest)' in text
    assert '"cmu_manifest_sha256": sha256_file(args.cmu_manifest)' in text
    assert (
        '"phase_definition": "canonical Ns mapping: 4=take_off, 5=cruise, 6=landing"' in text
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
    assert [row["mission_phase"] for row in rows] == ["take_off", "cruise", "landing", None]
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


def test_flatten_scalar_history_uses_real_epoch_seconds_when_provided():
    """Requirement #1: pretrain_med/train_source's own real per-epoch
    epoch_seconds (only produced when the caller passes
    record_epoch_seconds=True) must be threaded through as-is, never
    invented, and stays None when the caller doesn't supply it (zero_shot/
    source-pretraining validation fields also stay None/False either
    way)."""
    rows_without = _flatten_scalar_history(
        [1.0, 0.5], framework="sftnn_med", method="med_source_pretrain", seed=1, learning_rate=1e-3
    )
    assert [r["runtime_seconds"] for r in rows_without] == [None, None]

    rows_with = _flatten_scalar_history(
        [1.0, 0.5],
        framework="sftnn_med",
        method="med_source_pretrain",
        seed=1,
        learning_rate=1e-3,
        epoch_seconds=[0.12, 0.34],
    )
    assert [r["runtime_seconds"] for r in rows_with] == [0.12, 0.34]
    assert all(r["validation_mae"] is None and r["validation_evaluated"] is False for r in rows_with)


def _make_cycle(profile: str, n_windows: int, seed: int) -> ShenCycle:
    rng = np.random.default_rng(seed)
    n = n_windows * WINDOW
    features = rng.normal(size=(n, 3)).astype(np.float32)
    soc = np.clip(rng.uniform(0.1, 0.9, size=n).astype(np.float32), 0.0, 1.0)
    return ShenCycle(profile=profile, temperature=25, features=features, soc=soc, paths=())


def test_pretrain_med_and_train_source_record_epoch_seconds_backward_compatible():
    """Requirement #1/#8: default calls (no flag) keep the exact original
    2-tuple return and per-epoch training trace; record_epoch_seconds=True
    ADDS a third element -- a positive, per-epoch real wall-clock list of
    the same length as history -- without changing the trace itself."""
    device = torch.device("cpu")
    source_cycles = {p: _make_cycle(p, 2, i) for i, p in enumerate(PROFILES)}

    model, history = pretrain_med(
        source_cycles, epochs=2, batch_size=32, learning_rate=1e-3, device=device, seed=0
    )
    assert isinstance(history, list) and isinstance(history[0], float)

    model2, history2, epoch_seconds = pretrain_med(
        source_cycles, epochs=2, batch_size=32, learning_rate=1e-3, device=device, seed=0,
        record_epoch_seconds=True,
    )
    assert history2 == history
    assert len(epoch_seconds) == len(history) == 2
    assert all(s >= 0 for s in epoch_seconds)

    dann_model, dann_history = train_source(
        source_cycles, epochs=2, batch_size=32, learning_rate=1e-3, device=device, seed=0
    )
    dann_model2, dann_history2, dann_epoch_seconds = train_source(
        source_cycles, epochs=2, batch_size=32, learning_rate=1e-3, device=device, seed=0,
        record_epoch_seconds=True,
    )
    assert dann_history2 == dann_history
    assert len(dann_epoch_seconds) == len(dann_history) == 2
    assert all(s >= 0 for s in dann_epoch_seconds)


def test_med_adapt_validation_diagnostic_is_bit_identical_no_op():
    """Requirement #2/#8: toggling the diagnostic-only validation probe
    (and record_epoch_seconds) on/off for med_adapt must leave the global
    RNG, the final target_extractor state_dict (parameter-by-parameter),
    and the test predictions COMPLETELY unchanged -- it is read-only and
    must never feed back into training, pseudo-labeling, the scheduler,
    early stopping, checkpoint, or model selection."""
    device = torch.device("cpu")
    source_cycles = {p: _make_cycle(p, 2, i) for i, p in enumerate(PROFILES)}
    med_source, _ = pretrain_med(
        source_cycles, epochs=1, batch_size=32, learning_rate=1e-3, device=device, seed=0
    )
    target_cycles = [_make_cycle("HWFET", 3, 50), _make_cycle("LA92", 3, 51)]
    validation_cycles = [_make_cycle("NN", 2, 52)]
    target_x, target_y = windows(target_cycles)
    target_dataset = TensorDataset(torch.from_numpy(target_x), torch.from_numpy(target_y))

    torch.manual_seed(123)
    adapted_off, history_off = med_adapt(
        med_source, target_dataset, epochs=3, batch_size=32, learning_rate=1e-3,
        device=device, seed=42,
    )
    rng_after_off = torch.get_rng_state().clone()

    torch.manual_seed(123)
    adapted_on, history_on = med_adapt(
        med_source, target_dataset, epochs=3, batch_size=32, learning_rate=1e-3,
        device=device, seed=42,
        record_epoch_seconds=True, validation_cycles=validation_cycles, validation_interval=1,
    )
    rng_after_on = torch.get_rng_state().clone()

    assert torch.equal(rng_after_off, rng_after_on), "validation probe perturbed the global RNG"
    for p_off, p_on in zip(adapted_off.state_dict().values(), adapted_on.state_dict().values()):
        assert torch.equal(p_off, p_on), "validation probe changed the final state_dict"

    _, est_off = shen_predict(adapted_off, target_cycles, device)
    _, est_on = shen_predict(adapted_on, target_cycles, device)
    assert np.array_equal(est_off, est_on), "validation probe changed test predictions"

    for row in history_on:
        assert row["runtime_seconds"] >= 0
        assert row["validation_evaluated"] is True
        assert row["validation_mae"] is not None
        assert row["validation_rmse"] is not None
    for row in history_off:
        assert row["validation_evaluated"] is False
        assert row["validation_mae"] is None
        assert "runtime_seconds" not in row


def test_adapt_dann_mmd_validation_diagnostic_is_bit_identical_no_op():
    """Same guarantee as test_med_adapt_validation_diagnostic_is_bit_identical_no_op,
    for the DANN+MMD baseline's adapt_dann_mmd, plus a validation_interval
    check (only fires every Nth epoch, never every epoch when N>1)."""
    device = torch.device("cpu")
    source_cycles = {p: _make_cycle(p, 2, i) for i, p in enumerate(PROFILES)}
    target_cycles_by_profile = {p: _make_cycle(p, 2, 100 + i) for i, p in enumerate(PROFILES)}
    validation_cycles = [_make_cycle("NN", 2, 200)]
    dann_source, _ = train_source(
        source_cycles, epochs=1, batch_size=32, learning_rate=1e-3, device=device, seed=0
    )

    common_kwargs = dict(
        test_profile="UDDS", scenario="semisupervised", labeled_profile="Cycle_1",
        epochs=3, batch_size=32, learning_rate=1e-3, lambdas=(1.0, 0.1, 0.1),
        device=device, seed=7,
    )
    adapted_off, history_off, _ = adapt_dann_mmd(
        dann_source, source_cycles, target_cycles_by_profile, **common_kwargs
    )
    adapted_on, history_on, _ = adapt_dann_mmd(
        dann_source, source_cycles, target_cycles_by_profile, **common_kwargs,
        record_epoch_seconds=True, validation_cycles=validation_cycles, validation_interval=2,
    )

    for p_off, p_on in zip(adapted_off.state_dict().values(), adapted_on.state_dict().values()):
        assert torch.equal(p_off, p_on)
    _, est_off = shen_predict(adapted_off, list(target_cycles_by_profile.values()), device)
    _, est_on = shen_predict(adapted_on, list(target_cycles_by_profile.values()), device)
    assert np.array_equal(est_off, est_on)

    evaluated_flags = [row["validation_evaluated"] for row in history_on]
    assert evaluated_flags == [False, True, False], evaluated_flags


def test_budget_mission_ids_returns_manifest_validation_split(tmp_path):
    """The manifest's EXISTING 'validation'-split missions must be
    returned separately from train/test, never merged into either --
    this is what feeds the diagnostic-only validation probe."""
    manifest = pd.DataFrame(
        {
            "cell_id": ["VAH01"] * 6,
            "mission_cycle_id": ["m1", "m2", "m3", "m4", "m5", "m6"],
            "split": ["train", "train", "test", "test", "validation", "validation"],
        }
    )
    train_ids, test_ids, validation_ids = _budget_mission_ids(manifest, "VAH01")
    assert train_ids == {"m1", "m2"}
    assert test_ids == {"m3", "m4"}
    assert validation_ids == {"m5", "m6"}
    assert train_ids.isdisjoint(validation_ids) and test_ids.isdisjoint(validation_ids)


def test_budget_mission_ids_allows_empty_validation_split(tmp_path):
    """An empty validation split for a cell is not an error -- only an
    empty TRAIN split is (MED/DANN+MMD adaptation always needs unlabeled
    train-split data; the validation probe is simply diagnostic-only and
    optional)."""
    manifest = pd.DataFrame(
        {
            "cell_id": ["VAH01"] * 2,
            "mission_cycle_id": ["m1", "m2"],
            "split": ["train", "test"],
        }
    )
    train_ids, test_ids, validation_ids = _budget_mission_ids(manifest, "VAH01")
    assert train_ids == {"m1"}
    assert validation_ids == set()


def _write_test_predictions_parquet(path, rows):
    pd.DataFrame(rows).to_parquet(path)


def test_compute_phase_metrics_restricted_to_three_phases_and_recomputable(tmp_path):
    """Requirement #3/#8: phase_metrics.csv must be restricted to
    take_off/cruise/landing and every value must be EXACTLY recomputable
    from test_predictions.parquet's own soc_reference/soc_prediction/
    mission_phase columns -- read back from the FINAL, aggregated parquet,
    never re-derived from live model state, never a second evaluation
    set."""
    rng = np.random.default_rng(0)
    rows = []
    for phase in ("take_off", "cruise", "landing", "unmapped", None):
        for i in range(5):
            rows.append(
                {
                    "cell_id": "VAH01", "condition": "baseline protocol",
                    "method": "zero_shot", "mission_cycle_id": f"m{i}",
                    "mission_phase": phase,
                    "soc_reference": float(rng.uniform(0, 1)),
                    "soc_prediction": float(rng.uniform(0, 1)),
                }
            )
    path = tmp_path / "test_predictions.parquet"
    _write_test_predictions_parquet(path, rows)

    phase_metrics = _compute_phase_metrics(path)
    assert set(phase_metrics["mission_phase"]) == set(PHASE_SUMMARY_PHASES)
    assert PHASE_SUMMARY_PHASES == ("take_off", "cruise", "landing")

    raw = pd.read_parquet(path)
    for phase in PHASE_SUMMARY_PHASES:
        subset = raw.loc[raw["mission_phase"] == phase]
        expected_mae = float((subset["soc_prediction"] - subset["soc_reference"]).abs().mean() * 100)
        expected_rmse = float(
            np.sqrt(((subset["soc_prediction"] - subset["soc_reference"]) ** 2).mean()) * 100
        )
        row = phase_metrics.loc[phase_metrics["mission_phase"] == phase].iloc[0]
        assert row["n_points"] == len(subset)
        assert np.isclose(row["mae_percent"], expected_mae)
        assert np.isclose(row["rmse_percent"], expected_rmse)


def test_compute_phase_metrics_every_cell_method_covers_all_three_phases(tmp_path):
    """Requirement #3: verifies every (cell x method) combination has
    non-zero samples in all three flight phases -- the exact check the
    delivered final report must confirm for the real run."""
    rows = []
    for cell in ("VAH01", "VAH17"):
        for method in ("zero_shot", "med", "dann_mmd"):
            for phase in PHASE_SUMMARY_PHASES:
                for i in range(3):
                    rows.append(
                        {
                            "cell_id": cell, "condition": "baseline protocol",
                            "method": method, "mission_cycle_id": f"m{i}",
                            "mission_phase": phase,
                            "soc_reference": 0.5, "soc_prediction": 0.55,
                        }
                    )
    path = tmp_path / "test_predictions.parquet"
    _write_test_predictions_parquet(path, rows)
    phase_metrics = _compute_phase_metrics(path)
    coverage = phase_metrics.groupby(["cell_id", "method"])["mission_phase"].nunique()
    assert (coverage == len(PHASE_SUMMARY_PHASES)).all()
    assert (phase_metrics["n_points"] > 0).all()


def test_compute_phase_transfer_effects_delta_and_negative_transfer_flags():
    """Requirement #3: delta = adapted - zero_shot, delta > 0 means
    negative transfer; med and dann_mmd are each compared against
    zero_shot SEPARATELY (reference_method column), never against each
    other."""
    phase_metrics = pd.DataFrame(
        [
            {"cell_id": "VAH01", "condition": "c", "method": "zero_shot", "mission_phase": "cruise",
             "n_points": 10, "n_missions": 1, "mae_percent": 5.0, "rmse_percent": 6.0, "mean_signed_error_percent": 0.1},
            {"cell_id": "VAH01", "condition": "c", "method": "med", "mission_phase": "cruise",
             "n_points": 10, "n_missions": 1, "mae_percent": 8.0, "rmse_percent": 9.0, "mean_signed_error_percent": 0.2},
            {"cell_id": "VAH01", "condition": "c", "method": "dann_mmd", "mission_phase": "cruise",
             "n_points": 10, "n_missions": 1, "mae_percent": 3.0, "rmse_percent": 4.0, "mean_signed_error_percent": 0.05},
        ]
    )
    effects = _compute_phase_transfer_effects(phase_metrics)
    assert set(effects["reference_method"]) == {"zero_shot"}
    assert set(effects["method"]) == {"med", "dann_mmd"}

    med_row = effects.loc[effects["method"] == "med"].iloc[0]
    assert med_row["delta_mae_percent"] == pytest.approx(3.0)
    assert med_row["negative_transfer_mae"] is True or bool(med_row["negative_transfer_mae"]) is True

    dann_row = effects.loc[effects["method"] == "dann_mmd"].iloc[0]
    assert dann_row["delta_mae_percent"] == pytest.approx(-2.0)
    assert bool(dann_row["negative_transfer_mae"]) is False


def test_check_resume_result_sidecars_allows_collapsed_pseudo_label_row(tmp_path):
    """A genuinely collapsed pseudo-label run (0 windows survived
    confidence filtering) never produced a prediction/training-history
    sidecar -- that is the correct, honestly-recorded outcome, and must
    NOT be treated as a missing-artifact error on --resume."""
    cell_root = tmp_path / "VAH01"
    seed_root = tmp_path
    cell_root.mkdir()
    result_path = cell_root / "med.json"
    result_path.write_text(json.dumps({"pseudo_label_collapsed": True}), encoding="utf-8")
    _check_resume_result_sidecars(
        result_path=result_path, cell_root=cell_root, seed_root=seed_root, method="med",
    )  # must not raise


def test_check_resume_result_sidecars_rejects_missing_predictions(tmp_path):
    cell_root = tmp_path / "VAH01"
    seed_root = tmp_path
    cell_root.mkdir()
    result_path = cell_root / "zero_shot.json"
    result_path.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="prediction sidecar"):
        _check_resume_result_sidecars(
            result_path=result_path, cell_root=cell_root, seed_root=seed_root, method="zero_shot",
        )


def test_check_resume_result_sidecars_rejects_missing_adaptation_history(tmp_path):
    cell_root = tmp_path / "VAH01"
    seed_root = tmp_path
    cell_root.mkdir()
    (cell_root / "med_predictions.parquet").write_text("stub", encoding="utf-8")
    result_path = cell_root / "med.json"
    result_path.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="training-history sidecar"):
        _check_resume_result_sidecars(
            result_path=result_path, cell_root=cell_root, seed_root=seed_root, method="med",
        )


def test_check_resume_result_sidecars_rejects_missing_source_pretrain_evidence(tmp_path):
    cell_root = tmp_path / "VAH01"
    seed_root = tmp_path
    cell_root.mkdir()
    (cell_root / "med_predictions.parquet").write_text("stub", encoding="utf-8")
    (cell_root / "med_training_history.csv").write_text("stub", encoding="utf-8")
    result_path = cell_root / "med.json"
    result_path.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="source-pretraining"):
        _check_resume_result_sidecars(
            result_path=result_path, cell_root=cell_root, seed_root=seed_root, method="med",
        )


def test_check_resume_result_sidecars_accepts_complete_row(tmp_path):
    cell_root = tmp_path / "VAH01"
    seed_root = tmp_path
    cell_root.mkdir()
    (cell_root / "med_predictions.parquet").write_text("stub", encoding="utf-8")
    (cell_root / "med_training_history.csv").write_text("stub", encoding="utf-8")
    (seed_root / "med_source_pretrain_training_history.csv").write_text("stub", encoding="utf-8")
    result_path = cell_root / "med.json"
    result_path.write_text(json.dumps({}), encoding="utf-8")
    _check_resume_result_sidecars(
        result_path=result_path, cell_root=cell_root, seed_root=seed_root, method="med",
    )  # must not raise

    # zero_shot needs only the prediction sidecar, never an adaptation
    # training-history sidecar (it never adapts).
    (cell_root / "zero_shot_predictions.parquet").write_text("stub", encoding="utf-8")
    zero_shot_result = cell_root / "zero_shot.json"
    zero_shot_result.write_text(json.dumps({}), encoding="utf-8")
    _check_resume_result_sidecars(
        result_path=zero_shot_result, cell_root=cell_root, seed_root=seed_root, method="zero_shot",
    )  # must not raise


def test_run_sftnn_transfer_cmu_calls_resume_sidecar_check_before_skipping() -> None:
    """The new sidecar-existence resume check must actually be wired into
    main()'s per-(cell, method) resume-skip branch, alongside (never
    instead of) the existing cmu_input_sha256 check."""
    script_path = REPO_ROOT / "scripts/benchmarks/run_sftnn_transfer_cmu.py"
    text = script_path.read_text(encoding="utf-8")
    assert "_check_resume_cmu_input_sha256(result_path, actual)" in text
    assert "_check_resume_result_sidecars(" in text


def test_parse_args_validation_interval_default_and_wired_into_calls(monkeypatch) -> None:
    """Requirement #2: --validation-interval must exist (default 1) and
    actually be threaded into both med_adapt's and adapt_dann_mmd's call
    sites, never left unused."""
    monkeypatch.setattr(sys, "argv", ["run_sftnn_transfer_cmu.py"])
    args = parse_args()
    assert args.validation_interval == 1
    script_path = REPO_ROOT / "scripts/benchmarks/run_sftnn_transfer_cmu.py"
    text = script_path.read_text(encoding="utf-8")
    assert text.count("validation_interval=args.validation_interval") == 2
    assert text.count("validation_cycles=validation_cycles") == 2
    assert text.count("record_epoch_seconds=True") >= 4


def test_sftnn_notebook_displays_new_diagnostic_sections() -> None:
    """Requirement #5: the notebook must actually READ BACK and DISPLAY the
    new diagnostics (not just save them from the script side) -- checks
    for the specific file reads and section markers, not just any mention
    of the words."""
    notebook_path = REPO_ROOT / "notebooks/benchmarks/35_sftnn_med_transfer_panasonic_to_cmu.ipynb"
    text = notebook_path.read_text(encoding="utf-8")
    # Cell-level negative-transfer table (never only the 7-cell average).
    assert "negative_transfer_table" in text
    assert "negative_transfer_mae" in text and "negative_transfer_rmse" in text
    # Phase-level metrics/effects, read back from the files the script
    # writes (never re-derived from live model state in the notebook).
    assert "phase_metrics.csv" in text
    assert "phase_transfer_effects.csv" in text
    assert "reference_method" not in text or "phase_transfer_effects" in text
    # MED and DANN+MMD training curves shown separately, plus validation
    # MAE/RMSE curves.
    assert "for col, method in enumerate(['med', 'dann_mmd'])" in text
    assert "validation_mae" in text
    assert "validation_evaluated" in text
    # Source pretraining is once-per-seed, explicitly clarified and never
    # summed per-cell.
    assert "once per seed" in text
    assert "med_source_pretrain" in text and "dann_mmd_source_pretrain" in text
    assert "runtime_seconds" in text


def test_run_metadata_truthfully_records_validation_diagnostics_flag() -> None:
    """target_validation_labels_used_for_diagnostics must now be True
    (validation SOC labels ARE read by the read-only probe), while
    target_validation_labels_used_for_training/checkpoint_selection stay
    False -- the probe never feeds back into training or model
    selection."""
    script_path = REPO_ROOT / "scripts/benchmarks/run_sftnn_transfer_cmu.py"
    text = script_path.read_text(encoding="utf-8")
    assert '"target_validation_labels_used_for_diagnostics": True' in text
    assert '"target_validation_labels_used_for_training": False' in text
    assert '"target_validation_labels_used_for_checkpoint_selection": False' in text
    assert '"validation_interval": args.validation_interval' in text
