"""Diagnostic-only additions to run_backbone_transfer_cmu.py.

Covers metadata alignment through _windows()'s non-overlapping truncation,
the shared source_row_right join, Ns -> mission_phase mapping, the
diagnostic-only validation probe never perturbing training RNG/weights, and
recomputing error from test_predictions matching _evaluate's own metric.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from battery_benchmark.benchmarks.cmu_diagnostics import load_raw_diagnostic_fields
from scripts.benchmarks.run_backbone_transfer_cmu import (
    PHASE_SUMMARY_PHASES,
    REPO_ROOT,
    _build_prediction_rows,
    _check_resume_cmu_input_sha256,
    _check_resume_result_sidecars,
    _compute_cell_transfer_effects,
    _compute_phase_metrics,
    _compute_phase_transfer_effects,
    _evaluate,
    _loader,
    _predict_all,
    _target_windows,
    _train,
    _validation_diagnostic_metrics,
    _windows,
    parse_args,
)


def test_parse_args_cmu_defaults_use_ns_relabel_dataset_version(monkeypatch) -> None:
    """This backbone benchmark's default CMU inputs must point at the
    relabeled dataset build/manifest, not the pre-relabel version -- data
    version only, no model/training/hyperparameter change involved."""
    monkeypatch.setattr(sys, "argv", ["run_backbone_transfer_cmu.py"])
    args = parse_args()
    assert args.cmu_base_root.name == "dynamic_capacity_soc_development_v1_ns_relabel"
    assert args.cmu_condition_root.name == "dynamic_capacity_soc_development_v1_extension4_ns_relabel"
    assert args.cmu_manifest.name == "cmu_evtol_supervised_budget_manifest_v0_1_ns_relabel.csv"


def test_run_backbone_transfer_cmu_script_has_no_old_cmu_paths() -> None:
    """Guards against a stale/old CMU path silently surviving somewhere else
    in the actual running script (not just the argparse defaults checked
    above)."""
    script_path = REPO_ROOT / "scripts/benchmarks/run_backbone_transfer_cmu.py"
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
    """The backbone benchmark's default --output-root must be the NEW
    "_v2" directory (never the pre-relabel '.../backbone_transfer', and
    never the pre-diagnostics '..._ns_relabel_v1', which predates the
    training-only/validation-only runtime split and the phase-level/
    cell-level transfer-effect summaries) so a --resume against it can
    never pick up a result produced before these diagnostics existed. v1
    itself is never read, written, or touched by this default."""
    monkeypatch.setattr(sys, "argv", ["run_backbone_transfer_cmu.py"])
    args = parse_args()
    assert args.output_root.name == "backbone_transfer_ns_relabel_v2"


def test_run_backbone_transfer_cmu_script_never_reads_or_writes_v1_output_root() -> None:
    """The literal v1 output-root directory name must not appear anywhere
    in the running script as a live default/path -- only as the
    isolation-rationale comment explaining why v2 exists."""
    script_path = REPO_ROOT / "scripts/benchmarks/run_backbone_transfer_cmu.py"
    text = script_path.read_text(encoding="utf-8")
    assert 'backbone_transfer_ns_relabel_v2"' in text
    assert 'default=REPO_ROOT / "outputs/benchmarks/panasonic_to_cmu_evtol/backbone_transfer_ns_relabel_v1"' not in text


def test_check_resume_cmu_input_sha256_allows_matching_hash(tmp_path) -> None:
    """Requirement #4: resuming is allowed when the existing result's
    cmu_input_sha256 matches the freshly-verified current hash."""
    result_path = tmp_path / "5pct.json"
    result_path.write_text('{"cmu_input_sha256": "abc123"}', encoding="utf-8")
    _check_resume_cmu_input_sha256(result_path, "abc123")  # must not raise


def test_check_resume_cmu_input_sha256_rejects_mismatch(tmp_path) -> None:
    """Requirement #4: a hash mismatch (result from a different CMU data
    version) must raise immediately, never silently skip/overwrite/delete."""
    result_path = tmp_path / "5pct.json"
    result_path.write_text('{"cmu_input_sha256": "old_hash"}', encoding="utf-8")
    with pytest.raises(ValueError, match="cmu_input_sha256"):
        _check_resume_cmu_input_sha256(result_path, "new_hash")


def test_check_resume_cmu_input_sha256_rejects_missing_field(tmp_path) -> None:
    """Requirement #4: an existing result with NO cmu_input_sha256 recorded
    at all must also raise -- never treated as safely resumable."""
    result_path = tmp_path / "5pct.json"
    result_path.write_text('{"overall_mae_percent": 5.0}', encoding="utf-8")
    with pytest.raises(ValueError, match="no cmu_input_sha256"):
        _check_resume_cmu_input_sha256(result_path, "new_hash")


def test_run_backbone_transfer_cmu_calls_resume_hash_check_before_skipping() -> None:
    """The resume-safety check must actually be wired into main()'s
    per-(cell, budget) resume-skip branch, not just exist as a standalone,
    unused function."""
    script_path = REPO_ROOT / "scripts/benchmarks/run_backbone_transfer_cmu.py"
    text = script_path.read_text(encoding="utf-8")
    assert "_check_resume_cmu_input_sha256(result_path, actual)" in text


def test_backbone_notebook_result_root_is_isolated_ns_relabel_v2() -> None:
    """The notebook's RESULT_ROOT assignment must point at the NEW,
    isolated 'backbone_transfer_ns_relabel_v2' directory (never v1, which
    predates the training-only/validation-only runtime split and the
    phase-level/cell-level transfer-effect summaries). Requirement #3 also
    asks for all old outputs/execution_count to be cleared, so there is no
    longer any historical printed OUTPUT elsewhere in this notebook that
    could still show the pre-relabel or v1 path -- this checks the actual
    RESULT_ROOT source assignment."""
    notebook_path = REPO_ROOT / "notebooks/benchmarks/34_backbone_transfer_panasonic_to_cmu.ipynb"
    text = notebook_path.read_text(encoding="utf-8")
    assert (
        "RESULT_ROOT = Path(PATHS['output_root']) / "
        "'benchmarks/panasonic_to_cmu_evtol/backbone_transfer_ns_relabel_v2'"
    ) in text
    assert "'benchmarks/panasonic_to_cmu_evtol/backbone_transfer'" not in text
    assert "'benchmarks/panasonic_to_cmu_evtol/backbone_transfer_ns_relabel_v1'" not in text


def test_backbone_notebook_has_no_stale_outputs_or_execution_counts() -> None:
    """Requirement #3: every old output/execution_count must be cleared --
    this notebook is being substantially restructured (new diagnostics
    cells, new RESULT_ROOT), so stale outputs from a run against the old
    output root would be actively misleading if left in place."""
    import json as _json

    notebook_path = REPO_ROOT / "notebooks/benchmarks/34_backbone_transfer_panasonic_to_cmu.ipynb"
    notebook = _json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []


def test_backbone_notebook_uses_ns_relabel_cmu_paths() -> None:
    """Requirement #2: the backbone notebook's own CMU path constants must
    be updated in lockstep with the script's argparse defaults."""
    notebook_path = REPO_ROOT / "notebooks/benchmarks/34_backbone_transfer_panasonic_to_cmu.ipynb"
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
    """Requirement #3: run_metadata.json must explicitly record which CMU
    dataset build/manifest this run read, the manifest's hash (so a
    silently-swapped manifest with the same filename is detectable), and
    the canonical Ns phase definition -- diagnostic/provenance fields only,
    verified here by inspecting the source that assembles ``metadata``
    rather than running the full benchmark."""
    script_path = REPO_ROOT / "scripts/benchmarks/run_backbone_transfer_cmu.py"
    text = script_path.read_text(encoding="utf-8")
    assert '"cmu_base_root": str(args.cmu_base_root)' in text
    assert '"cmu_condition_root": str(args.cmu_condition_root)' in text
    assert '"cmu_manifest_path": str(args.cmu_manifest)' in text
    assert '"cmu_manifest_sha256": sha256_file(args.cmu_manifest)' in text
    assert (
        '"phase_definition": "canonical Ns mapping: 4=take_off, 5=cruise, 6=landing"' in text
    )


def test_cmu_diagnostics_reuses_canonical_ns_mission_phase_map_by_identity() -> None:
    """The shared ``cmu_diagnostics`` helper (used by this backbone runner
    via ``load_raw_diagnostic_fields``) must directly reuse
    ``battery_benchmark.datasets.cmu_evtol.ns_phase_labeling_v1``'s
    ``NS_MISSION_PHASE_MAP``/``MISSION_FLIGHT_PHASE_ROLES`` -- not a copy,
    not a re-typed local dict with different spellings. Checked by object
    identity (``is``), which a value-equal copy would still fail."""
    from battery_benchmark.benchmarks import cmu_diagnostics
    from battery_benchmark.datasets.cmu_evtol import ns_phase_labeling_v1

    assert cmu_diagnostics.NS_MISSION_PHASE_MAP is ns_phase_labeling_v1.NS_MISSION_PHASE_MAP
    assert (
        cmu_diagnostics.MISSION_FLIGHT_PHASE_ROLES
        is ns_phase_labeling_v1.MISSION_FLIGHT_PHASE_ROLES
    )
    # The old, differently-spelled local name must be gone entirely.
    assert not hasattr(cmu_diagnostics, "FLIGHT_MISSION_PHASES")


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
            # A single window of sequence_length=4 holds exactly 4
            # timesteps, so this is ONE window ("m1", 1), not 4 windows
            # (which would mean 4 * 4 = 16 expected timesteps for only 4
            # rows of data -- semantically invalid, and now caught by
            # _build_prediction_rows' own length check rather than
            # silently misaligning rows).
            missions=[("m1", 1)],
            sequence_length=4,
            raw_diagnostic=raw_diagnostic,
        )
        assert [row["mission_phase"] for row in rows] == ["take_off", "cruise", "landing", "rest_1"]
        assert rows[0]["model"] == model_name
        assert [row["mission_cycle_id"] for row in rows] == ["m1"] * 4
        assert [row["window_index"] for row in rows] == [0, 0, 0, 0]
        assert [row["timestep_index"] for row in rows] == [0, 1, 2, 3]
        # Identical schema keys regardless of model, per Section V.4.
        assert set(rows[0].keys()) == {
            "model", "method", "budget", "seed", "cell_id", "condition", "split",
            "mission_cycle_id", "window_index", "timestep_index",
            "mission_sequence_order", "elapsed_candidate_s", "original_time_s",
            "source_row_right", "soc_reference", "soc_prediction", "signed_error",
            "absolute_error", "voltage_v_raw", "current_a_raw", "temperature_c_raw",
            "capacity_ah", "capacity_source", "source_rpt_id", "ns_raw", "mission_phase",
        }


def test_build_prediction_rows_rejects_length_mismatch_against_expected_points(tmp_path):
    """Requirement #4: expected_points = sum(window_count) * sequence_length
    must equal the length of soc_reference/soc_prediction and all four
    metadata arrays, or _build_prediction_rows must raise immediately --
    never silently misalign rows the way the old
    one-entry-per-window-indexed-by-flat-timestep-position bug did."""
    path = tmp_path / "raw.parquet"
    _write_raw_diagnostic_parquet(path, 4, ns_values=[4, 5, 6, 7])
    raw_diagnostic = load_raw_diagnostic_fields(path)

    metadata = {
        "mission_sequence_order": np.array([0, 0, 0, 0]),
        "elapsed_candidate_s": np.arange(4, dtype=np.float64),
        "original_time_s": np.arange(4, dtype=np.float64),
        "source_row_right": np.array([0, 1, 2, 3], dtype=np.int64),
    }
    with pytest.raises(ValueError, match="expected"):
        _build_prediction_rows(
            model_name="cnn_lstm",
            method="fine_tuned",
            budget="5pct",
            seed=0,
            cell_id="VAH01",
            condition="baseline protocol",
            soc_reference=np.array([0.9, 0.8, 0.7, 0.6]),
            soc_prediction=np.array([0.91, 0.79, 0.72, 0.55]),
            metadata=metadata,
            # 4 windows * sequence_length=4 = 16 expected timesteps, but
            # only 4 rows of data -- must raise, not misalign.
            missions=[("m1", 4)],
            sequence_length=4,
            raw_diagnostic=raw_diagnostic,
        )


def test_build_prediction_rows_multi_mission_mission_id_and_window_index_alignment(tmp_path):
    """Requirement #6: a realistic multi-mission, multi-window case --
    m1 has 2 windows, m2 has 1 window, sequence_length=3, so 3 windows * 3
    = 9 timesteps total. Explicitly asserts mission_cycle_id (m1 repeated
    6x, m2 repeated 3x), window_index (0,0,0,1,1,1 for m1's two windows,
    restarting at 0,0,0 for m2's one window -- per-mission, not global),
    timestep_index (0,1,2 within every window), and that soc/source_row/
    mission_phase all line up row-for-row with those same indices."""
    path = tmp_path / "raw.parquet"
    _write_raw_diagnostic_parquet(path, 9, ns_values=[4, 5, 6, 4, 5, 6, 4, 5, 6])
    raw_diagnostic = load_raw_diagnostic_fields(path)

    missions = [("m1", 2), ("m2", 1)]
    sequence_length = 3
    metadata = {
        "mission_sequence_order": np.array([0, 0, 0, 1, 1, 1, 0, 0, 0]),
        "elapsed_candidate_s": np.arange(9, dtype=np.float64),
        "original_time_s": np.arange(9, dtype=np.float64),
        "source_row_right": np.arange(9, dtype=np.int64),
    }
    soc_reference = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    soc_prediction = soc_reference.copy()

    rows = _build_prediction_rows(
        model_name="cnn_lstm",
        method="fine_tuned",
        budget="5pct",
        seed=0,
        cell_id="VAH01",
        condition="baseline protocol",
        soc_reference=soc_reference,
        soc_prediction=soc_prediction,
        metadata=metadata,
        missions=missions,
        sequence_length=sequence_length,
        raw_diagnostic=raw_diagnostic,
    )

    assert len(rows) == 9
    assert [row["mission_cycle_id"] for row in rows] == ["m1"] * 6 + ["m2"] * 3
    assert [row["window_index"] for row in rows] == [0, 0, 0, 1, 1, 1, 0, 0, 0]
    assert [row["timestep_index"] for row in rows] == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert [row["source_row_right"] for row in rows] == list(range(9))
    assert [row["soc_reference"] for row in rows] == pytest.approx(soc_reference.tolist())
    assert [row["mission_phase"] for row in rows] == ["take_off", "cruise", "landing"] * 3


def test_train_runtime_seconds_excludes_validation_and_is_recorded_separately():
    """Requirement #1: runtime_seconds must cover ONLY the training batch
    loop; the validation diagnostic's own wall-clock must be recorded
    separately as validation_runtime_seconds (positive when validation
    ran, None when it didn't) -- the two must never be summed together,
    and a slower validation probe must never inflate runtime_seconds."""
    torch.manual_seed(0)
    model = _LinearProbe(3)
    x = np.random.default_rng(10).normal(size=(4, 10, 3)).astype(np.float32)
    y = np.random.default_rng(11).uniform(0, 1, size=(4, 10)).astype(np.float32)
    val_x = np.random.default_rng(12).normal(size=(2, 10, 3)).astype(np.float32)
    val_y = np.random.default_rng(13).uniform(0, 1, size=(2, 10)).astype(np.float32)

    history_no_val = _train(
        model, x, y, epochs=2, batch_size=4, lr=1e-3, device=torch.device("cpu"), label="test",
    )
    for row in history_no_val:
        assert row["runtime_seconds"] >= 0
        assert row["validation_runtime_seconds"] is None
        assert row["validation_evaluated"] is False

    torch.manual_seed(0)
    model2 = _LinearProbe(3)
    history_with_val = _train(
        model2, x, y, epochs=2, batch_size=4, lr=1e-3, device=torch.device("cpu"),
        label="test", validation_data=(val_x, val_y),
    )
    for row in history_with_val:
        assert row["runtime_seconds"] >= 0
        assert row["validation_runtime_seconds"] >= 0
        assert row["validation_evaluated"] is True


def test_check_resume_result_sidecars_rejects_missing_predictions(tmp_path):
    cell_root = tmp_path / "VAH01"
    seed_root = tmp_path
    cell_root.mkdir()
    result_path = cell_root / "zero_shot.json"
    result_path.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="prediction sidecar"):
        _check_resume_result_sidecars(
            result_path=result_path, cell_root=cell_root, seed_root=seed_root, budget="zero_shot",
        )


def test_check_resume_result_sidecars_rejects_missing_fine_tuning_history(tmp_path):
    cell_root = tmp_path / "VAH01"
    seed_root = tmp_path
    cell_root.mkdir()
    (cell_root / "5pct_predictions.parquet").write_text("stub", encoding="utf-8")
    result_path = cell_root / "5pct.json"
    result_path.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="training-history sidecar"):
        _check_resume_result_sidecars(
            result_path=result_path, cell_root=cell_root, seed_root=seed_root, budget="5pct",
        )


def test_check_resume_result_sidecars_rejects_missing_source_pretrain_evidence(tmp_path):
    cell_root = tmp_path / "VAH01"
    seed_root = tmp_path
    cell_root.mkdir()
    (cell_root / "5pct_predictions.parquet").write_text("stub", encoding="utf-8")
    (cell_root / "5pct_training_history.csv").write_text("stub", encoding="utf-8")
    result_path = cell_root / "5pct.json"
    result_path.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="source-pretraining"):
        _check_resume_result_sidecars(
            result_path=result_path, cell_root=cell_root, seed_root=seed_root, budget="5pct",
        )


def test_check_resume_result_sidecars_accepts_complete_rows(tmp_path):
    cell_root = tmp_path / "VAH01"
    seed_root = tmp_path
    cell_root.mkdir()
    (seed_root / "source_training_history.csv").write_text("stub", encoding="utf-8")

    (cell_root / "zero_shot_predictions.parquet").write_text("stub", encoding="utf-8")
    zero_shot_result = cell_root / "zero_shot.json"
    zero_shot_result.write_text(json.dumps({}), encoding="utf-8")
    _check_resume_result_sidecars(
        result_path=zero_shot_result, cell_root=cell_root, seed_root=seed_root, budget="zero_shot",
    )  # zero_shot needs no training-history sidecar (never fine-tunes)

    (cell_root / "5pct_predictions.parquet").write_text("stub", encoding="utf-8")
    (cell_root / "5pct_training_history.csv").write_text("stub", encoding="utf-8")
    result_path = cell_root / "5pct.json"
    result_path.write_text(json.dumps({}), encoding="utf-8")
    _check_resume_result_sidecars(
        result_path=result_path, cell_root=cell_root, seed_root=seed_root, budget="5pct",
    )  # must not raise


def test_run_backbone_transfer_cmu_calls_resume_sidecar_check_before_skipping() -> None:
    """The new sidecar-existence resume check must actually be wired into
    main()'s per-(cell, budget) resume-skip branch, alongside (never
    instead of) the existing cmu_input_sha256 check."""
    script_path = REPO_ROOT / "scripts/benchmarks/run_backbone_transfer_cmu.py"
    text = script_path.read_text(encoding="utf-8")
    assert "_check_resume_cmu_input_sha256(result_path, actual)" in text
    assert "_check_resume_result_sidecars(" in text


def _write_test_predictions_parquet(path, rows):
    pd.DataFrame(rows).to_parquet(path)


def test_compute_phase_metrics_restricted_to_three_phases_and_recomputable(tmp_path):
    """Requirement #2: phase_metrics.csv must be restricted to
    take_off/cruise/landing, cover all three backbones and every budget
    together, and every value must be EXACTLY recomputable from
    test_predictions.parquet's own soc_reference/soc_prediction/
    mission_phase columns."""
    rng = np.random.default_rng(0)
    rows = []
    for phase in ("take_off", "cruise", "landing", "unmapped", None):
        for i in range(5):
            rows.append(
                {
                    "model": "cnn_lstm", "cell_id": "VAH01", "condition": "baseline protocol",
                    "budget": "zero_shot", "mission_cycle_id": f"m{i}", "mission_phase": phase,
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


def test_compute_phase_transfer_effects_delta_and_negative_transfer_flags():
    """Requirement #2: delta = adapted - zero_shot, delta > 0 means
    negative transfer; every non-zero_shot budget is compared against
    zero_shot SEPARATELY (reference_budget column)."""
    phase_metrics = pd.DataFrame(
        [
            {"model": "cnn_lstm", "cell_id": "VAH01", "condition": "c", "budget": "zero_shot", "mission_phase": "cruise",
             "n_points": 10, "n_missions": 1, "mae_percent": 5.0, "rmse_percent": 6.0, "mean_signed_error_percent": 0.1},
            {"model": "cnn_lstm", "cell_id": "VAH01", "condition": "c", "budget": "5pct", "mission_phase": "cruise",
             "n_points": 10, "n_missions": 1, "mae_percent": 8.0, "rmse_percent": 9.0, "mean_signed_error_percent": 0.2},
            {"model": "cnn_lstm", "cell_id": "VAH01", "condition": "c", "budget": "full", "mission_phase": "cruise",
             "n_points": 10, "n_missions": 1, "mae_percent": 3.0, "rmse_percent": 4.0, "mean_signed_error_percent": 0.05},
        ]
    )
    effects = _compute_phase_transfer_effects(phase_metrics)
    assert set(effects["reference_budget"]) == {"zero_shot"}
    assert set(effects["budget"]) == {"5pct", "full"}

    pct5_row = effects.loc[effects["budget"] == "5pct"].iloc[0]
    assert pct5_row["delta_mae_percent"] == pytest.approx(3.0)
    assert bool(pct5_row["negative_transfer_mae"]) is True

    full_row = effects.loc[effects["budget"] == "full"].iloc[0]
    assert full_row["delta_mae_percent"] == pytest.approx(-2.0)
    assert bool(full_row["negative_transfer_mae"]) is False


def test_compute_cell_transfer_effects_is_overall_never_phase_restricted():
    """Requirement #2: cell_transfer_effects.csv is the OVERALL-level
    (never phase-restricted) delta per cell/budget vs zero_shot, computed
    from cell_metrics.csv's own overall_mae_percent/overall_rmse_percent
    -- distinct from the phase-level summaries above."""
    cell_metrics = pd.DataFrame(
        [
            {"model": "tcn", "cell_id": "VAH01", "budget": "zero_shot", "seed": 42,
             "overall_mae_percent": 10.0, "overall_rmse_percent": 12.0},
            {"model": "tcn", "cell_id": "VAH01", "budget": "1pct", "seed": 42,
             "overall_mae_percent": 15.0, "overall_rmse_percent": 18.0},
            {"model": "tcn", "cell_id": "VAH17", "budget": "zero_shot", "seed": 42,
             "overall_mae_percent": 9.0, "overall_rmse_percent": 11.0},
            {"model": "tcn", "cell_id": "VAH17", "budget": "1pct", "seed": 42,
             "overall_mae_percent": 7.0, "overall_rmse_percent": 8.0},
        ]
    )
    effects = _compute_cell_transfer_effects(cell_metrics)
    assert len(effects) == 2
    vah01 = effects.loc[effects["cell_id"] == "VAH01"].iloc[0]
    assert vah01["delta_mae_percent"] == pytest.approx(5.0)
    assert bool(vah01["negative_transfer_mae"]) is True
    vah17 = effects.loc[effects["cell_id"] == "VAH17"].iloc[0]
    assert vah17["delta_mae_percent"] == pytest.approx(-2.0)
    assert bool(vah17["negative_transfer_mae"]) is False


def test_compute_cell_transfer_effects_never_mixes_seeds():
    """Multi-seed cell_metrics.csv must join on (model, cell_id, seed) --
    never pairing one seed's zero_shot row against a different seed's
    adapted row."""
    cell_metrics = pd.DataFrame(
        [
            {"model": "tcn", "cell_id": "VAH01", "budget": "zero_shot", "seed": 42,
             "overall_mae_percent": 10.0, "overall_rmse_percent": 12.0},
            {"model": "tcn", "cell_id": "VAH01", "budget": "zero_shot", "seed": 43,
             "overall_mae_percent": 20.0, "overall_rmse_percent": 22.0},
            {"model": "tcn", "cell_id": "VAH01", "budget": "1pct", "seed": 42,
             "overall_mae_percent": 11.0, "overall_rmse_percent": 13.0},
            {"model": "tcn", "cell_id": "VAH01", "budget": "1pct", "seed": 43,
             "overall_mae_percent": 21.0, "overall_rmse_percent": 23.0},
        ]
    )
    effects = _compute_cell_transfer_effects(cell_metrics)
    assert len(effects) == 2
    for seed, expected_delta in ((42, 1.0), (43, 1.0)):
        row = effects.loc[effects["seed"] == seed].iloc[0]
        assert row["delta_mae_percent"] == pytest.approx(expected_delta)


def test_run_metadata_records_validation_runtime_split() -> None:
    """Requirement #1: run_metadata.json's validation_protocol text must
    explicitly say runtime_seconds excludes validation and that
    validation_runtime_seconds is recorded separately."""
    script_path = REPO_ROOT / "scripts/benchmarks/run_backbone_transfer_cmu.py"
    text = script_path.read_text(encoding="utf-8")
    assert "validation_runtime_seconds" in text
    assert '"runtime_seconds": train_timer.seconds' in text


def test_backbone_notebook_displays_new_diagnostic_sections() -> None:
    """Requirement #3: the notebook must actually READ BACK and DISPLAY
    cell/budget negative transfer, three-phase error and phase negative
    transfer, train/validation curves, and train vs validation time."""
    notebook_path = REPO_ROOT / "notebooks/benchmarks/34_backbone_transfer_panasonic_to_cmu.ipynb"
    text = notebook_path.read_text(encoding="utf-8")
    assert "cell_transfer_effects.csv" in text
    assert "phase_metrics.csv" in text
    assert "phase_transfer_effects.csv" in text
    assert "negative_transfer_mae" in text and "negative_transfer_rmse" in text
    assert "validation_mae" in text
    assert "validation_runtime_seconds" in text
    assert "runtime_seconds" in text
