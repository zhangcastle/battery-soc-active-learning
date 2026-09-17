"""Diagnostic-only additions to run_sftnn_transfer_cmu.py.

Covers metadata alignment through windows()'s non-overlapping WINDOW-length
truncation, the exact source_row_right join (shared with DARM/DTNN via
battery_benchmark.benchmarks.cmu_diagnostics), Ns -> mission_phase mapping,
and recomputing error from test_predictions matching _evaluate's own metric.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
import torch

from battery_benchmark.benchmarks.cmu_diagnostics import load_raw_diagnostic_fields
from battery_benchmark.reproductions.shen_2023.data import ShenCycle
from battery_benchmark.reproductions.shen_2023.model import MEDNet
from scripts.benchmarks.run_sftnn_transfer_cmu import (
    REPO_ROOT,
    WINDOW,
    _build_prediction_rows,
    _check_resume_cmu_input_sha256,
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


def test_parse_args_output_root_is_isolated_ns_relabel_v1(monkeypatch) -> None:
    """Requirement #3: the SFTNN/MED benchmark's default --output-root must
    be a NEW, isolated directory (never the pre-relabel
    '.../sftnn_med_transfer') so a --resume against it can never pick up a
    result produced from the old CMU data version, and the old results
    directory is never read, written, or touched by this default."""
    monkeypatch.setattr(sys, "argv", ["run_sftnn_transfer_cmu.py"])
    args = parse_args()
    assert args.output_root.name == "sftnn_med_transfer_ns_relabel_v1"


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


def test_sftnn_notebook_result_root_is_isolated_ns_relabel_v1() -> None:
    """Requirement #3: the notebook's RESULT_ROOT assignment must point at
    the new, isolated 'sftnn_med_transfer_ns_relabel_v1' directory -- an OLD
    run's cell output elsewhere in this notebook may still print the
    pre-relabel path (historical evidence, never edited), so this checks the
    actual RESULT_ROOT source assignment specifically, not just any
    occurrence of the string."""
    notebook_path = REPO_ROOT / "notebooks/benchmarks/35_sftnn_med_transfer_panasonic_to_cmu.ipynb"
    text = notebook_path.read_text(encoding="utf-8")
    assert (
        "'benchmarks/panasonic_to_cmu_evtol/sftnn_med_transfer_ns_relabel_v1'"
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
