"""Data-version, output-isolation, and resume-safety tests for run_tatn_cmu.py.

Covers only the first part of the TATN CMU benchmark diagnostics: the
default CMU/manifest paths, the isolated default output directory, and the
resume-safety check that must reject a stale results.json before ever
reusing it. Does not touch TATN's model, train_source/adapt/predict, or
evaluation logic.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.benchmarks.run_tatn_cmu import (
    REPO_ROOT,
    RESUME_CHECK_FIELDS,
    _check_resume_tatn_result,
    _to_shen_cycles,
    parse_args,
)


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


def test_output_root_default_is_isolated_tatn_ns_relabel_v1():
    args = _parse_args()
    assert args.output_root == (
        REPO_ROOT / "outputs/benchmarks/panasonic_to_cmu_evtol/tatn_ns_relabel_v1"
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
    }


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
    (tmp_path / "target.pt").write_bytes(b"")

    _check_resume_tatn_result(result_path=result_path, cell_output=tmp_path, expected=expected)


@pytest.mark.parametrize("field", RESUME_CHECK_FIELDS)
def test_check_resume_tatn_result_rejects_missing_field(tmp_path, field):
    """Every field listed in RESUME_CHECK_FIELDS must independently be
    required -- an old result predating any one of them must never be
    silently treated as resumable."""
    expected = _complete_expected()
    result_path = tmp_path / "results.json"
    _write_result(result_path, expected, drop=field)
    (tmp_path / "target.pt").write_bytes(b"")

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
        ("tensor_layout", "released_repo"),
        ("tatn_learning_rate", 1e-2),
        ("tatn_lambdas", [1.0, 0.5, 1.0]),
        ("source_early_stopping_patience", 5),
        ("source_min_epochs", 10),
        ("transfer_early_stopping_patience", 5),
        ("transfer_min_epochs", 10),
    ],
)
def test_check_resume_tatn_result_rejects_stale_field(tmp_path, field, stale_value):
    """A stale value for ANY tracked field (a different CMU data version,
    manifest, source checkpoint, scenario, or hyperparameter) must raise
    immediately, never silently reused."""
    expected = _complete_expected()
    result_path = tmp_path / "results.json"
    _write_result(result_path, expected, override={field: stale_value})
    (tmp_path / "target.pt").write_bytes(b"")

    with pytest.raises(ValueError, match=field):
        _check_resume_tatn_result(result_path=result_path, cell_output=tmp_path, expected=expected)


def test_check_resume_tatn_result_rejects_missing_target_checkpoint(tmp_path):
    """A fully-matching result JSON is still not resumable if its target.pt
    checkpoint sidecar is missing -- e.g. an interrupted prior run."""
    expected = _complete_expected()
    result_path = tmp_path / "results.json"
    _write_result(result_path, expected)
    # Deliberately do not create tmp_path / "target.pt".

    with pytest.raises(ValueError, match="target.pt"):
        _check_resume_tatn_result(result_path=result_path, cell_output=tmp_path, expected=expected)


def test_run_tatn_cmu_calls_resume_check_before_skipping():
    """The resume-safety check must actually be wired into main()'s
    per-cell resume-skip branch, not just exist as a standalone, unused
    function."""
    script_path = REPO_ROOT / "scripts/benchmarks/run_tatn_cmu.py"
    text = script_path.read_text(encoding="utf-8")
    assert "_check_resume_tatn_result(" in text
    # The wiring must run before the skip-and-continue, i.e. inside the
    # same `if args.resume and result_path.is_file():` branch.
    branch_start = text.index("if args.resume and result_path.is_file():")
    branch = text[branch_start : branch_start + 600]
    assert "_check_resume_tatn_result(" in branch
    assert "continue" in branch


def test_run_tatn_cmu_hash_checks_happen_before_resume_skip():
    """The CMU-input-hash-vs-manifest check must run before the resume-skip
    decision (so a resumed cell is still verified against the current data
    version), not only for freshly-computed cells."""
    script_path = REPO_ROOT / "scripts/benchmarks/run_tatn_cmu.py"
    text = script_path.read_text(encoding="utf-8")
    hash_check_index = text.index("CMU input hash does not match manifest")
    resume_branch_index = text.index("if args.resume and result_path.is_file():")
    assert hash_check_index < resume_branch_index


def test_tatn_notebook_result_root_is_isolated_ns_relabel_v1():
    notebook_path = REPO_ROOT / "notebooks/benchmarks/32_tatn_panasonic25_to_cmu.ipynb"
    text = notebook_path.read_text(encoding="utf-8")
    assert "benchmarks/panasonic_to_cmu_evtol/tatn_ns_relabel_v1" in text
    assert "'benchmarks/panasonic_to_cmu_evtol/tatn'" not in text


def test_tatn_notebook_cmu_paths_are_ns_relabel():
    notebook_path = REPO_ROOT / "notebooks/benchmarks/32_tatn_panasonic25_to_cmu.ipynb"
    text = notebook_path.read_text(encoding="utf-8")
    assert "dynamic_capacity_soc_development_v1_ns_relabel" in text
    assert "dynamic_capacity_soc_development_v1_extension4_ns_relabel" in text
    assert "cmu_evtol_supervised_budget_manifest_v0_1_ns_relabel.csv" in text
    assert "dynamic_capacity_soc_development_v1'" not in text
    assert "dynamic_capacity_soc_development_v1_extension4'" not in text
    assert "cmu_evtol_supervised_budget_manifest_v0_1.csv" not in text


def test_tatn_notebook_has_no_stale_outputs_or_execution_count():
    notebook_path = REPO_ROOT / "notebooks/benchmarks/32_tatn_panasonic25_to_cmu.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells, "expected at least one code cell"
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
