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
    _check_resume_tatn_result,
    _to_shen_cycles,
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
        "validation_interval": 10,
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
    (tmp_path / "training_history.csv").write_text("epoch\n1\n", encoding="utf-8")

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
    (tmp_path / "training_history.csv").write_text("epoch\n1\n", encoding="utf-8")

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
    (tmp_path / "target.pt").write_bytes(b"")
    (tmp_path / "training_history.csv").write_text("epoch\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        _check_resume_tatn_result(result_path=result_path, cell_output=tmp_path, expected=expected)


def test_check_resume_tatn_result_rejects_missing_target_checkpoint(tmp_path):
    """A fully-matching result JSON is still not resumable if its target.pt
    checkpoint sidecar is missing -- e.g. an interrupted prior run."""
    expected = _complete_expected()
    result_path = tmp_path / "results.json"
    _write_result(result_path, expected)
    (tmp_path / "training_history.csv").write_text("epoch\n1\n", encoding="utf-8")
    # Deliberately do not create tmp_path / "target.pt".

    with pytest.raises(ValueError, match="target.pt"):
        _check_resume_tatn_result(result_path=result_path, cell_output=tmp_path, expected=expected)


def test_check_resume_tatn_result_rejects_missing_training_history(tmp_path):
    """A fully-matching result JSON with target.pt present is still not
    resumable if its training_history.csv sidecar is missing -- e.g. a run
    from before this diagnostic was added, or an interrupted write."""
    expected = _complete_expected()
    result_path = tmp_path / "results.json"
    _write_result(result_path, expected)
    (tmp_path / "target.pt").write_bytes(b"")
    # Deliberately do not create tmp_path / "training_history.csv".

    with pytest.raises(ValueError, match="training_history.csv"):
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


def test_run_tatn_cmu_calls_train_source_with_record_epoch_seconds():
    """train_source() must be wired with record_epoch_seconds=True only in
    the actually-training branch (never fabricating a history when loading
    an existing checkpoint)."""
    script_path = REPO_ROOT / "scripts/benchmarks/run_tatn_cmu.py"
    text = script_path.read_text(encoding="utf-8")
    call_index = text.index("source_model, _, source_epoch_diagnostics = train_source(")
    call_block = text[call_index : call_index + 700]
    assert "record_epoch_seconds=True" in call_block
    assert "source_training_history.csv" in text


def test_run_tatn_cmu_calls_adapt_with_validation_wiring():
    """adapt() must be wired with record_epoch_seconds=True, the manifest's
    validation_cycles, and args.validation_interval."""
    script_path = REPO_ROOT / "scripts/benchmarks/run_tatn_cmu.py"
    text = script_path.read_text(encoding="utf-8")
    call_index = text.index("transferred, history, target_info = adapt(")
    call_block = text[call_index : call_index + 900]
    assert "record_epoch_seconds=True" in call_block
    assert "validation_cycles=validation_cycles" in call_block
    assert "validation_interval=args.validation_interval" in call_block


def test_run_tatn_cmu_writes_per_cell_and_top_level_training_history():
    """Every cell must write its own training_history.csv (with a cell_id
    column), and these must be aggregated into a top-level
    training_history.csv alongside condition_metrics.csv."""
    script_path = REPO_ROOT / "scripts/benchmarks/run_tatn_cmu.py"
    text = script_path.read_text(encoding="utf-8")
    assert 'cell_training_history.insert(0, "cell_id", cell_id)' in text
    assert 'cell_training_history.to_csv(cell_output / "training_history.csv"' in text
    assert "training_history_frames.append(cell_training_history)" in text
    assert (
        'pd.concat(training_history_frames, ignore_index=True).to_csv(\n'
        '            args.output_root / "training_history.csv", index=False\n'
        "        )" in text
        or "training_history_frames" in text
        and 'args.output_root / "training_history.csv"' in text
    )


def test_run_tatn_cmu_has_validation_interval_cli_arg_default_ten():
    args = _parse_args()
    assert args.validation_interval == 10


def test_tatn_notebook_has_no_stale_outputs_or_execution_count():
    notebook_path = REPO_ROOT / "notebooks/benchmarks/32_tatn_panasonic25_to_cmu.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells, "expected at least one code cell"
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


def test_tatn_notebook_has_training_monitoring_sections():
    """Requirement #6: the notebook must read the adaptation loss curves,
    validation MAE/RMSE curves, and a per-cell training/validation timing
    table directly from training_history.csv/source_training_history.csv
    -- no predictions/phase analysis is added here."""
    notebook_path = REPO_ROOT / "notebooks/benchmarks/32_tatn_panasonic25_to_cmu.ipynb"
    text = notebook_path.read_text(encoding="utf-8")
    assert "training_history.csv" in text
    assert "source_training_history.csv" in text
    for column in ("estimate_loss", "adversarial_loss", "mmd_loss", "total_loss"):
        assert column in text
    assert "validation_mae" in text
    assert "validation_rmse" in text
    assert "runtime_seconds" in text
    assert "validation_runtime_seconds" in text
