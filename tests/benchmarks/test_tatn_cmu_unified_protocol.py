"""Unified-protocol tests for the TATN CMU benchmark (output schema
``tatn_cmu_v2_diagnostic``).

Covers hash-level protocol verification, deterministic repeats, label-leak
guards, per-epoch monitoring schema, the read-only validation probe, prediction
and phase arithmetic, negative-transfer arithmetic, strict resume safety and a
small synthetic end-to-end smoke run of the whole pipeline (training ->
validation diagnostics -> predictions -> overall metrics -> phase metrics ->
timing -> metadata -> resume validation). No real data and no GPU are used.
"""

from __future__ import annotations

import json
import zlib
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch

from battery_benchmark.benchmarks.cmu_protocol_check import (
    ProtocolMismatchError,
    verify_cmu_protocol,
)
from battery_benchmark.reproductions.shen_2023.data import PROFILES, ShenCycle
from battery_benchmark.reproductions.shen_tatn import train as tatn_train
from battery_benchmark.reproductions.shen_tatn.train import (
    _validation_metrics_detailed,
    adapt,
    train_source,
)
from battery_benchmark.utils.hashing import sha256_file
from scripts.benchmarks import run_tatn_cmu as runner

WINDOW = 90


def _make_cycle(profile: str, n_windows: int, seed: int) -> ShenCycle:
    rng = np.random.default_rng(seed)
    n = n_windows * WINDOW
    return ShenCycle(
        profile=profile,
        temperature=25,
        features=rng.normal(size=(n, 3)).astype(np.float32),
        soc=np.clip(rng.uniform(0.1, 0.9, size=n).astype(np.float32), 0.0, 1.0),
        paths=(),
    )


# --------------------------------------------------------------------------
# Synthetic CMU world
# --------------------------------------------------------------------------
N_ROWS = 450  # five 90-step windows (> pseudo-label lookahead of 200)
SMOKE_CELLS = ("VAH01", "VAH30")


def _records(cell_index: int, ids: list[str], with_soc: bool = True):
    records = []
    for order, mission in enumerate(ids):
        rng = np.random.default_rng(1000 * cell_index + order + zlib.crc32(mission.encode()) % 7)
        base = 10_000 * (cell_index + 1) + 1_000 * order
        frame = pd.DataFrame(
            {
                "current": rng.normal(-5.0, 1.0, N_ROWS),
                "voltage": rng.normal(3.7, 0.1, N_ROWS),
                "temperature": rng.normal(25.0, 1.0, N_ROWS),
                "soc": np.linspace(0.9, 0.2, N_ROWS) if with_soc else 0.0,
                "elapsed_candidate_s": np.arange(N_ROWS, dtype=float),
                "original_time_s": 5_000.0 + np.arange(N_ROWS, dtype=float),
                "source_row_right": np.arange(base, base + N_ROWS, dtype=np.int64),
            }
        )
        records.append(
            SimpleNamespace(
                data=frame,
                metadata={
                    "mission_cycle_id": mission,
                    "sequence_order": order,
                    "source_path": f"cell{cell_index}.parquet",
                },
            )
        )
    return records


@pytest.fixture()
def world(tmp_path, monkeypatch):
    base_root, condition_root = tmp_path / "base", tmp_path / "cond"
    base_root.mkdir()
    condition_root.mkdir()
    manifest_rows, ids_by_cell = [], {}
    for index, cell in enumerate(SMOKE_CELLS):
        root = base_root if cell in runner.BASELINE_CELLS else condition_root
        artifact = root / f"{cell}_dynamic_soc.parquet"
        artifact.write_bytes(f"synthetic-{cell}".encode())
        digest = sha256_file(artifact)
        layout = [("train", 5_000.0, 2), ("validation", 1_000.0, 1), ("test", 1_000.0, 2)]
        ids = []
        for split, duration, count in layout:
            for rank in range(count):
                mission = f"{cell}_{split}_{rank}"
                ids.append(mission)
                manifest_rows.append(
                    {
                        "cell_id": cell, "mission_cycle_id": mission, "split": split,
                        "split_rank": rank, "mission_duration_s": duration,
                        "condition": "baseline protocol" if cell == "VAH01" else "35 °C chamber",
                        "source_artifact_sha256": digest,
                    }
                )
        ids_by_cell[cell] = ids
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    source_cycles = {p: _make_cycle(p, 2, i) for i, p in enumerate(PROFILES)}
    monkeypatch.setattr(
        runner, "load_temperature_cycles_and_range",
        lambda root, temperature: (
            source_cycles,
            np.array([-10.0, 3.0, 20.0]),
            np.array([5.0, 4.2, 30.0]),
        ),
    )

    def cell_index_of(path):
        return SMOKE_CELLS.index(path.name.split("_")[0])

    def fake_records(paths, target_hz, with_soc):
        cell = cell_index_of(paths[0])
        ids = ids_by_cell[SMOKE_CELLS[cell]]
        return SimpleNamespace(records=_records(cell, ids, with_soc))

    monkeypatch.setattr(runner, "load_cmu_features_for_uda", lambda p, target_hz: fake_records(p, target_hz, False))
    monkeypatch.setattr(runner, "load_cmu_dynamic_soc_for_vidal", lambda p, target_hz: fake_records(p, target_hz, True))

    def fake_raw(path):
        cell = cell_index_of(path)
        base = 10_000 * (cell + 1)
        n = 5_000
        rows = np.arange(base, base + n, dtype=np.int64)
        ns = np.where(np.arange(n) % 3 == 0, 4, np.where(np.arange(n) % 3 == 1, 5, 6))
        phase = np.array(["take_off", "cruise", "landing"], dtype=object)[ns - 4]
        return pd.DataFrame(
            {
                "source_row": rows, "voltage_v_raw": 3.7, "current_a_raw": -5.0,
                "temperature_c_raw": 25.0, "capacity_ah": 2.0, "capacity_source": "measured",
                "source_rpt_id": "rpt", "ns_raw": ns, "mission_phase": phase,
            }
        )

    monkeypatch.setattr(runner, "load_raw_diagnostic_fields", fake_raw)
    return SimpleNamespace(
        tmp=tmp_path, base_root=base_root, condition_root=condition_root,
        manifest=manifest_path, ids=ids_by_cell,
    )


def _argv(world, output, *extra):
    return [
        "--panasonic-data-root", str(world.tmp / "pan"),
        "--cmu-base-root", str(world.base_root),
        "--cmu-condition-root", str(world.condition_root),
        "--cmu-manifest", str(world.manifest),
        "--output-root", str(output),
        "--source-kind", "trained",
        "--cells", *SMOKE_CELLS,
        "--source-epochs", "2", "--transfer-epochs", "3", "--batch-size", "32",
        "--validation-interval", "2", "--device", "cpu",
        "--skip-reference-check",
        *extra,
    ]


@pytest.fixture()
def smoke_run(world):
    output = world.tmp / "run"
    runner.main(_argv(world, output, "--repeats", "2"))
    return world, output


# --------------------------------------------------------------------------
# Protocol check
# --------------------------------------------------------------------------
def _write_reference(root, manifest_sha, cell_hashes):
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_metadata.json").write_text(json.dumps({"cmu_manifest_sha256": manifest_sha}))
    for cell, digest in cell_hashes.items():
        (root / cell).mkdir(exist_ok=True)
        (root / cell / "results.json").write_text(json.dumps({"cell_id": cell, "cmu_input_sha256": digest}))


def _artifacts(world):
    return {
        cell: (world.base_root if cell in runner.BASELINE_CELLS else world.condition_root)
        / f"{cell}_dynamic_soc.parquet"
        for cell in SMOKE_CELLS
    }


def test_protocol_check_accepts_exact_manifest_and_reference_hashes(world):
    artifacts = _artifacts(world)
    hashes = {c: sha256_file(p) for c, p in artifacts.items()}
    reference = world.tmp / "ref"
    _write_reference(reference, sha256_file(world.manifest), hashes)
    record = verify_cmu_protocol(world.manifest, artifacts, reference_run_roots=[reference])
    assert record["manifest_hash_check"] == "passed"
    assert record["reference_check"] == "passed"
    assert record["cmu_artifact_sha256"] == hashes
    assert record["reference_runs"][0]["cells_verified"] == sorted(SMOKE_CELLS)


def test_protocol_check_rejects_artifact_not_matching_manifest(world):
    artifacts = _artifacts(world)
    artifacts["VAH01"].write_bytes(b"tampered")
    with pytest.raises(ProtocolMismatchError, match="VAH01"):
        verify_cmu_protocol(world.manifest, artifacts, require_reference=False)


def test_protocol_check_rejects_reference_with_different_hashes(world):
    artifacts = _artifacts(world)
    reference = world.tmp / "ref"
    _write_reference(reference, "0" * 64, {})
    with pytest.raises(ProtocolMismatchError, match="manifest sha256"):
        verify_cmu_protocol(world.manifest, artifacts, reference_run_roots=[reference])
    reference2 = world.tmp / "ref2"
    _write_reference(reference2, sha256_file(world.manifest), {"VAH30": "f" * 64})
    with pytest.raises(ProtocolMismatchError, match="VAH30"):
        verify_cmu_protocol(world.manifest, artifacts, reference_run_roots=[reference2])


def test_protocol_check_rejects_unverifiable_reference_and_missing_reference(world):
    artifacts = _artifacts(world)
    empty = world.tmp / "empty_ref"
    empty.mkdir()
    with pytest.raises(ProtocolMismatchError, match="cannot verify"):
        verify_cmu_protocol(world.manifest, artifacts, reference_run_roots=[empty])
    with pytest.raises(ProtocolMismatchError, match="reference"):
        verify_cmu_protocol(world.manifest, artifacts)  # reference required by default
    record = verify_cmu_protocol(world.manifest, artifacts, require_reference=False)
    assert record["reference_check"] == "skipped_by_user"


# --------------------------------------------------------------------------
# Seeds, independence, label guard
# --------------------------------------------------------------------------
def test_seed_rules_are_deterministic_and_repeat_one_matches_legacy():
    assert runner.repeat_seed(100, 1) == 100
    assert runner.repeat_seed(100, 3) == 2100
    assert runner.cell_seed(100, 1, "VAH01") == 100  # legacy: seed + cell_index
    assert runner.cell_seed(100, 1, "VAH11") == 106
    # independent of which cells were requested
    assert runner.cell_seed(100, 2, "VAH30") == 1103


def test_independent_repeats_give_independent_source_states_and_same_seed_is_identical():
    device = torch.device("cpu")
    cycles = {p: _make_cycle(p, 2, i) for i, p in enumerate(PROFILES)}
    kwargs = dict(epochs=2, batch_size=32, learning_rate=1e-3, device=device)
    a, _ = train_source(cycles, seed=runner.repeat_seed(100, 1), **kwargs)
    b, _ = train_source(cycles, seed=runner.repeat_seed(100, 2), **kwargs)
    a2, _ = train_source(cycles, seed=runner.repeat_seed(100, 1), **kwargs)
    assert runner._state_fingerprint(a) == runner._state_fingerprint(a2)
    assert runner._state_fingerprint(a) != runner._state_fingerprint(b)


def _guard(cycles, **overrides):
    kwargs = dict(
        scenario="unsupervised", labeled_profile=None,
        adaptation_ids={"a"}, validation_ids={"v"}, test_ids={"t"},
    )
    kwargs.update(overrides)
    runner._assert_no_target_label_leak(cycles, **kwargs)


def test_label_leak_guard_requires_placeholder_soc_and_disjoint_splits():
    clean = ShenCycle("a", 25, np.zeros((90, 3), np.float32), np.zeros(90, np.float32), ())
    _guard([clean])
    leaky = ShenCycle("a", 25, np.zeros((90, 3), np.float32), np.full(90, 0.5, np.float32), ())
    with pytest.raises(ValueError, match="non-placeholder SOC"):
        _guard([leaky])
    with pytest.raises(ValueError, match="labelled target mission"):
        _guard([clean], labeled_profile="a")
    with pytest.raises(ValueError, match="leakage"):
        _guard([clean], test_ids={"a"})


def test_adapt_unsupervised_never_reads_target_soc(monkeypatch):
    """Even if target SOC were non-zero the unsupervised path must ignore it:
    shuffling/replacing target soc cannot change the adapted weights."""
    device = torch.device("cpu")
    source_cycles = {p: _make_cycle(p, 2, i) for i, p in enumerate(PROFILES)}
    source, _ = train_source(
        source_cycles, epochs=1, batch_size=32, learning_rate=1e-3, device=device, seed=0
    )
    base = [_make_cycle("m1", 5, 11), _make_cycle("m2", 5, 12)]
    other = [
        ShenCycle(c.profile, c.temperature, c.features, 1.0 - c.soc, c.paths) for c in base
    ]
    kwargs = dict(
        test_profile=None, scenario="unsupervised", epochs=2, batch_size=32,
        learning_rate=1e-3, lambdas=(1.0, 1.0, 1.0), device=device, seed=3,
    )
    m1, _, _ = adapt(source, source_cycles, base, **kwargs)
    m2, _, _ = adapt(source, source_cycles, other, **kwargs)
    assert runner._state_fingerprint(m1) == runner._state_fingerprint(m2)


# --------------------------------------------------------------------------
# Validation probe + history schema
# --------------------------------------------------------------------------
def test_validation_probe_is_read_only_and_restores_modes_and_rng():
    device = torch.device("cpu")
    cycles = {p: _make_cycle(p, 2, i) for i, p in enumerate(PROFILES)}
    model, _ = train_source(cycles, epochs=1, batch_size=32, learning_rate=1e-3, device=device, seed=0)
    model.initialize_target()
    model.target_extractor.train()
    model.discriminator.train()
    model.source_extractor.eval()
    model.predictor.eval()
    modes = {id(m): m.training for m in model.modules()}
    state = {k: v.clone() for k, v in model.state_dict().items()}
    torch.manual_seed(5)
    np.random.seed(5)
    random.seed(5)
    rng = (torch.get_rng_state().clone(), np.random.get_state()[1].copy(), random.getstate())
    model.zero_grad(set_to_none=True)
    validation = [_make_cycle("v1", 2, 40), _make_cycle("v2", 3, 41)]
    result = _validation_metrics_detailed(model, validation, device)
    assert {id(m): m.training for m in model.modules()} == modes
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert torch.equal(rng[0], torch.get_rng_state())
    assert np.array_equal(rng[1], np.random.get_state()[1])
    assert rng[2] == random.getstate()
    assert all(p.grad is None for p in model.parameters())
    # arithmetic: pooled vs mission-macro
    per = []
    for cycle in validation:
        truth, est = tatn_train.predict(model, [cycle], device)
        per.append((truth, est))
    pooled_mae = np.mean(np.abs(np.concatenate([e - t for t, e in per]))) * 100
    macro_mae = np.mean([np.mean(np.abs(e - t)) for t, e in per]) * 100
    assert result["overall_mae"] == pytest.approx(pooled_mae, rel=1e-5)
    assert result["mission_macro_mae"] == pytest.approx(macro_mae, rel=1e-5)
    assert result["missions"] == 2


def test_adapt_history_has_required_monitoring_fields():
    device = torch.device("cpu")
    source_cycles = {p: _make_cycle(p, 2, i) for i, p in enumerate(PROFILES)}
    source, _ = train_source(source_cycles, epochs=1, batch_size=32, learning_rate=1e-3, device=device, seed=0)
    _, history, _ = adapt(
        source, source_cycles, [_make_cycle("m1", 5, 11), _make_cycle("m2", 5, 12)],
        test_profile=None, scenario="unsupervised", epochs=3, batch_size=32, learning_rate=1e-3,
        lambdas=(1.0, 1.0, 1.0), device=device, seed=3, record_epoch_seconds=True,
        validation_cycles=[_make_cycle("v", 2, 5)], validation_interval=1,
    )
    assert history
    for row in history:
        for key in (
            "epoch", "learning_rate", "runtime_seconds", "optimizer_steps", "total_loss",
            "adversarial_loss", "discriminator_loss", "mmd_loss", "pseudo_label_loss",
            "pseudo_accepted_windows", "pseudo_candidate_windows", "pseudo_accepted_fraction",
            "pseudo_threshold", "validation_overall_mae", "validation_overall_rmse",
            "validation_mission_macro_mae", "validation_mission_macro_rmse",
            "validation_evaluated",
        ):
            assert key in row, key
        assert row["runtime_seconds"] >= 0
        assert row["optimizer_steps"] > 0 and row["optimizer_steps"] % 2 == 0
        assert 0 <= row["pseudo_accepted_fraction"] <= 1
        assert row["pseudo_threshold"] == pytest.approx(0.08)
        assert row["validation_evaluated"] is True
    # source-only loss is not an optimised term during adaptation
    assert history[0]["source_soc_regression_loss"] is None


def test_train_source_diagnostics_carry_steps_and_regression_loss():
    cycles = {p: _make_cycle(p, 2, i) for i, p in enumerate(PROFILES)}
    _, history, diagnostics = train_source(
        cycles, epochs=2, batch_size=32, learning_rate=1e-3, device=torch.device("cpu"),
        seed=0, record_epoch_seconds=True,
    )
    for row, loss in zip(diagnostics, history):
        assert row["optimizer_steps"] > 0
        assert row["source_soc_regression_loss"] == loss == row["total_loss"]


# --------------------------------------------------------------------------
# Arithmetic: phase metrics, transfer effects, summaries
# --------------------------------------------------------------------------
def test_phase_metrics_arithmetic_with_repeats_and_bias(tmp_path):
    rows = []
    for repeat in (1, 2):
        for method, shift in (("Source only", 0.10), ("TATN", 0.04 * repeat)):
            for phase, truth in (("take_off", 0.5), ("cruise", 0.4), ("landing", 0.3)):
                for k, sign in enumerate((+1, -1, +1, +1)):
                    rows.append(
                        dict(
                            repeat=repeat, source_kind="trained", scenario="unsupervised",
                            cell_id="VAH01", condition="baseline protocol", method=method,
                            mission_phase=phase, mission_cycle_id=f"m{k % 2}",
                            soc_reference=truth,
                            soc_prediction=truth + (shift if sign > 0 else -shift / 2),
                        )
                    )
    path = tmp_path / "p.parquet"
    pd.DataFrame(rows).to_parquet(path)
    metrics = runner._compute_phase_metrics(path)
    got = metrics.loc[
        (metrics["repeat"] == 1) & (metrics["method"] == "Source only")
        & (metrics["mission_phase"] == "cruise")
    ].iloc[0]
    errors = np.array([0.10, -0.05, 0.10, 0.10])
    assert got["n_points"] == 4 and got["n_missions"] == 2
    assert got["mae_percent"] == pytest.approx(np.mean(np.abs(errors)) * 100)
    assert got["rmse_percent"] == pytest.approx(np.sqrt(np.mean(errors**2)) * 100)
    assert got["mean_signed_error_percent"] == pytest.approx(np.mean(errors) * 100)

    effects = runner._compute_phase_transfer_effects(metrics)
    assert len(effects) == 2 * 3  # repeat x phase
    row = effects.loc[(effects["repeat"] == 2) & (effects["mission_phase"] == "cruise")].iloc[0]
    assert row["delta_mae_percent"] == pytest.approx(row["tatn_mae_percent"] - row["source_only_mae_percent"])
    assert bool(row["negative_transfer_mae"]) == (row["delta_mae_percent"] > 0)
    # repeat 2: TATN shift 0.08 < source 0.10 -> improvement; repeat 1 shift 0.04
    assert (effects["delta_mae_percent"] < 0).all()


def test_full_mission_diagnostic_keeps_unmapped_phase_separate(tmp_path):
    frame = pd.DataFrame(
        [
            dict(repeat=1, source_kind="trained", scenario="unsupervised", cell_id="VAH01",
                 condition="c", method=m, mission_phase=ph, mission_cycle_id="m",
                 soc_reference=0.5, soc_prediction=0.6)
            for m in ("Source only", "TATN") for ph in ("take_off", None, "cc_charge")
        ]
    )
    path = tmp_path / "p.parquet"
    frame.to_parquet(path)
    primary = runner._compute_phase_metrics(path)
    full = runner._compute_phase_metrics(path, phases=None)
    assert set(primary["mission_phase"]) == {"take_off"}
    assert set(full["mission_phase"]) == {"take_off", "unmapped", "cc_charge"}


def test_cell_metrics_and_summary_arithmetic():
    rows = []
    for repeat, cell, so, ta in (
        (1, "VAH01", 5.0, 4.0), (2, "VAH01", 6.0, 7.0), (1, "VAH30", 8.0, 6.0), (2, "VAH30", 8.0, 8.0),
    ):
        for method, mae in (("Source only", so), ("TATN", ta)):
            rows.append(
                {"repeat": repeat, "seed": 1, "source_kind": "trained", "scenario": "unsupervised",
                 "cell_id": cell, "condition": "c", "method": method, "test_missions": 2,
                 "test_points": 180, **{k: mae for k in runner._OVERALL_METRICS}}
            )
    condition = pd.DataFrame(rows)
    info = pd.DataFrame(
        [{"repeat": r, "cell_id": c, "epochs_completed": 3, "adaptation_runtime_seconds": 2.0,
          "adaptation_wall_clock_seconds": 3.0, "pseudo_accepted_fraction": 0.5}
         for r in (1, 2) for c in ("VAH01", "VAH30")]
    )
    cell_metrics = runner.build_cell_metrics(condition, info)
    one = cell_metrics.loc[(cell_metrics["repeat"] == 2) & (cell_metrics["cell_id"] == "VAH01")].iloc[0]
    assert one["delta_overall_mae_percent"] == pytest.approx(1.0)
    assert bool(one["negative_transfer_overall_mae"]) is True
    zero = cell_metrics.loc[(cell_metrics["repeat"] == 2) & (cell_metrics["cell_id"] == "VAH30")].iloc[0]
    assert bool(zero["negative_transfer_overall_mae"]) is False  # delta == 0 is not negative transfer

    summary = runner.build_summary(cell_metrics)
    cell = summary.loc[(summary["scope"] == "cell_across_repeats") & (summary["cell_id"] == "VAH01")].iloc[0]
    assert cell["delta_overall_mae_percent_mean"] == pytest.approx(0.0)  # (-1 + 1) / 2
    assert cell["delta_overall_mae_percent_std"] == pytest.approx(np.std([-1.0, 1.0], ddof=1))
    assert cell["negative_transfer_overall_mae_rate"] == pytest.approx(0.5)
    core = summary.loc[summary["scope"] == "core7_across_repeats"].iloc[0]
    # repeat means over cells: r1 delta (-1 + -2)/2 = -1.5 ; r2 (1 + 0)/2 = 0.5
    assert core["delta_overall_mae_percent_mean"] == pytest.approx(-0.5)
    assert core["n"] == 2


# --------------------------------------------------------------------------
# End-to-end smoke + resume
# --------------------------------------------------------------------------
EXPECTED_TOP_LEVEL = {
    "run_metadata.json", "condition_metrics.csv", "cell_metrics.csv", "summary.csv",
    "training_history.csv", "source_training_history.csv", "test_predictions.parquet",
    "phase_metrics.csv", "phase_transfer_effects.csv", "phase_metrics_summary.csv",
    "phase_transfer_effects_summary.csv", "phase_metrics_full_mission_diagnostic.csv",
    "phase_transfer_effects_full_mission_diagnostic.csv",
}


def test_smoke_pipeline_outputs_schema_and_arithmetic(smoke_run):
    world, output = smoke_run
    assert EXPECTED_TOP_LEVEL <= {p.name for p in output.iterdir()}
    for repeat in (1, 2):
        for cell in SMOKE_CELLS:
            cell_dir = output / f"repeat_{repeat:02d}" / cell
            for name in ("results.json", "target.pt", "training_history.csv", "test_predictions.parquet"):
                assert (cell_dir / name).is_file()
    assert not list(output.rglob("*.incomplete"))

    predictions = pd.read_parquet(output / "test_predictions.parquet")
    assert list(predictions.columns) == runner.TEST_PREDICTIONS_SCHEMA.names
    assert set(predictions["method"]) == {"Source only", "TATN"}
    assert set(predictions["repeat"]) == {1, 2}
    assert set(predictions["cell_id"]) == set(SMOKE_CELLS)
    assert set(predictions["mission_phase"]) == {"take_off", "cruise", "landing"}
    meta = pq.read_schema(output / "test_predictions.parquet").metadata
    assert meta[b"soc_unit"] == b"fraction"
    assert meta[b"evaluation_unit"] == b"all_timesteps"
    # SOC / error arithmetic (fractions, never silently percent)
    assert predictions["soc_reference"].between(0, 1).all()
    assert np.allclose(predictions["signed_error"], predictions["soc_prediction"] - predictions["soc_reference"])
    assert np.allclose(predictions["absolute_error"], predictions["signed_error"].abs())
    # primary metrics recompute from the parquet
    cond = pd.read_csv(output / "condition_metrics.csv")
    for (repeat, cell, method), part in predictions.groupby(["repeat", "cell_id", "method"]):
        row = cond.loc[(cond["repeat"] == repeat) & (cond["cell_id"] == cell) & (cond["method"] == method)].iloc[0]
        assert part["absolute_error"].mean() * 100 == pytest.approx(row["overall_mae_percent"])
        assert np.sqrt((part["signed_error"] ** 2).mean()) * 100 == pytest.approx(row["overall_rmse_percent"])
        assert part["signed_error"].mean() * 100 == pytest.approx(row["overall_bias_percent"])
        assert len(part) == row["test_points"]

    # cell metrics: deltas and negative-transfer flags
    cells = pd.read_csv(output / "cell_metrics.csv")
    assert len(cells) == 2 * len(SMOKE_CELLS)
    assert np.allclose(
        cells["delta_overall_rmse_percent"],
        cells["tatn_overall_rmse_percent"] - cells["source_only_overall_rmse_percent"],
    )
    assert (cells["negative_transfer_overall_rmse"] == (cells["delta_overall_rmse_percent"] > 0)).all()
    # A cell whose frozen source yields no stable pseudo-label segment is a
    # recorded no-op (TATN == Source only, zero epochs) -- never hidden.
    adapted = cells["adaptation_status"] == "adapted"
    assert adapted.any() and (~adapted).any()  # the synthetic run exercises both paths
    assert (cells.loc[adapted, "epochs_completed"] > 0).all()
    assert (cells.loc[~adapted, "epochs_completed"] == 0).all()
    assert (cells.loc[~adapted, "delta_overall_mae_percent"] == 0).all()
    assert (cells["adaptation_runtime_seconds"] >= 0).all()

    # phase tables
    phases = pd.read_csv(output / "phase_metrics.csv")
    assert set(phases["mission_phase"]) == {"take_off", "cruise", "landing"}
    assert len(phases) == 2 * len(SMOKE_CELLS) * 2 * 3  # repeat x cell x method x phase
    effects = pd.read_csv(output / "phase_transfer_effects.csv")
    assert len(effects) == 2 * len(SMOKE_CELLS) * 3
    assert (effects["negative_transfer_mae"] == (effects["delta_mae_percent"] > 0)).all()
    summary = pd.read_csv(output / "phase_metrics_summary.csv")
    assert {"cell_across_repeats", "core7_across_repeats"} == set(summary["scope"])
    full = pd.read_csv(output / "phase_metrics_full_mission_diagnostic.csv")
    assert (full["evaluation_scope"] == "full_mission_diagnostic").all()

    # histories: schema, non-negative runtime, validation probe, source history
    history = pd.read_csv(output / "training_history.csv")
    source = pd.read_csv(output / "source_training_history.csv")
    for column in runner.HISTORY_COLUMNS:
        assert column in history.columns and column in source.columns, column
    assert (history["runtime_seconds"] >= 0).all() and (source["runtime_seconds"] >= 0).all()
    assert set(history["stage"]) == {"adaptation"} and set(source["stage"]) == {"source"}
    assert set(source["repeat"]) == {1, 2} and source["cell_id"].isna().all()
    assert history["validation_evaluated"].any()
    evaluated = history.loc[history["validation_evaluated"].astype(bool)]
    assert evaluated["validation_overall_mae"].notna().all()
    assert evaluated["validation_mission_macro_rmse"].notna().all()
    assert history["pseudo_accepted_fraction"].between(0, 1).all()

    # repeat independence + cell reset: one frozen source state per repeat
    results = {
        (r, c): json.loads((output / f"repeat_{r:02d}" / c / "results.json").read_text())
        for r in (1, 2) for c in SMOKE_CELLS
    }
    for repeat in (1, 2):
        assert len({results[(repeat, c)]["source_state_sha256"] for c in SMOKE_CELLS}) == 1
        assert len({results[(repeat, c)]["source_checkpoint_sha256"] for c in SMOKE_CELLS}) == 1
    assert results[(1, "VAH01")]["source_state_sha256"] != results[(2, "VAH01")]["source_state_sha256"]
    assert results[(1, "VAH01")]["seed"] == 100 and results[(2, "VAH01")]["seed"] == 1100
    assert results[(2, "VAH30")]["seed"] == 1103


def test_smoke_metadata_contract(smoke_run):
    world, output = smoke_run
    meta = json.loads((output / "run_metadata.json").read_text())
    assert meta["status"] == "completed"
    assert meta["output_schema_version"] == runner.OUTPUT_SCHEMA_VERSION
    assert meta["cmu_manifest_sha256"] == sha256_file(world.manifest)
    assert meta["protocol_verification"]["manifest_hash_check"] == "passed"
    assert set(meta["cmu_artifacts"]) == set(SMOKE_CELLS)
    assert len(meta["source_checkpoints"]) == 2
    assert all(len(c["sha256"]) == 64 for c in meta["source_checkpoints"])
    assert meta["source_independent_per_repeat"] is True
    assert meta["source_kind"] == "trained" and meta["scenario"] == "unsupervised"
    assert meta["tensor_layout"] == "paper"
    assert meta["evaluation_unit"] == "all_timesteps"
    assert meta["repeat_seeds"] == [100, 1100]
    # label-usage flags
    assert meta["target_soc_labels_used_for_training"] is False
    assert meta["target_soc_labels_used_for_adaptation"] is False
    assert meta["target_soc_labels_used_for_pseudo_labels"] is False
    assert meta["target_soc_labels_used_for_checkpoint_selection"] is False
    assert meta["target_soc_reference_used_for_posthoc_evaluation"] is True
    assert meta["validation_probe"]["diagnostic_only"] is True
    assert meta["validation_probe"]["used_for_checkpoint_selection"] is False
    assert meta["phase_mapping_version"] and len(meta["phase_mapping_sha256"]) == 64
    # not eligible: reference check skipped, only 2 repeats, not all seven cells
    assert meta["formal_benchmark_eligible"] is False
    assert any("skipped" in r for r in meta["formal_benchmark_ineligibility_reasons"])
    runtime = meta["runtime"]
    assert runtime["total_source_training_seconds"] > 0
    assert runtime["total_adaptation_training_seconds"] >= 0
    assert runtime["invocation_wall_clock_seconds"] > 0
    cost = meta["compute_cost"]
    assert cost["parameter_counts"]["total"] > 0
    assert cost["parameter_counts"]["trainable_in_adaptation"] > 0
    assert meta["history_columns"] == list(runner.HISTORY_COLUMNS)
    assert meta["transfer_hyperparameters"]["pseudo_label_threshold"] == pytest.approx(0.08)


def test_smoke_is_deterministic(world):
    runner.main(_argv(world, world.tmp / "a"))
    runner.main(_argv(world, world.tmp / "b"))
    a = pd.read_csv(world.tmp / "a" / "condition_metrics.csv")
    b = pd.read_csv(world.tmp / "b" / "condition_metrics.csv")
    metric_columns = [c for c in a.columns if c.endswith("_percent")]
    pd.testing.assert_frame_equal(a[metric_columns], b[metric_columns])
    pa_, pb_ = (pd.read_parquet(world.tmp / n / "test_predictions.parquet") for n in ("a", "b"))
    assert np.array_equal(pa_["soc_prediction"].to_numpy(), pb_["soc_prediction"].to_numpy())


def test_resume_skips_verified_cells_and_rejects_mismatches(smoke_run):
    world, output = smoke_run
    result = output / "repeat_01" / "VAH01" / "results.json"
    before = result.stat().st_mtime_ns
    runner.main(_argv(world, output, "--repeats", "2", "--resume"))
    assert result.stat().st_mtime_ns == before  # verified and skipped, not rerun

    with pytest.raises(ValueError, match="transfer_epochs"):
        runner.main(
            _argv(world, output, "--repeats", "2", "--resume") + ["--transfer-epochs", "4"]
        )
    with pytest.raises(ValueError, match="batch_size|source protocol"):
        runner.main(_argv(world, output, "--repeats", "2", "--resume", "--batch-size", "16"))
    with pytest.raises(FileExistsError):
        runner.main(_argv(world, output, "--repeats", "2"))  # non-empty, no --resume


def test_resume_rejects_changed_manifest(smoke_run):
    world, output = smoke_run
    other = world.tmp / "manifest_changed.csv"
    pd.read_csv(world.manifest).assign(extra=1).to_csv(other, index=False)
    argv = _argv(world, output, "--repeats", "2", "--resume")
    argv[argv.index("--cmu-manifest") + 1] = str(other)
    with pytest.raises(ValueError, match="manifest_sha256"):
        runner.main(argv)


def test_resume_rejects_legacy_output_root(world):
    legacy = world.tmp / "legacy"
    (legacy / "VAH01").mkdir(parents=True)
    (legacy / "VAH01" / "results.json").write_text(json.dumps({"cell_id": "VAH01"}))
    with pytest.raises(ValueError, match="legacy|run_metadata"):
        runner.main(_argv(world, legacy, "--resume"))
    # legacy run_metadata without the schema version
    (legacy / "run_metadata.json").write_text(json.dumps({"benchmark": "x"}))
    with pytest.raises(ValueError, match="legacy"):
        runner.main(_argv(world, legacy, "--resume"))
    assert (legacy / "VAH01" / "results.json").is_file()  # never deleted


def test_interrupted_parquet_write_is_never_resumable(smoke_run):
    world, output = smoke_run
    cell_dir = output / "repeat_01" / "VAH30"
    parquet = cell_dir / "test_predictions.parquet"
    parquet.replace(cell_dir / "test_predictions.parquet.incomplete")
    with pytest.raises(ValueError, match="test_predictions.parquet"):
        runner.main(_argv(world, output, "--repeats", "2", "--resume"))
    assert not parquet.exists()  # not silently regenerated or promoted


def test_unreadable_sidecar_blocks_resume(smoke_run):
    world, output = smoke_run
    (output / "repeat_02" / "VAH01" / "test_predictions.parquet").write_bytes(b"garbage")
    with pytest.raises(ValueError, match="unreadable"):
        runner.main(_argv(world, output, "--repeats", "2", "--resume"))


def test_resume_completes_after_missing_cell_result(smoke_run):
    world, output = smoke_run
    (output / "repeat_02" / "VAH30" / "results.json").unlink()
    runner.main(_argv(world, output, "--repeats", "2", "--resume"))
    cells = pd.read_csv(output / "cell_metrics.csv")
    assert len(cells) == 4


def test_trained_checkpoint_requires_single_repeat(world):
    with pytest.raises(SystemExit):
        runner.parse_args(_argv(world, world.tmp / "x", "--repeats", "2", "--trained-checkpoint", "a.pt"))


# --------------------------------------------------------------------------
# Notebook 32
# --------------------------------------------------------------------------
NOTEBOOK = runner.REPO_ROOT / "notebooks/benchmarks/32_tatn_panasonic25_to_cmu.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_is_clean_syntactically_valid_and_safe_by_default():
    import ast

    notebook = _notebook()
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        assert cell["execution_count"] is None and cell["outputs"] == []
        ast.parse("".join(cell["source"]))
    text = "".join("".join(c["source"]) for c in notebook["cells"])
    assert 'RUN_MODE = "screening"' in text
    assert "RUN_EXPERIMENT = False" in text
    assert '"screening": {"repeats": 1' in text and '"formal": {"repeats": 5' in text
    assert "tatn_ns_relabel_v2_unsupervised_" in text
    assert "tatn_ns_relabel_v1" not in text  # never the legacy root
    assert "dynamic_capacity_soc_development_v1_ns_relabel" in text
    assert "cmu_evtol_supervised_budget_manifest_v0_1_ns_relabel.csv" in text
    # thirteen numbered sections, in order
    headings = [
        line for c in notebook["cells"] if c["cell_type"] == "markdown"
        for line in "".join(c["source"]).splitlines() if line.startswith("## ")
    ]
    assert [h.split(".")[0].removeprefix("## ") for h in headings] == [str(i) for i in range(1, 14)]
    assert not any("Claude outputs" in "".join(c["source"]) for c in notebook["cells"])


def test_notebook_analysis_cells_run_against_runner_output(world, monkeypatch, tmp_path):
    """Execute every notebook cell except the Colab/Drive/git setup against a
    real (synthetic) runner output root: preflight, integrity checks, tables,
    plots and provenance report must all work from files on disk."""
    import sys
    import types

    import matplotlib

    matplotlib.use("Agg")
    shown = []
    ipython = types.ModuleType("IPython")
    display_module = types.ModuleType("IPython.display")
    display_module.display = lambda obj: shown.append(obj)
    ipython.display = display_module
    ipython.get_ipython = lambda: None
    ipython.version_info = (9, 0, 0)
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)

    drive = tmp_path / "drive"
    bench = drive / "benchmarks/panasonic_to_cmu_evtol"
    output = bench / "tatn_ns_relabel_v2_unsupervised_screening"
    runner.main(_argv(world, output, "--repeats", "1"))
    (tmp_path / "raw" / "Panasonic_18650PF").mkdir(parents=True)

    cells = ["".join(c["source"]) for c in _notebook()["cells"] if c["cell_type"] == "code"]
    setup, config = cells[0], cells[1]
    assert "drive.mount" in setup  # Colab-only cell is the only one skipped
    config = (
        config.replace(
            'PATHS = load_yaml(REPO / "configs/paths/colab_drive.yaml")',
            f'PATHS = {{"output_root": r"{drive}", "raw_data_root": r"{tmp_path / "raw"}"}}',
        )
        .replace(
            'CMU_BASE_ROOT = REPO / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1_ns_relabel"',
            f'CMU_BASE_ROOT = Path(r"{world.base_root}")',
        )
        .replace(
            'CMU_CONDITION_ROOT = REPO / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1_extension4_ns_relabel"',
            f'CMU_CONDITION_ROOT = Path(r"{world.condition_root}")',
        )
        .replace(
            'CMU_MANIFEST = REPO / "data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1_ns_relabel.csv"',
            f'CMU_MANIFEST = Path(r"{world.manifest}")',
        )
        .replace('CELLS = ["VAH01", "VAH17", "VAH27", "VAH30", "VAH15", "VAH12", "VAH11"]',
                 'CELLS = ["VAH01", "VAH30"]')
        .replace("VALIDATION_INTERVAL = 10", "VALIDATION_INTERVAL = 2")
        .replace('"screening": {"repeats": 1, "source_epochs": 300, "transfer_epochs": 150}',
                 '"screening": {"repeats": 1, "source_epochs": 2, "transfer_epochs": 3}')
        .replace("BATCH_SIZE = 64", "BATCH_SIZE = 32")
    )
    assert "CMU_MANIFEST = Path" in config and "PATHS = {" in config and 'CELLS = ["VAH01", "VAH30"]' in config
    namespace = {"REPO": runner.REPO_ROOT, "Path": __import__("pathlib").Path, "subprocess": __import__("subprocess"), "sys": sys}
    for source in [config, *cells[2:]]:
        exec(compile(source, "<notebook>", "exec"), namespace)
    assert namespace["RESULTS_READY"] is True
    assert namespace["RUN_EXPERIMENT"] is False
    assert "--skip-reference-check" in namespace["PRIMARY_COMMAND"]
    assert "--resume" in namespace["PRIMARY_COMMAND"]  # existing root is continued, not overwritten
    assert shown  # tables were produced
    overall = namespace["OVERALL_TABLE"]
    assert "core7" in overall.index and "RMSE | delta" in overall.columns
