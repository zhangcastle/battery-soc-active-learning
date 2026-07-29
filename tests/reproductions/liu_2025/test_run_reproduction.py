import math
import random
import sys

import numpy as np
import pandas as pd
import torch
from scripts.reproductions.liu_2025.run_reproduction import (
    _gradient_diagnostics,
    _legacy_source_protocol_sha256,
    _load_adaptation_state,
    _save_adaptation_state,
    fine_tune,
    parse_args,
    scheduled_adversarial_weight,
    supervised_loader,
)
from scripts.reproductions.liu_2025.run_temperature_suite import summarize

from battery_benchmark.reproductions.liu_2025.darm import DARM
from battery_benchmark.reproductions.liu_2025.protocol import source_protocol_sha256
from battery_benchmark.utils.hashing import sha256_file


def test_gradient_diagnostics_reports_opposing_directions():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    losses = {
        "adversarial": parameter.sum(),
        "reconstruction": -parameter.sum(),
        "mmd": parameter.square().sum(),
        "total": parameter.square().sum(),
    }
    _, cosines = _gradient_diagnostics(losses, [parameter])

    assert math.isclose(cosines["adversarial_vs_reconstruction"], -1.0)


def test_dann_adversarial_schedule_starts_at_zero_and_ends_near_one():
    assert scheduled_adversarial_weight(1.0, "dann", 0, 100) == 0.0
    assert scheduled_adversarial_weight(1.0, "dann", 99, 100) > 0.99


def test_new_cli_modes_default_to_historical_behavior(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_reproduction.py"])
    args = parse_args()
    assert args.target_bn_mode == "legacy"
    assert args.adversarial_update == "grl_joint"
    assert args.adversarial_schedule == "constant"
    assert args.grl_strength == 1.0
    assert args.source_normalization == "source_train"
    assert not args.source_only
    assert not args.include_fine_tuning
    assert not args.audit_target_bn_forward


def test_legacy_source_manifest_resolves_to_source_only_hash(tmp_path):
    manifest = tmp_path / "panasonic_25c_to_0c.csv"
    manifest.write_text(
        "record_id,role,source_sha256\n"
        "source,source_train,abc\n"
        "target,target_adapt,def\n",
        encoding="utf-8",
    )
    metadata = {
        "source_manifest": f"/content/repository/{manifest.name}",
        "source_manifest_sha256": sha256_file(manifest),
    }

    assert _legacy_source_protocol_sha256(metadata, manifest) == source_protocol_sha256(
        pd.read_csv(manifest, keep_default_na=False)
    )


def test_temperature_summary_keeps_missing_profiles_empty():
    result = {
        "target_source_only": {
            "LA92": {"rmse_percent": 2.0, "mae_percent": 1.0},
        },
        "target_darm": {
            "LA92": {"rmse_percent": 1.0, "mae_percent": 0.5},
        },
    }

    table = summarize({"s_10c": result})

    assert table["average_rmse_percent"].tolist() == [2.0, 1.0]
    assert table["rmse_percent_US06"].isna().all()


def test_fine_tuning_updates_target_and_predictor_but_not_source(tmp_path):
    torch.manual_seed(42)
    model = DARM()
    model.initialize_target()
    source_before = {
        name: value.detach().clone() for name, value in model.source_encoder.state_dict().items()
    }
    target_before = next(model.target_encoder.parameters()).detach().clone()
    predictor_before = next(model.predictor.parameters()).detach().clone()
    x = np.random.default_rng(42).normal(size=(8, 40, 3)).astype(np.float32)
    y = np.linspace(1.0, 0.0, len(x), dtype=np.float32)

    fine_tune(
        model,
        supervised_loader(x, y, batch_size=4, shuffle=False),
        epochs=1,
        learning_rate=1e-3,
        patience=0,
        device=torch.device("cpu"),
        output=tmp_path,
        resume=False,
        target_bn_mode="frozen_stats",
    )

    assert all(
        torch.equal(value, source_before[name])
        for name, value in model.source_encoder.state_dict().items()
    )
    assert not torch.equal(next(model.target_encoder.parameters()), target_before)
    assert not torch.equal(next(model.predictor.parameters()), predictor_before)


def test_adaptation_checkpoint_saves_and_restores_modes(tmp_path):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = DARM()
    model.initialize_target()
    model.set_target_bn_mode("frozen_stats")
    optimizers = {"joint": torch.optim.Adam(model.adaptation_parameters(), lr=1e-4)}
    expected = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    path = tmp_path / "adapt_last.pt"
    _save_adaptation_state(
        path,
        model,
        optimizers,
        1,
        0.5,
        0,
        {"total": 0.5},
        checkpoint_kind="last",
        target_bn_mode="frozen_stats",
        adversarial_update="grl_joint",
        adversarial_schedule="constant",
        grl_strength=0.1,
    )
    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(1)
    with torch.no_grad():
        next(model.target_encoder.parameters()).add_(1)

    state = _load_adaptation_state(
        path,
        model,
        optimizers,
        target_bn_mode="frozen_stats",
        adversarial_update="grl_joint",
        adversarial_schedule="constant",
        grl_strength=0.1,
        device="cpu",
    )

    assert state["metadata"] == {
        "checkpoint_kind": "last",
        "target_bn_mode": "frozen_stats",
        "adversarial_update": "grl_joint",
        "adversarial_schedule": "constant",
        "grl_strength": 0.1,
        "selection_uses_target_labels": False,
        "diagnostic": False,
    }
    assert all(
        torch.equal(parameter, expected[name]) for name, parameter in model.named_parameters()
    )
    assert model.target_bn_mode == "frozen_stats"
    assert random.random() == expected_python
    assert np.random.rand() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)
