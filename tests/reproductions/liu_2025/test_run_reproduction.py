import math
import random
import sys

import numpy as np
import torch
from scripts.reproductions.liu_2025.run_reproduction import (
    _gradient_diagnostics,
    _load_adaptation_state,
    _save_adaptation_state,
    parse_args,
    scheduled_adversarial_weight,
)

from battery_benchmark.reproductions.liu_2025.darm import DARM


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
    assert args.source_normalization == "source_train"
    assert not args.source_only
    assert not args.audit_target_bn_forward


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
        device="cpu",
    )

    assert state["metadata"] == {
        "checkpoint_kind": "last",
        "target_bn_mode": "frozen_stats",
        "adversarial_update": "grl_joint",
        "adversarial_schedule": "constant",
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
