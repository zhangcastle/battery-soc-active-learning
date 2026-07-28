import sys

import torch
from scripts.reproductions.liu_2025.run_reproduction import (
    _load_adaptation_state,
    _save_adaptation_state,
    parse_args,
)

from battery_benchmark.reproductions.liu_2025.darm import DARM


def test_new_cli_modes_default_to_historical_behavior(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_reproduction.py"])
    args = parse_args()
    assert args.target_bn_mode == "legacy"
    assert args.adversarial_update == "grl_joint"
    assert not args.audit_target_bn_forward


def test_adaptation_checkpoint_saves_and_restores_modes(tmp_path):
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
    )
    with torch.no_grad():
        next(model.target_encoder.parameters()).add_(1)

    state = _load_adaptation_state(
        path,
        model,
        optimizers,
        target_bn_mode="frozen_stats",
        adversarial_update="grl_joint",
        device="cpu",
    )

    assert state["metadata"] == {
        "checkpoint_kind": "last",
        "target_bn_mode": "frozen_stats",
        "adversarial_update": "grl_joint",
        "selection_uses_target_labels": False,
        "diagnostic": False,
    }
    assert all(
        torch.equal(parameter, expected[name]) for name, parameter in model.named_parameters()
    )
    assert model.target_bn_mode == "frozen_stats"
