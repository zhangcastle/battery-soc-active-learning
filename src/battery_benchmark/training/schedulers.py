"""Learning rate scheduler construction utilities."""

from __future__ import annotations

from typing import Any

import torch
import torch.optim as optim


def build_optimizer(
    model: torch.nn.Module,
    cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    """Build an optimizer from config.

    Config keys:  ``name`` (adam | sgd | adamw), ``lr``, ``weight_decay``.
    """
    opt_cfg = cfg.get("optimizer", {})
    name = opt_cfg.get("name", "adam").lower()
    lr = float(opt_cfg.get("lr", 1e-3))
    wd = float(opt_cfg.get("weight_decay", 0.0))

    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    raise ValueError(f"Unknown optimizer '{name}'. Supported: adam, adamw, sgd.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    steps_per_epoch: int = 1,
) -> Any | None:
    """Build an LR scheduler from config.  Returns None for ``name: none``."""
    sched_cfg = cfg.get("scheduler", {})
    name = sched_cfg.get("name", "none").lower()

    if name in ("none", "null", ""):
        return None

    if name == "step":
        step_size = int(sched_cfg.get("step_size", 30))
        gamma = float(sched_cfg.get("gamma", 0.1))
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name == "cosine":
        epochs = int(cfg.get("epochs", 100))
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if name == "plateau":
        patience = int(sched_cfg.get("patience", 10))
        factor = float(sched_cfg.get("factor", 0.5))
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, factor=factor
        )

    raise ValueError(f"Unknown scheduler '{name}'. Supported: none, step, cosine, plateau.")
