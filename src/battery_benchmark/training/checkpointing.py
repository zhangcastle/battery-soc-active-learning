"""Checkpoint save/load for Colab-safe training.

A checkpoint contains:
- model state dict
- optimizer state dict
- scheduler state dict (if any)
- current epoch
- best metric value
- random seed
- normalizer state dict (to restore scaler fitted on train data)
- resolved experiment config
- source/target domain metadata

Training can be resumed from any checkpoint via ``--resume path/to/ckpt.pt``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
    scheduler: Any | None = None,
    normalizer: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint to *path*.

    Parameters
    ----------
    path:
        File path for the ``.pt`` checkpoint.
    model, optimizer:
        PyTorch objects whose state dicts are saved.
    epoch:
        Current epoch number (0-indexed; resume will start at epoch+1).
    best_metric:
        Best validation metric seen so far.
    config:
        Resolved experiment config dict.
    scheduler:
        Optional LR scheduler.
    normalizer:
        Optional normaliser with a ``state_dict()`` method.
    extra:
        Any additional key-value pairs to store.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if normalizer is not None:
        payload["normalizer_state_dict"] = normalizer.state_dict()
    if extra:
        payload.update(extra)

    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    normalizer: Any | None = None,
    device: torch.device | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and restore state into provided objects.

    Returns
    -------
    dict
        Full checkpoint payload (includes ``epoch``, ``best_metric``, etc.).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_loc = device or "cpu"
    ckpt = torch.load(path, map_location=map_loc, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if normalizer is not None and "normalizer_state_dict" in ckpt:
        normalizer.load_state_dict(ckpt["normalizer_state_dict"])

    return ckpt
