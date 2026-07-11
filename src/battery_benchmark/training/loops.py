"""Core train/val loop functions used by Trainer."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Mask out NaN targets (unlabelled samples)
        valid = ~torch.isnan(y)
        if not valid.any():
            continue

        optimizer.zero_grad()
        pred = model(x)

        loss = criterion(pred[valid], y[valid])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one validation epoch and return mean loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        valid = ~torch.isnan(y)
        if not valid.any():
            continue

        pred = model(x)
        loss = criterion(pred[valid], y[valid])

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)
