"""Train from scratch on target-domain data only."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from battery_benchmark.models.base import BaseSOCModel
from battery_benchmark.transfer.base import BaseTransferMethod
from battery_benchmark.transfer.registry import register_transfer


@register_transfer("scratch")
class Scratch(BaseTransferMethod):
    """Train from random initialisation on labelled target data only.

    No source data is used.  Useful as a lower bound when target labels are
    available but source data provides little benefit.
    """

    requires_source_data = False
    requires_source_labels = False
    requires_target_data = True
    requires_target_labels = True
    uses_target_test_labels = False

    def fit_source(
        self,
        model: BaseSOCModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        trainer: Any,
    ) -> None:
        """No source training for scratch baseline."""

    def adapt_target(
        self,
        model: BaseSOCModel,
        target_loader: DataLoader,
        trainer: Any,
    ) -> None:
        """Train from scratch on target data."""
        trainer.fit(model, target_loader, target_loader)

    def predict(
        self,
        model: BaseSOCModel,
        loader: DataLoader,
        device: torch.device,
    ) -> torch.Tensor:
        model.eval()
        preds_list = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                preds_list.append(model(x).cpu())
        return torch.cat(preds_list, dim=0)
