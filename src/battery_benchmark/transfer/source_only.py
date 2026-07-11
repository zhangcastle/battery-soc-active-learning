"""Source-only transfer: train on source, apply to target without adaptation."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from battery_benchmark.models.base import BaseSOCModel
from battery_benchmark.transfer.base import BaseTransferMethod
from battery_benchmark.transfer.registry import register_transfer


@register_transfer("source_only")
class SourceOnly(BaseTransferMethod):
    """Train exclusively on labelled source data; no target adaptation.

    This is the simplest baseline: the source-trained model is applied
    directly to target data without any fine-tuning or alignment.
    """

    requires_source_data = True
    requires_source_labels = True
    requires_target_data = False
    requires_target_labels = False
    uses_target_test_labels = False

    def fit_source(
        self,
        model: BaseSOCModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        trainer: Any,
    ) -> None:
        """Train model on source data."""
        trainer.fit(model, train_loader, val_loader)

    def adapt_target(
        self,
        model: BaseSOCModel,
        target_loader: DataLoader,
        trainer: Any,
    ) -> None:
        """No target adaptation."""

    def predict(
        self,
        model: BaseSOCModel,
        loader: DataLoader,
        device: torch.device,
    ) -> torch.Tensor:
        """Run batch inference."""
        model.eval()
        preds_list = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                preds_list.append(model(x).cpu())
        return torch.cat(preds_list, dim=0)
