"""Fine-tuning transfer: pre-train on source, fine-tune on labelled target."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from battery_benchmark.models.base import BaseSOCModel
from battery_benchmark.transfer.base import BaseTransferMethod
from battery_benchmark.transfer.registry import register_transfer
from battery_benchmark.utils.logging import get_logger

logger = get_logger("transfer.finetune")


@register_transfer("finetune")
class Finetune(BaseTransferMethod):
    """Source pre-training followed by target fine-tuning.

    Parameters (in ``config["transfer"]``):
    - ``finetune_lr_scale`` (float, default 0.1): multiply source LR for fine-tuning.
    - ``freeze_layers`` (list[str], default []): layer name prefixes to freeze.
    """

    requires_source_data = True
    requires_source_labels = True
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
        """Pre-train on source data."""
        trainer.fit(model, train_loader, val_loader)

    def adapt_target(
        self,
        model: BaseSOCModel,
        target_loader: DataLoader,
        trainer: Any,
    ) -> None:
        """Fine-tune on target data with reduced learning rate."""
        transfer_cfg = self.config.get("transfer", {})
        lr_scale = float(transfer_cfg.get("finetune_lr_scale", 0.1))
        freeze_layers: list[str] = transfer_cfg.get("freeze_layers", [])

        # Freeze specified layers
        if freeze_layers:
            for name, param in model.named_parameters():
                if any(name.startswith(prefix) for prefix in freeze_layers):
                    param.requires_grad_(False)
                    logger.info("Freezing layer: %s", name)

        # Scale learning rate for fine-tuning
        for pg in trainer.optimizer.param_groups:
            pg["lr"] = pg["lr"] * lr_scale

        logger.info("Fine-tuning with LR scale %.3f", lr_scale)
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
