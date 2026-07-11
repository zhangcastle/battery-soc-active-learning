"""MMD domain adaptation — PLACEHOLDER."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from battery_benchmark.models.base import BaseSOCModel
from battery_benchmark.transfer.base import BaseTransferMethod
from battery_benchmark.transfer.registry import register_transfer


@register_transfer("mmd")
class MMDAdaptation(BaseTransferMethod):
    """Maximum Mean Discrepancy domain adaptation.

    NOT YET IMPLEMENTED.  This class provides the interface only.
    """

    requires_source_data = True
    requires_source_labels = True
    requires_target_data = True
    requires_target_labels = False
    uses_target_test_labels = False

    def fit_source(self, model, train_loader, val_loader, trainer):
        raise NotImplementedError("MMD adaptation is not yet implemented.")

    def adapt_target(self, model, target_loader, trainer):
        raise NotImplementedError("MMD adaptation is not yet implemented.")

    def predict(self, model, loader, device):
        raise NotImplementedError("MMD adaptation is not yet implemented.")
