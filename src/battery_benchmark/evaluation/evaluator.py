"""Evaluator: run inference on a DataLoader and compute all metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from battery_benchmark.evaluation.metrics import compute_metrics
from battery_benchmark.models.base import BaseSOCModel


class Evaluator:
    """Run batch inference and return metrics + raw predictions.

    Parameters
    ----------
    model:
        Trained SOC model.
    device:
        PyTorch device.
    """

    def __init__(self, model: BaseSOCModel, device: torch.device) -> None:
        self.model = model
        self.device = device

    def evaluate(
        self,
        loader: DataLoader,
        prefix: str = "",
    ) -> dict[str, Any]:
        """Run inference on *loader* and compute all supervised metrics.

        Parameters
        ----------
        loader:
            DataLoader with ``(x, y)`` batches.
        prefix:
            Metric key prefix (e.g. ``"test_"``).

        Returns
        -------
        dict with keys: metrics (dict), y_true (np), y_pred (np)
        """
        self.model.eval()
        y_true_list: list[np.ndarray] = []
        y_pred_list: list[np.ndarray] = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                pred = self.model(x).cpu().numpy()
                y_true_list.append(y.numpy())
                y_pred_list.append(pred)

        y_true = np.concatenate(y_true_list).ravel()
        y_pred = np.concatenate(y_pred_list).ravel()

        metrics = compute_metrics(y_true, y_pred, prefix=prefix)

        return {
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred,
        }
