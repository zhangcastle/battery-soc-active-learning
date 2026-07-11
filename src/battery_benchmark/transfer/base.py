"""Abstract base class for all transfer learning / domain adaptation methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import DataLoader

from battery_benchmark.models.base import BaseSOCModel


class BaseTransferMethod(ABC):
    """Interface for transfer learning and domain adaptation methods.

    Every method must declare its data requirements via class-level boolean
    attributes.  The runner uses these to decide which data to prepare.

    Invariant: ``uses_target_test_labels`` must always be False.
    Target test labels must never be used during training or adaptation.
    """

    # --- Data requirement declarations (must be overridden) ---
    requires_source_data: bool = NotImplemented
    requires_source_labels: bool = NotImplemented
    requires_target_data: bool = NotImplemented
    requires_target_labels: bool = NotImplemented
    uses_target_test_labels: bool = False  # MUST remain False

    def __init__(self, config: dict[str, Any]) -> None:
        if self.uses_target_test_labels:
            raise ValueError(
                f"{self.__class__.__name__} sets uses_target_test_labels=True, "
                "which is forbidden. Target test labels must never be used."
            )
        self.config = config

    @abstractmethod
    def fit_source(
        self,
        model: BaseSOCModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        trainer: Any,
    ) -> None:
        """Train the model on source-domain data."""

    @abstractmethod
    def adapt_target(
        self,
        model: BaseSOCModel,
        target_loader: DataLoader,
        trainer: Any,
    ) -> None:
        """Adapt the model to target-domain data (may be a no-op)."""

    @abstractmethod
    def predict(
        self,
        model: BaseSOCModel,
        loader: DataLoader,
        device: torch.device,
    ) -> torch.Tensor:
        """Run inference and return predictions."""

    def state_dict(self) -> dict[str, Any]:
        """Return serialisable state for checkpointing."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state from a dict."""
