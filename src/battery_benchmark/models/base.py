"""Abstract base class for all SOC estimation models."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class BaseSOCModel(nn.Module):
    """Common interface for all SOC estimation models.

    Subclasses must implement:
    - ``forward(x)``
    - ``get_model_metadata()``

    ``save_checkpoint``, ``load_checkpoint``, and ``predict_soc`` are provided
    as concrete methods with default behaviour.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, seq_len, n_features)``.

        Returns
        -------
        torch.Tensor
            SOC predictions of shape ``(batch, seq_len)`` or ``(batch,)``.
        """

    @abstractmethod
    def get_model_metadata(self) -> dict[str, Any]:
        """Return a dict describing model architecture and hyperparameters."""

    def predict_soc(
        self,
        x: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Run inference and return SOC predictions (no gradient)."""
        if device is not None:
            x = x.to(device)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            preds = self.forward(x)
        if was_training:
            self.train()
        return preds

    def save_checkpoint(
        self,
        path: str | Path,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Save model weights and metadata to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.state_dict(),
            "model_metadata": self.get_model_metadata(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> dict[str, Any]:
        """Load model weights from *path*.

        Returns
        -------
        dict
            Full checkpoint dict (useful for retrieving training state).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(ckpt["model_state_dict"], strict=strict)
        return ckpt

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
