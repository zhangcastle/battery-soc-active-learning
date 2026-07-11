"""General-purpose multi-layer LSTM for unified benchmark comparisons."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from battery_benchmark.models.base import BaseSOCModel
from battery_benchmark.models.registry import register_model


@register_model("lstm")
class LSTMModel(BaseSOCModel):
    """Multi-layer LSTM SOC estimator for unified benchmark mode."""

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_clip: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_clip = output_clip

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out).squeeze(-1)
        if self.output_clip:
            out = torch.clamp(out, 0.0, 1.0)
        return out

    def get_model_metadata(self) -> dict[str, Any]:
        return {
            "model_class": "LSTMModel",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "n_parameters": self.count_parameters(),
        }
