"""CNN-LSTM SOC estimator — placeholder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from battery_benchmark.models.base import BaseSOCModel
from battery_benchmark.models.registry import register_model


@register_model("cnn_lstm")
class CNNLSTMModel(BaseSOCModel):
    """1-D CNN feature extractor followed by LSTM SOC estimator."""

    def __init__(
        self,
        input_size: int = 3,
        cnn_channels: list[int] | None = None,
        cnn_kernel_size: int = 3,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        dropout: float = 0.1,
        output_clip: bool = True,
    ) -> None:
        super().__init__()
        cnn_channels = cnn_channels or [32, 64]
        self.output_clip = output_clip

        cnn_layers: list[nn.Module] = []
        in_ch = input_size
        for out_ch in cnn_channels:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) → (B, F, L) for CNN
        x_cnn = self.cnn(x.permute(0, 2, 1))  # (B, C, L)
        x_lstm = x_cnn.permute(0, 2, 1)        # (B, L, C)
        out, _ = self.lstm(x_lstm)
        out = self.fc(out).squeeze(-1)
        if self.output_clip:
            out = torch.clamp(out, 0.0, 1.0)
        return out

    def get_model_metadata(self) -> dict[str, Any]:
        return {"model_class": "CNNLSTMModel", "n_parameters": self.count_parameters()}
