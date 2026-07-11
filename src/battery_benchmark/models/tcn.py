"""Temporal Convolutional Network (TCN) SOC estimator — placeholder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from battery_benchmark.models.base import BaseSOCModel
from battery_benchmark.models.registry import register_model


class _CausalConv1d(nn.Module):
    """Causal dilated 1-D convolution block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad))
        self.chomp = pad
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv, nn.ReLU(), nn.Dropout(dropout))
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remove acausal padding
        out = self.conv(x)[:, :, :-self.chomp] if self.chomp else self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


@register_model("tcn")
class TCNModel(BaseSOCModel):
    """Temporal Convolutional Network for SOC estimation."""

    def __init__(
        self,
        input_size: int = 3,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        output_clip: bool = True,
    ) -> None:
        super().__init__()
        num_channels = num_channels or [32, 32, 64, 64]
        self.output_clip = output_clip

        layers: list[nn.Module] = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(_CausalConv1d(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x.permute(0, 2, 1))  # (B, C, L)
        out = self.fc(out.permute(0, 2, 1)).squeeze(-1)  # (B, L)
        if self.output_clip:
            out = torch.clamp(out, 0.0, 1.0)
        return out

    def get_model_metadata(self) -> dict[str, Any]:
        return {"model_class": "TCNModel", "n_parameters": self.count_parameters()}
