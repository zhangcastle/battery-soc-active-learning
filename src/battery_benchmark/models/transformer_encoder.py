"""Transformer Encoder SOC estimator — placeholder."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from battery_benchmark.models.base import BaseSOCModel
from battery_benchmark.models.registry import register_model


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


@register_model("transformer_encoder")
class TransformerEncoderModel(BaseSOCModel):
    """Transformer Encoder for SOC estimation."""

    def __init__(
        self,
        input_size: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        output_clip: bool = True,
    ) -> None:
        super().__init__()
        self.output_clip = output_clip
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = _PositionalEncoding(d_model, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pos_enc(self.input_proj(x))  # (B, L, d_model)
        out = self.transformer(out)
        out = self.fc(out).squeeze(-1)           # (B, L)
        if self.output_clip:
            out = torch.clamp(out, 0.0, 1.0)
        return out

    def get_model_metadata(self) -> dict[str, Any]:
        return {"model_class": "TransformerEncoderModel", "n_parameters": self.count_parameters()}
