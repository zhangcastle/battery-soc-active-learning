"""Minimal LSTM approximating the Vidal et al. architecture.

IMPORTANT UNCERTAINTY NOTICE
------------------------------
The exact details of the Vidal et al. (2022) sequence construction,
normalisation, and training procedure have NOT yet been confirmed from the
paper.  This implementation should NOT be used to claim paper-faithful
reproduction results until those details are verified.

See docs/known_uncertainties.md.

Architecture
------------
- Input: voltage, current, temperature  (3 features)
- Single LSTM layer with configurable hidden size (paper reports 10)
- Linear output layer mapping hidden → SOC
- Optional hard clamp of output to [0, 1]
- Sequence-to-sequence: SOC predicted at every timestep
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from battery_benchmark.models.base import BaseSOCModel
from battery_benchmark.models.registry import register_model


@register_model("vidal_lstm")
class VidalLSTM(BaseSOCModel):
    """Single-layer LSTM for SOC estimation (Vidal et al. scaffold).

    Parameters
    ----------
    input_size:
        Number of input features (default 3: voltage, current, temperature).
    hidden_size:
        LSTM hidden state dimension (paper reports 10).
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout between LSTM layers (only active for num_layers > 1).
    output_clip:
        If True, clamp output to [0, 1].
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 10,
        num_layers: int = 1,
        dropout: float = 0.0,
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
        """Forward pass.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, seq_len, input_size)``.

        Returns
        -------
        torch.Tensor
            SOC predictions of shape ``(batch, seq_len)``.
        """
        out, _ = self.lstm(x)           # (B, L, hidden_size)
        out = self.fc(out)              # (B, L, 1)
        out = out.squeeze(-1)           # (B, L)
        if self.output_clip:
            out = torch.clamp(out, 0.0, 1.0)
        return out

    def get_model_metadata(self) -> dict[str, Any]:
        return {
            "model_class": "VidalLSTM",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_clip": self.output_clip,
            "n_parameters": self.count_parameters(),
            "reproduction_status": "scaffold_only",
            "uncertainty_note": (
                "Sequence construction and normalisation details "
                "not yet confirmed from Vidal et al. paper."
            ),
        }
