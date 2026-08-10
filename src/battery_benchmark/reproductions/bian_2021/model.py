"""Paper-aligned DTNN and adaptive multiscale MMD."""

from __future__ import annotations

import torch
from torch import nn


class DTNN(nn.Module):
    """Two serial CNN layers, a two-layer BiLSTM, FC, and sequence SOC head."""

    def __init__(
        self,
        input_size: int = 2,
        cnn_filters: int = 256,
        bilstm_hidden: int = 64,
        fc_hidden: int = 256,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(input_size, cnn_filters, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size, padding=padding)
        self.bilstm = nn.LSTM(
            cnn_filters,
            bilstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * bilstm_hidden, fc_hidden)
        self.regression = nn.Linear(fc_hidden, 1)
        self.activation = nn.ReLU()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.activation(self.conv1(x.transpose(1, 2)))
        c2 = self.activation(self.conv2(c1)).transpose(1, 2)
        blstm2, _ = self.bilstm(c2)
        fc = self.activation(self.fc(blstm2))
        return c2, blstm2, fc

    def forward(
        self, x: torch.Tensor, *, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        features = self.encode(x)
        prediction = self.regression(features[-1]).squeeze(-1)
        return (prediction, features) if return_features else prediction


def feature_mmd(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Eq. 6 with DTNN as the learned RKHS mapping (no hand-set kernel)."""
    source = source.flatten(1)
    target = target.flatten(1)
    return (source.mean(0) - target.mean(0)).square().sum()


def adaptive_multiscale_mmd(
    source_features: tuple[torch.Tensor, ...],
    target_features: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eq. 8: k_l = 1 - MMD_l / sum(MMD), applied at C2, BLSTM2, and FC."""
    losses = torch.stack([feature_mmd(s, t) for s, t in zip(source_features, target_features)])
    denominator = losses.detach().sum()
    weights = (
        torch.ones_like(losses)
        if denominator <= torch.finfo(losses.dtype).eps
        else 1.0 - losses.detach() / denominator
    )
    return (weights * losses).sum(), losses, weights
