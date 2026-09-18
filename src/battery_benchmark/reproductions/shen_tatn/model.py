"""TATN architecture from Shen et al. (2023) and the authors' PyTorch repo.

Restored verbatim from commit ``973525c^`` (the parent of the SFTNN/MED
refactor), which is the last commit before this architecture was replaced in
``battery_benchmark.reproductions.shen_2023.model`` by MEDNet/DANNBaselineNet.
No algorithmic content has been changed from that historical version.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

WINDOW = 90
FEATURES = 3
FILTERS = 256
HIDDEN = 64


class FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, FILTERS, (3, 1), padding="same"),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(FILTERS, FILTERS, (3, 1), padding="same"),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # The official implementation uses two stacked, two-layer BiLSTMs.
        self.lstm1 = nn.LSTM(
            FILTERS * FEATURES, HIDDEN, 2, bidirectional=True, batch_first=True
        )
        self.lstm2 = nn.LSTM(
            HIDDEN * 2, HIDDEN, 2, bidirectional=True, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (WINDOW, FEATURES):
            raise ValueError(f"TATN expects (batch, {WINDOW}, {FEATURES}), got {tuple(x.shape)}")
        x = self.conv(x.transpose(1, 2).unsqueeze(1))
        x = x.reshape(x.shape[0], FILTERS * FEATURES, WINDOW).transpose(1, 2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x


class SOCPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN * 2),
            nn.ReLU(),
            nn.Linear(HIDDEN * 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


class DomainDiscriminator(nn.Module):
    """Binary discriminator used by the released implementation."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(WINDOW * HIDDEN * 2, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class TATN(nn.Module):
    """Separate frozen source and trainable target branches from Fig. 1."""

    def __init__(self) -> None:
        super().__init__()
        self.source_extractor = FeatureExtractor()
        self.target_extractor = FeatureExtractor()
        self.predictor = SOCPredictor()
        self.discriminator = DomainDiscriminator()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.target_extractor(x))

    def forward_source(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.source_extractor(x))

    def initialize_target(self) -> None:
        self.target_extractor.load_state_dict(self.source_extractor.state_dict())
        for parameter in (*self.source_extractor.parameters(), *self.predictor.parameters()):
            parameter.requires_grad_(False)
        self.source_extractor.eval()

    def source_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "extractor": deepcopy(self.source_extractor.state_dict()),
            "predictor": deepcopy(self.predictor.state_dict()),
        }

    def load_source_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        self.source_extractor.load_state_dict(state["extractor"])
        self.predictor.load_state_dict(state["predictor"])
        self.initialize_target()


def gaussian_mmd(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian-kernel MMD used by the released TATN implementation."""

    source = source.flatten(1)
    target = target.flatten(1)

    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(left[:, None, :] - right[None, :, :]).square().mean(dim=2))

    return kernel(source, source).mean() + kernel(target, target).mean() - 2 * kernel(
        source, target
    ).mean()
