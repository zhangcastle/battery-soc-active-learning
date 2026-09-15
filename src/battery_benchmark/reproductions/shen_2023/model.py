"""Architectures for the Shen et al. (2023) reproduction.

Two transfer mechanisms are implemented, matching two different rows the
paper itself compares in Table VI:

* ``MEDNet`` -- the paper's actual proposed method (Algorithm 1 / Eq. 2-19):
  a shared 2D-CNN + BiLSTM feature extractor feeding **two** differently
  initialised SOC estimator heads. At transfer time both heads are frozen;
  only a copy of the feature extractor is updated, driven purely by
  pseudo-labels (the frozen source heads' averaged output) plus a
  discrepancy term between the two heads. No source data, no target labels,
  and no domain discriminator or MMD term are used here -- this is the
  "source-free" / MED method itself.
* ``DANNBaselineNet`` -- a domain-adversarial-network + MMD baseline
  (single estimator head, a domain discriminator, Gaussian-kernel MMD
  alignment). The paper reports "adversarial" and "MMD" rows in Table VI as
  comparison baselines, not as the proposed method; this class exists only
  to reproduce those comparison rows and must never be reported as "the
  Shen et al. method" on its own.

Historical note: earlier code in this module used ``TATN``/``DomainDiscriminator``
as if the DANN+MMD mechanism *were* the paper's proposed method. It is not --
Algorithm 1 has no discriminator and no MMD term at all. That mechanism is
kept here, renamed, purely as the explicit "DANN+MMD baseline" comparison
point it actually is in the paper.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

WINDOW = 90
FEATURES = 3
FILTERS = 256
HIDDEN = 64

# Xavier-init gains for the two MED estimator heads. Differing gains are the
# paper's mechanism for making the two heads diverge enough to produce a
# useful discrepancy signal (Sec. III), rather than any extra loss term.
HEAD1_GAIN = 0.25
HEAD2_GAIN = 0.30


class FeatureExtractor(nn.Module):
    """Shared 2D-CNN + BiLSTM feature extractor (Fig. 1)."""

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
            raise ValueError(f"Expected (batch, {WINDOW}, {FEATURES}), got {tuple(x.shape)}")
        x = self.conv(x.transpose(1, 2).unsqueeze(1))
        x = x.reshape(x.shape[0], FILTERS * FEATURES, WINDOW).transpose(1, 2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x


class SOCHead(nn.Module):
    """One SOC estimator head, with a Xavier-init gain set at construction."""

    def __init__(self, gain: float) -> None:
        super().__init__()
        self.gain = gain
        self.layers = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN * 2),
            nn.ReLU(),
            nn.Linear(HIDDEN * 2, 1),
        )
        self.init_weight()

    def init_weight(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=self.gain)
                nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


def discrepancy(pred1: torch.Tensor, pred2: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    """Absolute discrepancy between the two heads' predictions (Eq. 9-10, 14).

    ``reduction="mean"`` matches the paper's Eq. (12)-(14). The collaborator's
    delivered reference code defaults to a *sum*-based loss (only switching to
    mean when a ``--paper_mean_loss`` flag is passed); this reproduction
    always uses the mean so results stay on the same scale the paper reports
    without needing an extra flag to remember.
    """

    diff = torch.abs(pred1 - pred2)
    return diff.mean() if reduction == "mean" else diff.sum()


class MEDNet(nn.Module):
    """Shen et al. (2023) proposed method: shared extractor, two frozen heads.

    ``source_extractor``/``head1``/``head2`` are trained jointly and
    supervised during pretraining, then frozen for the entire transfer
    stage (Algorithm 1, lines "freeze theta_E1, theta_E2"). ``target_extractor``
    starts as a copy of the source extractor and is the *only* thing updated
    during transfer, driven by pseudo-labels the frozen heads produce.
    """

    def __init__(self) -> None:
        super().__init__()
        self.source_extractor = FeatureExtractor()
        self.target_extractor = FeatureExtractor()
        self.head1 = SOCHead(HEAD1_GAIN)
        self.head2 = SOCHead(HEAD2_GAIN)

    def forward_source(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.source_extractor(x)
        return self.head1(features), self.head2(features)

    def forward_target(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.target_extractor(x)
        return self.head1(features), self.head2(features)

    def forward(self, x: torch.Tensor, branch: str = "target") -> torch.Tensor:
        """Average-of-heads SOC estimate (what gets reported/evaluated)."""

        pred1, pred2 = self.forward_source(x) if branch == "source" else self.forward_target(x)
        return (pred1 + pred2) / 2.0

    def estimate_source(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen-source SOC estimate averaged over both heads (Eq. 3) --
        this is what pseudo-labels are built from."""

        return self.forward(x, branch="source")

    def initialize_target(self) -> None:
        """Copy source -> target extractor and freeze both heads + the source branch.

        Only ``target_extractor`` keeps requires_grad=True after this call --
        this is what makes the transfer stage source-free: the heads (theta_E1,
        theta_E2) never move again, so pseudo-labels generated at the start of
        transfer stay a fixed, well-defined target for the whole run.
        """

        self.target_extractor.load_state_dict(self.source_extractor.state_dict())
        for parameter in (
            *self.source_extractor.parameters(),
            *self.head1.parameters(),
            *self.head2.parameters(),
        ):
            parameter.requires_grad_(False)
        self.source_extractor.eval()
        self.head1.eval()
        self.head2.eval()

    def source_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "extractor": deepcopy(self.source_extractor.state_dict()),
            "head1": deepcopy(self.head1.state_dict()),
            "head2": deepcopy(self.head2.state_dict()),
        }

    def load_source_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        self.source_extractor.load_state_dict(state["extractor"])
        self.head1.load_state_dict(state["head1"])
        self.head2.load_state_dict(state["head2"])
        self.initialize_target()


# ---------------------------------------------------------------------------
# DANN + MMD baseline (Table VI "adversarial" / "MMD" comparison rows only).
# ---------------------------------------------------------------------------


class SOCPredictor(nn.Module):
    """Single-head SOC predictor used by the DANN+MMD baseline."""

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
    """Binary domain discriminator used only by the DANN+MMD baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(WINDOW * HIDDEN * 2, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class DANNBaselineNet(nn.Module):
    """Domain-adversarial + MMD baseline -- a Table VI comparison row, not the
    paper's proposed method. Kept separate from ``MEDNet`` so the two are
    never conflated in results tables again."""

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

    def estimate_source(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_source(x)

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
    """Gaussian-kernel MMD used by the DANN+MMD baseline only."""

    source = source.flatten(1)
    target = target.flatten(1)

    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(left[:, None, :] - right[None, :, :]).square().mean(dim=2))

    return kernel(source, source).mean() + kernel(target, target).mean() - 2 * kernel(
        source, target
    ).mean()


# Backwards-compatible alias: earlier code called the DANN+MMD class "TATN"
# as if it were the paper's proposed method. Keep the name importable so any
# stale references fail loudly at the call site instead of silently
# re-training a baseline under the paper's headline name.
TATN = DANNBaselineNet
