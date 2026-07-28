"""Paper-faithful core of Liu et al.'s DARM SOC estimator.

The paper specifies the architecture and losses but does not publish code or
MK-MMD bandwidths. This reconstruction uses the table-I tensor shapes and a
median-distance multi-kernel RBF, keeping that single ambiguity explicit.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from battery_benchmark.models.base import BaseSOCModel

TARGET_BN_MODES = ("legacy", "frozen_stats", "fully_frozen")
ADVERSARIAL_UPDATES = ("grl_joint", "alternating_flipped", "eq12_joint")


class DepthwiseSeparable1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, final_relu: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        ]
        if final_relu:
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DWSResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            DepthwiseSeparable1d(in_channels, out_channels),
            DepthwiseSeparable1d(out_channels, out_channels, final_relu=False),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(x) + self.shortcut(x))


class FeatureEncoder(nn.Module):
    """Table-I encoder: (B, 3, 40) -> (B, 64, 5)."""

    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(DepthwiseSeparable1d(3, 8), DWSResBlock(8, 8))
        self.stage2 = nn.Sequential(DWSResBlock(8, 16), DWSResBlock(16, 16))
        self.stage3 = nn.Sequential(DWSResBlock(16, 32), DWSResBlock(32, 32))
        self.stage4 = nn.Sequential(DWSResBlock(32, 64), DWSResBlock(64, 64))
        self.pool = nn.MaxPool1d(2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        x1 = self.stage1(x.transpose(1, 2))
        x2 = self.stage2(self.pool(x1))
        x3 = self.stage3(self.pool(x2))
        latent = self.stage4(self.pool(x3))
        return latent, (x1, x2, x3)


class ReconstructionDecoder(nn.Module):
    """Symmetric table-I decoder with U-Net skip connections."""

    def __init__(self):
        super().__init__()
        self.stage3 = nn.Sequential(DWSResBlock(96, 32), DWSResBlock(32, 32))
        self.stage2 = nn.Sequential(DWSResBlock(48, 16), DWSResBlock(16, 16))
        self.stage1 = nn.Sequential(DWSResBlock(24, 8), DWSResBlock(8, 8))
        self.output = nn.Conv1d(8, 3, 1)

    @staticmethod
    def _up(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(
        self,
        latent: torch.Tensor,
        skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x1, x2, x3 = skips
        x = self.stage3(torch.cat((self._up(latent), x3), dim=1))
        x = self.stage2(torch.cat((self._up(x), x2), dim=1))
        x = self.stage1(torch.cat((self._up(x), x1), dim=1))
        return self.output(x).transpose(1, 2)


class AttentionHead(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.bilstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(128, 1)
        self.output = nn.Linear(128, output_size)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        sequence = latent.transpose(1, 2)
        states, _ = self.bilstm(sequence)
        weights = torch.softmax(self.attention(states), dim=1)
        return self.output((weights * states).sum(dim=1))


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


def gradient_reverse(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(x, strength)


def _domain_labels(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            torch.zeros(len(source), dtype=torch.long, device=source.device),
            torch.ones(len(target), dtype=torch.long, device=target.device),
        )
    )


def domain_statistics(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> dict[str, float]:
    source_prediction = source_logits.argmax(dim=1)
    target_prediction = target_logits.argmax(dim=1)
    source_accuracy = (source_prediction == 0).float().mean()
    target_accuracy = (target_prediction == 1).float().mean()
    overall_accuracy = torch.cat((source_prediction == 0, target_prediction == 1)).float().mean()
    return {
        "source_domain_accuracy": float(source_accuracy.detach()),
        "target_domain_accuracy": float(target_accuracy.detach()),
        "overall_domain_accuracy": float(overall_accuracy.detach()),
        "mean_p_source_given_source": float(source_logits.softmax(dim=1)[:, 0].mean().detach()),
        "mean_p_source_given_target": float(target_logits.softmax(dim=1)[:, 0].mean().detach()),
    }


def flipped_target_adversarial_loss(
    discriminator: nn.Module,
    target_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make target features look like source without updating the discriminator."""
    requires_grad = [parameter.requires_grad for parameter in discriminator.parameters()]
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    logits = discriminator(target_features)
    loss = F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))
    for parameter, enabled in zip(discriminator.parameters(), requires_grad):
        parameter.requires_grad_(enabled)
    return loss, logits


def target_bn_buffer_snapshot(model: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            "running_mean": module.running_mean.detach().clone(),
            "running_var": module.running_var.detach().clone(),
            "num_batches_tracked": module.num_batches_tracked.detach().clone(),
        }
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }


def target_bn_buffer_drift(
    before: dict[str, dict[str, torch.Tensor]],
    after: dict[str, dict[str, torch.Tensor]],
) -> dict[str, object]:
    layers = {}
    for name, initial in before.items():
        final = after[name]
        layers[name] = {
            "running_mean_max_abs_change": float(
                (final["running_mean"] - initial["running_mean"]).abs().max()
            ),
            "running_var_max_abs_change": float(
                (final["running_var"] - initial["running_var"]).abs().max()
            ),
            "num_batches_tracked_change": int(
                final["num_batches_tracked"] - initial["num_batches_tracked"]
            ),
        }
    return {
        "layers": layers,
        "max_running_mean_abs_change": max(
            (row["running_mean_max_abs_change"] for row in layers.values()), default=0.0
        ),
        "max_running_var_abs_change": max(
            (row["running_var_max_abs_change"] for row in layers.values()), default=0.0
        ),
        "max_num_batches_tracked_change": max(
            (abs(row["num_batches_tracked_change"]) for row in layers.values()), default=0
        ),
    }


def parameter_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def parameter_drift(
    before: dict[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, object]:
    changes = {
        name: float((parameter.detach() - before[name]).abs().max())
        for name, parameter in model.named_parameters()
    }
    return {
        "all_unchanged": not any(changes.values()),
        "changed_parameter_count": sum(value != 0.0 for value in changes.values()),
        "max_abs_change": max(changes.values(), default=0.0),
    }


def audit_target_bn_forward(
    model: "DARM",
    batches: Iterable[torch.Tensor],
    mode: str,
) -> dict[str, object]:
    """Run target data once without autograd or parameter updates."""
    model.set_target_bn_mode(mode)
    parameters_before = parameter_snapshot(model)
    buffers_before = target_bn_buffer_snapshot(model.target_encoder)
    model.train()
    with torch.no_grad():
        for batch in batches:
            model.encode_target(batch)
    return {
        "target_bn_mode": mode,
        "bn_buffer_drift": target_bn_buffer_drift(
            buffers_before, target_bn_buffer_snapshot(model.target_encoder)
        ),
        "parameter_drift": parameter_drift(parameters_before, model),
    }


def multi_kernel_mmd(
    source: torch.Tensor,
    target: torch.Tensor,
    multipliers: Iterable[float] = (0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    """Biased MK-MMD from equation (8), using median-distance bandwidth."""
    source = source.flatten(1)
    target = target.flatten(1)
    joined = torch.cat((source, target))
    distances = torch.cdist(joined, joined).square()
    positive = distances.detach()[distances.detach() > 0]
    base = positive.median() if positive.numel() else distances.new_tensor(1.0)

    kernel = torch.zeros_like(distances)
    for multiplier in multipliers:
        kernel += torch.exp(-distances / (2 * base * multiplier**2 + 1e-12))

    n_source = len(source)
    return (
        kernel[:n_source, :n_source].mean()
        + kernel[n_source:, n_source:].mean()
        - 2 * kernel[:n_source, n_source:].mean()
    )


class DARM(BaseSOCModel):
    """DARM with frozen source encoder/predictor during target adaptation."""

    def __init__(self):
        super().__init__()
        self.source_encoder = FeatureEncoder()
        self.target_encoder = deepcopy(self.source_encoder)
        self.decoder = ReconstructionDecoder()
        self.predictor = AttentionHead(1)
        self.discriminator = AttentionHead(2)
        self.target_bn_mode = "legacy"

    def encode_source(self, x: torch.Tensor) -> torch.Tensor:
        return self.source_encoder(x)[0]

    def encode_target(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        return self.target_encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, _ = self.target_encoder(x)
        return self.predictor(latent).squeeze(-1)

    def forward_source(self, x: torch.Tensor) -> torch.Tensor:
        latent, _ = self.source_encoder(x)
        return self.predictor(latent).squeeze(-1)

    def initialize_target(self) -> None:
        self.target_encoder.load_state_dict(self.source_encoder.state_dict())
        for parameter in (*self.source_encoder.parameters(), *self.predictor.parameters()):
            parameter.requires_grad_(False)
        self.source_encoder.eval()
        self.predictor.eval()

    def set_target_bn_mode(self, mode: str) -> None:
        if mode not in TARGET_BN_MODES:
            raise ValueError(f"Unknown target BN mode: {mode}")
        self.target_bn_mode = mode
        for module in self.target_encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                if module.affine:
                    module.weight.requires_grad_(mode != "fully_frozen")
                    module.bias.requires_grad_(mode != "fully_frozen")
                if mode != "legacy":
                    module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        frozen = not any(parameter.requires_grad for parameter in self.source_encoder.parameters())
        if mode and frozen:
            self.source_encoder.eval()
            self.predictor.eval()
        if mode and self.target_bn_mode != "legacy":
            for module in self.target_encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def source_parameters(self):
        return list(self.source_encoder.parameters()) + list(self.predictor.parameters())

    def adaptation_parameters(self):
        return [
            parameter
            for parameter in (
                *self.target_encoder.parameters(),
                *self.decoder.parameters(),
                *self.discriminator.parameters(),
            )
            if parameter.requires_grad
        ]

    def target_adaptation_parameters(self):
        return [
            parameter
            for parameter in (*self.target_encoder.parameters(), *self.decoder.parameters())
            if parameter.requires_grad
        ]

    def discriminator_step_loss(
        self,
        source_x: torch.Tensor,
        target_x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        with torch.no_grad():
            source_features = self.encode_source(source_x)
            target_features = self.encode_target(target_x)[0]
        source_logits = self.discriminator(source_features.detach())
        target_logits = self.discriminator(target_features.detach())
        logits = torch.cat((source_logits, target_logits))
        loss = F.cross_entropy(logits, _domain_labels(source_logits, target_logits))
        return loss, {
            "discriminator": float(loss.detach()),
            **domain_statistics(source_logits, target_logits),
        }

    def alternating_target_loss(
        self,
        source_x: torch.Tensor,
        target_x: torch.Tensor,
        *,
        adversarial_weight: float = 1.0,
        alpha: float = 10.0,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            source_features = self.encode_source(source_x)
        target_features, skips = self.encode_target(target_x)
        adversarial, _ = flipped_target_adversarial_loss(self.discriminator, target_features)
        reconstruction = F.mse_loss(self.decoder(target_features, skips), target_x)
        mmd = multi_kernel_mmd(source_features, target_features)
        total = adversarial_weight * adversarial + alpha * reconstruction + beta * mmd
        return total, {
            "adversarial": adversarial,
            "reconstruction": reconstruction,
            "mmd": mmd,
            "total": total,
        }

    def adaptation_loss(
        self,
        source_x: torch.Tensor,
        target_x: torch.Tensor,
        *,
        adversarial_weight: float = 1.0,
        grl_strength: float = 1.0,
        alpha: float = 10.0,
        beta: float = 1.0,
        adversarial_update: str = "grl_joint",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
        if adversarial_update not in {"grl_joint", "eq12_joint"}:
            raise ValueError(
                "adaptation_loss supports grl_joint and eq12_joint; "
                "use alternating_target_loss for alternating_flipped"
            )
        with torch.no_grad():
            source_features = self.encode_source(source_x)
        target_features, skips = self.encode_target(target_x)

        source_logits = self.discriminator(source_features.detach())
        discriminator_target_features = (
            gradient_reverse(target_features, grl_strength)
            if adversarial_update == "grl_joint"
            else target_features
        )
        target_logits = self.discriminator(discriminator_target_features)
        domain_logits = torch.cat((source_logits, target_logits))
        adversarial = F.cross_entropy(domain_logits, _domain_labels(source_logits, target_logits))
        reconstruction = F.mse_loss(self.decoder(target_features, skips), target_x)
        mmd = multi_kernel_mmd(source_features, target_features)
        total = adversarial_weight * adversarial + alpha * reconstruction + beta * mmd
        losses = {
            "adversarial": adversarial,
            "reconstruction": reconstruction,
            "mmd": mmd,
            "total": total,
        }
        return (
            total,
            losses,
            {
                "discriminator": float(adversarial.detach()),
                **domain_statistics(source_logits, target_logits),
            },
        )

    def get_model_metadata(self) -> dict[str, object]:
        return {
            "model_class": "DARM",
            "paper": "Liu et al. 2025, doi:10.1109/TTE.2024.3483973",
            "input_shape": [40, 3],
            "n_parameters": self.count_parameters(),
            "mk_mmd_bandwidth": "median x [0.5, 1, 2, 4] (paper unspecified)",
            "target_bn_mode": self.target_bn_mode,
        }
