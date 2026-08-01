"""Training and evaluation loops for the Shen et al. (2023) reproduction."""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from battery_benchmark.evaluation.metrics import compute_metrics
from battery_benchmark.reproductions.shen_2023.data import (
    PROFILES,
    ShenCycle,
    selected_windows,
    windows,
)
from battery_benchmark.reproductions.shen_2023.model import TATN, gaussian_mmd


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    if not len(x):
        raise ValueError("Cannot construct a loader from zero TATN windows")
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=min(batch_size, len(x)),
        shuffle=shuffle,
    )


@torch.no_grad()
def predict(
    model: TATN,
    cycles: list[ShenCycle],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, labels = [], []
    for cycle in cycles:
        x, y = windows([cycle])
        for xb, yb in _loader(x, y, 64, False):
            predictions.append(model(xb.to(device)).cpu().numpy())
            labels.append(yb.numpy())
    return np.concatenate(labels).ravel(), np.concatenate(predictions).ravel()


def train_source(
    cycles: dict[str, ShenCycle],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
    checkpoint: str | Path | None = None,
) -> tuple[TATN, list[float]]:
    seed_everything(seed)
    x, y = windows([cycles[p] for p in PROFILES if p in cycles])
    loader = _loader(x, y, batch_size, True)
    model = TATN().to(device)
    optimizer = torch.optim.Adam(
        (*model.source_extractor.parameters(), *model.predictor.parameters()),
        lr=learning_rate,
    )
    history, best_loss, best_state = [], math.inf, None
    model.source_extractor.train()
    model.predictor.train()
    for _ in range(epochs):
        total, samples = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model.forward_source(xb), yb, reduction="sum")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            samples += len(xb)
        epoch_loss = total / max(samples, 1)
        history.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = model.source_state_dict()
    assert best_state is not None
    model.load_source_state_dict(best_state)
    if checkpoint:
        path = Path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "source_state_dict": best_state,
                "history": history,
                "protocol": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "seed": seed,
                },
            },
            path,
        )
    return model, history


def load_source(checkpoint: str | Path, device: torch.device) -> TATN:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = TATN().to(device)
    model.load_source_state_dict(payload["source_state_dict"])
    return model


@torch.no_grad()
def _cycle_prediction(model: TATN, cycle: ShenCycle, device: torch.device) -> np.ndarray:
    model.eval()
    usable = len(cycle.soc) // 90 * 90
    output = np.empty(usable, dtype=np.float32)
    x, y = windows([cycle])
    offset = 0
    for xb, _ in _loader(x, y, 64, False):
        values = model.forward_source(xb.to(device)).cpu().numpy().ravel()
        output[offset : offset + len(values)] = values
        offset += len(values)
    return output


def target_training_windows(
    source_model: TATN,
    target_cycles: dict[str, ShenCycle],
    test_profile: str,
    scenario: str,
    device: torch.device,
    *,
    labeled_profile: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | str | None]]:
    if scenario not in {"semisupervised", "unsupervised"}:
        raise ValueError("scenario must be 'semisupervised' or 'unsupervised'")
    remaining = [p for p in PROFILES if p != test_profile]
    if scenario == "semisupervised":
        labeled_profile = labeled_profile or remaining[0]
        if labeled_profile not in remaining:
            raise ValueError("The labeled target profile must differ from the test profile")
    else:
        labeled_profile = None

    xs, ys, pseudo_windows = [], [], 0
    for profile in remaining:
        cycle = target_cycles[profile]
        if profile == labeled_profile:
            usable = (len(cycle.soc) // 2) // 90 * 90
            x, y = windows(
                [
                    ShenCycle(
                        profile=cycle.profile,
                        temperature=cycle.temperature,
                        features=cycle.features[:usable],
                        soc=cycle.soc[:usable],
                        paths=cycle.paths,
                    )
                ]
            )
        else:
            pseudo = _cycle_prediction(source_model, cycle, device)
            x, y = selected_windows(cycle, pseudo)
            pseudo_windows += len(x)
        if len(x):
            xs.append(x)
            ys.append(y)
    if not xs:
        raise RuntimeError(f"No target windows survived pseudo-label selection for {test_profile}")
    return np.concatenate(xs), np.concatenate(ys), {
        "labeled_profile": labeled_profile,
        "pseudo_windows": pseudo_windows,
        "total_windows": sum(len(x) for x in xs),
    }


def adapt(
    source_model: TATN,
    source_cycles: dict[str, ShenCycle],
    target_cycles: dict[str, ShenCycle],
    *,
    test_profile: str,
    scenario: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lambdas: tuple[float, float, float],
    device: torch.device,
    seed: int,
    labeled_profile: str | None = None,
) -> tuple[TATN, list[dict[str, float]], dict[str, int | str | None]]:
    seed_everything(seed)
    model = deepcopy(source_model).to(device)
    model.initialize_target()
    source_x, source_y = windows([source_cycles[p] for p in PROFILES])
    target_x, target_y, target_info = target_training_windows(
        model,
        target_cycles,
        test_profile,
        scenario,
        device,
        labeled_profile=labeled_profile,
    )
    source_loader = _loader(source_x, source_y, batch_size, True)
    target_loader = _loader(target_x, target_y, batch_size, True)
    extractor_optimizer = torch.optim.Adam(model.target_extractor.parameters(), lr=learning_rate)
    discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=learning_rate)
    estimate_weight, adversarial_weight, mmd_weight = lambdas
    history, best_loss, best_state = [], math.inf, None

    for _ in range(epochs):
        model.target_extractor.train()
        model.discriminator.train()
        model.source_extractor.eval()
        model.predictor.eval()
        totals = {"estimate": 0.0, "adversarial": 0.0, "mmd": 0.0, "total": 0.0}
        batches = 0
        for (source_batch, _), (target_batch, target_label) in zip(source_loader, target_loader):
            source_batch = source_batch.to(device)
            target_batch, target_label = target_batch.to(device), target_label.to(device)
            source_label = torch.ones((len(source_batch), 1), device=device)
            target_domain_label = torch.zeros((len(target_batch), 1), device=device)

            with torch.no_grad():
                source_features = model.source_extractor(source_batch)
            target_features = model.target_extractor(target_batch)
            discriminator_optimizer.zero_grad()
            discriminator_loss = F.binary_cross_entropy_with_logits(
                model.discriminator(source_features), source_label
            ) + F.binary_cross_entropy_with_logits(
                model.discriminator(target_features.detach()), target_domain_label
            )
            discriminator_loss.backward()
            discriminator_optimizer.step()

            extractor_optimizer.zero_grad()
            target_features = model.target_extractor(target_batch)
            estimate_loss = F.mse_loss(
                model.predictor(target_features), target_label, reduction="sum"
            )
            adversarial_loss = F.binary_cross_entropy_with_logits(
                model.discriminator(target_features), torch.ones_like(target_domain_label)
            )
            mmd_loss = gaussian_mmd(source_features, target_features)
            total_loss = (
                estimate_weight * estimate_loss
                + adversarial_weight * adversarial_loss
                + mmd_weight * mmd_loss
            )
            total_loss.backward()
            extractor_optimizer.step()
            for name, value in (
                ("estimate", estimate_loss),
                ("adversarial", adversarial_loss),
                ("mmd", mmd_loss),
                ("total", total_loss),
            ):
                totals[name] += float(value.detach())
            batches += 1
        epoch_metrics = {name: value / max(batches, 1) for name, value in totals.items()}
        history.append(epoch_metrics)
        # Paper does not specify model selection; use label-safe training objective.
        if epoch_metrics["total"] < best_loss:
            best_loss = epoch_metrics["total"]
            best_state = deepcopy(model.target_extractor.state_dict())
    assert best_state is not None
    model.target_extractor.load_state_dict(best_state)
    return model, history, target_info


def evaluate_cycle(model: TATN, cycle: ShenCycle, device: torch.device) -> dict[str, float]:
    truth, estimate = predict(model, [cycle], device)
    return compute_metrics(truth, estimate)


def save_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
