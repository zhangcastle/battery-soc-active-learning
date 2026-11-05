"""Pool-based target-window selection for Vidal Case 2 fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset

from battery_benchmark.reproductions.vidal_2019.model import VidalLSTM


@dataclass(frozen=True)
class SelectionResult:
    indices: np.ndarray
    uncertainty: np.ndarray


@torch.no_grad()
def model_outputs(
    model: VidalLSTM,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-window sequence predictions and mask-aware hidden embeddings."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    predictions, embeddings = [], []
    model.eval()
    model.to(device)
    for x, _, mask, _ in loader:
        x, mask = x.to(device), mask.to(device)
        hidden, _ = model.lstm(x)
        pred = model.fc(hidden).squeeze(-1)
        if model.output_clip:
            pred = pred.clamp(0.0, 1.0)
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        predictions.append(pred.cpu().numpy())
        embeddings.append(pooled.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(embeddings)


def committee_uncertainty(predictions: list[np.ndarray]) -> np.ndarray:
    """Mean timestep-wise standard deviation for each target window."""
    if len(predictions) < 2:
        raise ValueError("QBC and regression_clue require at least two committee models")
    shapes = {prediction.shape for prediction in predictions}
    if len(shapes) != 1:
        raise ValueError(f"Committee prediction shapes differ: {sorted(shapes)}")
    return np.std(np.stack(predictions), axis=0).mean(axis=1)


def _weighted_cluster_indices(
    embeddings: np.ndarray,
    uncertainty: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    if len(embeddings) != len(uncertainty):
        raise ValueError("Embeddings and uncertainty must have the same length")
    weights = np.asarray(uncertainty, dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Uncertainty must contain finite non-negative values")
    if not weights.any():
        weights = np.ones_like(weights)
    model = KMeans(n_clusters=count, random_state=seed, n_init=10)
    labels = model.fit_predict(embeddings, sample_weight=weights)
    selected: list[int] = []
    for cluster, center in enumerate(model.cluster_centers_):
        members = np.flatnonzero(labels == cluster)
        if not len(members):
            continue
        distances = np.linalg.norm(embeddings[members] - center, axis=1)
        selected.append(int(members[np.argmin(distances)]))
    if len(selected) != count:
        raise RuntimeError(f"Expected {count} selected windows, got {len(selected)}")
    return np.asarray(sorted(selected), dtype=int)


def select_indices(
    strategy: str,
    count: int,
    pool_size: int,
    seed: int,
    *,
    uncertainty: np.ndarray | None = None,
    embeddings: np.ndarray | None = None,
) -> SelectionResult:
    """Select target-window indices without reading their SOC labels."""
    if not 1 <= count <= pool_size:
        raise ValueError(f"Selection count must be in [1, {pool_size}], got {count}")
    if strategy == "all":
        indices = np.arange(pool_size, dtype=int)
        scores = np.zeros(pool_size, dtype=float)
    elif strategy == "random":
        indices = np.sort(np.random.default_rng(seed).choice(pool_size, count, replace=False))
        scores = np.zeros(pool_size, dtype=float)
    elif strategy == "qbc":
        if uncertainty is None or len(uncertainty) != pool_size:
            raise ValueError("QBC requires one uncertainty value per pool item")
        indices = np.sort(np.argsort(uncertainty, kind="stable")[-count:])
        scores = np.asarray(uncertainty, dtype=float)
    elif strategy == "regression_clue":
        if uncertainty is None or embeddings is None:
            raise ValueError("regression_clue requires uncertainty and embeddings")
        indices = _weighted_cluster_indices(embeddings, uncertainty, count, seed)
        scores = np.asarray(uncertainty, dtype=float)
    else:
        raise ValueError(f"Unsupported selection strategy: {strategy}")
    return SelectionResult(indices=indices, uncertainty=scores)
