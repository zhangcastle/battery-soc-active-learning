"""Reusable frozen source state for Liu 2025 downstream targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from battery_benchmark.preprocessing.normalization import MinMaxNormalizer
from battery_benchmark.reproductions.liu_2025.darm import DARM
from battery_benchmark.utils.hashing import sha256_file


def save_source_artifact(
    path: str | Path,
    model: DARM,
    normalizer: MinMaxNormalizer,
    metadata: dict[str, Any],
) -> str:
    """Save only the source encoder, predictor, scaler, and provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source_encoder_state_dict": model.source_encoder.state_dict(),
            "predictor_state_dict": model.predictor.state_dict(),
            "normalizer_state_dict": normalizer.state_dict(),
            "metadata": metadata,
        },
        path,
    )
    return sha256_file(path)


def load_source_artifact(
    path: str | Path,
    model: DARM,
    normalizer: MinMaxNormalizer,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Restore a frozen source and initialize a fresh target encoder from it."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    required = {
        "source_encoder_state_dict",
        "predictor_state_dict",
        "normalizer_state_dict",
        "metadata",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Source artifact missing fields: {sorted(missing)}")
    model.source_encoder.load_state_dict(payload["source_encoder_state_dict"])
    model.predictor.load_state_dict(payload["predictor_state_dict"])
    normalizer.load_state_dict(payload["normalizer_state_dict"])
    model.initialize_target()
    return dict(payload["metadata"])
