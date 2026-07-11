"""PyTorch Dataset wrapper for pre-built sequence arrays."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class CycleDataset(Dataset):
    """Wraps (X, y) sequence arrays as a PyTorch Dataset.

    Parameters
    ----------
    sequences:
        Feature array of shape ``(N, seq_len, n_features)``.
    targets:
        SOC array of shape ``(N, seq_len)`` or ``(N,)`` for scalar targets.
    metadata:
        Optional list of dicts with per-sequence metadata
        (cell_id, cycle_id, start_idx, label_available, etc.).
    """

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray | None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        if targets is not None:
            self.targets: torch.Tensor | None = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = None
        self.metadata = metadata or [{}] * len(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.sequences[idx]
        y = self.targets[idx] if self.targets is not None else torch.tensor(float("nan"))
        return x, y

    @property
    def n_features(self) -> int:
        return self.sequences.shape[-1]

    @property
    def seq_len(self) -> int:
        return self.sequences.shape[1]

    def has_labels(self) -> bool:
        """Return True if target labels are available and non-NaN."""
        if self.targets is None:
            return False
        return not torch.isnan(self.targets).all().item()
