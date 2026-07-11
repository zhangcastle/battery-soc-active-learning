"""Custom collate functions for variable-length sequences."""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


def pad_collate(batch):
    """Collate function that pads variable-length sequences.

    Expects each item to be ``(x, y)`` where ``x`` has shape ``(T, F)``
    and ``y`` has shape ``(T,)``.  Returns:
    - ``x_padded``: (B, T_max, F)
    - ``y_padded``: (B, T_max)
    - ``lengths``: (B,) original sequence lengths
    """
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs])
    x_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=float("nan"))
    return x_padded, y_padded, lengths


def default_collate(batch):
    """Standard collate for fixed-length sequences (no padding needed)."""
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)
