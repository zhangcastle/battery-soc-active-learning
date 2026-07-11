"""Deterministic seed utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility.

    Sets seeds for Python, NumPy, PyTorch (CPU and CUDA).
    Also enables deterministic CUDA operations where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms (may be slower but fully reproducible)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Allow non-deterministic if no CUDA to avoid slowing CPU training
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
