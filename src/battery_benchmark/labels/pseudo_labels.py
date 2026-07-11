"""Pseudo-label generation for unlabelled target-domain data.

WARNING: Pseudo-labels are NOT treated as ground truth.
They are model predictions used as soft training signals during
self-training iterations.  Evaluation always uses the hidden
proxy reference (or measured labels when available).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from battery_benchmark.data.schema import COLUMNS, LabelSource


def generate_pseudo_labels(
    model: Any,
    cycles: list[pd.DataFrame],
    feature_cols: list[str],
    normalizer: Any | None = None,
    device: str | torch.device = "cpu",
    confidence_threshold: float | None = None,
) -> list[pd.DataFrame]:
    """Generate pseudo-SOC labels for unlabelled target cycles.

    Parameters
    ----------
    model:
        A trained SOC model with a ``predict_soc`` method.
    cycles:
        Target-domain cycles (SOC column may be NaN or missing).
    feature_cols:
        Feature columns used for prediction.
    normalizer:
        Normaliser fitted on source training data.
    device:
        Device to run inference on.
    confidence_threshold:
        If set, only keep pseudo-labels where prediction std < threshold.
        Requires the model to return uncertainty estimates.

    Returns
    -------
    list[pd.DataFrame]
        Cycles with filled SOC column and LABEL_SOURCE = "pseudo_label".
    """
    raise NotImplementedError(
        "Pseudo-label generation is not yet implemented. "
        "This function will be populated when self-training is introduced."
    )
