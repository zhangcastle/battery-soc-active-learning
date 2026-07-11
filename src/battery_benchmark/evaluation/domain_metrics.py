"""Domain alignment metrics — PLACEHOLDERS.

These are diagnostic tools to measure how well a transfer method aligns
source and target feature distributions.  They are not SOC accuracy metrics.
"""

from __future__ import annotations

import numpy as np


def mmd_distance_placeholder(
    source_features: np.ndarray,
    target_features: np.ndarray,
) -> float:
    """Placeholder for Maximum Mean Discrepancy (RBF kernel).

    NOT YET IMPLEMENTED.
    """
    raise NotImplementedError(
        "MMD distance computation is not yet implemented. "
        "Will be added alongside the MMD transfer method."
    )


def domain_classifier_accuracy_placeholder(
    source_features: np.ndarray,
    target_features: np.ndarray,
) -> float:
    """Placeholder for domain classifier accuracy.

    A classifier accuracy near 0.5 indicates source and target are well-aligned.
    NOT YET IMPLEMENTED.
    """
    raise NotImplementedError(
        "Domain classifier accuracy is not yet implemented."
    )
