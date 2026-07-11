"""Plotting utilities for SOC estimation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for scripts and Colab
import matplotlib.pyplot as plt
import numpy as np


def plot_soc_trajectory(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "SOC Trajectory",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot predicted vs. ground-truth SOC over time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.arange(len(y_true))
    ax.plot(t, y_true, label="Ground truth", linewidth=1.5)
    ax.plot(t, y_pred, label="Predicted", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("SOC")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Prediction Error Distribution",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Histogram of prediction errors."""
    errors = (y_pred - y_true)[~(np.isnan(y_true) | np.isnan(y_pred))]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errors, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("SOC error (predicted − true)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_history(
    history: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot train/val loss curves."""
    epochs = [h["epoch"] for h in history]
    train_losses = [h.get("train_loss", float("nan")) for h in history]
    val_losses = [h.get("val_loss", float("nan")) for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="Train loss")
    ax.plot(epochs, val_losses, label="Val loss", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training History")
    ax.legend()
    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
