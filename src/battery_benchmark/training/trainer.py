"""Main Trainer class coordinating training loops, checkpointing, and callbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from battery_benchmark.training.callbacks import Callback, LoggingCallback
from battery_benchmark.training.checkpointing import save_checkpoint
from battery_benchmark.training.early_stopping import EarlyStopping
from battery_benchmark.training.loops import eval_epoch, train_epoch
from battery_benchmark.utils.logging import get_logger

logger = get_logger("training.trainer")


class Trainer:
    """Manages the training loop with checkpointing and optional early stopping.

    Parameters
    ----------
    optimizer:
        Constructed before calling ``fit``.
    epochs:
        Maximum number of epochs.
    device:
        PyTorch device.
    checkpoint_dir:
        Directory for checkpoint files.
    scheduler:
        Optional LR scheduler.
    normalizer:
        Optional normaliser for checkpoint state.
    config:
        Resolved experiment config (saved in checkpoints).
    callbacks:
        List of Callback instances.
    patience:
        Early stopping patience (None = disabled).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: torch.device,
        checkpoint_dir: str | Path,
        scheduler: Any | None = None,
        normalizer: Any | None = None,
        config: dict[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
        patience: int | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.scheduler = scheduler
        self.normalizer = normalizer
        self.config = config or {}
        self.callbacks: list[Callback] = callbacks or [LoggingCallback()]
        self.patience = patience

        self._criterion = nn.MSELoss()
        self.best_val_loss = float("inf")
        self.start_epoch = 0
        self.history: list[dict[str, float]] = []

    def resume_from(self, checkpoint_path: str | Path) -> None:
        """Load training state from a checkpoint file.

        After calling this, ``fit`` will resume from the saved epoch.
        """
        from battery_benchmark.training.checkpointing import load_checkpoint
        ckpt = load_checkpoint(
            checkpoint_path,
            model=None,  # model not loaded here; caller handles model
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            normalizer=self.normalizer,
            device=self.device,
        )
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_val_loss = ckpt.get("best_metric", float("inf"))
        logger.info(
            "Resumed from checkpoint '%s' at epoch %d (best_val_loss=%.6f)",
            checkpoint_path, self.start_epoch - 1, self.best_val_loss,
        )

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> list[dict[str, float]]:
        """Run the full training loop.

        Parameters
        ----------
        model:
            The SOC model to train.
        train_loader, val_loader:
            DataLoaders for train and validation splits.

        Returns
        -------
        list[dict[str, float]]
            Per-epoch metrics history.
        """
        model = model.to(self.device)
        early_stop = EarlyStopping(patience=self.patience) if self.patience else None

        for cb in self.callbacks:
            cb.on_train_start({})

        for epoch in range(self.start_epoch, self.epochs):
            logs: dict[str, float] = {"epoch": float(epoch)}

            train_loss = train_epoch(
                model, train_loader, self.optimizer, self._criterion, self.device
            )
            logs["train_loss"] = train_loss

            if len(val_loader.dataset) > 0:
                val_loss = eval_epoch(model, val_loader, self._criterion, self.device)
                logs["val_loss"] = val_loss
            else:
                val_loss = train_loss
                logs["val_loss"] = val_loss

            # Scheduler step
            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    if hasattr(self.scheduler, "metrics"):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

            for cb in self.callbacks:
                cb.on_epoch_end(epoch, logs)

            self.history.append(logs)

            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(
                    path=self.checkpoint_dir / "best.pt",
                    model=model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_metric=self.best_val_loss,
                    config=self.config,
                    scheduler=self.scheduler,
                    normalizer=self.normalizer,
                )

            # Save latest checkpoint (safe to overwrite; used for resume)
            save_checkpoint(
                path=self.checkpoint_dir / "last.pt",
                model=model,
                optimizer=self.optimizer,
                epoch=epoch,
                best_metric=self.best_val_loss,
                config=self.config,
                scheduler=self.scheduler,
                normalizer=self.normalizer,
            )

            # Early stopping
            if early_stop is not None and early_stop.step(val_loss, epoch):
                logger.info("Early stopping at epoch %d", epoch)
                break

        for cb in self.callbacks:
            cb.on_train_end({"history": self.history})

        return self.history
