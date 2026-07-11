"""Training callbacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Callback(ABC):
    """Base training callback."""

    def on_epoch_start(self, epoch: int, logs: dict[str, Any]) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        pass

    def on_train_start(self, logs: dict[str, Any]) -> None:
        pass

    def on_train_end(self, logs: dict[str, Any]) -> None:
        pass


class LoggingCallback(Callback):
    """Print epoch metrics to stdout."""

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        parts = [f"Epoch {epoch:4d}"]
        for k, v in logs.items():
            if isinstance(v, float):
                parts.append(f"{k}: {v:.6f}")
            else:
                parts.append(f"{k}: {v}")
        print("  ".join(parts))
