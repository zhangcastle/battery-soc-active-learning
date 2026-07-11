"""Early stopping callback."""

from __future__ import annotations


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience:
        Number of epochs with no improvement before stopping.
    min_delta:
        Minimum change to qualify as an improvement.
    mode:
        ``"min"`` for loss-like metrics, ``"max"`` for accuracy-like.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-5,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: float | None = None
        self._wait: int = 0
        self.stopped_epoch: int = 0

    def step(self, metric: float, epoch: int) -> bool:
        """Return True if training should stop."""
        improved = self._is_improved(metric)
        if improved:
            self._best = metric
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self.stopped_epoch = epoch
                return True
        return False

    def _is_improved(self, metric: float) -> bool:
        if self._best is None:
            return True
        if self.mode == "min":
            return metric < self._best - self.min_delta
        return metric > self._best + self.min_delta

    @property
    def best(self) -> float | None:
        return self._best
