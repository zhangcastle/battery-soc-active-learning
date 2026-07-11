"""Result schema and JSON serialisation for experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentResult:
    """Container for all outputs of one completed experiment."""

    experiment_name: str
    source_domain: str
    target_domain: str
    transfer_method: str
    model_name: str
    random_seed: int
    epochs_run: int

    # Supervised metrics (NaN when labels unavailable)
    train_mae: float = float("nan")
    train_rmse: float = float("nan")
    val_mae: float = float("nan")
    val_rmse: float = float("nan")
    test_mae: float = float("nan")
    test_rmse: float = float("nan")
    test_max_error: float = float("nan")
    test_bias: float = float("nan")
    test_r2: float = float("nan")

    n_parameters: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert NaN to None for JSON compatibility
        for k, v in d.items():
            if isinstance(v, float) and v != v:  # nan check
                d[k] = None
        return d

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentResult":
        d = dict(d)
        extra = d.pop("extra", {})
        # Replace None back to NaN
        for k, v in d.items():
            if v is None:
                d[k] = float("nan")
        return cls(**d, extra=extra)
