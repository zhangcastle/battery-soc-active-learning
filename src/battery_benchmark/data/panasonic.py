"""Panasonic NCR18650PF dataset adapter.

Source: McMaster University Battery Dataset
URL: https://data.mendeley.com/datasets/wykht8y7tg/1

Status: PLACEHOLDER — raises NotImplementedError.
See notebooks/02_prepare_panasonic.ipynb for preprocessing instructions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from battery_benchmark.data.base_dataset import BaseBatteryDataset
from battery_benchmark.data.registry import register_dataset


@register_dataset("panasonic_18650pf")
class PanasonicDataset(BaseBatteryDataset):
    """Adapter for the Panasonic NCR18650PF drive-cycle dataset."""

    def discover_files(self, raw_dir: str | Path) -> list[Path]:
        raise NotImplementedError(
            "PanasonicDataset is not yet implemented. "
            "See notebooks/02_prepare_panasonic.ipynb."
        )

    def build_manifest(self, raw_dir: str | Path) -> pd.DataFrame:
        raise NotImplementedError(
            "PanasonicDataset is not yet implemented."
        )

    def load_cycle(self, file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError("PanasonicDataset is not yet implemented.")

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("PanasonicDataset is not yet implemented.")

    def validate_cycle(self, df: pd.DataFrame) -> bool:
        raise NotImplementedError("PanasonicDataset is not yet implemented.")

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "panasonic_18650pf",
            "status": "placeholder",
            "nominal_capacity_ah": 2.9,
        }
