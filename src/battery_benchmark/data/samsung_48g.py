"""Samsung 48G dataset adapter.

Status: PLACEHOLDER — raises NotImplementedError.
Dataset availability and file structure are unconfirmed.
See docs/known_uncertainties.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from battery_benchmark.data.base_dataset import BaseBatteryDataset
from battery_benchmark.data.registry import register_dataset


@register_dataset("samsung_48g")
class Samsung48GDataset(BaseBatteryDataset):
    def discover_files(self, raw_dir: str | Path) -> list[Path]:
        raise NotImplementedError("Samsung48GDataset is not yet implemented.")

    def build_manifest(self, raw_dir: str | Path) -> pd.DataFrame:
        raise NotImplementedError("Samsung48GDataset is not yet implemented.")

    def load_cycle(self, file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError("Samsung48GDataset is not yet implemented.")

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Samsung48GDataset is not yet implemented.")

    def validate_cycle(self, df: pd.DataFrame) -> bool:
        raise NotImplementedError("Samsung48GDataset is not yet implemented.")

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "samsung_48g", "status": "placeholder"}
