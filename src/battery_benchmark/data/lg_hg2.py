"""LG HG2 dataset adapter.

Status: PLACEHOLDER — raises NotImplementedError.
See notebooks/03_prepare_lg_hg2.ipynb.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from battery_benchmark.data.base_dataset import BaseBatteryDataset
from battery_benchmark.data.registry import register_dataset


@register_dataset("lg_hg2")
class LGDataset(BaseBatteryDataset):
    """Adapter for the LG HG2 18650 drive-cycle dataset."""

    def discover_files(self, raw_dir: str | Path) -> list[Path]:
        raise NotImplementedError("LGDataset is not yet implemented.")

    def build_manifest(self, raw_dir: str | Path) -> pd.DataFrame:
        raise NotImplementedError("LGDataset is not yet implemented.")

    def load_cycle(self, file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError("LGDataset is not yet implemented.")

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("LGDataset is not yet implemented.")

    def validate_cycle(self, df: pd.DataFrame) -> bool:
        raise NotImplementedError("LGDataset is not yet implemented.")

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "lg_hg2", "status": "placeholder", "nominal_capacity_ah": 3.0}
