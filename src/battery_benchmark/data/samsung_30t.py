"""Samsung INR21700-30T dataset adapter.

Source: McMaster University Battery Dataset

``Samsung30TDataset`` is a legacy registry stub.  Use :class:`Samsung30TAdapter`
for the SequenceRecord-based API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from battery_benchmark.data.adapter_common import (
    AdapterSummary,
    ManifestMatAdapter,
    parse_manifest_soc_init,
)
from battery_benchmark.data.base_dataset import BaseBatteryDataset
from battery_benchmark.data.manifest_loader import ManifestEntry
from battery_benchmark.data.registry import register_dataset

__all__ = ["Samsung30TDataset", "Samsung30TAdapter", "AdapterSummary"]


@register_dataset("samsung_30t")
class Samsung30TDataset(BaseBatteryDataset):
    """Legacy adapter stub — kept for registry compatibility."""

    def discover_files(self, raw_dir: str | Path) -> list[Path]:
        raise NotImplementedError(
            "Samsung30TDataset is a placeholder. Use Samsung30TAdapter instead."
        )

    def build_manifest(self, raw_dir: str | Path) -> pd.DataFrame:
        raise NotImplementedError("Samsung30TDataset is a placeholder.")

    def load_cycle(self, file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError("Samsung30TDataset is a placeholder.")

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Samsung30TDataset is a placeholder.")

    def validate_cycle(self, df: pd.DataFrame) -> bool:
        raise NotImplementedError("Samsung30TDataset is a placeholder.")

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "samsung_30t",
            "status": "placeholder",
            "nominal_capacity_ah": 3.0,
            "note": "Use Samsung30TAdapter for the SequenceRecord API.",
        }


class Samsung30TAdapter(ManifestMatAdapter):
    """Vidal 2019 adapter for the Samsung INR21700-30T 25 °C dataset.

    SOC metadata is read directly from manifest ``soc_init`` and
    ``soc_anchor_source`` columns — no dataset-specific inference.
    """

    DATASET_NAME = "samsung_30t"
    SEQUENCE_ID_PREFIX = "samsung_30t_25c"
    DEFAULT_PAPER_BATTERY_NAME = "Samsung T30"
    DEFAULT_DATASET_IDENTITY_STATUS = "STRONGLY_SUPPORTED"

    def __init__(
        self,
        raw_root: str | Path,
        manifest_path: str | Path,
        capacity_hint_ah: float = 3.0,
    ) -> None:
        super().__init__(raw_root, manifest_path, capacity_hint_ah)

    def get_soc_init(self, entry: ManifestEntry) -> float | None:
        return parse_manifest_soc_init(entry)

    def get_soc_anchor_source(self, entry: ManifestEntry) -> str:
        return entry.soc_anchor_source.strip()
