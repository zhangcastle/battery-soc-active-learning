"""LG INR18650HG2 dataset adapter.

Source: McMaster University Battery Dataset

``LGDataset`` is a legacy registry stub.  Use :class:`LGHG2Adapter` for the
SequenceRecord-based API.
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

__all__ = ["LGDataset", "LGHG2Adapter", "AdapterSummary"]


@register_dataset("lg_hg2")
class LGDataset(BaseBatteryDataset):
    """Legacy adapter stub — kept for registry compatibility."""

    def discover_files(self, raw_dir: str | Path) -> list[Path]:
        raise NotImplementedError(
            "LGDataset is a placeholder. Use LGHG2Adapter instead."
        )

    def build_manifest(self, raw_dir: str | Path) -> pd.DataFrame:
        raise NotImplementedError("LGDataset is a placeholder.")

    def load_cycle(self, file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError("LGDataset is a placeholder.")

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("LGDataset is a placeholder.")

    def validate_cycle(self, df: pd.DataFrame) -> bool:
        raise NotImplementedError("LGDataset is a placeholder.")

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "lg_hg2",
            "status": "placeholder",
            "nominal_capacity_ah": 3.0,
            "note": "Use LGHG2Adapter for the SequenceRecord API.",
        }


def _lg_soc_anchor_source(entry: ManifestEntry) -> str:
    """Return LG soc_anchor_source from the manifest when available."""
    if entry.soc_anchor_source.strip():
        return entry.soc_anchor_source.strip()
    if entry.pairing_status == "not_applicable":
        return "not_applicable"
    if entry.file_type in ("charge", "characterization", "pause"):
        return "not_applicable"
    return "not_applicable"


class LGHG2Adapter(ManifestMatAdapter):
    """Vidal 2019 adapter for the LG HG2 25 °C dataset."""

    DATASET_NAME = "lg_hg2"
    SEQUENCE_ID_PREFIX = "lg_hg2_25c"
    DEFAULT_PAPER_BATTERY_NAME = "LG_HG2"
    DEFAULT_DATASET_IDENTITY_STATUS = "CONFIRMED"

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
        return _lg_soc_anchor_source(entry)
