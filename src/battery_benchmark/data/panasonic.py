"""Panasonic NCR18650PF dataset adapter.

Source: McMaster University Battery Dataset
URL: https://data.mendeley.com/datasets/wykht8y7tg/1

``PanasonicDataset`` is a legacy registry stub.  Use :class:`PanasonicAdapter`
for the SequenceRecord-based API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from battery_benchmark.data.adapter_common import (
    AdapterSummary,
    ManifestMatAdapter,
)
from battery_benchmark.data.base_dataset import BaseBatteryDataset
from battery_benchmark.data.manifest_loader import ManifestEntry
from battery_benchmark.data.registry import register_dataset
from battery_benchmark.data.sequence_record import SequenceRecord

# Re-export for backward compatibility
__all__ = ["PanasonicDataset", "PanasonicAdapter", "AdapterSummary"]


@register_dataset("panasonic_18650pf")
class PanasonicDataset(BaseBatteryDataset):
    """Legacy adapter stub — kept for registry compatibility."""

    def discover_files(self, raw_dir: str | Path) -> list[Path]:
        raise NotImplementedError(
            "PanasonicDataset is a placeholder. Use PanasonicAdapter instead."
        )

    def build_manifest(self, raw_dir: str | Path) -> pd.DataFrame:
        raise NotImplementedError("PanasonicDataset is a placeholder.")

    def load_cycle(self, file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError("PanasonicDataset is a placeholder.")

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("PanasonicDataset is a placeholder.")

    def validate_cycle(self, df: pd.DataFrame) -> bool:
        raise NotImplementedError("PanasonicDataset is a placeholder.")

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "panasonic_18650pf",
            "status": "placeholder",
            "nominal_capacity_ah": 2.9,
            "note": "Use PanasonicAdapter for the SequenceRecord API.",
        }


def _derive_soc_anchor_source(entry: ManifestEntry) -> str:
    """Infer soc_anchor_source from Panasonic manifest evidence."""
    if entry.soc_anchor_source:
        return entry.soc_anchor_source
    evidence = entry.soc_anchor_evidence.lower()
    if "full_charge_protocol" in evidence:
        return "full_charge_protocol"
    if entry.pairing_status == "confirmed":
        return "preceding_full_charge"
    if entry.pairing_status == "unresolved":
        return "unresolved"
    if entry.file_type in ("charge", "characterization"):
        return "not_applicable"
    return "not_applicable"


def _derive_soc_init(entry: ManifestEntry) -> float | None:
    """Infer soc_init from Panasonic manifest evidence."""
    evidence = entry.soc_anchor_evidence
    if "SOC_init=1.0" in evidence or entry.pairing_status in ("confirmed", "unresolved"):
        return 1.0
    return None


class PanasonicAdapter(ManifestMatAdapter):
    """Vidal 2019 adapter for the Panasonic NCR18650PF 25 °C dataset."""

    DATASET_NAME = "panasonic_18650pf"
    SEQUENCE_ID_PREFIX = "panasonic_18650pf_25c"
    DEFAULT_PAPER_BATTERY_NAME = "Panasonic_NCR18650PF"
    DEFAULT_DATASET_IDENTITY_STATUS = "CONFIRMED"

    def __init__(
        self,
        raw_root: str | Path,
        manifest_path: str | Path,
        capacity_hint_ah: float = 2.9,
    ) -> None:
        super().__init__(raw_root, manifest_path, capacity_hint_ah)

    def get_soc_init(self, entry: ManifestEntry) -> float | None:
        return _derive_soc_init(entry)

    def get_soc_anchor_source(self, entry: ManifestEntry) -> str:
        return _derive_soc_anchor_source(entry)
