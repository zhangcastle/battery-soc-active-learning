"""CMU eVTOL battery dataset adapter.

Target-domain SOC labels are NOT available as direct measured ground truth.
A proxy reference may be generated via quality-controlled coulomb counting
with dynamic capacity estimation.  See docs/cmu_evtol_protocol.md.

Status: PLACEHOLDER — raises NotImplementedError.
See notebooks/05_prepare_cmu_evtol.ipynb.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from battery_benchmark.data.base_dataset import BaseBatteryDataset
from battery_benchmark.data.registry import register_dataset


@register_dataset("cmu_evtol")
class CMUeVTOLDataset(BaseBatteryDataset):
    """Adapter for the CMU eVTOL battery dataset.

    Supports cell-level, mission-level, and cycle-level splits.
    Label availability is False by default; a proxy reference may be
    generated offline and stored as ``label_source = "coulomb_counted_dynamic_capacity"``.
    """

    def discover_files(self, raw_dir: str | Path) -> list[Path]:
        raise NotImplementedError(
            "CMUeVTOLDataset is not yet implemented. "
            "See notebooks/05_prepare_cmu_evtol.ipynb and docs/cmu_evtol_protocol.md."
        )

    def build_manifest(self, raw_dir: str | Path) -> pd.DataFrame:
        raise NotImplementedError("CMUeVTOLDataset is not yet implemented.")

    def load_cycle(self, file_path: str | Path, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError("CMUeVTOLDataset is not yet implemented.")

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("CMUeVTOLDataset is not yet implemented.")

    def validate_cycle(self, df: pd.DataFrame) -> bool:
        raise NotImplementedError("CMUeVTOLDataset is not yet implemented.")

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": "cmu_evtol",
            "status": "placeholder",
            "label_available": False,
            "label_source": "unavailable",
            "has_mission_segments": True,
        }
