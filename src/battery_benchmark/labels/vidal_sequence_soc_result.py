"""Data models for Vidal sequence-aware SOC v2 labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


LABEL_POLICY_VERSION = "vidal_sequence_v2"
INTEGRATION_METHOD = "trapezoidal_time_s"
ENDPOINT_CORRECTION_THRESHOLD = 0.03


@dataclass
class SequenceLabelRecordMeta:
    dataset: str
    protocol_split: str
    sequence_id: str
    sequence_order: int
    source_filename: str
    file_type: str
    profile: str
    record_role: str
    soc_init: float
    soc_end: float
    soc_init_source: str
    capacity_reference_ah: float
    owned_by_previous_sequence_id: str = ""
    anchor_source: str = ""
    reset_unwrap_applied: bool = False
    continuity_correction_applied: bool = False
    endpoint_correction_applied: bool = False
    raw_soc_end_before_correction: float | None = None
    corrected_soc_end: float | None = None
    endpoint_correction_magnitude: float = 0.0
    label_qc_status: str = "PASS"
    label_qc_reasons: list[str] = field(default_factory=list)
    soc_clip_count: int = 0
    soc_clip_max_excess: float = 0.0
    row_count: int = 0
    artifact_path: str = ""
    artifact_sha256: str = ""
    source_sha256: str = ""
    source_raw_path: str = ""
    sequence_segment_id: int = 1
    segment_order: int = 1
    segment_start_reason: str = ""
    sequence_segment_start: bool = False
    missing_preceding_charge: bool = False
    missing_following_charge: bool = False
    missing_charge_reason: str = ""
    confidence: str = ""
    owner_filename: str = ""
    owner_sequence_order: int | None = None
    git_dirty: bool = False


@dataclass
class SequenceSOCResult:
    """Pointwise labelled dataframe plus record metadata."""

    dataframe: pd.DataFrame
    meta: SequenceLabelRecordMeta

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta.__dict__,
            "columns": list(self.dataframe.columns),
            "row_count": len(self.dataframe),
        }


@dataclass
class AnchorMetadata:
    dataset: str
    filename: str
    next_selected_drive: str
    next_selected_profile: str
    anchor_soc: float = 1.0
    full_voltage_reached: bool = False
    current_taper_detected: bool = False
    anchor_qc_status: str = "PASS"
    anchor_qc_reasons: list[str] = field(default_factory=list)


OUTPUT_POINTWISE_COLUMNS = (
    "time_s",
    "timestamp",
    "voltage_v",
    "current_a",
    "temperature_c",
    "cycler_ah_raw",
    "cycler_ah_effective",
    "soc_cc_raw",
    "soc_sequence_v2",
)
