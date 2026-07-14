"""Standard column names and schema validation for battery cycle DataFrames.

Every dataset adapter must return DataFrames conforming to this schema.
Required columns must always be present and non-null.
Optional columns may be present or absent; absent optional columns are added
as NaN/None by ``validate_cycle``.

Adapter-level schema
--------------------
``REQUIRED_DATA_COLUMNS`` and ``REQUIRED_METADATA_FIELDS`` define the
intermediate representation produced by each dataset adapter *before* the
benchmark normalisation pipeline transforms it to the final ``COLUMNS`` schema.
``AdapterSplit`` and ``FileType`` are the controlled vocabularies for the
``split`` and ``file_type`` metadata fields of ``SequenceRecord``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Column name constants — use these everywhere instead of string literals
# ---------------------------------------------------------------------------

class COLUMNS:
    """Canonical column names for benchmark DataFrames."""

    # Time
    TIMESTAMP: str = "timestamp"          # seconds since cycle start (float)

    # Electrical measurements (required)
    VOLTAGE: str = "voltage"              # V
    CURRENT: str = "current"              # A (positive = charging)
    TEMPERATURE: str = "temperature"      # °C

    # Labels (optional — may be NaN for target domain)
    SOC: str = "soc"                      # [0, 1]
    SOH: str = "soh"                      # [0, 1]
    CAPACITY_AH: str = "capacity_ah"      # Ah

    # Identifiers (required)
    CELL_ID: str = "cell_id"
    CYCLE_ID: str = "cycle_id"            # integer, unique within cell
    DOMAIN_ID: str = "domain_id"
    PROFILE_NAME: str = "profile_name"

    # Dataset split
    SPLIT: str = "split"                  # "train" | "val" | "test" | "unsplit"

    # Optional identifiers
    MISSION_ID: str = "mission_id"        # for mission-based datasets (eVTOL)
    PHASE_NAME: str = "phase_name"        # e.g. "charge", "discharge", "rest"

    # Label metadata
    LABEL_AVAILABLE: str = "label_available"   # bool
    LABEL_SOURCE: str = "label_source"          # LabelSource value
    LABEL_QUALITY: str = "label_quality"        # float in [0, 1] or None
    SOURCE_DATASET: str = "source_dataset"
    SAMPLING_RATE_HZ: str = "sampling_rate_hz"


REQUIRED_COLS: list[str] = [
    COLUMNS.TIMESTAMP,
    COLUMNS.VOLTAGE,
    COLUMNS.CURRENT,
    COLUMNS.TEMPERATURE,
    COLUMNS.CELL_ID,
    COLUMNS.CYCLE_ID,
    COLUMNS.DOMAIN_ID,
    COLUMNS.PROFILE_NAME,
    COLUMNS.SPLIT,
]

OPTIONAL_COLS: list[str] = [
    COLUMNS.SOC,
    COLUMNS.SOH,
    COLUMNS.CAPACITY_AH,
    COLUMNS.MISSION_ID,
    COLUMNS.PHASE_NAME,
    COLUMNS.LABEL_AVAILABLE,
    COLUMNS.LABEL_SOURCE,
    COLUMNS.LABEL_QUALITY,
    COLUMNS.SOURCE_DATASET,
    COLUMNS.SAMPLING_RATE_HZ,
]


class LabelSource(str, Enum):
    """Enumeration of SOC label provenance."""

    MEASURED_REFERENCE = "measured_reference"
    CYCLER_REFERENCE = "cycler_reference"
    COULOMB_COUNTED_DYNAMIC_CAPACITY = "coulomb_counted_dynamic_capacity"
    COULOMB_COUNTED_FIXED_CAPACITY = "coulomb_counted_fixed_capacity"
    PSEUDO_LABEL = "pseudo_label"
    UNAVAILABLE = "unavailable"


class Split(str, Enum):
    """Dataset split labels."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    UNSPLIT = "unsplit"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_cycle(df: pd.DataFrame, raise_on_missing: bool = True) -> pd.DataFrame:
    """Validate a cycle DataFrame against the benchmark schema.

    Parameters
    ----------
    df:
        DataFrame to validate.
    raise_on_missing:
        If True, raise ValueError when a required column is absent.
        If False, warnings are printed and the DataFrame is returned as-is.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with optional missing columns added as NaN.

    Raises
    ------
    ValueError
        If required columns are missing and ``raise_on_missing=True``.
    """
    missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_required:
        msg = f"Cycle DataFrame is missing required columns: {missing_required}"
        if raise_on_missing:
            raise ValueError(msg)
        import warnings
        warnings.warn(msg, stacklevel=2)

    # Add absent optional columns as NaN
    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df = df.copy()
            df[col] = None

    return df


# ---------------------------------------------------------------------------
# Adapter-level schema — raw adapter output (pre-normalisation)
# ---------------------------------------------------------------------------

# Column names in the DataFrame carried by SequenceRecord.
# These use explicit unit suffixes to avoid ambiguity with the final COLUMNS.
REQUIRED_DATA_COLUMNS: list[str] = [
    "time_s",           # elapsed seconds since file start (float)
    "timestamp",        # wall-clock timestamp (datetime, str, or None — cleaned later)
    "voltage_v",        # cell terminal voltage in V
    "current_a",        # current in A  (positive = charge, negative = discharge)
    "temperature_c",    # battery case temperature in °C
    "cycler_ah",        # cumulative Ah as reported by cycler (may reset between files)
]

# Metadata fields required in every SequenceRecord.metadata dict.
REQUIRED_METADATA_FIELDS: frozenset[str] = frozenset({
    "dataset",              # str — canonical dataset name, e.g. "samsung_30t"
    "battery",              # str — public battery identifier
    "source_file",          # str — relative path to the source .mat file (non-empty)
    "source_sha256",        # str — SHA-256 of the source file (may be "" if not yet computed)
    "sequence_id",          # str — unique identifier for this record (non-empty)
    "profile",              # str — public dataset profile name, e.g. "Mixed1"
    "paper_profile",        # str — Vidal paper profile name, e.g. "Mix1"; "" for excluded files
    "file_type",            # str — one of ALLOWED_FILE_TYPES
    "split",                # str — one of ALLOWED_ADAPTER_SPLITS
    "temperature_nominal_c", # float — nominal chamber temperature
    "soc_init",             # float | None — SOC at file start (None = unknown)
    "soc_anchor_source",    # str — e.g. "full_charge_protocol" or "preceding_full_charge"
    "capacity_hint_ah",     # float | None — nominal cell capacity hint (None = unknown)
})


class AdapterSplit(str, Enum):
    """Split vocabulary at the adapter / manifest level.

    Distinct from the benchmark-level ``Split`` enum which uses ``train``/
    ``val``/``test``/``unsplit``.  These map the manifest boolean flags to a
    single string label carried in ``SequenceRecord.metadata["split"]``.
    """

    FULL_TRAIN = "full_train"       # Mix1–8 (all training data)
    PARTIAL_TRAIN = "partial_train" # Mix1–3 (limited target-domain data)
    TEST = "test"                   # UDDS, LA92, US06
    EXCLUDE = "exclude"             # characterisation, HWFET, PausCycl, etc.
    UNSPECIFIED = "unspecified"     # not yet classified


class FileType(str, Enum):
    """File-type vocabulary for adapter-level classification."""

    DRIVE_CYCLE = "drive_cycle"
    CHARGE = "charge"
    CHARACTERIZATION = "characterization"
    OTHER = "other"
    UNKNOWN = "unknown"


# Convenience sets for validation
ALLOWED_ADAPTER_SPLITS: frozenset[str] = frozenset(s.value for s in AdapterSplit)
ALLOWED_FILE_TYPES: frozenset[str] = frozenset(t.value for t in FileType)


def get_schema_info() -> dict[str, Any]:
    """Return a summary of the schema for documentation."""
    return {
        "required_columns": REQUIRED_COLS,
        "optional_columns": OPTIONAL_COLS,
        "label_sources": [s.value for s in LabelSource],
        "splits": [s.value for s in Split],
        "adapter_splits": [s.value for s in AdapterSplit],
        "file_types": [t.value for t in FileType],
        "required_data_columns": REQUIRED_DATA_COLUMNS,
    }
