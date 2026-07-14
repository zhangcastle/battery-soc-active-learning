"""SequenceRecord: the standard output of a battery dataset adapter.

Each ``SequenceRecord`` corresponds to exactly one raw source file.  It carries
the time-series data as a pandas DataFrame, a rich metadata dict, and a
``QCResult`` that summarises any data-quality findings.

This is the *intermediate* representation produced by adapters.  The benchmark
normalisation pipeline later transforms it into the final ``COLUMNS`` schema
(``schema.py``) for model training.

Design invariants
-----------------
- ``data`` is always a *copy* of whatever DataFrame was passed in, so the
  caller's DataFrame is never mutated.
- Validation runs automatically in ``__post_init__``; there is no way to
  construct an invalid ``SequenceRecord`` silently.
- SOC calculation, resampling, normalisation, and column renaming are all
  *outside* the scope of this class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from battery_benchmark.data.adapter_qc import (
    AdapterSchemaError,
    InvalidAdapterSplitError,
    InvalidFileTypeError,
    MissingDataColumnError,
    MissingMetadataFieldError,
    QCResult,
)
from battery_benchmark.data.schema import (
    ALLOWED_ADAPTER_SPLITS,
    ALLOWED_FILE_TYPES,
    REQUIRED_DATA_COLUMNS,
    REQUIRED_METADATA_FIELDS,
)


@dataclass
class SequenceRecord:
    """Container for a single raw file's data after adapter loading.

    Parameters
    ----------
    data:
        Time-series DataFrame.  Must contain all columns listed in
        :data:`~battery_benchmark.data.schema.REQUIRED_DATA_COLUMNS`.
        A copy is stored internally; the original DataFrame is not modified.
    metadata:
        Dict of file-level metadata.  Must contain all fields listed in
        :data:`~battery_benchmark.data.schema.REQUIRED_METADATA_FIELDS`.
    qc:
        QC result for this file.  Defaults to a fresh ``QCResult()``.

    Raises
    ------
    TypeError
        If *data* is not a ``pandas.DataFrame``.
    MissingDataColumnError
        If any column in ``REQUIRED_DATA_COLUMNS`` is absent from *data*.
    MissingMetadataFieldError
        If any field in ``REQUIRED_METADATA_FIELDS`` is absent from *metadata*.
    AdapterSchemaError
        If ``metadata["split"]`` or ``metadata["file_type"]`` contains an
        unrecognised value, or if ``metadata["source_file"]`` /
        ``metadata["sequence_id"]`` is empty.
    """

    data: pd.DataFrame
    metadata: dict[str, Any]
    qc: QCResult = field(default_factory=QCResult)

    def __post_init__(self) -> None:
        # Store a copy so the caller's DataFrame is never mutated.
        object.__setattr__(self, "data", self.data.copy())
        self.validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Run all structural checks.  Raises on the first violation found.

        Called automatically during construction.  May also be called
        explicitly after external modifications to :attr:`metadata`.
        """
        self._check_data_type()
        self._check_data_columns()
        self._check_metadata_fields()
        self._check_metadata_values()

    def _check_data_type(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError(
                f"SequenceRecord.data must be a pandas DataFrame, "
                f"got {type(self.data).__name__!r}."
            )

    def _check_data_columns(self) -> None:
        missing = [c for c in REQUIRED_DATA_COLUMNS if c not in self.data.columns]
        if missing:
            raise MissingDataColumnError(missing)

    def _check_metadata_fields(self) -> None:
        missing = sorted(REQUIRED_METADATA_FIELDS - set(self.metadata))
        if missing:
            raise MissingMetadataFieldError(missing)

    def _check_metadata_values(self) -> None:
        # source_file must be a non-empty string (not a list, not "")
        source_file = self.metadata["source_file"]
        if not isinstance(source_file, str):
            raise AdapterSchemaError(
                f"metadata['source_file'] must be a non-empty string, "
                f"got {type(source_file).__name__!r}: {source_file!r}."
            )
        if not source_file.strip():
            raise AdapterSchemaError(
                "metadata['source_file'] must not be empty."
            )

        # sequence_id must be a non-empty string
        seq_id = self.metadata["sequence_id"]
        if not isinstance(seq_id, str):
            raise AdapterSchemaError(
                f"metadata['sequence_id'] must be a non-empty string, "
                f"got {type(seq_id).__name__!r}: {seq_id!r}."
            )
        if not seq_id.strip():
            raise AdapterSchemaError(
                "metadata['sequence_id'] must not be empty."
            )

        # split must be a recognised AdapterSplit value
        split_val = self.metadata["split"]
        if split_val not in ALLOWED_ADAPTER_SPLITS:
            raise InvalidAdapterSplitError(split_val, ALLOWED_ADAPTER_SPLITS)

        # file_type must be a recognised FileType value
        ft_val = self.metadata["file_type"]
        if ft_val not in ALLOWED_FILE_TYPES:
            raise InvalidFileTypeError(ft_val, ALLOWED_FILE_TYPES)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_rows(self) -> int:
        """Number of rows in the data DataFrame."""
        return len(self.data)

    @property
    def columns(self) -> list[str]:
        """Ordered list of column names in the data DataFrame."""
        return list(self.data.columns)

    @property
    def has_required_columns(self) -> bool:
        """True when all required data columns are present."""
        return all(c in self.data.columns for c in REQUIRED_DATA_COLUMNS)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a flat dict summarising this record (suitable for logging).

        Does not include the full DataFrame — only key statistics.
        """
        return {
            "dataset": self.metadata.get("dataset", ""),
            "battery": self.metadata.get("battery", ""),
            "source_file": self.metadata.get("source_file", ""),
            "sequence_id": self.metadata.get("sequence_id", ""),
            "profile": self.metadata.get("profile", ""),
            "paper_profile": self.metadata.get("paper_profile", ""),
            "file_type": self.metadata.get("file_type", ""),
            "split": self.metadata.get("split", ""),
            "temperature_nominal_c": self.metadata.get("temperature_nominal_c"),
            "soc_init": self.metadata.get("soc_init"),
            "soc_anchor_source": self.metadata.get("soc_anchor_source", ""),
            "capacity_hint_ah": self.metadata.get("capacity_hint_ah"),
            "n_rows": self.n_rows,
            "columns": self.columns,
            "qc_status": self.qc.status,
            "qc_issues": list(self.qc.issues),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SequenceRecord("
            f"seq_id={self.metadata.get('sequence_id', '?')!r}, "
            f"profile={self.metadata.get('profile', '?')!r}, "
            f"split={self.metadata.get('split', '?')!r}, "
            f"n_rows={self.n_rows}, "
            f"qc={self.qc.status!r})"
        )
