"""QC result container and custom exceptions for battery dataset adapters.

This module defines:

- Custom exception hierarchy for adapter schema errors.
- ``QCResult`` — a dataclass that accumulates quality-control findings for a
  single raw file.  It does *not* perform any data inspection itself; callers
  populate it by calling ``add_warning`` / ``add_failure`` / ``add_note``.

Usage example::

    qc = QCResult()
    if nan_count > 0:
        qc.add_warning(f"{nan_count} NaN values detected in voltage_v")
    if negative_dt_count > 0:
        qc.add_failure("strictly-decreasing timestamps found")
    assert qc.status in ("PASS", "WARN", "FAIL")
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# QC status constants
# ---------------------------------------------------------------------------

QC_PASS = "PASS"
QC_WARN = "WARN"
QC_FAIL = "FAIL"
ALLOWED_QC_STATUS: frozenset[str] = frozenset({QC_PASS, QC_WARN, QC_FAIL})


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class AdapterSchemaError(ValueError):
    """Base class for all adapter-level schema violations."""


class MissingDataColumnError(AdapterSchemaError):
    """Raised when a required data column is absent from the DataFrame."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(
            f"SequenceRecord data is missing required column(s): {missing}"
        )


class MissingMetadataFieldError(AdapterSchemaError):
    """Raised when a required metadata field is absent."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(
            f"SequenceRecord metadata is missing required field(s): {missing}"
        )


class InvalidQCStatusError(AdapterSchemaError):
    """Raised when an unrecognised QC status string is supplied."""

    def __init__(self, status: str) -> None:
        super().__init__(
            f"Invalid QC status {status!r}. Must be one of {sorted(ALLOWED_QC_STATUS)}."
        )


class InvalidAdapterSplitError(AdapterSchemaError):
    """Raised when metadata['split'] contains an unrecognised value."""

    def __init__(self, value: str, allowed: frozenset[str]) -> None:
        super().__init__(
            f"Invalid split value {value!r}. Must be one of {sorted(allowed)}."
        )


class InvalidFileTypeError(AdapterSchemaError):
    """Raised when metadata['file_type'] contains an unrecognised value."""

    def __init__(self, value: str, allowed: frozenset[str]) -> None:
        super().__init__(
            f"Invalid file_type value {value!r}. Must be one of {sorted(allowed)}."
        )


# ---------------------------------------------------------------------------
# QCResult
# ---------------------------------------------------------------------------

@dataclass
class QCResult:
    """Accumulates quality-control findings for a single raw adapter file.

    Parameters
    ----------
    status:
        Initial status string; must be ``"PASS"``, ``"WARN"``, or ``"FAIL"``.
    issues:
        List of warning/failure messages.  Defaults to a new empty list.
    notes:
        Informational messages that do not affect status.  Defaults to a new
        empty list.
    duplicate_timestamp_count:
        Number of time steps where ``dt == 0`` (cycler artifact).
    negative_time_step_count:
        Number of time steps where ``dt < 0`` (strictly non-monotonic).
    nan_count:
        Total NaN values across all numeric columns.
    inf_count:
        Total ±Inf values across all numeric columns.
    ah_reset_detected:
        True if the Ah counter was observed to reset mid-file.

    Notes
    -----
    This class is a *container* — it does not perform any data inspection.
    Populate it by calling :meth:`add_warning`, :meth:`add_failure`, and
    :meth:`add_note`.
    """

    status: str = QC_PASS
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    duplicate_timestamp_count: int = 0
    negative_time_step_count: int = 0
    nan_count: int = 0
    inf_count: int = 0
    ah_reset_detected: bool = False

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_QC_STATUS:
            raise InvalidQCStatusError(self.status)
        self._validate_counts()

    def _validate_counts(self) -> None:
        for attr in (
            "duplicate_timestamp_count",
            "negative_time_step_count",
            "nan_count",
            "inf_count",
        ):
            val = getattr(self, attr)
            if val < 0:
                raise ValueError(
                    f"QCResult.{attr} must be >= 0, got {val!r}."
                )

    # ------------------------------------------------------------------
    # Mutating helpers
    # ------------------------------------------------------------------

    def add_warning(self, message: str) -> None:
        """Append *message* to issues and set status to WARN (unless already FAIL)."""
        self.issues.append(message)
        if self.status != QC_FAIL:
            self.status = QC_WARN

    def add_failure(self, message: str) -> None:
        """Append *message* to issues and unconditionally set status to FAIL."""
        self.issues.append(message)
        self.status = QC_FAIL

    def add_note(self, message: str) -> None:
        """Append an informational *message* that does not affect status."""
        self.notes.append(message)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_failures(self) -> bool:
        """True when status is FAIL."""
        return self.status == QC_FAIL

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain-dict representation suitable for logging or CSV export."""
        return {
            "status": self.status,
            "issues": list(self.issues),
            "notes": list(self.notes),
            "duplicate_timestamp_count": self.duplicate_timestamp_count,
            "negative_time_step_count": self.negative_time_step_count,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "ah_reset_detected": self.ah_reset_detected,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"QCResult(status={self.status!r}, "
            f"issues={len(self.issues)}, "
            f"notes={len(self.notes)})"
        )
