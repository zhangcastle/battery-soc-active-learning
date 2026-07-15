"""Generic CSV manifest loader for battery dataset adapters.

A *manifest* is a CSV file that assigns every raw ``.mat`` file in one dataset
session to a split role (full_train / partial_train / test / exclude).  This
module reads that CSV and returns a typed list of :class:`ManifestEntry`
objects that adapters can consume without re-parsing CSV details.

Design rules
------------
- The manifest file is never modified.
- All validation happens at load time; no silent coercion.
- Every split flag is validated for logical consistency.
- Extra columns are preserved in ``ManifestEntry.raw_row``.
- Row order is always preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import math

import pandas as pd

from battery_benchmark.data.schema import ALLOWED_FILE_TYPES, ALLOWED_RECORD_ROLES


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ManifestError(ValueError):
    """Base exception for manifest loading / selection failures."""


class MissingManifestColumnError(ManifestError):
    """Raised when a required CSV column is absent."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__(
            f"Manifest CSV is missing required column(s): {sorted(missing)}"
        )


class InvalidManifestRowError(ManifestError):
    """Raised when a single manifest row violates a consistency rule."""

    def __init__(self, row_idx: int, reason: str) -> None:
        self.row_idx = row_idx
        self.reason = reason
        super().__init__(f"Invalid manifest row {row_idx}: {reason}")


class UnsupportedManifestSplitError(ManifestError):
    """Raised when an unrecognised split label is requested."""

    def __init__(self, split: str) -> None:
        self.split = split
        super().__init__(
            f"Unsupported split {split!r}. "
            f"Must be one of: {sorted(SUPPORTED_SPLITS)}."
        )


class SHA256MismatchError(ManifestError):
    """Raised when a file's actual SHA-256 differs from the manifest record."""

    def __init__(self, path: Path, expected: str, actual: str) -> None:
        self.path = path
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"SHA-256 mismatch for {path.name}: "
            f"manifest={expected[:16]}… actual={actual[:16]}…"
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Columns that every manifest CSV must contain.
REQUIRED_MANIFEST_COLUMNS: frozenset[str] = frozenset({
    "battery",
    "filename",
    "relative_path",
    "file_type",
    "full_train",
    "partial_train",
    "test",
    "exclude",
    "qc_status",
    "profile",
    "temperature_c",
    "record_role",
})

#: Split labels understood by :func:`select_manifest_entries`.
SUPPORTED_SPLITS: frozenset[str] = frozenset({
    "full_train",
    "partial_train",
    "test",
    "exclude",
})

#: Allowed manifest QC status values (normalised to upper case).
ALLOWED_QC_STATUSES: frozenset[str] = frozenset({"PASS", "WARN", "FAIL"})


# ---------------------------------------------------------------------------
# Flag parsing helper
# ---------------------------------------------------------------------------

def _parse_bool_flag(value: Any, column: str, row_idx: int) -> int:
    """Normalise boolean split flags to integer 0 or 1.

    Accepts bool, int 0/1, float 0.0/1.0, and string ``0``/``1``/``true``/``false``.
    Rejects truncated floats (e.g. 0.5), out-of-range integers, NaN, and Inf.

    Raises
    ------
    InvalidManifestRowError
        When the value cannot be interpreted as a boolean flag.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        if value in (0, 1):
            return value
    elif isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise InvalidManifestRowError(
                row_idx,
                f"Cannot parse boolean flag for column {column!r}: {value!r}. "
                "Expected one of: 0, 1, '0', '1', True, False.",
            )
        if value == 0.0:
            return 0
        if value == 1.0:
            return 1
    if isinstance(value, str):
        s = value.strip()
        if s in ("0", "False", "false"):
            return 0
        if s in ("1", "True", "true"):
            return 1
    raise InvalidManifestRowError(
        row_idx,
        f"Cannot parse boolean flag for column {column!r}: {value!r}. "
        "Expected one of: 0, 1, '0', '1', True, False.",
    )


def _safe_str(value: Any) -> str:
    """Return *value* as a stripped string, or '' for NaN/None."""
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    return str(value).strip()


def _parse_temperature_c(value: Any, row_idx: int) -> float:
    """Parse ``temperature_c`` as a finite float.

    Raises
    ------
    InvalidManifestRowError
        When the value is empty, non-numeric, NaN, or Inf.
    """
    raw = value
    if value is None:
        text = ""
    elif isinstance(value, float) and value != value:  # NaN
        raise InvalidManifestRowError(
            row_idx,
            f"Invalid temperature_c: column='temperature_c', raw_value={raw!r} (NaN).",
        )
    else:
        text = str(value).strip()

    if not text:
        raise InvalidManifestRowError(
            row_idx,
            f"Invalid temperature_c: column='temperature_c', raw_value={raw!r} (empty).",
        )

    try:
        temp = float(text)
    except (ValueError, TypeError):
        raise InvalidManifestRowError(
            row_idx,
            f"Invalid temperature_c: column='temperature_c', raw_value={raw!r} (non-numeric).",
        ) from None

    if not math.isfinite(temp):
        raise InvalidManifestRowError(
            row_idx,
            f"Invalid temperature_c: column='temperature_c', raw_value={raw!r} (non-finite).",
        )
    return temp


def _parse_qc_status(value: Any, row_idx: int) -> str:
    """Normalise and validate ``qc_status`` as PASS / WARN / FAIL."""
    text = _safe_str(value).upper()
    if text not in ALLOWED_QC_STATUSES:
        raise InvalidManifestRowError(
            row_idx,
            f"Invalid qc_status: column='qc_status', raw_value={value!r}. "
            f"Must be one of: {sorted(ALLOWED_QC_STATUSES)}.",
        )
    return text


def _parse_file_type(value: Any, row_idx: int) -> str:
    """Validate ``file_type`` against :data:`ALLOWED_FILE_TYPES`."""
    text = _safe_str(value).lower()
    if not text:
        raise InvalidManifestRowError(
            row_idx,
            f"Invalid file_type: column='file_type', raw_value={value!r} (empty).",
        )
    if text not in ALLOWED_FILE_TYPES:
        raise InvalidManifestRowError(
            row_idx,
            f"Invalid file_type: column='file_type', raw_value={value!r}. "
            f"Must be one of: {sorted(ALLOWED_FILE_TYPES)}.",
        )
    return text


def _parse_record_role(value: Any, row_idx: int) -> str:
    """Validate ``record_role`` against :data:`ALLOWED_RECORD_ROLES`."""
    text = _safe_str(value).lower()
    if not text:
        raise InvalidManifestRowError(
            row_idx,
            f"Invalid record_role: column='record_role', raw_value={value!r} (empty). "
            f"Must be one of: {sorted(ALLOWED_RECORD_ROLES)}.",
        )
    if text not in ALLOWED_RECORD_ROLES:
        raise InvalidManifestRowError(
            row_idx,
            f"Invalid record_role: column='record_role', raw_value={value!r}. "
            f"Must be one of: {sorted(ALLOWED_RECORD_ROLES)}.",
        )
    return text


def _validate_record_role_consistency(
    row_idx: int,
    record_role: str,
    file_type: str,
    full_train: int,
    partial_train: int,
    test_: int,
    exclude: int,
) -> None:
    """Cross-check ``record_role`` against split flags and ``file_type``."""
    is_selected = bool(full_train or partial_train or test_)

    if record_role == "model_input":
        if file_type not in ("drive_cycle", "charge"):
            raise InvalidManifestRowError(
                row_idx,
                f"record_role='model_input' requires file_type in "
                f"{{'drive_cycle', 'charge'}}, got file_type={file_type!r}.",
            )
        if not is_selected:
            raise InvalidManifestRowError(
                row_idx,
                "record_role='model_input' requires membership in "
                "full_train, partial_train, or test.",
            )
        if exclude:
            raise InvalidManifestRowError(
                row_idx,
                "record_role='model_input' cannot coexist with exclude=1.",
            )
    elif record_role == "soc_anchor":
        if file_type != "charge":
            raise InvalidManifestRowError(
                row_idx,
                f"record_role='soc_anchor' requires file_type='charge', "
                f"got file_type={file_type!r}.",
            )
        if not is_selected:
            raise InvalidManifestRowError(
                row_idx,
                "record_role='soc_anchor' requires membership in "
                "full_train, partial_train, or test.",
            )
        if exclude:
            raise InvalidManifestRowError(
                row_idx,
                "record_role='soc_anchor' cannot coexist with exclude=1.",
            )
    elif record_role == "excluded":
        if not exclude:
            raise InvalidManifestRowError(
                row_idx,
                "record_role='excluded' requires exclude=1.",
            )
        if is_selected:
            raise InvalidManifestRowError(
                row_idx,
                "record_role='excluded' cannot belong to a selected split.",
            )


def entry_manifest_splits(entry: ManifestEntry) -> list[str]:
    """Return manifest membership labels for *entry* in stable order."""
    splits: list[str] = []
    if entry.full_train == 1:
        splits.append("full_train")
    if entry.partial_train == 1:
        splits.append("partial_train")
    if entry.test == 1:
        splits.append("test")
    if entry.exclude == 1:
        splits.append("exclude")
    return splits


# ---------------------------------------------------------------------------
# ManifestEntry dataclass
# ---------------------------------------------------------------------------

@dataclass
class ManifestEntry:
    """One row from a split manifest CSV, with typed attributes.

    Required fields are validated at construction time.  Optional fields that
    are absent from a particular manifest default to empty string / None.
    All original CSV columns are available in :attr:`raw_row`.

    Notes
    -----
    ``split`` flags are stored as ``int`` 0/1 (not bool) to avoid ambiguity
    when serialising back to CSV.
    """

    # ---- always present ----
    battery: str
    temperature_c: float
    filename: str
    relative_path: str
    profile: str
    file_type: str
    full_train: int
    partial_train: int
    test: int
    exclude: int
    qc_status: str
    record_role: str

    # ---- optional — present in some manifests ----
    sha256: str = ""
    paper_battery_name: str = ""
    dataset_identity_status: str = ""
    paper_profile: str = ""
    profile_number: str = ""
    start_datetime: str = ""
    pairing_status: str = ""
    soc_anchor_evidence: str = ""
    soc_anchor_source: str = ""      # explicit if Samsung; derived by adapter for Panasonic
    soc_init_str: str = ""           # raw string; adapter converts to float | None
    confidence: str = ""
    evidence: str = ""
    notes: str = ""
    qc_notes: str = ""

    #: Original CSV row dict — preserves all columns not listed above.
    raw_row: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_selected(self) -> bool:
        """True when the entry belongs to at least one active split."""
        return bool(self.full_train or self.partial_train or self.test)

    @property
    def is_excluded(self) -> bool:
        """True when exclude == 1."""
        return self.exclude == 1


# ---------------------------------------------------------------------------
# Row → ManifestEntry converter
# ---------------------------------------------------------------------------

def _row_to_entry(row: pd.Series, row_idx: int) -> ManifestEntry:
    """Parse one DataFrame row into a :class:`ManifestEntry`.

    Raises :class:`InvalidManifestRowError` on any constraint violation.
    """
    s = _safe_str

    filename = s(row.get("filename", ""))
    relative_path = s(row.get("relative_path", ""))

    if not filename.strip():
        raise InvalidManifestRowError(row_idx, "filename is empty.")
    if not relative_path.strip():
        raise InvalidManifestRowError(row_idx, "relative_path is empty.")

    # Parse split flags
    def _flag(col: str) -> int:
        return _parse_bool_flag(row.get(col, 0), col, row_idx)

    full_train = _flag("full_train")
    partial_train = _flag("partial_train")
    test_ = _flag("test")
    exclude = _flag("exclude")

    # Validate logical consistency
    if exclude and (full_train or partial_train or test_):
        raise InvalidManifestRowError(
            row_idx,
            f"exclude=1 conflicts with a selected split flag "
            f"(full_train={full_train}, partial_train={partial_train}, test={test_}). "
            "A file cannot be both excluded and selected."
        )
    if full_train and test_:
        raise InvalidManifestRowError(
            row_idx,
            "full_train=1 and test=1 on the same row — train/test overlap is not permitted."
        )
    if partial_train and test_:
        raise InvalidManifestRowError(
            row_idx,
            "partial_train=1 and test=1 on the same row — train/test overlap is not permitted."
        )
    if partial_train and not full_train:
        raise InvalidManifestRowError(
            row_idx,
            "partial_train=1 but full_train=0 — partial_train must be a subset of full_train."
        )

    file_type = _parse_file_type(row.get("file_type"), row_idx)
    record_role = _parse_record_role(row.get("record_role"), row_idx)
    _validate_record_role_consistency(
        row_idx, record_role, file_type, full_train, partial_train, test_, exclude
    )

    return ManifestEntry(
        battery=s(row.get("battery", "")),
        temperature_c=_parse_temperature_c(row.get("temperature_c"), row_idx),
        filename=filename,
        relative_path=relative_path,
        profile=s(row.get("profile", "")),
        file_type=file_type,
        full_train=full_train,
        partial_train=partial_train,
        test=test_,
        exclude=exclude,
        qc_status=_parse_qc_status(row.get("qc_status"), row_idx),
        record_role=record_role,
        sha256=s(row.get("sha256", "")),
        paper_battery_name=s(row.get("paper_battery_name", "")),
        dataset_identity_status=s(row.get("dataset_identity_status", "")),
        paper_profile=s(row.get("paper_profile", "")),
        profile_number=s(row.get("profile_number", "")),
        start_datetime=s(row.get("start_datetime", "")),
        pairing_status=s(row.get("pairing_status", "")),
        soc_anchor_evidence=s(row.get("soc_anchor_evidence", "")),
        soc_anchor_source=s(row.get("soc_anchor_source", "")),
        soc_init_str=s(row.get("soc_init", "")),
        confidence=s(row.get("confidence", "")),
        evidence=s(row.get("evidence", "")),
        notes=s(row.get("notes", "")),
        qc_notes=s(row.get("qc_notes", "")),
        raw_row=row.to_dict(),
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def load_manifest(path: str | Path) -> list[ManifestEntry]:
    """Load a split manifest CSV and return an ordered list of entries.

    Parameters
    ----------
    path:
        Path to the manifest ``.csv`` file.

    Returns
    -------
    list[ManifestEntry]
        One entry per CSV row, in the same order as the file.

    Raises
    ------
    ManifestError
        If the file does not exist or is not a ``.csv``.
    MissingManifestColumnError
        If any required column is absent.
    InvalidManifestRowError
        If any row violates a split-consistency rule.
    """
    path = Path(path)

    if not path.exists():
        raise ManifestError(f"Manifest file not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ManifestError(
            f"Expected a .csv manifest file, got extension {path.suffix!r}: {path}"
        )

    # Read all columns as strings to avoid dtype surprises; keep empty cells as ""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(df.columns))
    if missing:
        raise MissingManifestColumnError(missing)

    entries: list[ManifestEntry] = []
    for idx, row in df.iterrows():
        entry = _row_to_entry(row, int(idx))
        entries.append(entry)

    return entries


def select_manifest_entries(
    entries: list[ManifestEntry],
    split: str,
) -> list[ManifestEntry]:
    """Return the subset of *entries* belonging to *split*.

    Parameters
    ----------
    entries:
        Full manifest entry list as returned by :func:`load_manifest`.
    split:
        One of ``"full_train"``, ``"partial_train"``, ``"test"``,
        ``"exclude"``.

    Returns
    -------
    list[ManifestEntry]
        Filtered list in the same order as *entries*.

    Raises
    ------
    UnsupportedManifestSplitError
        If *split* is not in :data:`SUPPORTED_SPLITS`.
    """
    if split not in SUPPORTED_SPLITS:
        raise UnsupportedManifestSplitError(split)

    selectors = {
        "full_train": lambda e: e.full_train == 1,
        "partial_train": lambda e: e.partial_train == 1,
        "test": lambda e: e.test == 1,
        "exclude": lambda e: e.exclude == 1,
    }
    pred = selectors[split]
    return [e for e in entries if pred(e)]
