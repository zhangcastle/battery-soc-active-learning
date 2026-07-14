"""Generic read-only MATLAB .mat file loader for battery dataset adapters.

This module provides a single public function:

    result = load_mat_file(path)

It returns a :class:`MatLoadResult` dataclass containing the raw measurement
arrays extracted from the ``meas`` MATLAB struct, the file's SHA-256 hash, and
any non-fatal warnings encountered during loading.

Design invariants
-----------------
- The source file is **never modified**.
- No data cleaning is performed here (no deduplication, no Ah-reset correction,
  no sign normalisation, no resampling).
- The ``TimeStamp`` field is preserved as-is (object / string array); callers
  are responsible for parsing.
- Every numeric measurement field is returned as a one-dimensional
  ``numpy.ndarray``, regardless of whether MATLAB stored it as (n,1) or (1,n).

MATLAB format support
---------------------
- MATLAB v4 / v5 / v6: supported via ``scipy.io.loadmat``.
- MATLAB v7.3 (HDF5): **not yet supported** — ``h5py`` is not a current project
  dependency.  A clear :class:`UnsupportedMatFormatError` is raised, along with
  a message explaining how to add support.

If you need v7.3 support, add ``h5py>=3.0`` to ``pyproject.toml`` and the
fallback branch in ``_load_with_backend`` will become active automatically.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class MatFileError(OSError):
    """Base exception for .mat loading failures."""


class MissingMeasurementStructError(MatFileError):
    """Raised when the expected ``meas`` top-level struct is absent."""

    def __init__(self, path: Path, root_keys: list[str]) -> None:
        super().__init__(
            f"No 'meas' struct found in {path}. "
            f"Top-level keys present: {root_keys}. "
            "Ensure the file contains a MATLAB struct named 'meas'."
        )
        self.path = path
        self.root_keys = root_keys


class UnsupportedMatFormatError(MatFileError):
    """Raised for MATLAB v7.3 / HDF5 files when h5py is not available.

    To add support, install h5py (``pip install h5py>=3.0``) and add it to
    ``pyproject.toml`` under ``[project] dependencies``.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"{path} appears to be a MATLAB v7.3 / HDF5 file. "
            "scipy.io.loadmat does not support this format. "
            "h5py is not installed. "
            "To add support: pip install h5py>=3.0 and add 'h5py>=3.0' to "
            "pyproject.toml dependencies."
        )
        self.path = path


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

# Fields that are always attempted when extracting from the meas struct.
# Additional fields present in the struct are also preserved.
_KNOWN_FIELDS: tuple[str, ...] = (
    "Time",
    "TimeStamp",
    "Voltage",
    "Current",
    "Ah",
    "Battery_Temp_degC",
    "Wh",
    "Power",
)


@dataclass
class MatLoadResult:
    """Result of loading a single MATLAB .mat file.

    Parameters
    ----------
    path:
        Resolved, absolute path to the loaded file.
    sha256:
        Hex-encoded SHA-256 digest of the file bytes (computed before loading).
    root_keys:
        Top-level MATLAB variable names (metadata keys like ``__header__``
        excluded).
    measurements:
        Dict mapping field name → 1-D numpy array.  Contains every numeric or
        object-array field found inside the ``meas`` struct.
    loader_backend:
        ``"scipy"`` or ``"h5py"`` (the latter requires optional h5py install).
    warnings:
        Non-fatal messages encountered during loading (e.g. unexpected field
        shapes, missing optional fields).
    """

    path: Path
    sha256: str
    root_keys: list[str]
    measurements: dict[str, np.ndarray]
    loader_backend: str
    warnings: list[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        fields = sorted(self.measurements)
        return (
            f"MatLoadResult(path={self.path.name!r}, "
            f"backend={self.loader_backend!r}, "
            f"sha256={self.sha256[:8]}…, "
            f"fields={fields})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256_of_file(path: Path) -> str:
    """Return hex SHA-256 of *path* contents, reading in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_1d_array(value: Any) -> np.ndarray:
    """Convert *value* to a one-dimensional numpy array.

    Handles MATLAB column vectors (n,1), row vectors (1,n), and scalar-wrapped
    arrays.  Object arrays (e.g. TimeStamp strings) are preserved as dtype=object.
    """
    arr = np.asarray(value)
    return arr.ravel()


def _extract_meas_dict(meas_raw: Any, path: Path, root_keys: list[str]) -> dict[str, Any]:
    """Normalise *meas_raw* (as returned by scipy) to a plain Python dict.

    scipy with ``simplify_cells=True`` returns structs as dicts.  Without it
    (or for older/unusual layouts), it may return a numpy void / record array
    or a 1×1 object array wrapping a dict.  This function handles all cases.
    """
    # Fast path: already a dict
    if isinstance(meas_raw, dict):
        return meas_raw

    # numpy void / structured array (dtype has named fields) → dict
    if isinstance(meas_raw, np.ndarray) and meas_raw.dtype.names:
        squeezed = meas_raw.squeeze()
        if squeezed.ndim == 0:
            # 0-d structured array — item() gives a numpy.void
            item = squeezed.item()
            if hasattr(item, "dtype") and item.dtype.names:
                return {n: item[n] for n in item.dtype.names}
        elif squeezed.ndim == 1 and squeezed.size == 1:
            item = squeezed[0]
            return {n: item[n] for n in squeezed.dtype.names}
        # Multi-element struct array: return first element
        item = squeezed.flat[0]
        if hasattr(item, "dtype") and hasattr(item.dtype, "names") and item.dtype.names:
            return {n: item[n] for n in item.dtype.names}

    # 1-element object array wrapping something else (recurse once)
    if isinstance(meas_raw, np.ndarray) and meas_raw.size == 1:
        inner = meas_raw.flat[0]
        if not isinstance(inner, np.ndarray) or inner.dtype.names:
            return _extract_meas_dict(inner, path, root_keys)

    raise MissingMeasurementStructError(path, root_keys)


def _load_with_scipy(path: Path) -> tuple[dict[str, Any], str]:
    """Load *path* with scipy.  Returns ``(mat_dict, backend_name)``.

    Raises
    ------
    UnsupportedMatFormatError
        When scipy detects a MATLAB v7.3 / HDF5 file.
    MatFileError
        On any other scipy load error.
    """
    try:
        mat = scipy.io.loadmat(str(path), simplify_cells=True)
        return mat, "scipy"
    except NotImplementedError as exc:
        # scipy raises NotImplementedError for HDF5 / v7.3
        if "HDF" in str(exc) or "7.3" in str(exc) or "hdf" in str(exc).lower():
            # Try h5py if available
            try:
                return _load_with_h5py(path)
            except ImportError:
                raise UnsupportedMatFormatError(path) from exc
        raise MatFileError(f"scipy failed to load {path}: {exc}") from exc
    except Exception as exc:
        raise MatFileError(f"Failed to load {path}: {exc}") from exc


def _load_with_h5py(path: Path) -> tuple[dict[str, Any], str]:
    """Load a MATLAB v7.3 / HDF5 file using h5py (optional dependency).

    Raises
    ------
    ImportError
        When h5py is not installed.
    MatFileError
        On read errors.
    """
    import h5py  # optional; raises ImportError if not installed

    try:
        result: dict[str, Any] = {}
        with h5py.File(path, "r") as f:
            root_keys = list(f.keys())
            result["__root_keys__"] = root_keys
            if "meas" not in f:
                return result, "h5py"
            meas_group = f["meas"]
            meas_dict: dict[str, Any] = {}
            for key in meas_group.keys():
                ds = meas_group[key]
                if hasattr(ds, "__iter__") or hasattr(ds, "shape"):
                    try:
                        meas_dict[key] = np.array(ds)
                    except Exception:
                        pass  # skip unreadable datasets
            result["meas"] = meas_dict
        return result, "h5py"
    except Exception as exc:
        raise MatFileError(f"h5py failed to load {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def load_mat_file(path: str | Path) -> MatLoadResult:
    """Load a MATLAB .mat file and return raw measurements from its ``meas`` struct.

    Parameters
    ----------
    path:
        Path to the ``.mat`` file (string or ``pathlib.Path``).

    Returns
    -------
    MatLoadResult
        Contains SHA-256, root keys, measurement arrays (all 1-D), backend
        name, and any non-fatal warnings.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    MatFileError
        If the file extension is not ``.mat``, or if loading fails for any
        other reason.
    MissingMeasurementStructError
        If no top-level ``meas`` variable is found in the file.
    UnsupportedMatFormatError
        If the file is MATLAB v7.3 / HDF5 and h5py is not installed.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    if path.suffix.lower() != ".mat":
        raise MatFileError(
            f"Expected a .mat file, got extension {path.suffix!r}: {path}"
        )

    # Compute SHA-256 before loading (proves we never write to the file)
    sha256 = _sha256_of_file(path)

    # Load raw MATLAB data
    mat, backend = _load_with_scipy(path)

    # Derive root keys (filter MATLAB metadata keys)
    root_keys = sorted(k for k in mat if not k.startswith("__"))

    # Locate and normalise the meas struct
    if "meas" not in mat:
        raise MissingMeasurementStructError(path, root_keys)

    load_warnings: list[str] = []
    meas_dict = _extract_meas_dict(mat["meas"], path, root_keys)

    # Convert every field to a 1-D numpy array
    measurements: dict[str, np.ndarray] = {}
    for key, value in meas_dict.items():
        try:
            arr = _to_1d_array(value)
            measurements[key] = arr
        except Exception as exc:  # noqa: BLE001
            load_warnings.append(
                f"Field {key!r} could not be converted to ndarray and was skipped: {exc}"
            )

    # Warn about missing expected fields (informational only — not required at this layer)
    for fname in _KNOWN_FIELDS:
        if fname not in measurements:
            load_warnings.append(f"Optional field {fname!r} not found in meas struct.")

    return MatLoadResult(
        path=path,
        sha256=sha256,
        root_keys=root_keys,
        measurements=measurements,
        loader_backend=backend,
        warnings=load_warnings,
    )
