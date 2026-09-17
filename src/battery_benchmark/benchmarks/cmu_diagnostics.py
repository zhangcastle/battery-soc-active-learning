"""Shared, read-only CMU eVTOL diagnostic helpers for benchmark runners.

This module factors out the Ns -> mission_phase vocabulary and the
source_row_right -> source_row raw-parquet join pattern that
``scripts/benchmarks/run_bian_cmu.py`` (DTNN) proved out first, so that
TATN/SFTNN/DARM/backbone runners can reuse an *identical* mapping and join
implementation instead of each reinventing (and potentially drifting from)
their own copy.

Deliberately NOT used to touch or refactor Vidal's (``run_cmu_transfer.py``)
or DTNN's (``run_bian_cmu.py``) own existing, frozen copies of this same
logic -- both already have their own working, tested versions and changing
them to "call the shared helper instead" would be a pure code-tidiness
change with zero user-visible benefit and non-zero risk to already-frozen,
paper-reproducing pipelines. This module exists only for the *remaining*
runners (SFTNN/MED, DARM, the three-backbone runner) that do not yet have
this diagnostic join at all.

Every function here is read-only / non-feedback: nothing in this module
touches model weights, optimizers, training data, or RNG state. It only
reads a raw CMU parquet's diagnostic columns and joins them onto whatever
source_row_right array a caller already computed while building its own
test windows.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

# Ns -> mission-phase label, identical across every CMU benchmark runner
# that adopts this shared module (Section II.C of the unified diagnostics
# spec explicitly requires one vocabulary, not a per-framework variant).
# Any Ns value not present as a key here maps to None via .map(), never
# raises, never invents a label.
NS_MISSION_PHASE_MAP: dict[int, str] = {
    0: "cc_charge",
    1: "cv_charge",
    2: "other_segment",
    3: "pre_mission_rest",
    4: "takeoff",
    5: "cruise",
    6: "landing",
    7: "post_flight_rest",
}
# Flight-phase SOC analysis (notebook/reporting) should focus on these three.
FLIGHT_MISSION_PHASES: tuple[str, ...] = ("takeoff", "cruise", "landing")

# CMU core-seven cell -> human-readable condition label. Identical across
# every CMU benchmark runner (DARM already had its own private copy of this
# exact mapping; SFTNN/backbone previously had none at all) -- kept here as
# a single shared source of truth so a "condition" column means the same
# thing regardless of which runner produced it.
CELL_CONDITIONS: dict[str, str] = {
    "VAH01": "baseline protocol",
    "VAH17": "baseline protocol",
    "VAH27": "baseline protocol",
    "VAH30": "35 °C chamber",
    "VAH15": "extended 1000 s cruise",
    "VAH12": "short 400 s cruise",
    "VAH11": "20% lower discharge power",
}

# Raw CMU parquet column name -> test_predictions.parquet output field name.
# voltage_v/current_a/Temperature__C are read straight from the raw parquet
# (never from resampled/normalized model input) and renamed to *_raw so
# they can't be mistaken for model-input features. Identical to DTNN's
# RAW_DIAGNOSTIC_SOURCE_COLUMNS (kept here as a single shared source of
# truth for every OTHER runner that adopts this module).
RAW_DIAGNOSTIC_SOURCE_COLUMNS: dict[str, str] = {
    "voltage_v": "voltage_v_raw",
    "current_a": "current_a_raw",
    "Temperature__C": "temperature_c_raw",
    "capacity_ah": "capacity_ah",
    "capacity_source": "capacity_source",
    "source_rpt_id": "source_rpt_id",
    # Raw tester-native Ns, untouched -- mission_phase is derived FROM this
    # column (after the rename below), never the reverse.
    "Ns": "ns_raw",
}
DIAGNOSTIC_METADATA_COLUMNS: tuple[str, ...] = tuple(
    RAW_DIAGNOSTIC_SOURCE_COLUMNS.values()
) + ("mission_phase",)

# How a caller should cast each joined diagnostic value when building a
# prediction record, applied only to a non-None value -- a real null from
# the join must be passed through as None, never cast (never "nan").
DIAGNOSTIC_METADATA_CASTERS: dict[str, type] = {
    "voltage_v_raw": float,
    "current_a_raw": float,
    "temperature_c_raw": float,
    "capacity_ah": float,
    "capacity_source": str,
    "source_rpt_id": str,
    "ns_raw": int,
    "mission_phase": str,
}


def load_raw_diagnostic_fields(path: Path) -> pd.DataFrame:
    """One per-cell read of the raw CMU parquet's diagnostic columns (plus
    ``source_row``, the join key), read directly from ``path`` -- the same
    raw parquet already hash-verified against the manifest for this cell --
    never from resampled or normalized data.

    ``mission_phase`` is added as a purely derived column computed from the
    just-renamed ``ns_raw`` via ``NS_MISSION_PHASE_MAP`` (vectorized
    ``.map()``): an unmapped Ns value becomes a real ``None``, never a
    fabricated label. ``ns_raw`` itself is kept as-is alongside it.
    """
    frame = pd.read_parquet(path, columns=["source_row", *RAW_DIAGNOSTIC_SOURCE_COLUMNS])
    if not frame["source_row"].is_unique:
        raise ValueError(f"{path}: source_row must be unique for a diagnostic join")
    frame = frame.rename(columns=RAW_DIAGNOSTIC_SOURCE_COLUMNS)
    frame["mission_phase"] = frame["ns_raw"].map(NS_MISSION_PHASE_MAP)
    return frame


def join_diagnostic_fields_by_source_row(source_row_right, raw_diagnostic: pd.DataFrame):
    """Vectorized (pandas ``DataFrame.merge``) join of
    ``DIAGNOSTIC_METADATA_COLUMNS`` onto every prediction referenced in
    ``source_row_right`` (any shape), keyed by source_row_right -> the raw
    parquet's own ``source_row``. One per-cell ``merge`` call, never a
    Python dict accumulated across timesteps/cells/the whole experiment.

    A ``source_row_right`` with no matching row in ``raw_diagnostic``
    yields real nulls (``None``) for that entry's diagnostic fields,
    handled defensively via a vectorized boolean-mask assignment (never
    ``.astype(str)``, which would turn a missing value into the literal
    string ``"nan"``).
    """
    shape = source_row_right.shape
    left = pd.DataFrame({"source_row": source_row_right.reshape(-1)})
    merged = left.merge(
        raw_diagnostic[["source_row", *DIAGNOSTIC_METADATA_COLUMNS]],
        on="source_row",
        how="left",
        sort=False,
    )
    if len(merged) != len(left):
        raise ValueError("source_row_right diagnostic join changed row count")
    result = {}
    for column in DIAGNOSTIC_METADATA_COLUMNS:
        values = merged[column].to_numpy(dtype=object).copy()
        values[pd.isna(values)] = None
        result[column] = values.reshape(shape)
    return result


def get_full_commit_hash(repo_path: str | Path | None = None) -> str:
    """Duplicated (not imported) from ``battery_benchmark.utils.git_info``
    on purpose: kept here too so callers of this diagnostics module have a
    single import surface, while the canonical implementation used
    elsewhere in the repo is untouched. Returns ``"unknown"`` if git is
    unavailable."""
    cwd = str(repo_path) if repo_path is not None else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def is_dirty(repo_path: str | Path | None = None) -> bool:
    cwd = str(repo_path) if repo_path is not None else None
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
