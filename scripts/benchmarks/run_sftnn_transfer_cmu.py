#!/usr/bin/env python3
"""Run the Shen et al. (2023) MED source-free method on the CMU eVTOL target
domain (Panasonic 25C source -> CMU eVTOL core-seven cells).

This is the natural "zero_shot" comparison point for
``run_backbone_transfer_cmu.py``: MED never sees any CMU labels during
adaptation (no real target SOC, no source replay), so there is no notion of
a label budget here the way the backbone-transfer sweep has one. Per cell,
per seed, this script reports exactly two (optionally three) rows:

* ``zero_shot``    -- the frozen Panasonic source model applied directly to
                       the CMU target domain with no adaptation at all.
* ``med``          -- Algorithm 1's source-free transfer: the target feature
                       extractor is adapted using only pseudo-labels drawn
                       from the frozen source model's own averaged-head
                       output on CMU's *unlabeled* train-split missions.
                       Real CMU labels are used only afterwards, to score
                       the held-out test-split missions.
* ``dann_mmd``     -- optional (``--include-dann-mmd-baseline``): the
                       paper's own domain-adversarial + MMD comparison
                       baseline, run the same way.

Like ``run_backbone_transfer_cmu.py``, this script does not import
``battery_benchmark.reproductions.vidal_2019`` case code directly and does
not touch the already-published, formal-status Vidal LSTM results. CMU
``soc_dynamic`` is a development dynamic-capacity reference, not measured
ground truth, so every row this script writes is
``label_is_ground_truth=false`` / ``formal_benchmark_eligible=false`` --
identical convention to the backbone-transfer benchmark.

Pseudo-label confidence filtering (``build_pseudo_target_dataset``) runs
per CMU *mission cycle*, exactly like it runs per Panasonic *profile* in the
in-paper suite -- this avoids the cross-file-boundary artifact identified in
the collaborator's delivered reference code, where the same filter is
applied to samples flattened across unrelated drive-cycle files first.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from battery_benchmark.benchmarks.cmu_diagnostics import (
    CELL_CONDITIONS,
    DIAGNOSTIC_METADATA_CASTERS,
    join_diagnostic_fields_by_source_row,
    load_raw_diagnostic_fields,
)
from battery_benchmark.datasets.cmu_evtol.vidal_adapter import load_cmu_dynamic_soc_for_vidal
from battery_benchmark.evaluation.metrics import compute_metrics
from battery_benchmark.reproductions.shen_2023.data import (
    PROFILES,
    ShenCycle,
    load_temperature_cycles_and_range,
    windows,
)
from battery_benchmark.reproductions.shen_2023.model import MEDNet, DANNBaselineNet
from battery_benchmark.reproductions.shen_2023.train import (
    adapt_dann_mmd as _adapt_dann_mmd,
    build_pseudo_target_dataset,
    load_med_source,
    load_source as load_dann_source,
    med_adapt,
    pretrain_med,
    predict,
    train_source as train_dann_source,
)
from battery_benchmark.utils.device import get_device
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty
from battery_benchmark.utils.hashing import sha256_file

CELLS = ("VAH01", "VAH17", "VAH27", "VAH30", "VAH15", "VAH12", "VAH11")
BASELINE_CELLS = {"VAH01", "VAH17", "VAH27"}
FEATURES = ("current", "voltage", "temperature")
WINDOW = 90  # fixed by the MED/TATN architecture (Conv2d/LSTM shapes in model.py)
METHOD_CHOICES = ("zero_shot", "med", "dann_mmd")

# `windows()` (shen_2023/data.py) builds non-overlapping WINDOW-length
# windows and predict()/`_evaluate` ravel BOTH the model's per-timestep
# output and the per-timestep label across every window -- i.e. this
# benchmark's real evaluation convention is all_timesteps (like DTNN/
# backbone), not last_step_per_window (unlike DARM). Recorded truthfully
# in run metadata rather than assumed.
EVALUATION_UNIT = "all_timesteps"

# Fixed schema for test_predictions.parquet -- one row per raw CMU
# timestep (not per window), mirroring run_bian_cmu.py's
# TEST_PREDICTIONS_SCHEMA. method distinguishes zero_shot/med/dann_mmd so
# their prediction-level diagnostics are never conflated.
TEST_PREDICTIONS_SCHEMA = pa.schema(
    [
        ("framework", pa.string()),
        ("method", pa.string()),
        ("seed", pa.int64()),
        ("cell_id", pa.string()),
        ("condition", pa.string()),
        ("split", pa.string()),
        ("mission_cycle_id", pa.string()),
        ("window_index", pa.int64()),
        ("timestep_index", pa.int64()),
        ("mission_sequence_order", pa.int64()),
        ("elapsed_candidate_s", pa.float64()),
        ("original_time_s", pa.float64()),
        ("source_row_right", pa.int64()),
        ("voltage_v_raw", pa.float64()),
        ("current_a_raw", pa.float64()),
        ("temperature_c_raw", pa.float64()),
        ("capacity_ah", pa.float64()),
        ("capacity_source", pa.string()),
        ("source_rpt_id", pa.string()),
        ("ns_raw", pa.int64()),
        ("mission_phase", pa.string()),
        ("soc_reference", pa.float64()),
        ("soc_prediction", pa.float64()),
        ("signed_error", pa.float64()),
        ("absolute_error", pa.float64()),
    ]
)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


class _Timer:
    """Wall-clock stopwatch. CUDA ops are async, so any timed block that ran
    on GPU is synced before stopping -- otherwise the measured duration would
    just be launch-overhead, not actual compute time."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def __enter__(self) -> "_Timer":
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self._start


def _check_resume_cmu_input_sha256(result_path: Path, expected_sha256: str) -> None:
    """Resume-safety check, run ONLY at the point where --resume is about to
    skip re-running a (cell, method) because ``result_path`` already exists
    -- never anywhere else, and never changes what a non-resumed run
    trains/predicts/evaluates.

    ``expected_sha256`` is this invocation's freshly-computed, already
    manifest-verified CMU input hash for this cell. The existing result
    JSON's own ``cmu_input_sha256`` must equal it exactly -- a missing
    field (an old result predating hash tracking) or a mismatch (a result
    produced from a different CMU data version, e.g. before the
    _ns_relabel migration) means this on-disk result cannot be trusted to
    represent the CURRENT data version, and must never be silently
    resumed, overwritten, or deleted. Raises immediately instead."""
    existing = json.loads(result_path.read_text(encoding="utf-8"))
    existing_sha256 = existing.get("cmu_input_sha256")
    if existing_sha256 is None:
        raise ValueError(
            f"{result_path}: existing result has no cmu_input_sha256 recorded -- "
            "cannot verify it was produced from the current CMU data version; "
            "refusing to --resume from it. Rerun without --resume, or use a "
            "fresh --output-root."
        )
    if existing_sha256 != expected_sha256:
        raise ValueError(
            f"{result_path}: existing result's cmu_input_sha256={existing_sha256!r} "
            f"does not match the current CMU data version's hash "
            f"{expected_sha256!r} -- refusing to --resume from a different CMU "
            "data version. Rerun without --resume, or use a fresh --output-root."
        )


def _check_resume_result_sidecars(
    *, result_path: Path, cell_root: Path, seed_root: Path, method: str
) -> None:
    """Resume-safety check, run at the SAME point as
    ``_check_resume_cmu_input_sha256`` and in ADDITION to it: verifies the
    sidecar files a complete result actually depends on are still present
    on disk, never trusting a result JSON alone. A prior run could have
    been interrupted after writing ``<method>.json`` but before its
    prediction/training-history sidecars finished, or (with
    ``--no-save-checkpoints``) its source checkpoint may legitimately be
    absent while its source-pretraining history sidecar still exists (or
    vice versa) -- in every such case, silently resuming would leave the
    aggregated cell_metrics.csv/test_predictions.parquet/
    training_history.csv looking complete while actually missing fields.
    Raises ValueError immediately instead of skipping, overwriting, or
    deleting anything."""
    existing = json.loads(result_path.read_text(encoding="utf-8"))
    if existing.get("pseudo_label_collapsed"):
        # A genuinely collapsed pseudo-label run (0 windows survived
        # confidence filtering) never produced a prediction/training-
        # history sidecar in the first place -- that is the CORRECT,
        # honestly-recorded outcome (see the `dataset` == 0 branch in
        # main()), not a missing artifact.
        return
    predictions_path = cell_root / f"{method}_predictions.parquet"
    if not predictions_path.is_file():
        raise ValueError(
            f"{result_path}: --resume would skip this row, but its "
            f"prediction sidecar {predictions_path} is missing. Rerun "
            "without --resume, or use a fresh --output-root."
        )
    if method in ("med", "dann_mmd"):
        history_path = cell_root / f"{method}_training_history.csv"
        if not history_path.is_file():
            raise ValueError(
                f"{result_path}: --resume would skip this row, but its "
                f"adaptation training-history sidecar {history_path} is "
                "missing. Rerun without --resume, or use a fresh "
                "--output-root."
            )
        source_name = "med_source" if method == "med" else "dann_mmd_source"
        source_history = seed_root / f"{source_name}_pretrain_training_history.csv"
        source_checkpoint = seed_root / (
            "med_source.pt" if method == "med" else "dann_source.pt"
        )
        if not source_history.is_file() and not source_checkpoint.is_file():
            raise ValueError(
                f"{result_path}: --resume would skip this row, but neither "
                f"its source-pretraining history {source_history} nor "
                f"checkpoint {source_checkpoint} is present -- this "
                "method's source pretraining cannot be verified. Rerun "
                "without --resume, or use a fresh --output-root."
            )


def _cell_path(cell: str, base_root: Path, condition_root: Path) -> Path:
    root = base_root if cell in BASELINE_CELLS else condition_root
    return root / f"{cell}_dynamic_soc.parquet"


def _mission_cycles(
    records, mission_ids: set[str], feature_min: np.ndarray, feature_max: np.ndarray
) -> list[ShenCycle]:
    """Build one ShenCycle per CMU mission, normalised with the Panasonic
    source-fitted min-max range (source-fitted normalisation applied
    unchanged to the target domain, same convention as the backbone-transfer
    benchmark)."""

    if not mission_ids:
        raise ValueError("mission_ids is empty")
    scale = np.where(feature_max == feature_min, 1.0, feature_max - feature_min)
    cycles = []
    for record in records:
        mission = str(record.metadata["mission_cycle_id"])
        if mission not in mission_ids:
            continue
        frame = record.data
        x = ((frame.loc[:, list(FEATURES)].to_numpy(dtype=np.float64) - feature_min) / scale).astype(
            np.float32
        )
        y = frame["soc"].to_numpy(dtype=np.float64).astype(np.float32)
        cycles.append(ShenCycle(profile=mission, temperature=None, features=x, soc=y, paths=()))
    if not cycles:
        raise ValueError("No CMU records matched the requested mission_ids")
    return cycles


def _mission_metadata(records, mission_ids: set[str]) -> dict[str, dict[str, np.ndarray]]:
    """Diagnostic-only companion to ``_mission_cycles``: for each mission in
    ``mission_ids``, reads mission_sequence_order/elapsed_candidate_s/
    original_time_s/source_row_right straight from ``record.data`` -- the
    exact same frame ``_mission_cycles`` reads ``FEATURES``/``soc`` from,
    in the same per-record iteration order -- so these full-length
    (untruncated) per-row arrays can later be truncated/reshaped by
    ``_windows_metadata`` with the IDENTICAL ``usable = len(...) // WINDOW
    * WINDOW`` logic ``windows()`` itself uses, keeping every value aligned
    to the exact SOC timestep it describes."""
    metadata = {}
    for record in records:
        mission = str(record.metadata["mission_cycle_id"])
        if mission not in mission_ids:
            continue
        frame = record.data
        metadata[mission] = {
            "mission_sequence_order": np.full(
                len(frame), int(record.metadata["sequence_order"]), dtype=np.int64
            ),
            "elapsed_candidate_s": frame["elapsed_candidate_s"].to_numpy(dtype=np.float64),
            "original_time_s": frame["original_time_s"].to_numpy(dtype=np.float64),
            "source_row_right": frame["source_row_right"].to_numpy(dtype=np.int64),
        }
    return metadata


def _windows_metadata(values: np.ndarray, length: int = WINDOW) -> np.ndarray:
    """Truncates/reshapes a full-length per-row metadata array with the
    SAME ``usable = len(values) // length * length`` logic ``windows()``
    (shen_2023/data.py) applies to features/soc for the same cycle."""
    usable = len(values) // length * length
    return values[:usable].reshape(-1, length)


@torch.no_grad()
def _predict_all_with_metadata(
    model,
    cycles: list[ShenCycle],
    metadata_by_mission: dict[str, dict[str, np.ndarray]],
    device: torch.device,
    branch: str = "target",
):
    """Diagnostic-only companion to ``predict``/``_evaluate``: for each
    cycle, builds x/y via the SAME ``windows([cycle], tensor_layout)`` call
    ``predict`` uses, runs the model with the IDENTICAL loader construction
    (batch_size=64, ``shuffle=False``), and also reshapes this mission's
    metadata with ``_windows_metadata`` -- so predictions, labels, and
    metadata all ravel into 1:1 aligned flat arrays, in the same per-cycle
    order ``_evaluate`` iterates. A test recomputes overall_mae_percent/
    overall_rmse_percent from these predictions/labels and asserts they
    match ``_evaluate``'s own metrics exactly (after the same [0, 1]
    clip ``_evaluate`` applies)."""
    model.eval()
    predictions, references, mission_ids, window_indices, timestep_indices = [], [], [], [], []
    metadata_out = {
        "mission_sequence_order": [],
        "elapsed_candidate_s": [],
        "original_time_s": [],
        "source_row_right": [],
    }
    for cycle in cycles:
        x, y = windows([cycle])
        if not len(x):
            continue
        n_windows = len(x)
        mission_metadata = metadata_by_mission[cycle.profile]
        windowed_metadata = {
            name: _windows_metadata(mission_metadata[name]) for name in metadata_out
        }
        cycle_predictions = []
        for (xb,) in _loader_no_labels(x, 64):
            output = (
                model.estimate_source(xb.to(device)) if branch == "source" else model(xb.to(device))
            )
            cycle_predictions.append(output.cpu().numpy())
        cycle_prediction = np.concatenate(cycle_predictions)
        predictions.append(cycle_prediction.ravel())
        references.append(y.ravel())
        mission_ids.extend([cycle.profile] * (n_windows * WINDOW))
        window_indices.extend(np.repeat(np.arange(n_windows), WINDOW))
        timestep_indices.extend(np.tile(np.arange(WINDOW), n_windows))
        for name in metadata_out:
            metadata_out[name].append(windowed_metadata[name].ravel())
    if not predictions:
        empty_float = np.empty((0,), dtype=np.float64)
        empty_int = np.empty((0,), dtype=np.int64)
        return (
            empty_float,
            empty_float,
            np.empty((0,), dtype=object),
            empty_int,
            empty_int,
            {
                "mission_sequence_order": empty_int,
                "elapsed_candidate_s": empty_float,
                "original_time_s": empty_float,
                "source_row_right": empty_int,
            },
        )
    prediction = np.concatenate(predictions).astype(np.float64)
    reference = np.concatenate(references).astype(np.float64)
    metadata_flat = {
        name: np.concatenate(metadata_out[name])
        for name in metadata_out
    }
    metadata_flat["mission_sequence_order"] = metadata_flat["mission_sequence_order"].astype(np.int64)
    metadata_flat["source_row_right"] = metadata_flat["source_row_right"].astype(np.int64)
    return (
        prediction,
        reference,
        np.array(mission_ids, dtype=object),
        np.array(window_indices, dtype=np.int64),
        np.array(timestep_indices, dtype=np.int64),
        metadata_flat,
    )


def _loader_no_labels(x: np.ndarray, batch_size: int):
    """Same TensorDataset/DataLoader construction as ``_loader`` in
    shen_2023/train.py, minus the label tensor -- used only so
    ``_predict_all_with_metadata`` can iterate x in the IDENTICAL
    batch order (``shuffle=False``) as ``predict``'s own loader over
    ``(x, y)``, without needing a second copy of y."""
    from torch.utils.data import DataLoader, TensorDataset

    return DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=min(batch_size, len(x)),
        shuffle=False,
    )


def _build_prediction_rows(
    *,
    framework: str,
    method: str,
    seed: int,
    cell_id: str,
    condition: str,
    soc_reference: np.ndarray,
    soc_prediction: np.ndarray,
    mission_ids: np.ndarray,
    window_indices: np.ndarray,
    timestep_indices: np.ndarray,
    metadata: dict[str, np.ndarray],
    raw_diagnostic: pd.DataFrame,
) -> list[dict]:
    """Assembles one test_predictions.parquet row per raw timestep,
    joining Ns/mission_phase/raw V-I-T/capacity provenance by this
    timestep's own source_row_right -- never reverse-derived from x."""
    if not len(soc_reference):
        return []
    joined = join_diagnostic_fields_by_source_row(metadata["source_row_right"], raw_diagnostic)
    clipped_prediction = np.clip(soc_prediction, 0.0, 1.0)
    rows = []
    n = len(soc_reference)
    for i in range(n):
        row = {
            "framework": framework,
            "method": method,
            "seed": seed,
            "cell_id": cell_id,
            "condition": condition,
            "split": "test",
            "mission_cycle_id": str(mission_ids[i]),
            "window_index": int(window_indices[i]),
            "timestep_index": int(timestep_indices[i]),
            "mission_sequence_order": int(metadata["mission_sequence_order"][i]),
            "elapsed_candidate_s": float(metadata["elapsed_candidate_s"][i]),
            "original_time_s": float(metadata["original_time_s"][i]),
            "source_row_right": int(metadata["source_row_right"][i]),
            "soc_reference": float(soc_reference[i]),
            "soc_prediction": float(clipped_prediction[i]),
            "signed_error": float(clipped_prediction[i] - soc_reference[i]),
            "absolute_error": float(abs(clipped_prediction[i] - soc_reference[i])),
        }
        for column in (
            "voltage_v_raw",
            "current_a_raw",
            "temperature_c_raw",
            "capacity_ah",
            "capacity_source",
            "source_rpt_id",
            "ns_raw",
            "mission_phase",
        ):
            value = joined[column][i]
            row[column] = None if value is None else DIAGNOSTIC_METADATA_CASTERS[column](value)
        rows.append(row)
    return rows


def _flatten_scalar_history(
    history: list[float],
    *,
    framework: str,
    method: str,
    seed: int,
    learning_rate: float,
    epoch_seconds: list[float] | None = None,
) -> list[dict]:
    """For pretrain_med/train_dann_source, whose returned ``history`` is
    just a per-epoch scalar loss (see shen_2023/train.py) -- exports it
    as-is, without retraining, and without inventing a per-epoch learning
    rate the underlying loop doesn't actually track (this constant is the
    optimizer's configured LR; pretrain_med/train_source have no
    scheduler here since early_stopping_patience drives a
    ReduceLROnPlateau only for the *adaptation* loops, not source
    pretraining -- see the checkpoint's own "learning_rate" field, which
    stays fixed for these two functions).

    ``epoch_seconds``, when supplied, is ``pretrain_med``'s/
    ``train_source``'s own real per-epoch wall-clock list (returned only
    when the caller passes ``record_epoch_seconds=True``), aligned
    index-for-index with ``history`` -- exported as-is, never re-measured
    or invented here. ``validation_mae``/``validation_rmse``/
    ``validation_evaluated`` stay None/False unconditionally for source
    pretraining: this is zero_shot/source-pretraining diagnostic history,
    which must never fabricate a validation trace (only MED/DANN+MMD
    *adaptation* records real validation diagnostics -- see
    ``_flatten_dict_history``)."""
    seconds = epoch_seconds if epoch_seconds is not None else [None] * len(history)
    return [
        {
            "framework": framework,
            "method": method,
            "seed": seed,
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "train_loss": value,
            "runtime_seconds": seconds[epoch],
            "validation_mae": None,
            "validation_rmse": None,
            "validation_evaluated": False,
        }
        for epoch, value in enumerate(history)
    ]


def _flatten_dict_history(
    history: list[dict], *, framework: str, method: str, seed: int, cell_id: str
) -> list[dict]:
    """For med_adapt/adapt_dann_mmd, whose returned ``history`` is a list
    of real per-epoch loss dicts (see shen_2023/train.py: {"soc","dis",
    "total","learning_rate"} for MED, {"estimate","adversarial","mmd",
    "total","learning_rate"} for DANN+MMD) -- exports every key genuinely
    present, never inventing a field the code doesn't compute, and never
    conflating MED's field names with DANN+MMD's.

    When the caller passed ``record_epoch_seconds=True``/
    ``validation_cycles=...`` into ``med_adapt``/``adapt_dann_mmd``, each
    ``entry`` dict already carries real ``runtime_seconds``/
    ``validation_mae``/``validation_rmse``/``validation_evaluated`` values
    (see ``_validation_metrics`` in shen_2023/train.py) -- ``row.update
    (entry)`` below picks those up automatically, overriding the
    placeholder defaults set first. If the caller omitted those flags,
    ``entry`` simply doesn't contain those keys and the None/False
    defaults survive untouched -- this function needs no branching on
    which mode was requested."""
    rows = []
    for epoch, entry in enumerate(history):
        row = {
            "framework": framework,
            "method": method,
            "seed": seed,
            "cell_id": cell_id,
            "epoch": epoch + 1,
            "runtime_seconds": None,
            "validation_mae": None,
            "validation_rmse": None,
            "validation_evaluated": False,
        }
        row.update(entry)
        rows.append(row)
    return rows


def _budget_mission_ids(
    manifest: pd.DataFrame, cell_id: str
) -> tuple[set[str], set[str], set[str]]:
    """Every train-split mission is unlabeled adaptation data for MED (no
    label budget applies -- MED never uses target labels for training).

    ``validation_ids`` (the manifest's EXISTING ``split == "validation"``
    missions) is diagnostic-only: read-only labels used exclusively by the
    ``_validation_metrics`` probe wired into ``med_adapt``/
    ``adapt_dann_mmd`` below (see ``validation_cycles=``), never for
    training, pseudo-labeling, the scheduler, early stopping, checkpoint
    selection, or model selection, and never mixed into ``train_ids``/
    ``test_ids``. An empty validation set for a cell is not an error --
    the diagnostic probe is simply skipped for that cell (unlike
    ``train_ids``, which every cell must have)."""

    rows = manifest.loc[manifest["cell_id"] == cell_id]
    if rows.empty:
        raise ValueError(f"{cell_id}: no manifest rows")
    train_ids = set(rows.loc[rows["split"] == "train", "mission_cycle_id"].astype(str))
    test_ids = set(rows.loc[rows["split"] == "test", "mission_cycle_id"].astype(str))
    validation_ids = set(
        rows.loc[rows["split"] == "validation", "mission_cycle_id"].astype(str)
    )
    if not train_ids:
        raise ValueError(f"{cell_id}: empty train-split mission set")
    return train_ids, test_ids, validation_ids


@torch.no_grad()
def _evaluate(model, cycles: list[ShenCycle], device: torch.device, branch: str = "target") -> dict[str, float]:
    truth, estimate = predict(model, cycles, device, branch=branch)
    estimate = np.clip(estimate, 0.0, 1.0)
    metrics = compute_metrics(truth, estimate)
    per_mission = [
        compute_metrics(*predict(model, [cycle], device, branch=branch)) for cycle in cycles
    ]
    return {
        "overall_mae_percent": metrics["mae"] * 100,
        "overall_rmse_percent": metrics["rmse"] * 100,
        "overall_max_error_percent": metrics["max_error"] * 100,
        "overall_bias_percent": metrics["bias"] * 100,
        "overall_r2": metrics["r2"],
        "mission_macro_mae_percent": float(np.mean([m["mae"] for m in per_mission]) * 100),
        "mission_macro_rmse_percent": float(np.mean([m["rmse"] for m in per_mission]) * 100),
        "missions": len(cycles),
        "windows": sum(len(windows([c])[0]) for c in cycles),
    }


# Canonical Ns -> mission-phase vocabulary this benchmark's phase-level
# summaries cover -- matches the canonical Ns mapping recorded in
# run_metadata.json's "phase_definition" field. Rows whose mission_phase
# is None (unmapped Ns) or any other phase are excluded from both
# summaries below.
PHASE_SUMMARY_PHASES = ("take_off", "cruise", "landing")


def _compute_phase_metrics(predictions_path: Path) -> pd.DataFrame:
    """Diagnostic-only summary, built by READING BACK the final, top-level
    (post-merge) test_predictions.parquet -- never re-derived from live
    model state, never a second evaluation set -- grouped by cell_id/
    condition/method/mission_phase and restricted to
    PHASE_SUMMARY_PHASES (take_off/cruise/landing). Every value here is
    directly, exactly recomputable from that same parquet's
    soc_reference/soc_prediction/mission_phase columns, mirroring
    run_liu_cmu_darm.py's ``_compute_phase_metrics``."""
    frame = pd.read_parquet(
        predictions_path,
        columns=[
            "cell_id", "condition", "method", "mission_phase",
            "mission_cycle_id", "soc_reference", "soc_prediction",
        ],
    )
    frame = frame.loc[frame["mission_phase"].isin(PHASE_SUMMARY_PHASES)].copy()
    frame["signed_error"] = frame["soc_prediction"] - frame["soc_reference"]
    frame["absolute_error"] = frame["signed_error"].abs()
    grouped = frame.groupby(
        ["cell_id", "condition", "method", "mission_phase"], sort=False, as_index=False
    )
    summary = grouped.agg(
        n_points=("soc_reference", "size"),
        n_missions=("mission_cycle_id", "nunique"),
        mae_percent=("absolute_error", lambda s: float(s.mean()) * 100),
        rmse_percent=("signed_error", lambda s: float(np.sqrt((s**2).mean())) * 100),
        mean_signed_error_percent=("signed_error", lambda s: float(s.mean()) * 100),
    )
    return summary


def _compute_phase_transfer_effects(phase_metrics: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic-only: pairs each (cell_id, condition, mission_phase)'s
    ``zero_shot`` row from ``phase_metrics`` (see ``_compute_phase_metrics``)
    against ``med``'s and, separately, ``dann_mmd``'s row -- one merge per
    adapted method, never mixing the two -- and reports the
    adapted-minus-zero_shot deltas plus a boolean negative-transfer flag
    for both MAE and RMSE. ``reference_method`` is always "zero_shot".
    delta = adapted - zero_shot, so a POSITIVE delta means adaptation made
    this (cell, condition, mission_phase) worse than zero-shot, i.e.
    negative transfer. A (cell_id, condition, mission_phase) group missing
    either method is simply dropped (inner merge) rather than producing a
    bogus delta."""
    columns = [
        "cell_id", "condition", "method", "reference_method", "mission_phase",
        "zero_shot_mae_percent", "adapted_mae_percent", "delta_mae_percent",
        "zero_shot_rmse_percent", "adapted_rmse_percent", "delta_rmse_percent",
        "negative_transfer_mae", "negative_transfer_rmse",
    ]
    if phase_metrics.empty:
        return pd.DataFrame(columns=columns)
    zero_shot = phase_metrics.loc[phase_metrics["method"] == "zero_shot"]
    frames = []
    for adapted_method in ("med", "dann_mmd"):
        adapted = phase_metrics.loc[phase_metrics["method"] == adapted_method]
        if zero_shot.empty or adapted.empty:
            continue
        merged = zero_shot.merge(
            adapted,
            on=["cell_id", "condition", "mission_phase"],
            suffixes=("_zero_shot", "_adapted"),
            how="inner",
        )
        if merged.empty:
            continue
        merged["method"] = adapted_method
        merged["reference_method"] = "zero_shot"
        merged["zero_shot_mae_percent"] = merged["mae_percent_zero_shot"]
        merged["adapted_mae_percent"] = merged["mae_percent_adapted"]
        merged["delta_mae_percent"] = (
            merged["mae_percent_adapted"] - merged["mae_percent_zero_shot"]
        )
        merged["zero_shot_rmse_percent"] = merged["rmse_percent_zero_shot"]
        merged["adapted_rmse_percent"] = merged["rmse_percent_adapted"]
        merged["delta_rmse_percent"] = (
            merged["rmse_percent_adapted"] - merged["rmse_percent_zero_shot"]
        )
        merged["negative_transfer_mae"] = merged["delta_mae_percent"] > 0
        merged["negative_transfer_rmse"] = merged["delta_rmse_percent"] > 0
        frames.append(merged[columns])
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panasonic-data-root", type=Path, default=REPO_ROOT / "data/raw/Panasonic_18650PF"
    )
    parser.add_argument(
        "--cmu-base-root", type=Path,
        default=REPO_ROOT / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1_ns_relabel",
    )
    parser.add_argument(
        "--cmu-condition-root", type=Path,
        default=REPO_ROOT
        / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1_extension4_ns_relabel",
    )
    parser.add_argument(
        "--cmu-manifest", type=Path,
        default=REPO_ROOT
        / "data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1_ns_relabel.csv",
    )
    parser.add_argument(
        "--output-root", type=Path,
        # Isolated from BOTH the pre-_ns_relabel results (previously
        # ".../sftnn_med_transfer") AND from the pre-diagnostics
        # "_ns_relabel_v1" results (which predate per-epoch runtime,
        # diagnostic-only validation, and phase-level summaries) -- a
        # --resume against this new "_v2" directory can never
        # accidentally pick up a result produced before these diagnostics
        # existed, since v1 is never written to or read from by this
        # default. v1 itself is left exactly as it was: never deleted,
        # overwritten, or migrated.
        default=REPO_ROOT / "outputs/benchmarks/panasonic_to_cmu_evtol/sftnn_med_transfer_ns_relabel_v2",
    )
    parser.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS))
    parser.add_argument("--methods", nargs="+", choices=METHOD_CHOICES, default=["zero_shot", "med"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--source-epochs", type=int, default=2000)
    parser.add_argument("--transfer-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--transfer-lr", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument(
        "--validation-interval", type=int, default=1,
        help="run the diagnostic-only validation probe (manifest 'validation'-"
        "split missions) every N adaptation epochs, for MED and DANN+MMD "
        "adaptation only -- never zero-shot or source pretraining. Read-only: "
        "never affects training, pseudo-labeling, the scheduler, early "
        "stopping, checkpoint, or model selection.",
    )
    parser.add_argument("--pseudo-threshold", type=float, default=0.08)
    parser.add_argument("--pseudo-lookahead", type=int, default=200)
    parser.add_argument("--med-lambda-soc", type=float, default=1.0)
    parser.add_argument("--med-lambda-dis", type=float, default=0.1)
    parser.add_argument(
        "--dann-lambdas", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        metavar=("ESTIMATE", "ADVERSARIAL", "MMD"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume", action="store_true",
        help="skip source pretraining and (method, seed, cell) evaluations that "
        "already have a result file on disk",
    )
    parser.add_argument(
        "--no-save-checkpoints", action="store_true",
        help="do not write model checkpoints; result JSON rows are always written",
    )
    return parser.parse_args()


def _save_timing(seed_root: Path, name: str, *, seconds: float | None, epochs: int | None, device: str) -> None:
    """Persist a pretraining timing record so it survives --resume: if this
    run resumed from a checkpoint instead of retraining, we still want the
    original run's timing available for the rows written this time."""

    seed_root.mkdir(parents=True, exist_ok=True)
    path = seed_root / f"{name}_timing.json"
    if seconds is None:
        return  # nothing new measured this run; leave any existing file alone
    path.write_text(
        json.dumps({"seconds": seconds, "epochs": epochs, "device": device}, indent=2),
        encoding="utf-8",
    )


def _load_timing(seed_root: Path, name: str) -> dict:
    path = seed_root / f"{name}_timing.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"seconds": None, "epochs": None, "device": None}


def _rebuild_cell_metrics(output_root: Path) -> pd.DataFrame:
    """Aggregate every per-(method, seed, cell) result JSON under output_root,
    the same merge-not-overwrite convention run_backbone_transfer_cmu.py
    uses for cell_metrics.csv."""

    all_rows: list[dict] = []
    for result_path in sorted(output_root.glob("seed_*/*/*.json")):
        all_rows.append(json.loads(result_path.read_text(encoding="utf-8")))
    table = pd.DataFrame(all_rows)
    table.to_csv(output_root / "cell_metrics.csv", index=False)
    return table


def _rebuild_test_predictions(output_root: Path) -> None:
    """Aggregate every per-(seed, cell, method) test_predictions parquet
    sidecar (``seed_*/<cell>/<method>_predictions.parquet``) into the
    top-level test_predictions.parquet -- never a single truncate-on-open
    writer for the whole run, same merge-not-overwrite rationale as
    ``_rebuild_cell_metrics``."""
    paths = sorted(output_root.glob("seed_*/*/*_predictions.parquet"))
    if not paths:
        return
    pq.write_table(pa.concat_tables([pq.read_table(path) for path in paths]), output_root / "test_predictions.parquet")


def _rebuild_training_history(output_root: Path) -> None:
    """Aggregate every per-(seed[, cell]) training-history sidecar (source
    pretraining under ``seed_*/*_training_history.csv``, adaptation under
    ``seed_*/<cell>/<method>_training_history.csv``) into the top-level
    training_history.csv."""
    paths = sorted(output_root.glob("seed_*/*_training_history.csv")) + sorted(
        output_root.glob("seed_*/*/*_training_history.csv")
    )
    if not paths:
        return
    frames = [pd.read_csv(path) for path in paths]
    pd.concat(frames, ignore_index=True).to_csv(output_root / "training_history.csv", index=False)


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    device = get_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    print("loading Panasonic 25C source cycles (MED source-free pretraining) ...")
    source_cycles, feature_min, feature_max = load_temperature_cycles_and_range(
        args.panasonic_data_root, 25
    )
    manifest = pd.read_csv(args.cmu_manifest)

    for seed in args.seeds:
        seed_root = args.output_root / f"seed_{seed}"
        med_checkpoint = seed_root / "med_source.pt"
        dann_checkpoint = seed_root / "dann_source.pt"

        med_pretrain_seconds = None
        med_pretrain_epochs = None
        _seed(seed)
        if args.resume and med_checkpoint.is_file():
            med_source = load_med_source(med_checkpoint, device)
            print(f"=== seed {seed}: resumed MED source checkpoint from {med_checkpoint} ===")
        else:
            print(f"=== seed {seed}: MED source pretraining, max {args.source_epochs} epochs ===")
            with _Timer(device) as timer:
                med_source, med_pretrain_history, med_pretrain_epoch_seconds = pretrain_med(
                    source_cycles,
                    epochs=args.source_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    device=device,
                    seed=seed,
                    checkpoint=None if args.no_save_checkpoints else med_checkpoint,
                    early_stopping_patience=args.early_stopping_patience,
                    record_epoch_seconds=True,
                )
            med_pretrain_seconds = timer.seconds
            med_pretrain_epochs = len(med_pretrain_history)
            print(f"    done in {med_pretrain_seconds:.1f}s ({med_pretrain_epochs} epochs)")
            # Diagnostic-only: exports pretrain_med's own returned
            # per-epoch scalar loss history (and real per-epoch
            # runtime_seconds, from record_epoch_seconds=True above) --
            # never retrains, never a second call to pretrain_med.
            seed_root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                _flatten_scalar_history(
                    med_pretrain_history,
                    framework="sftnn_med",
                    method="med_source_pretrain",
                    seed=seed,
                    learning_rate=args.lr,
                    epoch_seconds=med_pretrain_epoch_seconds,
                )
            ).to_csv(seed_root / "med_source_pretrain_training_history.csv", index=False)
        _save_timing(
            seed_root, "med_source_pretrain",
            seconds=med_pretrain_seconds, epochs=med_pretrain_epochs, device=str(device),
        )
        med_pretrain_timing = _load_timing(seed_root, "med_source_pretrain")

        dann_source = None
        dann_pretrain_seconds = None
        dann_pretrain_epochs = None
        if "dann_mmd" in args.methods:
            if args.resume and dann_checkpoint.is_file():
                dann_source = load_dann_source(dann_checkpoint, device)
                print(f"=== seed {seed}: resumed DANN+MMD source checkpoint from {dann_checkpoint} ===")
            else:
                print(f"=== seed {seed}: DANN+MMD source pretraining, max {args.source_epochs} epochs ===")
                with _Timer(device) as timer:
                    dann_source, dann_pretrain_history, dann_pretrain_epoch_seconds = train_dann_source(
                        source_cycles,
                        epochs=args.source_epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.lr,
                        device=device,
                        seed=seed,
                        checkpoint=None if args.no_save_checkpoints else dann_checkpoint,
                        early_stopping_patience=args.early_stopping_patience,
                        record_epoch_seconds=True,
                    )
                dann_pretrain_seconds = timer.seconds
                dann_pretrain_epochs = len(dann_pretrain_history)
                print(f"    done in {dann_pretrain_seconds:.1f}s ({dann_pretrain_epochs} epochs)")
                seed_root.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    _flatten_scalar_history(
                        dann_pretrain_history,
                        framework="sftnn_med",
                        method="dann_mmd_source_pretrain",
                        seed=seed,
                        learning_rate=args.lr,
                        epoch_seconds=dann_pretrain_epoch_seconds,
                    )
                ).to_csv(seed_root / "dann_mmd_source_pretrain_training_history.csv", index=False)
            _save_timing(
                seed_root, "dann_source_pretrain",
                seconds=dann_pretrain_seconds, epochs=dann_pretrain_epochs, device=str(device),
            )
        dann_pretrain_timing = _load_timing(seed_root, "dann_source_pretrain")

        for cell in args.cells:
            cell_root = seed_root / cell
            path = _cell_path(cell, args.cmu_base_root, args.cmu_condition_root)
            expected = set(
                manifest.loc[manifest["cell_id"] == cell, "source_artifact_sha256"].astype(str)
            )
            actual = sha256_file(path)
            if expected != {actual}:
                raise ValueError(f"{cell}: CMU input hash does not match manifest")
            records = load_cmu_dynamic_soc_for_vidal([path], target_hz=1.0).records
            train_ids, test_ids, validation_ids = _budget_mission_ids(manifest, cell)
            test_cycles = _mission_cycles(records, test_ids, feature_min, feature_max)
            train_cycles = _mission_cycles(records, train_ids, feature_min, feature_max)
            # Diagnostic-only, read-only validation probe input: the
            # manifest's EXISTING "validation"-split missions for this
            # cell (never used for training/pseudo-labeling/early-
            # stopping/checkpoint selection -- see med_adapt's/
            # adapt_dann_mmd's validation_cycles= docstring). An empty
            # validation_ids set for this cell is not an error; the probe
            # is simply skipped (validation_cycles left as None) and
            # validation_evaluated stays honestly False for every epoch.
            validation_cycles = (
                _mission_cycles(records, validation_ids, feature_min, feature_max)
                if validation_ids
                else None
            )
            # Diagnostic-only: per-test-mission metadata (for
            # test_predictions.parquet) and the raw-parquet Ns/V/I/T/
            # capacity join table for this cell, built once per cell and
            # reused across every method below.
            test_metadata = _mission_metadata(records, test_ids)
            raw_diagnostic = load_raw_diagnostic_fields(path)

            for method in args.methods:
                result_path = cell_root / f"{method}.json"
                if args.resume and result_path.is_file():
                    # Resume safety: an existing result JSON must have been
                    # produced from the SAME CMU data version this
                    # invocation just hash-verified above (`actual`) --
                    # never silently resumed from a stale/mismatched data
                    # version. Raises immediately rather than skipping,
                    # overwriting, or deleting anything.
                    _check_resume_cmu_input_sha256(result_path, actual)
                    _check_resume_result_sidecars(
                        result_path=result_path, cell_root=cell_root,
                        seed_root=seed_root, method=method,
                    )
                    print(f"  {cell}/{method}: resumed from {result_path}")
                    continue

                extra = {}
                timing = {"pseudo_label_seconds": None, "adapt_seconds": None}
                diagnostic_model, diagnostic_branch, history_rows = None, "target", []
                if method == "zero_shot":
                    with _Timer(device) as eval_timer:
                        metrics = _evaluate(med_source, test_cycles, device, branch="source")
                    diagnostic_model, diagnostic_branch = med_source, "source"
                elif method == "med":
                    with _Timer(device) as pseudo_timer:
                        dataset, info = build_pseudo_target_dataset(
                            med_source, train_cycles, device,
                            threshold=args.pseudo_threshold, lookahead=args.pseudo_lookahead,
                        )
                    if len(dataset) == 0:
                        # Zero-shot source predictions on this cell are too
                        # noisy/discontinuous for any window to survive the
                        # smoothness filter (this can legitimately happen --
                        # MED's own Algorithm 1 has no fallback for it). Do
                        # NOT crash the whole multi-cell sweep over one cell:
                        # record it as an honest failed/skipped row (no
                        # metrics, status flagged) and move on, the same way
                        # a formal_benchmark_eligible=false row already
                        # signals "don't trust this number" elsewhere in this
                        # repo.
                        print(
                            f"  {cell}/med: WARNING -- no pseudo-label windows survived "
                            f"confidence filtering (threshold={args.pseudo_threshold}, "
                            f"{info.get('total_windows', 0)} candidate windows, 0 passed). "
                            "Recording as a failed/skipped row instead of aborting the run."
                        )
                        row = {
                            "method": method,
                            "cell_id": cell,
                            "seed": seed,
                            "device": str(device),
                            "med_source_pretrain_seconds": med_pretrain_timing["seconds"],
                            "med_source_pretrain_epochs": med_pretrain_timing["epochs"],
                            "dann_source_pretrain_seconds": dann_pretrain_timing["seconds"],
                            "dann_source_pretrain_epochs": dann_pretrain_timing["epochs"],
                            "pseudo_label_seconds": pseudo_timer.seconds,
                            "adapt_seconds": None,
                            "eval_seconds": None,
                            "cmu_input_sha256": actual,
                            "n_train_missions": len(train_cycles),
                            "n_test_missions": len(test_cycles),
                            "label_is_ground_truth": False,
                            "formal_benchmark_eligible": False,
                            "pseudo_label_info": info,
                            "pseudo_label_collapsed": True,
                            "epochs_completed": 0,
                        }
                        cell_root.mkdir(parents=True, exist_ok=True)
                        result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
                        _rebuild_cell_metrics(args.output_root)
                        continue
                    with _Timer(device) as adapt_timer:
                        adapted, history = med_adapt(
                            med_source, dataset,
                            epochs=args.transfer_epochs, batch_size=args.batch_size,
                            learning_rate=args.transfer_lr, device=device, seed=seed,
                            lambdas=(args.med_lambda_soc, args.med_lambda_dis),
                            early_stopping_patience=args.early_stopping_patience,
                            record_epoch_seconds=True,
                            validation_cycles=validation_cycles,
                            validation_interval=args.validation_interval,
                        )
                    with _Timer(device) as eval_timer:
                        metrics = _evaluate(adapted, test_cycles, device, branch="target")
                    timing["pseudo_label_seconds"] = pseudo_timer.seconds
                    timing["adapt_seconds"] = adapt_timer.seconds
                    extra = {"pseudo_label_info": info, "epochs_completed": len(history)}
                    diagnostic_model, diagnostic_branch = adapted, "target"
                    history_rows = _flatten_dict_history(
                        history, framework="sftnn_med", method="med", seed=seed, cell_id=cell
                    )
                    if not args.no_save_checkpoints:
                        cell_root.mkdir(parents=True, exist_ok=True)
                        torch.save(adapted.target_extractor.state_dict(), cell_root / "med.pt")
                elif method == "dann_mmd":
                    assert dann_source is not None
                    with _Timer(device) as adapt_timer:
                        adapted, history, info = _adapt_dann_mmd(
                            dann_source, source_cycles, train_cycles,
                            test_profile=None, scenario="unsupervised",
                            epochs=args.transfer_epochs, batch_size=args.batch_size,
                            learning_rate=args.transfer_lr, lambdas=tuple(args.dann_lambdas),
                            device=device, seed=seed,
                            record_epoch_seconds=True,
                            validation_cycles=validation_cycles,
                            validation_interval=args.validation_interval,
                        )
                    with _Timer(device) as eval_timer:
                        metrics = _evaluate(adapted, test_cycles, device, branch="target")
                    timing["adapt_seconds"] = adapt_timer.seconds
                    extra = {"pseudo_label_info": info, "epochs_completed": len(history)}
                    diagnostic_model, diagnostic_branch = adapted, "target"
                    history_rows = _flatten_dict_history(
                        history, framework="sftnn_med", method="dann_mmd", seed=seed, cell_id=cell
                    )
                    if not args.no_save_checkpoints:
                        cell_root.mkdir(parents=True, exist_ok=True)
                        torch.save(adapted.target_extractor.state_dict(), cell_root / "dann_mmd.pt")
                else:
                    raise ValueError(method)

                # Diagnostic-only: predictions/labels/metadata via the
                # SAME per-cycle windows()/loader construction _evaluate
                # itself uses, joined with Ns/mission_phase/raw V-I-T/
                # capacity provenance. Never feeds back into metrics or
                # model state -- `metrics` above was already computed by
                # the untouched _evaluate call.
                prediction, reference, mission_ids_arr, window_idx, timestep_idx, pred_metadata = (
                    _predict_all_with_metadata(
                        diagnostic_model, test_cycles, test_metadata, device, branch=diagnostic_branch
                    )
                )
                prediction_rows = _build_prediction_rows(
                    framework="sftnn_med",
                    method=method,
                    seed=seed,
                    cell_id=cell,
                    condition=CELL_CONDITIONS.get(cell, cell),
                    soc_reference=reference,
                    soc_prediction=prediction,
                    mission_ids=mission_ids_arr,
                    window_indices=window_idx,
                    timestep_indices=timestep_idx,
                    metadata=pred_metadata,
                    raw_diagnostic=raw_diagnostic,
                )
                cell_root.mkdir(parents=True, exist_ok=True)
                if prediction_rows:
                    pq.write_table(
                        pa.Table.from_pylist(prediction_rows, schema=TEST_PREDICTIONS_SCHEMA),
                        cell_root / f"{method}_predictions.parquet",
                    )
                if history_rows:
                    pd.DataFrame(history_rows).to_csv(
                        cell_root / f"{method}_training_history.csv", index=False
                    )

                row = {
                    "method": method,
                    "cell_id": cell,
                    "seed": seed,
                    "device": str(device),
                    "med_source_pretrain_seconds": med_pretrain_timing["seconds"],
                    "med_source_pretrain_epochs": med_pretrain_timing["epochs"],
                    "dann_source_pretrain_seconds": dann_pretrain_timing["seconds"],
                    "dann_source_pretrain_epochs": dann_pretrain_timing["epochs"],
                    "pseudo_label_seconds": timing["pseudo_label_seconds"],
                    "adapt_seconds": timing["adapt_seconds"],
                    "eval_seconds": eval_timer.seconds,
                    "cmu_input_sha256": actual,
                    "n_train_missions": len(train_cycles),
                    "n_test_missions": len(test_cycles),
                    "label_is_ground_truth": False,
                    "formal_benchmark_eligible": False,
                    **extra,
                    **metrics,
                }
                row_seconds = sum(
                    s for s in (timing["pseudo_label_seconds"], timing["adapt_seconds"], eval_timer.seconds)
                    if s is not None
                )
                print(
                    f"  {cell}/{method}: MAE={metrics['overall_mae_percent']:.3f}% "
                    f"RMSE={metrics['overall_rmse_percent']:.3f}% ({row_seconds:.1f}s)"
                )
                cell_root.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
                _rebuild_cell_metrics(args.output_root)

    table = _rebuild_cell_metrics(args.output_root)
    _rebuild_test_predictions(args.output_root)
    _rebuild_training_history(args.output_root)

    # Diagnostic-only phase summaries, read back from the just-rebuilt
    # top-level test_predictions.parquet (never from in-memory rows, never
    # a second evaluation set) so they always reflect every cell on disk,
    # including any --resume'd ones. Skipped (never writing empty/
    # placeholder files) if the merged parquet doesn't exist yet or has no
    # take_off/cruise/landing rows.
    predictions_path = args.output_root / "test_predictions.parquet"
    if predictions_path.is_file():
        phase_metrics = _compute_phase_metrics(predictions_path)
        if not phase_metrics.empty:
            phase_metrics.to_csv(args.output_root / "phase_metrics.csv", index=False)
            phase_transfer_effects = _compute_phase_transfer_effects(phase_metrics)
            phase_transfer_effects.to_csv(
                args.output_root / "phase_transfer_effects.csv", index=False
            )

    metadata = {
        "benchmark": "sftnn_med_transfer_panasonic_25c_to_cmu_core7",
        "note": (
            "MED (Shen et al. 2023 Algorithm 1) source-free transfer from "
            "Panasonic 25C to the CMU eVTOL core-seven cells, plus an "
            "optional zero-shot baseline and an optional DANN+MMD "
            "comparison baseline. MED's transfer stage uses no CMU labels "
            "and no Panasonic source data/labels -- only pseudo-labels from "
            "the frozen source model's averaged-head output on CMU's "
            "unlabeled train-split missions. Real CMU labels are used only "
            "for post-hoc scoring of the held-out test-split missions."
        ),
        "methods": list(args.methods),
        "cells": list(args.cells),
        # Explicit record of which CMU dataset build and manifest this run
        # read from, and which Ns -> mission-phase vocabulary its
        # diagnostic fields use -- diagnostic/provenance-only, does not
        # affect model, pseudo-labeling, training, or evaluation logic.
        "cmu_base_root": str(args.cmu_base_root),
        "cmu_condition_root": str(args.cmu_condition_root),
        "cmu_manifest_path": str(args.cmu_manifest),
        "cmu_manifest_sha256": sha256_file(args.cmu_manifest),
        "phase_definition": "canonical Ns mapping: 4=take_off, 5=cruise, 6=landing",
        "seeds": list(args.seeds),
        "feature_order": list(FEATURES),
        "window": WINDOW,
        "source_profiles": list(PROFILES),
        "source_temperature_c": 25,
        "source_feature_min": feature_min.tolist(),
        "source_feature_max": feature_max.tolist(),
        "normalization": "Panasonic 25C source-fitted min-max applied unchanged to CMU target",
        "pseudo_threshold": args.pseudo_threshold,
        "pseudo_lookahead": args.pseudo_lookahead,
        "med_lambdas": [args.med_lambda_soc, args.med_lambda_dis],
        "dann_lambdas": args.dann_lambdas,
        "source_epochs": args.source_epochs,
        "transfer_epochs": args.transfer_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "transfer_lr": args.transfer_lr,
        "evaluation_unit": EVALUATION_UNIT,
        "evaluation_unit_note": (
            "windows() builds non-overlapping WINDOW-length windows and "
            "predict()/_evaluate ravel per-timestep output AND per-timestep "
            "label across every window -- all_timesteps, like DTNN/backbone, "
            "unlike DARM's last_step_per_window."
        ),
        "metric_units": {
            "soc_reference": "fraction",
            "soc_prediction": "fraction",
            "signed_error": "fraction",
            "absolute_error": "fraction",
            "overall_mae_percent": "percent",
            "overall_rmse_percent": "percent",
            "overall_max_error_percent": "percent",
            "overall_bias_percent": "percent",
            "mission_macro_mae_percent": "percent",
            "mission_macro_rmse_percent": "percent",
        },
        "target_soc_labels_used_for_training": False,
        "target_soc_labels_used_for_adaptation": False,
        "target_soc_labels_used_for_checkpoint_selection": False,
        # Manifest "validation"-split mission SOC labels ARE now read by a
        # read-only diagnostic probe (see med_adapt's/adapt_dann_mmd's
        # validation_cycles=/_validation_metrics in shen_2023/train.py),
        # for MED and DANN+MMD *adaptation* only -- never zero-shot, never
        # source pretraining, and never fed back into training,
        # pseudo-labeling, the scheduler, early stopping, checkpoint
        # selection, or model selection.
        "target_validation_labels_used_for_diagnostics": True,
        "target_validation_labels_used_for_training": False,
        "target_validation_labels_used_for_checkpoint_selection": False,
        "validation_interval": args.validation_interval,
        "target_soc_reference_used_for_posthoc_evaluation": True,
        "label_is_ground_truth": False,
        "formal_benchmark_eligible": False,
        "git_commit": get_full_commit_hash(REPO_ROOT),
        "git_dirty": is_dirty(REPO_ROOT),
        # Wall-clock time spent by *this invocation* only -- with --resume,
        # anything skipped because a result file already existed is not
        # counted here. Per-row timing breakdowns (pretrain/pseudo-label/
        # adapt/eval seconds, plus the device they ran on) live in
        # cell_metrics.csv, which is cumulative across every invocation.
        "this_invocation_wall_seconds": time.perf_counter() - run_start,
        "device": str(device),
    }
    metadata_path = args.output_root / "run_metadata.json"
    if metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["methods"] = sorted(set(previous.get("methods", [])) | set(metadata["methods"]))
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nthis invocation took {metadata['this_invocation_wall_seconds']:.1f}s wall-clock")


if __name__ == "__main__":
    main()
