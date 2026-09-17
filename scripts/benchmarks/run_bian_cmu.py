#!/usr/bin/env python3
"""Run paper-aligned supervised Bian DTNN adaptation on CMU development SOC."""

# ruff: noqa: E402 -- executable from a source checkout without installation

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from battery_benchmark.datasets.cmu_evtol.ns_phase_labeling_v1 import (
    MISSION_FLIGHT_PHASE_ROLES,
    NS_MISSION_PHASE_MAP,
)
from battery_benchmark.datasets.cmu_evtol.vidal_adapter import load_cmu_dynamic_soc_for_vidal
from battery_benchmark.evaluation.metrics import compute_metrics
from battery_benchmark.reproductions.bian_2021 import DTNN, adaptive_multiscale_mmd
from battery_benchmark.reproductions.liu_2025.cmu_protocol import select_duration_matched_missions
from battery_benchmark.reproductions.shen_2023.data import (
    TEMPERATURES,
    load_temperature_cycles_and_range,
)
from battery_benchmark.utils.device import get_device
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty
from battery_benchmark.utils.hashing import sha256_file

CELLS = ("VAH01", "VAH17", "VAH27", "VAH30", "VAH15", "VAH12", "VAH11")
BASELINE_CELLS = {"VAH01", "VAH17", "VAH27"}
SOURCE_PROFILES = ("Cycle_1", "Cycle_2", "Cycle_3", "Cycle_4", "NN", "US06", "HWFET")
FEATURES = ("current", "voltage")

# Fixed schema for test_predictions.parquet, shared by every chunk written
# via _write_prediction_chunk below -- guarantees identical column order
# and types across all repeat x cell x method chunks (and across separate
# runs), regardless of which cells/methods happened to produce rows first.
TEST_PREDICTIONS_SCHEMA = pa.schema(
    [
        ("framework", pa.string()),
        ("method", pa.string()),
        ("repeat", pa.int64()),
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
        # Raw-parquet diagnostic fields, joined per cell by source_row_right
        # (see _load_raw_diagnostic_fields / _join_diagnostic_fields_by_
        # source_row below). All nullable: a genuinely missing
        # capacity_source/source_rpt_id (or, defensively, an unmatched
        # source_row_right) round-trips as a real parquet null, never the
        # text "nan".
        ("development_phase_role", pa.string()),
        ("voltage_v_raw", pa.float64()),
        ("current_a_raw", pa.float64()),
        ("temperature_c_raw", pa.float64()),
        ("capacity_ah", pa.float64()),
        ("capacity_source", pa.string()),
        ("source_rpt_id", pa.string()),
        ("soc_reference", pa.float64()),
        ("soc_prediction", pa.float64()),
        ("signed_error", pa.float64()),
        ("absolute_error", pa.float64()),
        # Appended, not reordered into the block above, so every existing
        # column keeps its position -- see _load_raw_diagnostic_fields /
        # NS_MISSION_PHASE_MAP. ns_raw is the untouched tester-native
        # Ns value; mission_phase is a derived, purely diagnostic label
        # computed from it (never the reverse -- ns_raw is never dropped
        # in favor of the label). Both nullable, same as the other
        # diagnostic-join columns.
        ("ns_raw", pa.int64()),
        ("mission_phase", pa.string()),
    ]
)

# Ns -> mission-phase label for this DTNN benchmark's diagnostic
# mission_phase column. This is the SAME canonical, protocol-level mapping
# (imported above, not copied) that
# battery_benchmark.datasets.cmu_evtol.ns_phase_labeling_v1 uses for its own
# full-raw-timeline diagnostic: take_off/cruise/landing for Ns 4/5/6,
# cc_charge/cv_charge/rest_2/rest_1 for Ns 0/1/3/7, and a real None for any
# unmapped Ns (including 2 and 8) -- never a fabricated label. DTNN
# previously defined its own, differently-spelled local copy of this map
# (BIAN_NS_MISSION_PHASE_MAP: takeoff/other_segment/pre_mission_rest/
# post_flight_rest); that local copy is retired in favor of directly reusing
# the canonical module-level dict so "mission_phase" means exactly the same
# thing everywhere it appears across the CMU eVTOL diagnostics.
# Flight-phase SOC analysis (notebook/reporting) should focus on these three.
# NS_MISSION_PHASE_MAP/MISSION_FLIGHT_PHASE_ROLES are used directly
# (imported above), not aliased to a DTNN-local name, so a `grep` for the
# canonical names finds every consumer, including this one.

# Raw CMU parquet column name -> test_predictions.parquet output field name,
# for the diagnostic fields joined onto each test timestep by
# source_row_right. voltage_v/current_a/Temperature__C are deliberately
# read straight from the raw parquet here (never from the resampled/
# normalized x used for training/adaptation/prediction) and renamed to
# *_raw so they can't be mistaken for model-input features.
RAW_DIAGNOSTIC_SOURCE_COLUMNS = {
    "development_phase_role": "development_phase_role",
    "voltage_v": "voltage_v_raw",
    "current_a": "current_a_raw",
    "Temperature__C": "temperature_c_raw",
    "capacity_ah": "capacity_ah",
    "capacity_source": "capacity_source",
    "source_rpt_id": "source_rpt_id",
    # Raw tester-native Ns, untouched -- see NS_MISSION_PHASE_MAP and
    # _load_raw_diagnostic_fields (mission_phase is derived FROM this
    # column, after the rename below, not read from the raw parquet).
    "Ns": "ns_raw",
}
# mission_phase is derived (see _load_raw_diagnostic_fields), not a
# raw-parquet column, so it isn't a value in RAW_DIAGNOSTIC_SOURCE_COLUMNS
# -- appended here so the rest of the join/cast/schema pipeline (which all
# key off DIAGNOSTIC_METADATA_COLUMNS) picks it up automatically.
DIAGNOSTIC_METADATA_COLUMNS = tuple(RAW_DIAGNOSTIC_SOURCE_COLUMNS.values()) + ("mission_phase",)

# How _evaluate() casts each joined diagnostic value when building a
# prediction record -- applied only to a non-None value (see
# _DIAGNOSTIC_METADATA_CASTERS' use in _evaluate: a real null from the join
# is passed through as None, never cast).
_DIAGNOSTIC_METADATA_CASTERS = {
    "development_phase_role": str,
    "voltage_v_raw": float,
    "current_a_raw": float,
    "temperature_c_raw": float,
    "capacity_ah": float,
    "capacity_source": str,
    "source_rpt_id": str,
    "ns_raw": int,
    "mission_phase": str,
}


def _check_ns_mission_phase_consistency(frame: pd.DataFrame, path: Path) -> None:
    """Pre-run consistency check (diagnostic-only, read-only -- raises
    before any test_predictions.parquet row is written for this cell,
    never after): for every row whose raw ``ns_raw`` is 4/5/6, the derived
    ``mission_phase`` must equal take_off/cruise/landing respectively (this
    is tautologically true given a correct NS_MISSION_PHASE_MAP, but is
    checked explicitly so a corrupted/edited map fails loudly rather than
    silently mislabeling every downstream row). Separately, and more
    substantively, wherever the INDEPENDENTLY-derived
    ``development_phase_role`` (built by
    ``development_pipeline_v0.enumerate_diagnostic_candidates`` from its own
    source-row-range candidate detection, not from ``Ns`` directly) is one
    of take_off/cruise/landing, it must agree with ``mission_phase`` for
    that same row. A mismatch means the Ns-based labeling and the
    candidate-detection-based labeling disagree about what a row is, which
    would silently corrupt every phase-conditioned analysis downstream --
    this raises immediately instead."""
    expected_by_ns = {4: "take_off", 5: "cruise", 6: "landing"}
    for ns_value, expected_phase in expected_by_ns.items():
        mask = frame["ns_raw"] == ns_value
        if not mask.any():
            continue
        bad = frame.loc[mask & (frame["mission_phase"] != expected_phase)]
        if not bad.empty:
            raise ValueError(
                f"{path}: {len(bad)} row(s) with ns_raw={ns_value} have "
                f"mission_phase != {expected_phase!r} (NS_MISSION_PHASE_MAP "
                "may have been edited/corrupted)"
            )

    role_mask = frame["development_phase_role"].isin(MISSION_FLIGHT_PHASE_ROLES)
    if role_mask.any():
        mismatched = frame.loc[
            role_mask & (frame["mission_phase"] != frame["development_phase_role"])
        ]
        if not mismatched.empty:
            counts = mismatched["development_phase_role"].value_counts().to_dict()
            raise ValueError(
                f"{path}: {len(mismatched)} row(s) where development_phase_role "
                f"disagrees with the Ns-derived mission_phase ({counts}) -- Ns-based "
                "and candidate-detection-based flight-phase labeling have diverged "
                "for this cell's raw parquet"
            )


def _load_raw_diagnostic_fields(path: Path) -> pd.DataFrame:
    """One per-cell read of the raw CMU parquet's diagnostic columns (plus
    source_row, the join key), read directly from ``path`` -- the same raw
    parquet already hash-verified against the manifest for this cell --
    never from resampled or normalized data. Columns are renamed
    immediately to the test_predictions.parquet output field names via
    RAW_DIAGNOSTIC_SOURCE_COLUMNS.

    ``mission_phase`` is then added as a purely derived column, computed
    from the just-renamed ``ns_raw`` via the CANONICAL, directly-reused
    ``NS_MISSION_PHASE_MAP.get`` (``.map()``, vectorized) -- an unmapped Ns
    value (2, 8, or anything else outside the map) becomes a real ``None``,
    never a fabricated label. ``ns_raw`` itself is kept as-is alongside
    it -- the semantic label is additive, never a replacement for the raw
    value. Before returning, ``_check_ns_mission_phase_consistency`` runs a
    read-only cross-check between this Ns-derived label and the
    independently-built ``development_phase_role`` column, raising
    immediately on any disagreement."""
    frame = pd.read_parquet(path, columns=["source_row", *RAW_DIAGNOSTIC_SOURCE_COLUMNS])
    if not frame["source_row"].is_unique:
        raise ValueError(f"{path}: source_row must be unique for a diagnostic join")
    frame = frame.rename(columns=RAW_DIAGNOSTIC_SOURCE_COLUMNS)
    frame["mission_phase"] = frame["ns_raw"].map(NS_MISSION_PHASE_MAP)
    _check_ns_mission_phase_consistency(frame, path)
    return frame


def _join_diagnostic_fields_by_source_row(source_row_right, raw_diagnostic):
    """Vectorized (pandas ``DataFrame.merge``) join of DIAGNOSTIC_METADATA_COLUMNS
    (the raw-parquet diagnostic columns, plus the derived ``mission_phase``
    already attached by ``_load_raw_diagnostic_fields``) onto every timestep referenced in
    ``source_row_right`` (any shape -- typically (n_windows, seq_len)),
    keyed by source_row_right -> the raw parquet's own source_row. This is
    a single per-cell ``merge`` call, never a Python dict accumulated
    across timesteps, cells, or the whole experiment: ``raw_diagnostic``
    (see _load_raw_diagnostic_fields) is loaded once per cell, and this
    function is called once per cell with the full array of that cell's
    test-split source_row_right values.

    A ``source_row_right`` with no matching row in ``raw_diagnostic``
    (should not happen in practice -- source_row_right always comes from
    this same cell's own raw parquet -- but handled defensively rather
    than silently producing a bogus value) yields real nulls (``None``)
    for that timestep's diagnostic fields. A genuinely missing individual
    field value (e.g. an absent source_rpt_id/capacity_source, matching
    the existing convention already used for the Vidal CMU script) is
    also a real ``None`` -- constructed via ``pd.isna()`` + a vectorized
    boolean-mask assignment, never ``.astype(str)``, which would turn a
    missing value into the literal string ``"nan"``."""
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


def _write_prediction_chunk(writer: pq.ParquetWriter, rows: list[dict]) -> None:
    """Writes one repeat x cell x method chunk of prediction records to an
    already-open ParquetWriter, using TEST_PREDICTIONS_SCHEMA so every
    chunk (from this call or any other) has identical column order/types.
    A no-op on an empty chunk (pa.Table.from_pylist tolerates it, but there
    is nothing to write and nothing to gain from writing a zero-row
    record batch)."""
    if not rows:
        return
    writer.write_table(pa.Table.from_pylist(rows, schema=TEST_PREDICTIONS_SCHEMA))


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


class _Timer:
    """Wall-clock stopwatch. CUDA ops are async, so a timed block that ran
    on GPU is synced before stopping -- otherwise the measured duration
    would just be launch overhead, not actual compute time. Diagnostic
    only: never wraps anything that changes what gets trained."""

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


# Per-timestep metadata columns carried through _windows/_target_data in
# lockstep with x/y whenever include_metadata=True -- see _target_parts.
METADATA_COLUMNS = (
    "mission_sequence_order",
    "elapsed_candidate_s",
    "original_time_s",
    "source_row_right",
)


def _windows(
    parts: list[tuple[np.ndarray, np.ndarray, str]]
    | list[tuple[np.ndarray, np.ndarray, str, dict]],
    length: int,
    *,
    include_metadata: bool = False,
):
    """Unchanged reshape/truncation logic for x/y/missions when
    ``include_metadata=False`` (every pre-existing caller): identical
    output to before this parameter existed.

    When ``include_metadata=True``, each part is a 4-tuple
    ``(x, y, mission, metadata)`` where ``metadata`` maps each name in
    METADATA_COLUMNS to a 1-D per-timestep array the same length as
    ``x``/``y`` for that part (see _target_parts). Every metadata array is
    truncated and reshaped with the EXACT SAME ``usable``/``count`` values
    computed from x/y -- not a separate slicing pass -- so metadata stays
    index-for-index aligned with the SOC timestep it describes, including
    across truncated/dropped trailing timesteps and windows dropped
    entirely for a mission with no complete window."""
    xs, ys, missions = [], [], []
    metadata_lists = {name: [] for name in METADATA_COLUMNS} if include_metadata else None
    for part in parts:
        if include_metadata:
            x, y, mission, metadata = part
        else:
            x, y, mission = part
        usable = min(len(x), len(y)) // length * length
        if usable:
            count = usable // length
            xs.append(x[:usable].reshape(count, length, x.shape[-1]))
            ys.append(y[:usable].reshape(count, length))
            missions.append((mission, count))
            if include_metadata:
                for name in METADATA_COLUMNS:
                    metadata_lists[name].append(metadata[name][:usable].reshape(count, length))
    if not xs:
        raise ValueError("No complete non-overlapping windows")
    x_out = np.concatenate(xs).astype(np.float32)
    y_out = np.concatenate(ys).astype(np.float32)
    if not include_metadata:
        return x_out, y_out, missions
    metadata_out = {name: np.concatenate(metadata_lists[name]) for name in METADATA_COLUMNS}
    return x_out, y_out, missions, metadata_out


def _source_data(root: Path, length: int):
    raw_parts = []
    for temperature in TEMPERATURES:
        cycles, feature_min, feature_max = load_temperature_cycles_and_range(root, temperature)
        scale = np.where(feature_max == feature_min, 1.0, feature_max - feature_min)
        raw_parts.extend(
            (
                cycles[name].features[:, :2] * scale[:2] + feature_min[:2],
                cycles[name].soc,
                f"{temperature}C/{name}",
            )
            for name in SOURCE_PROFILES
        )
    all_features = np.concatenate([part[0] for part in raw_parts])
    feature_min, feature_max = all_features.min(0), all_features.max(0)
    scale = np.where(feature_max == feature_min, 1.0, feature_max - feature_min)
    parts = [((x - feature_min) / scale, y, name) for x, y, name in raw_parts]
    x, y, _ = _windows(parts, length)
    return x, y, feature_min, feature_max


def _target_parts(records, mission_ids, initial_soc, fraction, portion, *, include_metadata=False):
    """``include_metadata=False`` (every pre-existing caller: adapt data,
    validation diagnostic data): unchanged output, byte-for-byte the same
    as before this parameter existed.

    ``include_metadata=True`` (test data only): for each part, also builds
    the four METADATA_COLUMNS arrays (mission_sequence_order,
    elapsed_candidate_s, original_time_s, source_row_right) directly from
    the SAME frame this part's raw/y features/labels come from, and
    slices them with the EXACT SAME two-step slicing raw/y already use --
    first ``[starts[0]:]`` (drop everything before the mission first
    reaches initial_soc), then ``[selection]`` (the adapt/test 30/70
    split). This is not a second, independent slicing pass computed
    after the fact: it's the same starts/selection indices applied to a
    parallel array, which is what keeps every metadata value aligned to
    the exact SOC timestep it describes."""
    if portion not in {"adapt", "test"}:
        raise ValueError("portion must be adapt or test")
    parts = []
    for record in records:
        mission = str(record.metadata["mission_cycle_id"])
        if mission not in mission_ids:
            continue
        frame = record.data
        y = frame["soc"].to_numpy(dtype=np.float64)
        starts = np.flatnonzero(y <= initial_soc)
        if not len(starts):
            continue
        raw = frame.loc[:, FEATURES].to_numpy(dtype=np.float64)[starts[0] :]
        y = y[starts[0] :]
        boundary = int(len(y) * fraction)
        selection = slice(0, boundary) if portion == "adapt" else slice(boundary, None)
        if not include_metadata:
            parts.append((raw[selection], y[selection], mission))
            continue
        sequence_order = int(record.metadata["sequence_order"])
        metadata_full = {
            "mission_sequence_order": np.full(len(frame), sequence_order, dtype=np.int64),
            "elapsed_candidate_s": frame["elapsed_candidate_s"].to_numpy(dtype=np.float64),
            "original_time_s": frame["original_time_s"].to_numpy(dtype=np.float64),
            "source_row_right": frame["source_row_right"].to_numpy(dtype=np.int64),
        }
        metadata_selected = {
            name: values[starts[0] :][selection] for name, values in metadata_full.items()
        }
        parts.append((raw[selection], y[selection], mission, metadata_selected))
    return parts


def _target_data(parts, length, feature_min, feature_max, *, include_metadata=False):
    """``include_metadata`` must match what ``parts`` was built with (see
    ``_target_parts``): it only changes whether each part is unpacked as a
    3- or 4-tuple before scaling x, and is passed straight through to
    ``_windows`` so the metadata arrays get the identical
    truncation/reshape as x/y."""
    scale = np.where(feature_max == feature_min, 1.0, feature_max - feature_min)
    if include_metadata:
        parts = [
            ((x - feature_min) / scale, y, mission, metadata)
            for x, y, mission, metadata in parts
        ]
    else:
        parts = [((x - feature_min) / scale, y, mission) for x, y, mission in parts]
    return _windows(parts, length, include_metadata=include_metadata)


def _loader(x, y, batch_size, shuffle, *, generator=None):
    """Unchanged for every existing (training) call site: ``generator``
    defaults to ``None``, so the DataLoader falls back to consuming
    ``torch.default_generator`` for its internal base-seed draw exactly as
    before -- this is what the (pre-existing) shuffled source/target
    training loaders rely on for their epoch-to-epoch reshuffling.

    ``_validation_diagnostic_metrics`` (diagnostic-only) passes its own
    local ``torch.Generator()`` instead: PyTorch's DataLoader always draws
    one value from its generator to seed itself -- even with
    ``num_workers=0`` and ``shuffle=False`` -- so constructing/iterating a
    validation loader with the default (``generator=None``) would silently
    advance ``torch.default_generator`` and perturb every later epoch's
    training-loader shuffle. Passing a separate, local generator keeps that
    internal draw off the global RNG stream, so the validation diagnostic
    cannot affect training determinism."""
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=min(batch_size, len(x)),
        shuffle=shuffle,
        generator=generator,
    )


def _train_source(model, x, y, epochs, batch_size, lr, device):
    """Runs source pretraining exactly as before (same loader, same
    optimizer, same loss, same epoch count). The only addition is a
    per-epoch diagnostic history list returned alongside training -- pure
    bookkeeping (a wall-clock timer around the existing training-only loop,
    plus the same weighted-mean MSE that was already computed for the
    print statement) that does not change what is trained or in what
    order."""
    loader = _loader(x, y, batch_size, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        with _Timer(device) as timer:
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(xb)
        runtime_seconds = timer.seconds
        source_mse = total / len(x)
        history.append(
            {
                "epoch": epoch + 1,
                "source_mse": source_mse,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "runtime_seconds": runtime_seconds,
            }
        )
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % max(epochs // 10, 1) == 0:
            print(f"  source epoch {epoch + 1}/{epochs}: mse={source_mse:.6f}")
    return history


LAYER_NAMES = ("cnn2", "bilstm2", "fc")  # matches adaptive_multiscale_mmd's (c2, blstm2, fc) order


@torch.no_grad()
def _validation_diagnostic_metrics(model, x, y, batch_size, device):
    """Lightweight MAE/RMSE, as [0,1] fractions (NOT percent, unlike
    _evaluate's overall_mae_percent/overall_rmse_percent), for the periodic
    validation_mae/validation_rmse diagnostic in training_history.csv only.
    Never used for early stopping or checkpoint selection. A pure eval-mode
    forward pass under no_grad; the model's training/eval mode is restored
    to whatever it was before this call, so the caller's training loop is
    unaffected either way."""

    was_training = model.training
    model.eval()
    # A local, non-global generator: see _loader's docstring -- this keeps
    # the DataLoader's internal base-seed draw off torch.default_generator,
    # so this diagnostic call never perturbs the training loaders' RNG
    # stream (and therefore never perturbs training determinism), whether
    # or not this call happens at all.
    validation_generator = torch.Generator()
    prediction = torch.cat(
        [
            model(xb.to(device)).cpu()
            for xb, _ in _loader(x, y, batch_size, False, generator=validation_generator)
        ]
    ).numpy()
    prediction = np.clip(prediction, 0.0, 1.0)
    if was_training:
        model.train()
    metrics = compute_metrics(y, prediction)
    return float(metrics["mae"]), float(metrics["rmse"])


def _adapt(
    model, source_x, source_y, target_x, target_y, args, device,
    *, validation_x=None, validation_y=None,
):
    """Runs the adaptation loop exactly as before (same batches, same order,
    same optimizer.step() calls, same final-batch return value used for
    ``training_final_batch``). The only addition is a second return value:
    one diagnostic row per epoch, each field averaged over every optimizer
    step that ran within that epoch (not just the last batch). This is
    read-only bookkeeping on values already computed by the existing
    forward/backward pass -- it does not change what is trained, in what
    order, or the final model weights.

    Timing: exactly one _Timer per epoch (one CUDA synchronize before the
    epoch's target_loader loop starts, one right after it ends) -- not one
    per optimizer step. runtime_seconds is therefore the epoch's real
    adaptation wall-clock time, including the lightweight diagnostic
    float-extraction/accumulation interleaved with the training steps
    (negligible next to forward/backward/optimizer.step(), and read-only,
    so it doesn't change what's trained -- only how long the epoch's
    wall-clock is reported as taking). Validation evaluation (below) runs
    strictly after this timer closes, so it is never counted in
    runtime_seconds either.

    Validation (diagnostic only, controlled by args.validation_interval):
    every ``validation_interval`` epochs, plus always on epoch 1 and the
    final epoch, the current model is evaluated in eval mode under
    torch.no_grad() on validation_x/validation_y (if given) and then
    restored to train mode -- a pure forward pass on already-adapted
    weights. It is never used for early stopping, checkpoint selection, or
    anything that feeds back into training: no gradients, no
    optimizer/scheduler interaction, and it runs after optimizer.step() for
    that epoch has already happened. validation_interval=0, or omitting
    validation_x/validation_y, disables it entirely (validation_evaluated
    is False and validation_mae/validation_rmse are NaN on every row).
    """
    source_loader = _loader(source_x, source_y, args.transfer_batch_size, True)
    target_loader = _loader(target_x, target_y, args.transfer_batch_size, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.transfer_lr)
    criterion = nn.MSELoss()
    validation_interval = getattr(args, "validation_interval", 10)
    final = {}
    history = []
    for epoch in range(args.transfer_epochs):
        model.train()
        source_batches = cycle(source_loader)
        epoch_source_mse_sum = 0.0
        epoch_target_mse_sum = 0.0
        epoch_adaptive_mmd_sum = 0.0
        epoch_total_loss_sum = 0.0
        epoch_layer_mmd_sum = [0.0] * len(LAYER_NAMES)
        epoch_layer_weight_sum = [0.0] * len(LAYER_NAMES)
        epoch_steps = 0
        # One CUDA sync pair per epoch, not per optimizer step: sync+start
        # before the epoch's target_loader loop, sync+stop right after it
        # ends. runtime_seconds is therefore the epoch's actual adaptation
        # wall-clock time -- training steps plus the lightweight diagnostic
        # float-extraction/accumulation interleaved with them below, which
        # is negligible next to the forward/backward/optimizer.step() cost
        # and is read-only bookkeeping that never feeds back into training.
        with _Timer(device) as epoch_timer:
            for target_batch in target_loader:
                source_batch = next(source_batches)
                sx, sy = (item.to(device) for item in source_batch)
                tx, ty = (item.to(device) for item in target_batch)
                optimizer.zero_grad(set_to_none=True)
                sp, sf = model(sx, return_features=True)
                tp, tf = model(tx, return_features=True)
                source_loss = criterion(sp, sy)
                target_loss = criterion(tp, ty)
                mmd, layer_mmd, weights = adaptive_multiscale_mmd(sf, tf)
                loss = source_loss + args.target_weight * target_loss + args.mmd_weight * mmd
                loss.backward()
                optimizer.step()
                layer_mmd_values = [float(v) for v in layer_mmd.detach().cpu()]
                layer_weight_values = [float(v) for v in weights.cpu()]
                final = {
                    "source_mse": float(source_loss.detach()),
                    "target_mse": float(target_loss.detach()),
                    "adaptive_mmd": float(mmd.detach()),
                    "layer_mmd": layer_mmd_values,
                    "layer_weights": layer_weight_values,
                }
                # Diagnostic-only accumulation for the per-epoch average
                # below -- reads values already produced by the step above,
                # does not feed back into the model, optimizer, or loss in
                # any way. Left inside the timed region (see epoch_timer
                # above) rather than given its own synchronize/timer.
                epoch_source_mse_sum += final["source_mse"]
                epoch_target_mse_sum += final["target_mse"]
                epoch_adaptive_mmd_sum += final["adaptive_mmd"]
                epoch_total_loss_sum += float(loss.detach())
                for index, value in enumerate(layer_mmd_values):
                    epoch_layer_mmd_sum[index] += value
                for index, value in enumerate(layer_weight_values):
                    epoch_layer_weight_sum[index] += value
                epoch_steps += 1
        epoch_training_seconds = epoch_timer.seconds
        if epoch_steps:
            row = {
                "epoch": epoch + 1,
                "source_mse": epoch_source_mse_sum / epoch_steps,
                "target_mse": epoch_target_mse_sum / epoch_steps,
                "adaptive_mmd": epoch_adaptive_mmd_sum / epoch_steps,
                "total_loss": epoch_total_loss_sum / epoch_steps,
            }
            for index, name in enumerate(LAYER_NAMES):
                row[f"layer_mmd_{name}"] = epoch_layer_mmd_sum[index] / epoch_steps
            for index, name in enumerate(LAYER_NAMES):
                row[f"layer_weight_{name}"] = epoch_layer_weight_sum[index] / epoch_steps
            row["learning_rate"] = optimizer.param_groups[0]["lr"]
            row["runtime_seconds"] = epoch_training_seconds
            row["optimizer_steps"] = epoch_steps

            # Diagnostic only -- runs after optimizer.step() for this epoch
            # has already happened, so it cannot influence training. Never
            # used for early stopping or checkpoint selection.
            epoch_number = epoch + 1
            should_run_validation = (
                validation_interval > 0
                and validation_x is not None
                and validation_y is not None
                and len(validation_x) > 0
                and (
                    epoch_number == 1
                    or epoch_number == args.transfer_epochs
                    or epoch_number % validation_interval == 0
                )
            )
            if should_run_validation:
                validation_mae, validation_rmse = _validation_diagnostic_metrics(
                    model, validation_x, validation_y, args.batch_size, device
                )
                row["validation_mae"] = validation_mae
                row["validation_rmse"] = validation_rmse
                row["validation_evaluated"] = True
            else:
                row["validation_mae"] = math.nan
                row["validation_rmse"] = math.nan
                row["validation_evaluated"] = False

            history.append(row)
    final["optimizer_steps"] = args.transfer_epochs * len(target_loader)
    return final, history


@torch.no_grad()
def _evaluate(
    model, x, y, missions, batch_size, device, *, return_predictions=False, metadata=None
):
    """Existing metric computation (``overall``) is completely unchanged --
    still built from the same ``prediction``/``y`` arrays via the same
    ``compute_metrics`` calls in the same order. The only addition is an
    optional second return value: when ``return_predictions=True``, also
    return a list of per-timestep prediction records (mission_cycle_id,
    window_index, timestep_index, soc_reference, soc_prediction,
    signed_error, absolute_error) built read-only from those same
    ``prediction``/``y`` arrays after ``overall`` has already been computed.
    Callers that don't pass ``return_predictions`` get exactly the previous
    single-value return, so every existing call site is unaffected.

    ``metadata`` (only meaningful together with ``return_predictions=True``)
    is the optional dict of METADATA_COLUMNS arrays that ``_target_data(...,
    include_metadata=True)`` returns for the test split -- each array shaped
    exactly like ``y`` (n_windows, seq_len), built by the SAME
    truncation/reshape ``_windows`` applied to ``x``/``y`` themselves (see
    ``_target_parts``/``_windows``), so indexing it at
    ``[global_window_index, timestep_index]`` is guaranteed aligned to the
    SOC timestep being recorded -- never a separately-computed guess.
    ``metadata=None`` (the default, used by any caller not evaluating the
    test split) simply omits those four fields from each record.
    ``metadata`` may also carry any of DIAGNOSTIC_METADATA_COLUMNS (see
    _join_diagnostic_fields_by_source_row) -- present keys are copied into
    each record the same way, with a real ``None`` passed straight through
    (never cast) whenever the joined value was itself a null; keys not in
    ``metadata`` are simply skipped, so this stays compatible with a
    ``metadata`` dict that only has the four geometry columns."""
    model.eval()
    prediction = torch.cat(
        [model(xb.to(device)).cpu() for xb, _ in _loader(x, y, batch_size, False)]
    ).numpy()
    prediction = np.clip(prediction, 0.0, 1.0)
    metrics = compute_metrics(y, prediction)
    overall = {
        "overall_mae_percent": metrics["mae"] * 100,
        "overall_rmse_percent": metrics["rmse"] * 100,
        "overall_max_error_percent": metrics["max_error"] * 100,
        "overall_bias_percent": metrics["bias"] * 100,
        "overall_r2": metrics["r2"],
    }
    start, per_mission = 0, []
    for _, count in missions:
        stop = start + count
        per_mission.append(compute_metrics(y[start:stop], prediction[start:stop]))
        start = stop
    overall.update(
        {
            "mission_macro_rmse_percent": float(np.mean([m["rmse"] for m in per_mission]) * 100),
            "mission_macro_mae_percent": float(np.mean([m["mae"] for m in per_mission]) * 100),
            "test_missions": len(missions),
            "test_windows": len(x),
        }
    )
    if not return_predictions:
        return overall

    # Diagnostic/reporting only, from here down: per-timestep prediction
    # records for the saved test_predictions.parquet. Read-only reuse of
    # the ``prediction``/``y`` arrays already computed above -- does not
    # recompute or alter anything ``overall`` reports. ``window_index`` is
    # the window's position within its own mission (0-based), matching
    # ``timestep_index`` being the position within that window (0-based).
    records = []
    start = 0
    for mission, count in missions:
        for local_window_index in range(count):
            global_window_index = start + local_window_index
            for timestep_index in range(x.shape[1]):
                reference = float(y[global_window_index, timestep_index])
                predicted = float(prediction[global_window_index, timestep_index])
                record = {
                    "mission_cycle_id": mission,
                    "window_index": local_window_index,
                    "timestep_index": timestep_index,
                    "soc_reference": reference,
                    "soc_prediction": predicted,
                    "signed_error": predicted - reference,
                    "absolute_error": abs(predicted - reference),
                }
                if metadata is not None:
                    record["mission_sequence_order"] = int(
                        metadata["mission_sequence_order"][global_window_index, timestep_index]
                    )
                    record["elapsed_candidate_s"] = float(
                        metadata["elapsed_candidate_s"][global_window_index, timestep_index]
                    )
                    record["original_time_s"] = float(
                        metadata["original_time_s"][global_window_index, timestep_index]
                    )
                    record["source_row_right"] = int(
                        metadata["source_row_right"][global_window_index, timestep_index]
                    )
                    for name, caster in _DIAGNOSTIC_METADATA_CASTERS.items():
                        if name not in metadata:
                            continue
                        raw_value = metadata[name][global_window_index, timestep_index]
                        record[name] = None if raw_value is None else caster(raw_value)
                records.append(record)
        start += count
    return overall, records


def _cell_path(cell, base_root, condition_root):
    root = base_root if cell in BASELINE_CELLS else condition_root
    return root / f"{cell}_dynamic_soc.parquet"


def _build_run_metadata(args, runs):
    """Builds the run.json metadata dict -- pure bookkeeping, no
    computation. Factored out of main() so it can be unit tested without
    running the full CMU/Panasonic pipeline. validation_interval and the
    three target_validation_labels_used_for_* flags document the
    (diagnostic-only) validation split's role: it feeds
    validation_mae/validation_rmse in training_history.csv (see _adapt's
    docstring), and nothing else -- never training, adaptation, or
    checkpoint/model selection.

    This is also where DTNN documents the deliberate points of difference
    from the Vidal CMU benchmark (see scripts/benchmarks/run_cmu_transfer.py)
    for cross-benchmark comparison/Results & Discussion -- evaluation_unit
    and metric_units below -- without changing DTNN's own training,
    adaptation, or evaluation behavior in any way."""
    cmu_manifest_path = getattr(args, "cmu_manifest", None)
    cmu_manifest_sha256 = (
        sha256_file(cmu_manifest_path)
        if cmu_manifest_path is not None and Path(cmu_manifest_path).exists()
        else None
    )
    return {
        "benchmark": "bian_2021_panasonic_multitemperature_to_cmu_supervised_development",
        "architecture": "CNN1->CNN2->two-layer-BiLSTM->FC->sequence-regression",
        "mmd_layers": ["CNN2", "BiLSTM2", "FC"],
        "objective": "source_mse + 5*target_mse + 0.1*adaptive_multiscale_mmd",
        "source_profiles": list(SOURCE_PROFILES),
        "source_temperatures_c": list(TEMPERATURES),
        "feature_order": list(FEATURES),
        "normalization": "source-global min-max applied unchanged to target",
        "target_protocol": "start at initial SOC; first 30% adapt; remaining 70% test",
        "validation_protocol": (
            "duration-matched missions disjoint from adapt/test; same "
            "initial-SOC/trailing-fraction slicing as the test split; "
            "diagnostic-only (see target_validation_labels_used_for_* below)"
        ),
        "target_soc_labels_used_for_adaptation": True,
        # DTNN has no separate purely-supervised target training step
        # distinct from adaptation -- target_mse (from target_y) is one of
        # the three terms optimizer.step() backprops on every adaptation
        # step (see _adapt). So "used for training" and "used for
        # adaptation" name the same step here; kept as two explicit flags
        # so a reader comparing run.json files doesn't have to know that.
        "target_soc_labels_used_for_training": True,
        "target_soc_labels_used_for_checkpoint_selection": False,
        "target_soc_reference_used_for_posthoc_evaluation": True,
        # Metadata only, describing the (diagnostic-only) validation split
        # recorded per-run as validation_missions_diagnostic in each
        # `runs` entry -- does not change what's computed.
        # validation_interval mirrors args.validation_interval (also
        # present under "arguments" below); repeated here at the top
        # level, alongside the target_soc_labels_* flags, so the
        # validation split's role is discoverable without having to know
        # it lives under "arguments".
        "validation_interval": args.validation_interval,
        "target_validation_labels_used_for_diagnostics": True,
        "target_validation_labels_used_for_training": False,
        "target_validation_labels_used_for_checkpoint_selection": False,
        # Diagnostic-only, metadata-only: documents (never changes) the
        # evaluation convention DTNN's existing _evaluate() already uses
        # (metrics computed over every timestep of every window -- no
        # last-step slicing anywhere in _evaluate) versus Vidal's
        # last_step_per_window convention (run_cmu_transfer.py's
        # evaluate_target_last). Neither script's evaluation behavior is
        # touched by this field; a future independent canonical post-hoc
        # evaluator is expected to reconcile the two, not this script.
        "evaluation_unit": "all_timesteps",
        "evaluation_unit_note": (
            "DTNN evaluates every timestep of every test window "
            "(all_timesteps); the Vidal CMU benchmark "
            "(run_cmu_transfer.py) evaluates only each window's final "
            "timestep (last_step_per_window). Unchanged by this run; "
            "documented here for cross-benchmark comparison only."
        ),
        # Diagnostic-only, metadata-only: documents each metric's unit
        # scale without changing any of them. overall_*_percent/
        # mission_macro_*_percent in metrics.csv/summary.csv are legacy,
        # percent-scale ([0,100]) metrics -- unchanged. Every
        # machine-readable per-timestep/per-epoch diagnostic added by the
        # Vidal-alignment pass (soc_reference/soc_prediction/signed_error/
        # absolute_error in test_predictions.parquet, validation_mae/
        # validation_rmse in training_history.csv) is already a raw [0,1]
        # fraction, matching Vidal's own prediction-level fields and
        # compute_metrics' native output -- so no unit change was needed
        # for those, only this documentation.
        "metric_units": {
            "overall_mae_percent": "percent (0-100); legacy, unchanged",
            "overall_rmse_percent": "percent (0-100); legacy, unchanged",
            "overall_max_error_percent": "percent (0-100); legacy, unchanged",
            "overall_bias_percent": "percent (0-100); legacy, unchanged",
            "mission_macro_mae_percent": "percent (0-100); legacy, unchanged",
            "mission_macro_rmse_percent": "percent (0-100); legacy, unchanged",
            "validation_mae": "fraction (0-1)",
            "validation_rmse": "fraction (0-1)",
            "soc_reference": "fraction (0-1)",
            "soc_prediction": "fraction (0-1)",
            "signed_error": "fraction (0-1)",
            "absolute_error": "fraction (0-1)",
        },
        "label_is_ground_truth": False,
        "formal_benchmark_eligible": False,
        # Git/data provenance -- get_full_commit_hash/is_dirty never raise
        # (they return "unknown"/False when git is unavailable or the
        # checkout isn't a git repo, e.g. a CI/sandbox snapshot), and
        # cmu_manifest_sha256 is None whenever args has no cmu_manifest
        # attribute or that path doesn't exist (e.g. the unit tests below,
        # which build `args` as a bare SimpleNamespace) -- so this never
        # breaks a caller that only supplies the fields it cares about.
        "git_commit": get_full_commit_hash(REPO_ROOT),
        "git_dirty": is_dirty(REPO_ROOT),
        "cmu_manifest_sha256": cmu_manifest_sha256,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "runs": runs,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panasonic-data-root", type=Path, default=REPO_ROOT / "data/raw/Panasonic_18650PF"
    )
    parser.add_argument(
        "--cmu-base-root",
        type=Path,
        default=REPO_ROOT
        / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1_ns_relabel",
    )
    parser.add_argument(
        "--cmu-condition-root",
        type=Path,
        default=REPO_ROOT
        / "outputs/datasets/cmu_evtol/dynamic_capacity_soc_development_v1_extension4_ns_relabel",
    )
    parser.add_argument(
        "--cmu-manifest",
        type=Path,
        default=REPO_ROOT
        / "data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1_ns_relabel.csv",
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "outputs/benchmarks/bian_2021_cmu"
    )
    parser.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS))
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--adapt-fraction", type=float, default=0.30)
    parser.add_argument("--initial-soc", type=float, default=0.70)
    parser.add_argument("--source-epochs", type=int, default=2000)
    parser.add_argument("--transfer-epochs", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--transfer-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--transfer-lr", type=float, default=1e-4)
    parser.add_argument("--target-weight", type=float, default=5.0)
    parser.add_argument("--mmd-weight", type=float, default=0.1)
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=10,
        help="record validation_mae/validation_rmse in training_history.csv "
        "every N adaptation epochs, plus always on epoch 1 and the final "
        "epoch; diagnostic only, never used for early stopping, checkpoint "
        "selection, or training. 0 disables validation recording entirely.",
    )
    parser.add_argument("--budget-seconds", type=float, default=8778.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lightweight", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        not 0 < args.adapt_fraction < 1
        or not 0 < args.initial_soc <= 1
        or min(args.source_epochs, args.transfer_epochs, args.repeats) <= 0
    ):
        raise ValueError("fractions/SOC and epoch/repeat counts are invalid")
    device = get_device(args.device)
    source_x, source_y, source_feature_min, source_feature_max = _source_data(
        args.panasonic_data_root, args.sequence_length
    )
    manifest = pd.read_csv(args.cmu_manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows, runs, training_history_rows, source_training_history_rows = [], [], [], []
    widths = (64, 32, 64) if args.lightweight else (256, 64, 256)

    # test_predictions.parquet is written incrementally, one repeat x cell x
    # method chunk at a time -- never by accumulating every per-timestep
    # record for the whole experiment in memory (test_prediction_rows was
    # removed). Chunks go to a ".incomplete" temp file via ParquetWriter;
    # only after every chunk below has been written without error is the
    # writer closed and the temp file atomically replaced onto the final
    # "test_predictions.parquet" name. On any exception the writer is still
    # closed (so the temp file has a valid parquet footer and is readable
    # for inspection) but is left at the ".incomplete" name -- an explicit,
    # unambiguous marker that the file is a partial write, never silently
    # promoted to the final name.
    test_predictions_tmp_path = args.output_root / "test_predictions.parquet.incomplete"
    test_predictions_final_path = args.output_root / "test_predictions.parquet"
    test_predictions_writer = pq.ParquetWriter(test_predictions_tmp_path, TEST_PREDICTIONS_SCHEMA)
    test_predictions_completed = False
    try:
        for repeat in range(args.repeats):
            seed = args.seed + repeat
            _seed(seed)
            print(
                f"repeat {repeat + 1}/{args.repeats}: pretraining on {len(source_x)} source windows"
            )
            source_model = DTNN(2, *widths).to(device)
            source_history = _train_source(
                source_model, source_x, source_y, args.source_epochs, args.batch_size, args.lr, device
            )
            for history_row in source_history:
                source_training_history_rows.append(
                    {"repeat": repeat + 1, "seed": seed, **history_row}
                )
            source_state = copy.deepcopy(source_model.state_dict())

            for cell in args.cells:
                path = _cell_path(cell, args.cmu_base_root, args.cmu_condition_root)
                expected = set(
                    manifest.loc[manifest["cell_id"] == cell, "source_artifact_sha256"].astype(str)
                )
                actual = sha256_file(path)
                if expected != {actual}:
                    raise ValueError(f"{cell}: CMU input hash does not match manifest")
                adapt_ids, validation_ids, test_ids, budget = select_duration_matched_missions(
                    manifest, cell, budget_seconds=args.budget_seconds
                )
                records = load_cmu_dynamic_soc_for_vidal([path], target_hz=1.0).records
                adapt_parts = _target_parts(
                    records, adapt_ids, args.initial_soc, args.adapt_fraction, "adapt"
                )
                adapt_x, adapt_y, adapt_missions = _target_data(
                    adapt_parts, args.sequence_length, source_feature_min, source_feature_max
                )
                # include_metadata=True: test_metadata carries
                # mission_sequence_order/elapsed_candidate_s/
                # original_time_s/source_row_right per timestep, sliced
                # and reshaped by the exact same starts/selection/
                # usable/count logic as test_x/test_y themselves (see
                # _target_parts/_windows) -- never a second, independent
                # slicing pass computed after the fact. Only the test
                # split carries this metadata (adapt/validation data
                # above/below are unaffected, still 3-tuples).
                test_parts = _target_parts(
                    records, test_ids, args.initial_soc, args.adapt_fraction, "test",
                    include_metadata=True,
                )
                test_x, test_y, test_missions, test_metadata = _target_data(
                    test_parts, args.sequence_length, source_feature_min, source_feature_max,
                    include_metadata=True,
                )
                # Diagnostic-only: join development_phase_role/voltage_v_raw/
                # current_a_raw/temperature_c_raw/capacity_ah/capacity_source/
                # source_rpt_id onto every test timestep by source_row_right,
                # straight from this cell's raw parquet at `path` (the same
                # file already hash-verified above) -- never from x (the
                # scaler-normalized model input) or from resampled data. One
                # per-cell vectorized merge (_join_diagnostic_fields_by_
                # source_row), not a Python dict accumulated across the
                # experiment; test_metadata['source_row_right'] is exactly
                # what _target_parts/_windows already built above, so the
                # join key is the same one used for the geometry columns.
                raw_diagnostic = _load_raw_diagnostic_fields(path)
                test_metadata = {
                    **test_metadata,
                    **_join_diagnostic_fields_by_source_row(
                        test_metadata["source_row_right"], raw_diagnostic
                    ),
                }
                # Diagnostic-only validation data: the existing validation_ids
                # split (already computed by select_duration_matched_missions
                # above, previously unused for anything but a mission count),
                # sliced with the SAME rule as test ("test" portion -- start at
                # initial_soc, keep the trailing (1 - adapt_fraction) of each
                # mission). Never touches adapt_ids/adapt_x, so it cannot affect
                # what the model trains on. A validation split that yields no
                # complete windows (e.g. very few validation missions) simply
                # disables the diagnostic for this cell rather than failing the
                # whole run.
                validation_x = validation_y = None
                if validation_ids:
                    try:
                        validation_parts = _target_parts(
                            records, validation_ids, args.initial_soc, args.adapt_fraction, "test"
                        )
                        validation_x, validation_y, _validation_missions = _target_data(
                            validation_parts, args.sequence_length,
                            source_feature_min, source_feature_max,
                        )
                    except ValueError:
                        validation_x = validation_y = None
                condition = str(manifest.loc[manifest["cell_id"] == cell, "condition"].iloc[0])
                model = DTNN(2, *widths).to(device)
                model.load_state_dict(source_state)
                source_only, source_only_predictions = _evaluate(
                    model, test_x, test_y, test_missions, args.batch_size, device,
                    return_predictions=True, metadata=test_metadata,
                )
                training, adapt_history = _adapt(
                    model, source_x, source_y, adapt_x, adapt_y, args, device,
                    validation_x=validation_x, validation_y=validation_y,
                )
                adapted, adapted_predictions = _evaluate(
                    model, test_x, test_y, test_missions, args.batch_size, device,
                    return_predictions=True, metadata=test_metadata,
                )
                for history_row in adapt_history:
                    training_history_rows.append(
                        {"repeat": repeat + 1, "seed": seed, "cell_id": cell, **history_row}
                    )
                for method, metrics in (("Source only", source_only), ("Bian MDA", adapted)):
                    rows.append(
                        {
                            "repeat": repeat + 1,
                            "seed": seed,
                            "cell_id": cell,
                            "initial_soc_percent": int(round(args.initial_soc * 100)),
                            "method": method,
                            **metrics,
                        }
                    )
                # Diagnostic/reporting only: per-timestep prediction records for
                # test_predictions.parquet, unified across methods with the same
                # framework/method/repeat/seed/cell_id/condition/split columns
                # used elsewhere in this benchmark suite. Built from the same
                # per-timestep records _evaluate() already computed above (see
                # its return_predictions=True branch) -- does not change any
                # metric computation. Written immediately as one chunk per
                # repeat x cell x method rather than accumulated in memory
                # for the whole experiment (see test_predictions_writer
                # above).
                for method, predictions in (
                    ("Source only", source_only_predictions),
                    ("Bian MDA", adapted_predictions),
                ):
                    chunk_rows = [
                        {
                            "framework": "bian_2021",
                            "method": method,
                            "repeat": repeat + 1,
                            "seed": seed,
                            "cell_id": cell,
                            "condition": condition,
                            "split": "test",
                            **record,
                        }
                        for record in predictions
                    ]
                    _write_prediction_chunk(test_predictions_writer, chunk_rows)
                runs.append(
                    {
                        "repeat": repeat + 1,
                        "seed": seed,
                        "cell_id": cell,
                        "initial_soc_percent": int(round(args.initial_soc * 100)),
                        "cmu_input_sha256": actual,
                        "adaptation_budget": budget,
                        "adaptation_fraction_within_mission": args.adapt_fraction,
                        "adaptation_missions": len(adapt_missions),
                        "adaptation_windows": len(adapt_x),
                        # Renamed from validation_missions_unused: these
                        # missions ARE used now, for the periodic
                        # validation_mae/validation_rmse diagnostic in
                        # training_history.csv (see validation_interval and
                        # the target_validation_labels_used_for_* flags
                        # below) -- just never for training, adaptation, or
                        # checkpoint/model selection. Value is unchanged
                        # (still len(validation_ids)).
                        "validation_missions_diagnostic": len(validation_ids),
                        "training_final_batch": training,
                    }
                )
                print(
                    f"  {cell}: source={source_only['overall_rmse_percent']:.3f}% "
                    f"MDA={adapted['overall_rmse_percent']:.3f}%"
                )
        test_predictions_completed = True
    finally:
        # Always close the writer, even on failure, so the temp file has a
        # valid parquet footer and can be opened for inspection. Only a
        # clean, exception-free completion of every chunk above is
        # promoted (via an atomic replace) to the final
        # "test_predictions.parquet" name -- on any exception the file
        # stays at its ".incomplete" name, and the exception itself still
        # propagates (no except clause here) so the run is reported as
        # failed rather than silently swallowed.
        test_predictions_writer.close()
        if test_predictions_completed:
            test_predictions_tmp_path.replace(test_predictions_final_path)

    table = pd.DataFrame(rows)
    table.to_csv(args.output_root / "metrics.csv", index=False)
    training_history = pd.DataFrame(
        training_history_rows,
        columns=[
            "repeat", "seed", "cell_id", "epoch",
            "source_mse", "target_mse", "adaptive_mmd", "total_loss",
            "layer_mmd_cnn2", "layer_mmd_bilstm2", "layer_mmd_fc",
            "layer_weight_cnn2", "layer_weight_bilstm2", "layer_weight_fc",
            "learning_rate", "runtime_seconds", "optimizer_steps",
            "validation_mae", "validation_rmse", "validation_evaluated",
        ],
    )
    training_history.to_csv(args.output_root / "training_history.csv", index=False)
    source_training_history = pd.DataFrame(
        source_training_history_rows,
        columns=["repeat", "seed", "epoch", "source_mse", "learning_rate", "runtime_seconds"],
    )
    source_training_history.to_csv(
        args.output_root / "source_training_history.csv", index=False
    )
    summary = (
        table.groupby(["cell_id", "initial_soc_percent", "method"], sort=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    summary.to_csv(args.output_root / "summary.csv", index=False)
    metadata = _build_run_metadata(args, runs)
    (args.output_root / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
