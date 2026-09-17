#!/usr/bin/env python3
"""Run CNN-LSTM, TCN, and Transformer-encoder backbones through a Vidal-style
supervised transfer protocol on the CMU eVTOL core-seven cells.

This script deliberately does NOT import from
``battery_benchmark.reproductions.vidal_2019`` (case1.py / case2.py /
windows.py / resampling.py / sequence_v2.py). Those modules produced the
already-published, formal-status results in
``docs/benchmarks/vidal_panasonic_to_cmu_evtol_results_v0_1.md`` and must not
be touched by this experiment. Instead this script reimplements the same
*protocol* -- Panasonic 25 C, 1 Hz, non-overlapping fixed-length windows,
source-fitted min-max normalisation, nested chronological target-label
budgets, full-parameter fine-tuning restarted from the frozen source
checkpoint for every budget -- using only shared dataset adapters
(``datasets.cmu_evtol``, ``reproductions.shen_2023.data``) and the generic
model registry (``models.cnn_lstm`` / ``models.tcn`` /
``models.transformer_encoder``).

Budgets reuse the nested mission-budget flags already frozen in
``data/manifests/cmu_evtol/cmu_evtol_supervised_budget_manifest_v0_1_ns_relabel.csv``
(``budget_1pct``, ``budget_5pct``, ``budget_10pct``, ``vidal_partial``,
``vidal_full``), so the resulting budget points line up exactly with the
already-published Vidal LSTM sweep.

CMU ``soc_dynamic`` is a development dynamic-capacity reference, not
measured ground truth. Every run written by this script is
``label_is_ground_truth=false`` and ``formal_benchmark_eligible=false``.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
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

from battery_benchmark.benchmarks.cmu_diagnostics import (
    CELL_CONDITIONS,
    DIAGNOSTIC_METADATA_CASTERS,
    join_diagnostic_fields_by_source_row,
    load_raw_diagnostic_fields,
)
from battery_benchmark.datasets.cmu_evtol.vidal_adapter import load_cmu_dynamic_soc_for_vidal
from battery_benchmark.evaluation.metrics import compute_metrics
from battery_benchmark.models.registry import ModelRegistry
import battery_benchmark.models.cnn_lstm  # noqa: F401 -- registers "cnn_lstm"
import battery_benchmark.models.tcn  # noqa: F401 -- registers "tcn"
import battery_benchmark.models.transformer_encoder  # noqa: F401 -- registers "transformer_encoder"
from battery_benchmark.reproductions.shen_2023.data import PROFILES, load_temperature_cycles_and_range
from battery_benchmark.utils.device import get_device
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty
from battery_benchmark.utils.hashing import sha256_file

CELLS = ("VAH01", "VAH17", "VAH27", "VAH30", "VAH15", "VAH12", "VAH11")
BASELINE_CELLS = {"VAH01", "VAH17", "VAH27"}
FEATURES = ("current", "voltage", "temperature")
MODEL_NAMES = ("cnn_lstm", "tcn", "transformer_encoder")
BUDGETS = ("zero_shot", "1pct", "5pct", "10pct", "partial", "full")
BUDGET_COLUMNS = {
    "1pct": "budget_1pct",
    "5pct": "budget_5pct",
    "10pct": "budget_10pct",
    "partial": "vidal_partial",
    "full": "vidal_full",
}
# output_clip=False during training: all three models clamp their final layer
# to [0, 1] by default, and torch.clamp has exactly zero gradient outside that
# range. If a model's untrained output happens to start outside [0, 1] (this
# reliably happens for cnn_lstm, not for tcn/transformer_encoder, given the
# same seed/data/lr -- verified empirically: with the clamp active, cnn_lstm's
# training loss is bit-for-bit identical across every epoch and every
# fine-tuning budget, because no gradient ever reaches its weights), the
# model is permanently stuck predicting a constant and never trains at all.
# _evaluate() already clips its predictions to [0, 1] for the reported
# metrics, so this only removes a redundant, gradient-killing clamp from the
# training path -- it does not change what gets reported.
MODEL_CONFIG = {"input_size": len(FEATURES), "output_clip": False}

# `_windows`/`_evaluate` below build non-overlapping fixed-length windows
# and evaluate every raw timestep in them (no last-step slicing) --
# all_timesteps, identical convention to DTNN, unlike DARM's
# last_step_per_window.
EVALUATION_UNIT = "all_timesteps"

# Fixed schema for test_predictions.parquet -- one row per raw CMU
# timestep, identical shape across all three backbones (no per-model
# divergence), mirroring run_bian_cmu.py's TEST_PREDICTIONS_SCHEMA.
TEST_PREDICTIONS_SCHEMA = pa.schema(
    [
        ("model", pa.string()),
        ("method", pa.string()),
        ("budget", pa.string()),
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


def _windows(
    parts: list[tuple[np.ndarray, np.ndarray, str]]
    | list[tuple[np.ndarray, np.ndarray, str, dict]],
    length: int,
    *,
    include_metadata: bool = False,
):
    """Reshape a list of (features, soc, id) time series into non-overlapping
    windows. ``include_metadata=False`` (every pre-existing caller):
    unchanged output, byte-for-byte identical to before this parameter
    existed.

    ``include_metadata=True`` (diagnostic test-window construction only):
    each part is instead a 4-tuple ``(x, y, item_id, metadata)`` where
    ``metadata`` maps each of mission_sequence_order/elapsed_candidate_s/
    original_time_s/source_row_right to a 1-D per-row array the same
    length as x/y for that part. Every metadata array is truncated and
    reshaped with the EXACT SAME ``usable``/``count`` values computed from
    x/y -- not a separate slicing pass -- so metadata stays index-for-index
    aligned with the SOC timestep it describes."""
    xs, ys, ids = [], [], []
    metadata_lists = (
        {
            "mission_sequence_order": [],
            "elapsed_candidate_s": [],
            "original_time_s": [],
            "source_row_right": [],
        }
        if include_metadata
        else None
    )
    for part in parts:
        if include_metadata:
            x, y, item_id, metadata = part
        else:
            x, y, item_id = part
        usable = min(len(x), len(y)) // length * length
        if usable:
            count = usable // length
            xs.append(x[:usable].reshape(count, length, x.shape[-1]))
            ys.append(y[:usable].reshape(count, length))
            ids.append((item_id, count))
            if include_metadata:
                for name in metadata_lists:
                    metadata_lists[name].append(metadata[name][:usable].reshape(count, length))
    if not xs:
        raise ValueError("No complete non-overlapping windows")
    x_out = np.concatenate(xs).astype(np.float32)
    y_out = np.concatenate(ys).astype(np.float32)
    if not include_metadata:
        return x_out, y_out, ids
    metadata_out = {name: np.concatenate(values) for name, values in metadata_lists.items()}
    return x_out, y_out, ids, metadata_out


def _source_data(root: Path, length: int):
    """Panasonic 25 C, all nine profiles, already min-max normalised to [0, 1]."""
    cycles, feature_min, feature_max = load_temperature_cycles_and_range(root, 25)
    parts = [(cycles[name].features, cycles[name].soc, name) for name in PROFILES]
    x, y, _ = _windows(parts, length)
    return x, y, feature_min, feature_max


def _cell_path(cell: str, base_root: Path, condition_root: Path) -> Path:
    root = base_root if cell in BASELINE_CELLS else condition_root
    return root / f"{cell}_dynamic_soc.parquet"


def _budget_mission_ids(manifest: pd.DataFrame, cell_id: str, budget: str) -> tuple[set[str], set[str]]:
    rows = manifest.loc[manifest["cell_id"] == cell_id]
    if rows.empty:
        raise ValueError(f"{cell_id}: no manifest rows")
    test_ids = set(rows.loc[rows["split"] == "test", "mission_cycle_id"].astype(str))
    if budget == "zero_shot":
        return set(), test_ids
    column = BUDGET_COLUMNS[budget]
    train = rows.loc[rows["split"] == "train"]
    adapt_ids = set(train.loc[train[column].astype(bool), "mission_cycle_id"].astype(str))
    if not adapt_ids:
        raise ValueError(f"{cell_id}/{budget}: empty adaptation mission set")
    return adapt_ids, test_ids


def _target_windows(
    records, mission_ids: set[str], feature_min, feature_max, length: int,
    *, include_metadata: bool = False,
):
    """``include_metadata=False`` (every pre-existing caller): unchanged
    output. ``include_metadata=True`` (diagnostic test-window construction
    only): also builds mission_sequence_order/elapsed_candidate_s/
    original_time_s/source_row_right directly from the SAME frame x/y come
    from, passed straight through to ``_windows`` for the identical
    truncation/reshape."""
    if not mission_ids:
        raise ValueError("mission_ids is empty")
    scale = np.where(feature_max == feature_min, 1.0, feature_max - feature_min)
    parts = []
    for record in records:
        mission = str(record.metadata["mission_cycle_id"])
        if mission not in mission_ids:
            continue
        frame = record.data
        x = (frame.loc[:, list(FEATURES)].to_numpy(dtype=np.float64) - feature_min) / scale
        y = frame["soc"].to_numpy(dtype=np.float64)
        if not include_metadata:
            parts.append((x, y, mission))
            continue
        metadata = {
            "mission_sequence_order": np.full(
                len(frame), int(record.metadata["sequence_order"]), dtype=np.int64
            ),
            "elapsed_candidate_s": frame["elapsed_candidate_s"].to_numpy(dtype=np.float64),
            "original_time_s": frame["original_time_s"].to_numpy(dtype=np.float64),
            "source_row_right": frame["source_row_right"].to_numpy(dtype=np.int64),
        }
        parts.append((x, y, mission, metadata))
    if not parts:
        raise ValueError("No CMU records matched the requested mission_ids")
    return _windows(parts, length, include_metadata=include_metadata)


def _loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, *, generator=None
) -> DataLoader:
    """``generator`` defaults to ``None`` for every pre-existing (training)
    call site, unchanged. ``_validation_diagnostic_metrics`` passes its own
    local ``torch.Generator()`` instead -- PyTorch's DataLoader always
    draws one value from its generator to seed itself, even with
    ``shuffle=False``, so constructing/iterating a validation loader with
    the default generator would silently advance ``torch.default_generator``
    and perturb later training-loader shuffles. A separate, local generator
    keeps that draw off the global RNG stream."""
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=min(batch_size, len(x)),
        shuffle=shuffle,
        generator=generator,
    )


@torch.no_grad()
def _validation_diagnostic_metrics(model, x, y, batch_size, device):
    """Lightweight MAE/RMSE (fraction scale, NOT percent), for the periodic
    validation_mae/validation_rmse diagnostic in training_history.csv only.
    Never used for early stopping or checkpoint selection -- ``_train``
    below always restarts full-parameter fine-tuning from the frozen
    source checkpoint for every budget regardless of what this returns. A
    pure eval-mode forward pass under no_grad; the model's training/eval
    mode is restored to whatever it was before this call, so the caller's
    training loop is unaffected either way. Uses a local ``torch.Generator``
    (see ``_loader``'s docstring) so this diagnostic never perturbs the
    training loader's RNG stream."""
    was_training = model.training
    model.eval()
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


def _train(
    model: nn.Module, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int,
    lr: float, device: torch.device, label: str, *, validation_data=None,
) -> list[dict]:
    """Full-parameter supervised MSE training (used for both source pretraining
    and target fine-tuning -- Vidal Case 2 unlocks every trainable parameter
    for target fine-tuning rather than freezing any layer). Unchanged
    training loop/optimizer/loss/data order from before this docstring --
    the only addition is capturing each epoch's already-computed mean loss
    into a returned list (previously discarded except for the periodic
    print) and, when ``validation_data=(val_x, val_y)`` is supplied,
    running the diagnostic-only validation pass above after each epoch.
    ``validation_data`` defaults to ``None`` (source pretraining has no
    natural held-out split -- Section IV explicitly says not to fabricate
    one), in which case validation_evaluated is honestly False for every
    epoch."""
    loader = _loader(x, y, batch_size, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history: list[dict] = []
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(xb)
        epoch_mse = total / len(x)
        validation_mae = validation_rmse = None
        validation_evaluated = False
        if validation_data is not None:
            val_x, val_y = validation_data
            validation_mae, validation_rmse = _validation_diagnostic_metrics(
                model, val_x, val_y, batch_size, device
            )
            validation_evaluated = True
        history.append(
            {
                "epoch": epoch + 1,
                "train_mse": epoch_mse,
                "learning_rate": lr,
                "runtime_seconds": time.perf_counter() - epoch_started,
                "validation_mae": validation_mae,
                "validation_rmse": validation_rmse,
                "validation_evaluated": validation_evaluated,
            }
        )
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % max(epochs // 5, 1) == 0:
            print(f"    {label} epoch {epoch + 1}/{epochs}: mse={epoch_mse:.6f}")
    return history


@torch.no_grad()
def _evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, missions, batch_size: int,
              device: torch.device) -> dict[str, float]:
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
            "mission_macro_mae_percent": float(np.mean([m["mae"] for m in per_mission]) * 100),
            "mission_macro_rmse_percent": float(np.mean([m["rmse"] for m in per_mission]) * 100),
            "missions": len(missions),
            "windows": len(x),
        }
    )
    return overall


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
        default=REPO_ROOT / "outputs/benchmarks/panasonic_to_cmu_evtol/backbone_transfer",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS))
    parser.add_argument("--budgets", nargs="+", choices=BUDGETS, default=list(BUDGETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--source-epochs", type=int, default=200)
    parser.add_argument("--transfer-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--transfer-lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume", action="store_true",
        help="skip (model, seed) source pretraining and (model, seed, cell, budget) "
        "evaluations that already have a result file on disk",
    )
    parser.add_argument(
        "--no-save-checkpoints", action="store_true",
        help="do not write per-budget model checkpoints (source.pt / <budget>.pt); "
        "result JSON rows are always written",
    )
    return parser.parse_args()


def _save_timing(seed_root: Path, name: str, *, seconds: float | None, epochs: int | None, device: str) -> None:
    """Persist a timing record so it survives --resume: if this run resumed
    from a checkpoint (or an already-loaded cell's records) instead of
    redoing the work, we still want the original run's timing available for
    the rows written this time."""

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


@torch.no_grad()
def _predict_all(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    """Diagnostic-only companion to ``_evaluate``: returns the raw
    per-timestep prediction array using the IDENTICAL loader construction
    (same TensorDataset, same batch_size, ``shuffle=False``) as
    ``_evaluate``, so predictions line up 1:1, in the same order, with the
    metadata arrays from ``_target_windows(..., include_metadata=True)``. A
    test recomputes overall_mae_percent/overall_rmse_percent from these
    (clipped) predictions and asserts they match ``_evaluate``'s output."""
    model.eval()
    dummy_y = np.zeros((len(x), x.shape[1]), dtype=np.float32)
    prediction = torch.cat(
        [model(xb.to(device)).cpu() for xb, _ in _loader(x, dummy_y, batch_size, False)]
    ).numpy()
    return prediction.astype(np.float64)


def _build_prediction_rows(
    *,
    model_name: str,
    method: str,
    budget: str,
    seed: int,
    cell_id: str,
    condition: str,
    soc_reference: np.ndarray,
    soc_prediction: np.ndarray,
    metadata: dict[str, np.ndarray],
    missions: list[tuple[str, int]],
    sequence_length: int,
    raw_diagnostic: pd.DataFrame,
) -> list[dict]:
    """Assembles one test_predictions.parquet row per raw CMU timestep,
    identical shape/logic across all three backbones. Joins Ns/mission_phase/
    raw V-I-T/capacity provenance by this timestep's own source_row_right --
    never reverse-derived from x.

    ``missions`` is a list of ``(mission_id, window_count)`` -- window
    counts, NOT timestep counts. Every per-timestep array (``soc_reference``,
    ``soc_prediction``, and the four ``metadata`` arrays) must have length
    ``sum(window_count for _, window_count in missions) * sequence_length``;
    a mismatch raises immediately rather than silently misaligning rows
    (this is exactly the bug this function previously had: it built
    ``mission_ids``/``window_index`` at one-entry-per-WINDOW length but then
    indexed them with the flat per-TIMESTEP position ``i``).

    ``mission_cycle_id`` and ``window_index`` are built per-window first
    (one entry per window: the window's mission id, and its 0-based index
    WITHIN that mission, restarting at 0 for every mission), then each
    repeated ``sequence_length`` times to reach timestep-level length.
    ``timestep_index`` is 0..sequence_length-1 within every window."""
    n_windows_total = sum(window_count for _, window_count in missions)
    expected_points = n_windows_total * sequence_length
    for array_name, array in (
        ("soc_reference", soc_reference),
        ("soc_prediction", soc_prediction),
        ("metadata['mission_sequence_order']", metadata["mission_sequence_order"]),
        ("metadata['elapsed_candidate_s']", metadata["elapsed_candidate_s"]),
        ("metadata['original_time_s']", metadata["original_time_s"]),
        ("metadata['source_row_right']", metadata["source_row_right"]),
    ):
        if len(array) != expected_points:
            raise ValueError(
                f"_build_prediction_rows: {array_name} has length {len(array)}, "
                f"expected {expected_points} "
                "(= sum(window_count for missions) * sequence_length)"
            )
    n = expected_points
    if not n:
        return []
    joined = join_diagnostic_fields_by_source_row(metadata["source_row_right"], raw_diagnostic)
    clipped_prediction = np.clip(soc_prediction, 0.0, 1.0)
    # Per-window mission id and per-mission-restarting window index, built
    # ONE ENTRY PER WINDOW first, then each repeated sequence_length times
    # -- never indexed directly by the flat timestep position i.
    window_mission_ids = [
        mission for mission, window_count in missions for _ in range(window_count)
    ]
    window_indices_within_mission = (
        np.concatenate([np.arange(window_count) for _, window_count in missions])
        if missions
        else np.array([], dtype=np.int64)
    )
    mission_ids = np.repeat(np.array(window_mission_ids, dtype=object), sequence_length)
    window_indices = np.repeat(window_indices_within_mission, sequence_length)
    timestep_indices = np.tile(np.arange(sequence_length), n_windows_total)
    rows = []
    for i in range(n):
        row = {
            "model": model_name,
            "method": method,
            "budget": budget,
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


def _rebuild_test_predictions(output_root: Path) -> None:
    """Aggregate every per-(model, seed, cell, budget) test_predictions
    parquet sidecar into the top-level test_predictions.parquet -- never a
    single truncate-on-open writer for the whole (multi-process, per-model)
    run, same merge-not-overwrite rationale as ``_rebuild_cell_metrics``."""
    paths = sorted(output_root.glob("*/seed_*/*/*_predictions.parquet"))
    if not paths:
        return
    pq.write_table(
        pa.concat_tables([pq.read_table(path) for path in paths]),
        output_root / "test_predictions.parquet",
    )


def _rebuild_training_history(output_root: Path) -> None:
    """Aggregate every per-(model, seed) source training-history sidecar and
    every per-(model, seed, cell, budget) target training-history sidecar
    into the top-level training_history.csv."""
    paths = sorted(output_root.glob("*/seed_*/source_training_history.csv")) + sorted(
        output_root.glob("*/seed_*/*/*_training_history.csv")
    )
    if not paths:
        return
    frames = [pd.read_csv(path) for path in paths]
    pd.concat(frames, ignore_index=True).to_csv(output_root / "training_history.csv", index=False)


def _validation_mission_ids(manifest: pd.DataFrame, cell_id: str) -> set[str]:
    rows = manifest.loc[manifest["cell_id"] == cell_id]
    return set(rows.loc[rows["split"] == "validation", "mission_cycle_id"].astype(str))


def _rebuild_cell_metrics(output_root: Path) -> pd.DataFrame:
    """Aggregate every per-(model, seed, cell, budget) result JSON under output_root.

    cell_metrics.csv is shared across models, and each backbone is normally
    launched as its own subprocess/notebook cell (one ``--models`` value per
    invocation). Writing it from the current process's in-memory rows alone
    would make the last model to run silently overwrite the others' rows in
    the CSV, even though their result files are untouched on disk. This
    rebuilds the CSV from every <output_root>/<model>/seed_<seed>/<cell>/<budget>.json
    found, so the file always reflects everything completed so far --
    regardless of which invocation ran last -- and matches the same
    per-budget file layout the resume logic below reads and writes.
    """
    all_rows: list[dict] = []
    for result_path in sorted(output_root.glob("*/seed_*/*/*.json")):
        all_rows.append(json.loads(result_path.read_text(encoding="utf-8")))
    table = pd.DataFrame(all_rows)
    table.to_csv(output_root / "cell_metrics.csv", index=False)
    return table


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    device = get_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    print("loading Panasonic 25C source windows ...")
    source_x, source_y, feature_min, feature_max = _source_data(
        args.panasonic_data_root, args.sequence_length
    )
    manifest = pd.read_csv(args.cmu_manifest)

    rows: list[dict] = []
    param_counts: dict[str, int] = {}

    for model_name in args.models:
        for seed in args.seeds:
            seed_root = args.output_root / model_name / f"seed_{seed}"
            source_checkpoint = seed_root / "source.pt"

            model = ModelRegistry.build(model_name, MODEL_CONFIG).to(device)
            param_counts[model_name] = model.count_parameters()

            source_pretrain_seconds = None
            if args.resume and source_checkpoint.is_file():
                model.load_state_dict(torch.load(source_checkpoint, map_location=device))
                print(f"=== {model_name} / seed {seed}: resumed source checkpoint from {source_checkpoint} ===")
            else:
                _seed(seed)
                print(f"=== {model_name} / seed {seed}: pretraining on {len(source_x)} source windows ===")
                with _Timer(device) as timer:
                    source_history = _train(
                        model, source_x, source_y, args.source_epochs, args.batch_size, args.lr, device, "source"
                    )
                source_pretrain_seconds = timer.seconds
                print(f"    done in {source_pretrain_seconds:.1f}s")
                # Diagnostic-only: exports _train's own returned per-epoch
                # loss history (previously discarded) -- never retrains.
                seed_root.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    [{"model": model_name, "seed": seed, **entry} for entry in source_history]
                ).to_csv(seed_root / "source_training_history.csv", index=False)
                if not args.no_save_checkpoints:
                    seed_root.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), source_checkpoint)
            _save_timing(
                seed_root, "source_pretrain",
                seconds=source_pretrain_seconds, epochs=args.source_epochs, device=str(device),
            )
            source_pretrain_timing = _load_timing(seed_root, "source_pretrain")
            source_state = copy.deepcopy(model.state_dict())

            for cell in args.cells:
                cell_root = seed_root / cell
                path = None
                records = None
                actual = None
                cell_data_load_seconds = None
                raw_diagnostic = None

                for budget in args.budgets:
                    result_path = cell_root / f"{budget}.json"
                    if args.resume and result_path.is_file():
                        rows.append(json.loads(result_path.read_text(encoding="utf-8")))
                        print(f"  {cell}/{budget}: resumed from {result_path}")
                        continue

                    if records is None:
                        with _Timer(device) as data_timer:
                            path = _cell_path(cell, args.cmu_base_root, args.cmu_condition_root)
                            expected = set(
                                manifest.loc[manifest["cell_id"] == cell, "source_artifact_sha256"].astype(str)
                            )
                            actual = sha256_file(path)
                            if expected != {actual}:
                                raise ValueError(f"{cell}: CMU input hash does not match manifest")
                            records = load_cmu_dynamic_soc_for_vidal([path], target_hz=1.0).records
                        cell_data_load_seconds = data_timer.seconds
                        raw_diagnostic = load_raw_diagnostic_fields(path)
                    cell_root.mkdir(parents=True, exist_ok=True)
                    # Stored at seed_root (not cell_root) with the cell name in
                    # the filename: cell_root already holds <budget>.json result
                    # files, and _rebuild_cell_metrics globs *every* json file
                    # four levels under output_root -- a timing file living
                    # inside cell_root would get silently ingested as a bogus
                    # result row.
                    _save_timing(
                        seed_root, f"{cell}_data_load",
                        seconds=cell_data_load_seconds, epochs=None, device=str(device),
                    )
                    cell_data_load_timing = _load_timing(seed_root, f"{cell}_data_load")

                    adapt_ids, test_ids = _budget_mission_ids(manifest, cell, budget)
                    test_x, test_y, test_missions, test_metadata = _target_windows(
                        records, test_ids, feature_min, feature_max, args.sequence_length,
                        include_metadata=True,
                    )
                    eval_model = ModelRegistry.build(model_name, MODEL_CONFIG).to(device)
                    eval_model.load_state_dict(source_state)
                    adapt_missions = 0
                    adapt_seconds = None
                    target_history: list[dict] = []
                    if budget != "zero_shot":
                        adapt_x, adapt_y, adapt_mission_ids = _target_windows(
                            records, adapt_ids, feature_min, feature_max, args.sequence_length
                        )
                        adapt_missions = len(adapt_mission_ids)
                        # Diagnostic-only: the CMU supervised-budget manifest
                        # already defines a "validation" split, disjoint from
                        # train/test, unused anywhere else in this script.
                        # When this cell has validation-split missions, they
                        # are used ONLY for a periodic read-only MAE/RMSE
                        # probe inside _train (see
                        # _validation_diagnostic_metrics) -- never for
                        # checkpoint selection or early stopping, which this
                        # script does not do at all (every budget always
                        # fine-tunes for the full args.transfer_epochs from
                        # the frozen source checkpoint).
                        validation_ids = _validation_mission_ids(manifest, cell)
                        validation_data = None
                        if validation_ids:
                            try:
                                val_x, val_y, _ = _target_windows(
                                    records, validation_ids, feature_min, feature_max,
                                    args.sequence_length,
                                )
                                validation_data = (val_x, val_y)
                            except ValueError:
                                validation_data = None  # no complete validation windows for this cell
                        _seed(seed)
                        with _Timer(device) as adapt_timer:
                            target_history = _train(
                                eval_model, adapt_x, adapt_y, args.transfer_epochs,
                                args.batch_size, args.transfer_lr, device, budget,
                                validation_data=validation_data,
                            )
                        adapt_seconds = adapt_timer.seconds
                        pd.DataFrame(
                            [
                                {"model": model_name, "seed": seed, "cell_id": cell, "budget": budget, **entry}
                                for entry in target_history
                            ]
                        ).to_csv(cell_root / f"{budget}_training_history.csv", index=False)
                    with _Timer(device) as eval_timer:
                        metrics = _evaluate(eval_model, test_x, test_y, test_missions, args.batch_size, device)

                    # Diagnostic-only: predictions via the SAME loader
                    # construction _evaluate uses, joined with
                    # Ns/mission_phase/raw V-I-T/capacity provenance. Never
                    # feeds back into metrics or model state -- `metrics`
                    # above was already computed by the untouched _evaluate
                    # call.
                    raw_prediction = _predict_all(eval_model, test_x, args.batch_size, device)
                    prediction_rows = _build_prediction_rows(
                        model_name=model_name,
                        method="zero_shot" if budget == "zero_shot" else "fine_tuned",
                        budget=budget,
                        seed=seed,
                        cell_id=cell,
                        condition=CELL_CONDITIONS.get(cell, cell),
                        soc_reference=test_y.ravel().astype(np.float64),
                        soc_prediction=raw_prediction.ravel(),
                        metadata={name: values.ravel() for name, values in test_metadata.items()},
                        missions=test_missions,
                        sequence_length=args.sequence_length,
                        raw_diagnostic=raw_diagnostic,
                    )
                    if prediction_rows:
                        pq.write_table(
                            pa.Table.from_pylist(prediction_rows, schema=TEST_PREDICTIONS_SCHEMA),
                            cell_root / f"{budget}_predictions.parquet",
                        )
                    row_seconds = sum(
                        s for s in (cell_data_load_seconds, adapt_seconds, eval_timer.seconds) if s is not None
                    )
                    row = {
                        "model": model_name,
                        "cell_id": cell,
                        "budget": budget,
                        "seed": seed,
                        "device": str(device),
                        "source_pretrain_seconds": source_pretrain_timing["seconds"],
                        "source_pretrain_epochs": source_pretrain_timing["epochs"],
                        "cell_data_load_seconds": cell_data_load_timing["seconds"],
                        "adapt_seconds": adapt_seconds,
                        "eval_seconds": eval_timer.seconds,
                        "row_seconds": row_seconds,
                        "cmu_input_sha256": actual,
                        "adaptation_missions": adapt_missions,
                        "n_parameters": param_counts[model_name],
                        "label_is_ground_truth": False,
                        "formal_benchmark_eligible": False,
                        **metrics,
                    }
                    print(
                        f"  {cell}/{budget}: MAE={metrics['overall_mae_percent']:.3f}% "
                        f"RMSE={metrics['overall_rmse_percent']:.3f}% ({row_seconds:.1f}s)"
                    )

                    result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
                    if not args.no_save_checkpoints and budget != "zero_shot":
                        torch.save(eval_model.state_dict(), cell_root / f"{budget}.pt")
                    rows.append(row)
                    _rebuild_cell_metrics(args.output_root)

    table = _rebuild_cell_metrics(args.output_root)
    _rebuild_test_predictions(args.output_root)
    _rebuild_training_history(args.output_root)
    metadata = {
        "benchmark": "backbone_transfer_panasonic_25c_to_cmu_core7",
        "note": (
            "Reimplements the Vidal transfer protocol (source-fitted 1Hz "
            "windows, nested chronological target-label budgets, "
            "restart-from-source full-parameter fine-tuning) for CNN-LSTM, "
            "TCN, and Transformer-encoder backbones. Does not import "
            "battery_benchmark.reproductions.vidal_2019; the published Vidal "
            "LSTM CMU results are untouched by this script."
        ),
        "models": sorted(set(table["model"]) | set(args.models)) if not table.empty else list(args.models),
        "n_parameters": param_counts,
        "cells": list(args.cells),
        # Explicit record of which CMU dataset build and manifest this run
        # read from, and which Ns -> mission-phase vocabulary its diagnostic
        # fields use -- diagnostic/provenance-only, does not affect model,
        # training, or evaluation logic.
        "cmu_base_root": str(args.cmu_base_root),
        "cmu_condition_root": str(args.cmu_condition_root),
        "cmu_manifest_path": str(args.cmu_manifest),
        "cmu_manifest_sha256": sha256_file(args.cmu_manifest),
        "phase_definition": "canonical Ns mapping: 4=take_off, 5=cruise, 6=landing",
        "budgets": list(args.budgets),
        "budget_columns": BUDGET_COLUMNS,
        "seeds": list(args.seeds),
        "feature_order": list(FEATURES),
        "sequence_length": args.sequence_length,
        "source_profiles": list(PROFILES),
        "source_temperature_c": 25,
        "source_feature_min": feature_min.tolist(),
        "source_feature_max": feature_max.tolist(),
        "normalization": "Panasonic 25C source-fitted min-max applied unchanged to CMU target",
        "source_epochs": args.source_epochs,
        "transfer_epochs": args.transfer_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "transfer_lr": args.transfer_lr,
        "evaluation_unit": EVALUATION_UNIT,
        "evaluation_unit_note": (
            "_windows()/_evaluate build non-overlapping fixed-length windows "
            "and score every raw timestep in them -- all_timesteps, "
            "identical convention to DTNN, unlike DARM's last_step_per_window."
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
        "validation_protocol": (
            "Diagnostic-only: when the CMU manifest's 'validation' split has "
            "complete windows for a cell, _train() probes validation_mae/"
            "validation_rmse (fraction) after every fine-tuning epoch via a "
            "read-only, no_grad forward pass. Never used for early stopping "
            "or checkpoint selection -- every budget always fine-tunes for "
            "the full --transfer-epochs from the frozen source checkpoint, "
            "unchanged by this diagnostic."
        ),
        "target_soc_labels_used_for_training": True,
        "target_soc_labels_used_for_adaptation": True,
        "target_soc_labels_used_for_checkpoint_selection": False,
        "target_validation_labels_used_for_diagnostics": True,
        "target_validation_labels_used_for_training": False,
        "target_validation_labels_used_for_checkpoint_selection": False,
        "target_soc_reference_used_for_posthoc_evaluation": True,
        "label_is_ground_truth": False,
        "formal_benchmark_eligible": False,
        "git_commit": get_full_commit_hash(REPO_ROOT),
        "git_dirty": is_dirty(REPO_ROOT),
        # Wall-clock time spent by *this invocation* only -- with --resume,
        # anything skipped because a result file already existed is not
        # counted here. Per-row timing breakdowns (pretrain/data-load/adapt/
        # eval seconds, plus the device they ran on) live in cell_metrics.csv,
        # which is cumulative across every invocation.
        "this_invocation_wall_seconds": time.perf_counter() - run_start,
        "device": str(device),
    }
    metadata_path = args.output_root / "run_metadata.json"
    if metadata_path.is_file():
        # run_metadata.json is shared across models the same way cell_metrics.csv
        # is (each backbone is normally launched as its own invocation): merge
        # into the existing file instead of overwriting other models' entries.
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["models"] = sorted(set(previous.get("models", [])) | set(metadata["models"]))
        metadata["n_parameters"] = {**previous.get("n_parameters", {}), **metadata["n_parameters"]}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nthis invocation took {metadata['this_invocation_wall_seconds']:.1f}s wall-clock")


if __name__ == "__main__":
    main()
