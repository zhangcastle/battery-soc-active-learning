"""Training and evaluation loops for the Shen et al. (2023) reproduction.

Two independent transfer paths live here:

* ``pretrain_med`` / ``build_pseudo_target_dataset`` / ``med_adapt`` --
  the paper's actual proposed method (Algorithm 1). Source-free: the
  transfer stage never touches source data, source labels, or target
  labels. Pseudo-labels come once from the frozen source model's
  averaged-head output; only the target feature extractor is trained.
* ``train_source`` (DANN+MMD baseline) / ``adapt_dann_mmd`` -- the
  domain-adversarial + MMD comparison baseline the paper reports in
  Table VI alongside its proposed method. Not source-free in the same
  sense (it does use source data, batch-for-batch, throughout transfer),
  and must never be reported as "the Shen et al. method."

``build_pseudo_target_dataset`` is deliberately generic (it only needs a
list of ``ShenCycle``-shaped windows), so both the in-paper Panasonic
cross-temperature suite (``suite.py``) and the CMU eVTOL target-domain
benchmark (``scripts/benchmarks/run_sftnn_transfer_cmu.py``) can reuse the
exact same pseudo-labeling and confidence-filtering code.
"""

from __future__ import annotations

import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from battery_benchmark.evaluation.metrics import compute_metrics
from battery_benchmark.reproductions.shen_2023.data import (
    PROFILES,
    ShenCycle,
    selected_windows,
    windows,
)
from battery_benchmark.reproductions.shen_2023.model import (
    DANNBaselineNet,
    MEDNet,
    discrepancy,
    gaussian_mmd,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _improved(value: float, best: float, min_delta: float) -> bool:
    return not math.isfinite(best) or value < best * (1.0 - min_delta)


class _EpochTimer:
    """Opt-in, diagnostic-only per-epoch wall-clock timer for the training
    loops below. Entirely inert unless a caller passes
    ``record_epoch_seconds=True`` -- when it is not requested, this class
    is never constructed and adds zero overhead to the default (backward-
    compatible) code path. CUDA ops are async, so a timed block that ran
    on GPU is synced before both the start and end timestamps -- at most
    one synchronize() pair per epoch, skipped entirely on CPU."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def __enter__(self) -> "_EpochTimer":
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self._start


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    if not len(x):
        raise ValueError("Cannot construct a loader from zero windows")
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=min(batch_size, len(x)),
        shuffle=shuffle,
    )


@torch.no_grad()
def predict(
    model: MEDNet | DANNBaselineNet,
    cycles: list[ShenCycle],
    device: torch.device,
    tensor_layout: str = "paper",
    branch: str = "target",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, labels = [], []
    for cycle in cycles:
        x, y = windows([cycle], tensor_layout)
        for xb, yb in _loader(x, y, 64, False):
            xb = xb.to(device)
            output = (
                model.estimate_source(xb) if branch == "source" else model(xb)
            )
            predictions.append(output.cpu().numpy())
            labels.append(yb.numpy())
    return np.concatenate(labels).ravel(), np.concatenate(predictions).ravel()


def evaluate_cycle(
    model: MEDNet | DANNBaselineNet,
    cycle: ShenCycle,
    device: torch.device,
    tensor_layout: str = "paper",
    branch: str = "target",
) -> dict[str, float]:
    truth, estimate = predict(model, [cycle], device, tensor_layout, branch=branch)
    return compute_metrics(truth, estimate)


def save_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _snapshot_training_modes(model: torch.nn.Module) -> dict[int, bool]:
    """Capture every submodule's exact ``.training`` boolean, keyed by
    ``id()``. Used by ``_validation_metrics`` to restore a deliberately
    mixed train/eval configuration (e.g. ``med_adapt`` keeps
    ``target_extractor`` in ``.train()`` while ``source_extractor``/the
    heads stay frozen in ``.eval()`` for the whole adaptation loop) --
    a single top-level ``model.training`` flag cannot capture this."""

    return {id(module): module.training for module in model.modules()}


def _restore_training_modes(model: torch.nn.Module, modes: dict[int, bool]) -> None:
    for module in model.modules():
        if id(module) in modes:
            module.training = modes[id(module)]


def _diagnostic_loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    """Read-only loader for ``_validation_metrics`` only.

    Identical construction to ``_loader`` (``shuffle=False``) except it
    passes an explicit LOCAL ``torch.Generator()`` instead of relying on
    PyTorch's global default generator. This matters even with
    ``shuffle=False``: constructing a ``DataLoader`` iterator always draws
    one value from its generator to derive worker/base seeding, even with
    ``num_workers=0`` -- so without an explicit local generator here, every
    validation probe would silently advance ``torch.default_generator`` and
    perturb the exact shuffle order of whichever training loader
    (``med_adapt``'s/``adapt_dann_mmd``'s own ``shuffle=True`` loaders,
    seeded via ``seed_everything``) runs afterward. This is the same
    precaution used by the CMU DARM benchmark's own
    ``_validation_loader``."""

    if not len(x):
        raise ValueError("Cannot construct a diagnostic loader from zero windows")
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=min(batch_size, len(x)),
        shuffle=False,
        generator=torch.Generator(),
    )


@torch.no_grad()
def _validation_metrics(
    model: MEDNet | DANNBaselineNet,
    cycles: list[ShenCycle],
    device: torch.device,
    tensor_layout: str = "paper",
    branch: str = "target",
) -> tuple[float, float]:
    """Read-only, diagnostic-only validation probe used by ``med_adapt``
    and ``adapt_dann_mmd``.

    This is intentionally NOT a thin wrapper around ``predict()`` left as
    a black box: ``predict()`` unconditionally calls ``model.eval()``,
    which recursively forces *every* submodule into eval mode. During
    adaptation, ``med_adapt``/``adapt_dann_mmd`` deliberately keep some
    submodules frozen in ``.eval()`` while others stay in ``.train()`` --
    calling ``model.eval()`` without restoring that exact per-submodule
    state afterward would silently change dropout behavior (``MEDNet``'s
    ``FeatureExtractor`` has active ``nn.Dropout`` layers) on the very next
    training epoch. So this helper snapshots every submodule's exact mode
    first, runs the probe in full eval mode for deterministic metrics, and
    restores the exact prior configuration before returning -- regardless
    of exceptions. It also builds its own loader via ``_diagnostic_loader``
    (a LOCAL generator, never the global one -- see that function's
    docstring) instead of ``predict``'s shared ``_loader``, so this probe
    can never perturb the global RNG stream a later training-loader
    shuffle depends on.

    Never computes gradients (``@torch.no_grad()``) and never updates any
    parameter. Returns ``(mae_percent, rmse_percent)`` to match the
    percent-scale convention already used for MAE/RMSE throughout the CMU
    transfer scripts (see ``run_sftnn_transfer_cmu.py``'s ``_evaluate``).
    """

    modes = _snapshot_training_modes(model)
    try:
        model.eval()
        predictions, labels = [], []
        for cycle in cycles:
            x, y = windows([cycle], tensor_layout)
            for xb, yb in _diagnostic_loader(x, y, 64):
                xb = xb.to(device)
                output = (
                    model.estimate_source(xb) if branch == "source" else model(xb)
                )
                predictions.append(output.cpu().numpy())
                labels.append(yb.numpy())
        truth = np.concatenate(labels).ravel()
        estimate = np.concatenate(predictions).ravel()
    finally:
        _restore_training_modes(model, modes)
    metrics = compute_metrics(truth, estimate)
    return metrics["mae"] * 100.0, metrics["rmse"] * 100.0


# ---------------------------------------------------------------------------
# MED: the paper's proposed source-free method (Algorithm 1).
# ---------------------------------------------------------------------------


def pretrain_med(
    cycles: dict[str, ShenCycle],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
    checkpoint: str | Path | None = None,
    early_stopping_patience: int | None = None,
    min_epochs: int = 0,
    min_delta: float = 1e-4,
    tensor_layout: str = "paper",
    record_epoch_seconds: bool = False,
) -> tuple[MEDNet, list[float]] | tuple[MEDNet, list[float], list[float]]:
    """Joint supervised pretraining of the shared extractor + both heads.

    Both heads see the same batches and the same labels; the only thing
    that makes them diverge is their different Xavier-init gain (set in
    ``SOCHead``). No diversity-promoting loss term is used -- that mirrors
    the paper, which describes divergent initialization as the mechanism,
    not an extra objective.

    ``record_epoch_seconds`` defaults to False, preserving the exact
    original behavior and 2-tuple return for every existing caller
    (``suite.py``'s in-paper cross-temperature runs, and any other
    unmodified call site). When explicitly set True, this ALSO returns a
    third element: a ``list[float]`` of real per-epoch wall-clock seconds
    (see ``_EpochTimer``), aligned index-for-index with ``history`` --
    purely diagnostic, never read by the training loop itself, never
    changes what is trained, in what order, or with what hyperparameters.
    """

    seed_everything(seed)
    x, y = windows([cycles[p] for p in PROFILES if p in cycles], tensor_layout)
    loader = _loader(x, y, batch_size, True)
    model = MEDNet().to(device)
    optimizer = torch.optim.Adam(
        (
            *model.source_extractor.parameters(),
            *model.head1.parameters(),
            *model.head2.parameters(),
        ),
        lr=learning_rate,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=max(5, early_stopping_patience // 3),
            threshold=min_delta,
            min_lr=learning_rate * 0.01,
        )
        if early_stopping_patience is not None
        else None
    )
    history, best_loss, best_state, stale_epochs = [], math.inf, None, 0
    epoch_seconds: list[float] = []
    model.source_extractor.train()
    model.head1.train()
    model.head2.train()
    for _ in range(epochs):
        timer = _EpochTimer(device) if record_epoch_seconds else None
        if timer is not None:
            timer.__enter__()
        total, samples = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred1, pred2 = model.forward_source(xb)
            loss = F.mse_loss(pred1, yb, reduction="sum") + F.mse_loss(pred2, yb, reduction="sum")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            samples += 2 * len(xb)
        epoch_loss = total / max(samples, 1)
        history.append(epoch_loss)
        if timer is not None:
            timer.__exit__()
            epoch_seconds.append(timer.seconds)
        if scheduler is not None:
            scheduler.step(epoch_loss)
        if _improved(epoch_loss, best_loss, min_delta):
            best_loss = epoch_loss
            best_state = model.source_state_dict()
            stale_epochs = 0
        else:
            stale_epochs += 1
        if (
            early_stopping_patience is not None
            and len(history) >= min_epochs
            and stale_epochs >= early_stopping_patience
        ):
            break
    assert best_state is not None
    model.load_source_state_dict(best_state)
    if checkpoint:
        path = Path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "source_state_dict": best_state,
                "history": history,
                "protocol": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "epochs_completed": len(history),
                    "early_stopping_patience": early_stopping_patience,
                    "head1_gain": model.head1.gain,
                    "head2_gain": model.head2.gain,
                },
            },
            path,
        )
    if record_epoch_seconds:
        return model, history, epoch_seconds
    return model, history


def load_med_source(checkpoint: str | Path, device: torch.device) -> MEDNet:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = MEDNet().to(device)
    model.load_source_state_dict(payload["source_state_dict"])
    return model


def build_pseudo_target_dataset(
    source_model: MEDNet,
    cycles: list[ShenCycle],
    device: torch.device,
    *,
    threshold: float = 0.08,
    lookahead: int = 200,
    tensor_layout: str = "paper",
) -> tuple[TensorDataset, dict[str, int]]:
    """Build a pseudo-labeled target dataset from a frozen MED source model.

    Confidence filtering (``selected_windows``) runs **per cycle**, before
    any concatenation across cycles/files -- this is deliberate: the
    delivered reference implementation's equivalent filter flattens all
    target files into one array first and then applies a 200-sample
    lookahead across that flattened array, which can span an unrelated file
    boundary. Filtering per cycle first avoids that cross-file artifact
    entirely, at zero extra cost, since ``selected_windows`` already worked
    this way for the in-paper cross-temperature transfer.
    """

    source_model.eval()
    xs, ys, pseudo_windows, total_windows = [], [], 0, 0
    for cycle in cycles:
        usable = len(cycle.soc) // 90 * 90
        if not usable:
            continue
        probe = ShenCycle(
            profile=cycle.profile,
            temperature=cycle.temperature,
            features=cycle.features[:usable],
            soc=cycle.soc[:usable],
            paths=cycle.paths,
        )
        x_probe, _ = windows([probe], tensor_layout)
        with torch.no_grad():
            pseudo = np.concatenate(
                [
                    source_model.estimate_source(xb.to(device)).cpu().numpy().ravel()
                    for xb, _ in _loader(x_probe, np.zeros((len(x_probe), 90), np.float32), 64, False)
                ]
            )
        total_windows += len(x_probe)
        x, y = selected_windows(
            probe, pseudo, threshold=threshold, lookahead=lookahead, tensor_layout=tensor_layout
        )
        if len(x):
            xs.append(x)
            ys.append(y)
            pseudo_windows += len(x)
    if not xs:
        empty_x, empty_y = windows([])
        dataset = TensorDataset(torch.from_numpy(empty_x), torch.from_numpy(empty_y))
        return dataset, {
            "pseudo_windows": 0,
            "total_windows": total_windows,
            "status": "skipped_no_pseudo_labels",
        }
    x, y = np.concatenate(xs), np.concatenate(ys)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return dataset, {
        "pseudo_windows": pseudo_windows,
        "total_windows": total_windows,
        "status": "adapted",
    }


def med_adapt(
    source_model: MEDNet,
    target_dataset: TensorDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
    lambdas: tuple[float, float] = (1.0, 0.1),
    early_stopping_patience: int | None = None,
    min_epochs: int = 0,
    min_delta: float = 1e-4,
    record_epoch_seconds: bool = False,
    validation_cycles: list[ShenCycle] | None = None,
    validation_interval: int = 1,
    tensor_layout: str = "paper",
) -> tuple[MEDNet, list[dict[str, float]]]:
    """Source-free transfer (Algorithm 1): only ``target_extractor`` trains.

    Both heads and the source extractor are frozen by ``initialize_target``
    below and stay frozen for the entire loop -- no source data and no real
    target labels are used anywhere in this function, only the fixed
    pseudo-labels already baked into ``target_dataset``.

    ``record_epoch_seconds`` and ``validation_cycles``/``validation_interval``
    are purely diagnostic and default to inert (``False``/``None``), so the
    default call signature and returned ``history`` schema are unchanged
    for every existing caller (``suite.py``'s in-paper runs). When
    ``record_epoch_seconds=True``, each ``epoch_metrics`` dict gains a real
    ``runtime_seconds`` key (at most one CUDA sync pair per epoch -- see
    ``_EpochTimer``); this never changes what is trained, in what order, or
    with what hyperparameters. When ``validation_cycles`` is supplied, every
    ``validation_interval``-th epoch also runs a read-only diagnostic probe
    (see ``_validation_metrics``) against those cycles and records
    ``validation_mae``/``validation_rmse``/``validation_evaluated`` -- this
    is evaluated *after* the epoch's optimizer step and never feeds back
    into training, pseudo-labeling, the scheduler, early stopping,
    checkpoint selection, or model selection, all of which continue to use
    only the training objective exactly as before.
    """

    if len(target_dataset) == 0:
        raise ValueError("target_dataset is empty; nothing to adapt on")
    seed_everything(seed)
    model = deepcopy(source_model).to(device)
    model.initialize_target()
    loader = DataLoader(
        target_dataset, batch_size=min(batch_size, len(target_dataset)), shuffle=True
    )
    optimizer = torch.optim.Adam(model.target_extractor.parameters(), lr=learning_rate)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=max(5, early_stopping_patience // 3),
            threshold=min_delta,
            min_lr=learning_rate * 0.01,
        )
        if early_stopping_patience is not None
        else None
    )
    lambda_soc, lambda_dis = lambdas
    history, best_loss, best_state, stale_epochs = [], math.inf, None, 0
    validation_interval = max(validation_interval, 1)

    for epoch_index in range(epochs):
        timer = _EpochTimer(device) if record_epoch_seconds else None
        if timer is not None:
            timer.__enter__()
        model.target_extractor.train()
        model.source_extractor.eval()
        model.head1.eval()
        model.head2.eval()
        totals = {"soc": 0.0, "dis": 0.0, "total": 0.0}
        batches = 0
        for xb, pseudo_y in loader:
            xb, pseudo_y = xb.to(device), pseudo_y.to(device)
            optimizer.zero_grad()
            pred1, pred2 = model.forward_target(xb)
            loss_soc = F.mse_loss(pred1, pseudo_y, reduction="mean") + F.mse_loss(
                pred2, pseudo_y, reduction="mean"
            )
            loss_dis = discrepancy(pred1, pred2, reduction="mean")
            loss = lambda_soc * loss_soc + lambda_dis * loss_dis
            loss.backward()
            optimizer.step()
            totals["soc"] += float(loss_soc.detach())
            totals["dis"] += float(loss_dis.detach())
            totals["total"] += float(loss.detach())
            batches += 1
        epoch_metrics = {name: value / max(batches, 1) for name, value in totals.items()}
        epoch_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        if timer is not None:
            timer.__exit__()
            epoch_metrics["runtime_seconds"] = timer.seconds
        if validation_cycles and (epoch_index + 1) % validation_interval == 0:
            validation_mae, validation_rmse = _validation_metrics(
                model, validation_cycles, device, tensor_layout, branch="target"
            )
            epoch_metrics["validation_mae"] = validation_mae
            epoch_metrics["validation_rmse"] = validation_rmse
            epoch_metrics["validation_evaluated"] = True
        else:
            epoch_metrics["validation_mae"] = None
            epoch_metrics["validation_rmse"] = None
            epoch_metrics["validation_evaluated"] = False
        history.append(epoch_metrics)
        if scheduler is not None:
            scheduler.step(epoch_metrics["total"])
        # No target labels are available for model selection (source-free);
        # select on the training objective itself, as the released reference
        # implementation does.
        if _improved(epoch_metrics["total"], best_loss, min_delta):
            best_loss = epoch_metrics["total"]
            best_state = deepcopy(model.target_extractor.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if (
            early_stopping_patience is not None
            and len(history) >= min_epochs
            and stale_epochs >= early_stopping_patience
        ):
            break
    assert best_state is not None
    model.target_extractor.load_state_dict(best_state)
    return model, history


def target_cycles_excluding(
    target_cycles: dict[str, ShenCycle], test_profile: str
) -> list[ShenCycle]:
    """Every target-domain profile except the held-out test profile."""

    return [target_cycles[p] for p in PROFILES if p in target_cycles and p != test_profile]


def adapt_med(
    source_model: MEDNet,
    target_cycles: dict[str, ShenCycle] | list[ShenCycle],
    *,
    test_profile: str | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
    lambdas: tuple[float, float] = (1.0, 0.1),
    pseudo_threshold: float = 0.08,
    pseudo_lookahead: int = 200,
    early_stopping_patience: int | None = None,
    min_epochs: int = 0,
    min_delta: float = 1e-4,
    tensor_layout: str = "paper",
) -> tuple[MEDNet, list[dict[str, float]], dict[str, int]]:
    """Paper-faithful, source-free MED transfer for the in-paper suite.

    ``target_cycles`` may be the full per-temperature profile dict (paper
    cross-temperature transfer, ``test_profile`` held out) or an arbitrary
    list of cycles (e.g. CMU eVTOL missions with no notion of "profile").
    """

    if isinstance(target_cycles, dict):
        if test_profile is None:
            raise ValueError("test_profile is required when target_cycles is a profile dict")
        adaptation_cycles = target_cycles_excluding(target_cycles, test_profile)
    else:
        adaptation_cycles = list(target_cycles)

    dataset, info = build_pseudo_target_dataset(
        source_model,
        adaptation_cycles,
        device,
        threshold=pseudo_threshold,
        lookahead=pseudo_lookahead,
        tensor_layout=tensor_layout,
    )
    if len(dataset) == 0:
        return deepcopy(source_model), [], info
    model, history = med_adapt(
        source_model,
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
        lambdas=lambdas,
        early_stopping_patience=early_stopping_patience,
        min_epochs=min_epochs,
        min_delta=min_delta,
    )
    return model, history, info


# ---------------------------------------------------------------------------
# DANN + MMD baseline (Table VI "adversarial"/"MMD" comparison rows).
# ---------------------------------------------------------------------------


def train_source(
    cycles: dict[str, ShenCycle],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
    checkpoint: str | Path | None = None,
    early_stopping_patience: int | None = None,
    min_epochs: int = 0,
    min_delta: float = 1e-4,
    tensor_layout: str = "paper",
    record_epoch_seconds: bool = False,
) -> tuple[DANNBaselineNet, list[float]] | tuple[DANNBaselineNet, list[float], list[float]]:
    """Single-head source pretraining for the DANN+MMD baseline only.

    ``record_epoch_seconds`` defaults to False (unchanged 2-tuple return
    for every existing caller). See ``pretrain_med``'s docstring for the
    exact same opt-in, backward-compatible convention."""

    seed_everything(seed)
    x, y = windows([cycles[p] for p in PROFILES if p in cycles], tensor_layout)
    loader = _loader(x, y, batch_size, True)
    model = DANNBaselineNet().to(device)
    optimizer = torch.optim.Adam(
        (*model.source_extractor.parameters(), *model.predictor.parameters()),
        lr=learning_rate,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=max(5, early_stopping_patience // 3),
            threshold=min_delta,
            min_lr=learning_rate * 0.01,
        )
        if early_stopping_patience is not None
        else None
    )
    history, best_loss, best_state, stale_epochs = [], math.inf, None, 0
    epoch_seconds: list[float] = []
    model.source_extractor.train()
    model.predictor.train()
    for _ in range(epochs):
        timer = _EpochTimer(device) if record_epoch_seconds else None
        if timer is not None:
            timer.__enter__()
        total, samples = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model.forward_source(xb), yb, reduction="sum")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            samples += len(xb)
        epoch_loss = total / max(samples, 1)
        history.append(epoch_loss)
        if timer is not None:
            timer.__exit__()
            epoch_seconds.append(timer.seconds)
        if scheduler is not None:
            scheduler.step(epoch_loss)
        if _improved(epoch_loss, best_loss, min_delta):
            best_loss = epoch_loss
            best_state = model.source_state_dict()
            stale_epochs = 0
        else:
            stale_epochs += 1
        if (
            early_stopping_patience is not None
            and len(history) >= min_epochs
            and stale_epochs >= early_stopping_patience
        ):
            break
    assert best_state is not None
    model.load_source_state_dict(best_state)
    if checkpoint:
        path = Path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "source_state_dict": best_state,
                "history": history,
                "protocol": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "epochs_completed": len(history),
                    "early_stopping_patience": early_stopping_patience,
                },
            },
            path,
        )
    if record_epoch_seconds:
        return model, history, epoch_seconds
    return model, history


def load_source(checkpoint: str | Path, device: torch.device) -> DANNBaselineNet:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = DANNBaselineNet().to(device)
    model.load_source_state_dict(payload["source_state_dict"])
    return model


@torch.no_grad()
def _cycle_prediction(
    model: DANNBaselineNet,
    cycle: ShenCycle,
    device: torch.device,
    tensor_layout: str = "paper",
) -> np.ndarray:
    model.eval()
    usable = len(cycle.soc) // 90 * 90
    output = np.empty(usable, dtype=np.float32)
    x, y = windows([cycle], tensor_layout)
    offset = 0
    for xb, _ in _loader(x, y, 64, False):
        values = model.forward_source(xb.to(device)).cpu().numpy().ravel()
        output[offset : offset + len(values)] = values
        offset += len(values)
    return output


def target_training_windows(
    source_model: DANNBaselineNet,
    target_cycles: dict[str, ShenCycle] | list[ShenCycle],
    test_profile: str | None,
    scenario: str,
    device: torch.device,
    *,
    labeled_profile: str | None = None,
    tensor_layout: str = "paper",
) -> tuple[np.ndarray, np.ndarray, dict[str, int | str | None]]:
    if scenario not in {"semisupervised", "unsupervised"}:
        raise ValueError("scenario must be 'semisupervised' or 'unsupervised'")
    if isinstance(target_cycles, dict):
        if test_profile is None:
            raise ValueError("test_profile is required for paper temperature folds")
        remaining = [p for p in PROFILES if p != test_profile]
        candidates = [(profile, target_cycles[profile]) for profile in remaining]
    else:
        remaining = [cycle.profile for cycle in target_cycles]
        candidates = [(cycle.profile, cycle) for cycle in target_cycles]

    if scenario == "semisupervised":
        labeled_profile = labeled_profile or remaining[0]
        if labeled_profile not in remaining:
            raise ValueError("The labeled target profile must differ from the test profile")
    else:
        labeled_profile = None

    xs, ys, pseudo_windows = [], [], 0
    for profile, cycle in candidates:
        if profile == labeled_profile:
            usable = (len(cycle.soc) // 2) // 90 * 90
            x, y = windows(
                [
                    ShenCycle(
                        profile=cycle.profile,
                        temperature=cycle.temperature,
                        features=cycle.features[:usable],
                        soc=cycle.soc[:usable],
                        paths=cycle.paths,
                    )
                ],
                tensor_layout,
            )
        else:
            pseudo = _cycle_prediction(source_model, cycle, device, tensor_layout)
            x, y = selected_windows(cycle, pseudo, tensor_layout=tensor_layout)
            pseudo_windows += len(x)
        if len(x):
            xs.append(x)
            ys.append(y)
    if not xs:
        x, y = windows([])
        return x, y, {
            "labeled_profile": labeled_profile,
            "pseudo_windows": 0,
            "total_windows": 0,
            "status": "skipped_no_pseudo_labels",
        }
    return np.concatenate(xs), np.concatenate(ys), {
        "labeled_profile": labeled_profile,
        "pseudo_windows": pseudo_windows,
        "total_windows": sum(len(x) for x in xs),
        "status": "adapted",
    }


def adapt_dann_mmd(
    source_model: DANNBaselineNet,
    source_cycles: dict[str, ShenCycle],
    target_cycles: dict[str, ShenCycle] | list[ShenCycle],
    *,
    test_profile: str | None,
    scenario: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lambdas: tuple[float, float, float],
    device: torch.device,
    seed: int,
    labeled_profile: str | None = None,
    early_stopping_patience: int | None = None,
    min_epochs: int = 0,
    min_delta: float = 1e-4,
    tensor_layout: str = "paper",
    record_epoch_seconds: bool = False,
    validation_cycles: list[ShenCycle] | None = None,
    validation_interval: int = 1,
) -> tuple[DANNBaselineNet, list[dict[str, float]], dict[str, int | str | None]]:
    """DANN + MMD baseline adaptation.

    ``record_epoch_seconds`` and ``validation_cycles``/``validation_interval``
    are purely diagnostic and default to inert, so the default call
    signature and returned ``history`` schema are unchanged for every
    existing caller. See ``med_adapt``'s docstring for the exact same
    opt-in, backward-compatible convention -- the same guarantee applies
    here: this never changes training data, batch order, the adversarial/
    MMD objective, the scheduler, early stopping, or model selection."""

    seed_everything(seed)
    model = deepcopy(source_model).to(device)
    model.initialize_target()
    source_x, source_y = windows([source_cycles[p] for p in PROFILES], tensor_layout)
    target_x, target_y, target_info = target_training_windows(
        model,
        target_cycles,
        test_profile,
        scenario,
        device,
        labeled_profile=labeled_profile,
        tensor_layout=tensor_layout,
    )
    if not len(target_x):
        return model, [], target_info
    source_loader = _loader(source_x, source_y, batch_size, True)
    target_loader = _loader(target_x, target_y, batch_size, True)
    extractor_optimizer = torch.optim.Adam(model.target_extractor.parameters(), lr=learning_rate)
    discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=learning_rate)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            extractor_optimizer,
            factor=0.5,
            patience=max(5, early_stopping_patience // 3),
            threshold=min_delta,
            min_lr=learning_rate * 0.01,
        )
        if early_stopping_patience is not None
        else None
    )
    estimate_weight, adversarial_weight, mmd_weight = lambdas
    history, best_loss, best_state, stale_epochs = [], math.inf, None, 0
    validation_interval = max(validation_interval, 1)

    for epoch_index in range(epochs):
        timer = _EpochTimer(device) if record_epoch_seconds else None
        if timer is not None:
            timer.__enter__()
        model.target_extractor.train()
        model.discriminator.train()
        model.source_extractor.eval()
        model.predictor.eval()
        totals = {"estimate": 0.0, "adversarial": 0.0, "mmd": 0.0, "total": 0.0}
        batches = 0
        for (source_batch, _), (target_batch, target_label) in zip(source_loader, target_loader):
            source_batch = source_batch.to(device)
            target_batch, target_label = target_batch.to(device), target_label.to(device)
            source_label = torch.ones((len(source_batch), 1), device=device)
            target_domain_label = torch.zeros((len(target_batch), 1), device=device)

            with torch.no_grad():
                source_features = model.source_extractor(source_batch)
            target_features = model.target_extractor(target_batch)
            discriminator_optimizer.zero_grad()
            discriminator_loss = F.binary_cross_entropy_with_logits(
                model.discriminator(source_features), source_label
            ) + F.binary_cross_entropy_with_logits(
                model.discriminator(target_features.detach()), target_domain_label
            )
            discriminator_loss.backward()
            discriminator_optimizer.step()

            extractor_optimizer.zero_grad()
            target_features = model.target_extractor(target_batch)
            estimate_loss = F.mse_loss(
                model.predictor(target_features), target_label, reduction="sum"
            )
            adversarial_loss = F.binary_cross_entropy_with_logits(
                model.discriminator(target_features), torch.ones_like(target_domain_label)
            )
            mmd_loss = gaussian_mmd(source_features, target_features)
            total_loss = (
                estimate_weight * estimate_loss
                + adversarial_weight * adversarial_loss
                + mmd_weight * mmd_loss
            )
            total_loss.backward()
            extractor_optimizer.step()
            for name, value in (
                ("estimate", estimate_loss),
                ("adversarial", adversarial_loss),
                ("mmd", mmd_loss),
                ("total", total_loss),
            ):
                totals[name] += float(value.detach())
            batches += 1
        epoch_metrics = {name: value / max(batches, 1) for name, value in totals.items()}
        epoch_metrics["learning_rate"] = extractor_optimizer.param_groups[0]["lr"]
        if timer is not None:
            timer.__exit__()
            epoch_metrics["runtime_seconds"] = timer.seconds
        if validation_cycles and (epoch_index + 1) % validation_interval == 0:
            validation_mae, validation_rmse = _validation_metrics(
                model, validation_cycles, device, tensor_layout, branch="target"
            )
            epoch_metrics["validation_mae"] = validation_mae
            epoch_metrics["validation_rmse"] = validation_rmse
            epoch_metrics["validation_evaluated"] = True
        else:
            epoch_metrics["validation_mae"] = None
            epoch_metrics["validation_rmse"] = None
            epoch_metrics["validation_evaluated"] = False
        history.append(epoch_metrics)
        if scheduler is not None:
            scheduler.step(epoch_metrics["total"])
        # Paper does not specify model selection; use label-safe training objective.
        if _improved(epoch_metrics["total"], best_loss, min_delta):
            best_loss = epoch_metrics["total"]
            best_state = deepcopy(model.target_extractor.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if (
            early_stopping_patience is not None
            and len(history) >= min_epochs
            and stale_epochs >= early_stopping_patience
        ):
            break
    assert best_state is not None
    model.target_extractor.load_state_dict(best_state)
    return model, history, target_info
