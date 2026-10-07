"""Training and evaluation loops for the Shen et al. (2023) TATN reproduction.

Restored verbatim from commit ``973525c^`` (the parent of the SFTNN/MED
refactor). The only changes from that historical version are import paths:
this module's own sibling import (``model``) now points at
``battery_benchmark.reproductions.shen_tatn.model`` instead of
``battery_benchmark.reproductions.shen_2023.model``. The ``data`` import is
unchanged, because ``battery_benchmark.reproductions.shen_2023.data`` is
byte-identical to its pre-refactor version and is shared, CMU-agnostic
data-loading plumbing rather than TATN-specific algorithmic code.
"""

from __future__ import annotations

import contextlib
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
from battery_benchmark.reproductions.shen_tatn.model import TATN, gaussian_mmd


# Default threshold of ``shen_2023.data.selected_windows`` (paper/repo value).
PSEUDO_LABEL_THRESHOLD = 0.08


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
    ``record_epoch_seconds=True`` (``train_source``) or unconditionally used
    only around the read-only validation probe in ``adapt`` -- when not
    requested, this class is never constructed and adds zero overhead to
    the default (backward-compatible) code path. CUDA ops are async, so a
    timed block that ran on GPU is synced before both the start and end
    timestamps -- at most one synchronize() pair per epoch (never per
    batch), skipped entirely on CPU."""

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
        raise ValueError("Cannot construct a loader from zero TATN windows")
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=min(batch_size, len(x)),
        shuffle=shuffle,
    )


@torch.no_grad()
def predict(
    model: TATN,
    cycles: list[ShenCycle],
    device: torch.device,
    tensor_layout: str = "paper",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, labels = [], []
    for cycle in cycles:
        x, y = windows([cycle], tensor_layout)
        for xb, yb in _loader(x, y, 64, False):
            predictions.append(model(xb.to(device)).cpu().numpy())
            labels.append(yb.numpy())
    return np.concatenate(labels).ravel(), np.concatenate(predictions).ravel()


def _snapshot_training_modes(model: torch.nn.Module) -> dict[int, bool]:
    """Capture every submodule's exact ``.training`` boolean, keyed by
    ``id()``. Used by ``_validation_metrics`` to restore a deliberately
    mixed train/eval configuration (``adapt`` keeps ``target_extractor``/
    ``discriminator`` in ``.train()`` while ``source_extractor``/
    ``predictor`` stay frozen in ``.eval()`` for the whole adaptation loop)
    -- a single top-level ``model.training`` flag cannot capture this."""

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
    perturb the exact shuffle order of ``adapt``'s own ``shuffle=True``
    loaders (seeded via ``seed_everything``). Same precaution already used
    by ``shen_2023.train``'s and the CMU DARM benchmark's own diagnostic
    loaders."""

    if not len(x):
        raise ValueError("Cannot construct a diagnostic loader from zero TATN windows")
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=min(batch_size, len(x)),
        shuffle=False,
        generator=torch.Generator(),
    )


@contextlib.contextmanager
def _preserve_global_rng():
    """Snapshot and restore every global RNG (python, numpy, torch CPU, CUDA)
    so a read-only diagnostic probe can never perturb the training stream."""

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@torch.no_grad()
def _validation_metrics_detailed(
    model: TATN,
    cycles: list[ShenCycle],
    device: torch.device,
    tensor_layout: str = "paper",
    branch: str = "target",
) -> dict[str, float]:
    """Read-only, diagnostic-only validation probe (``diagnostic_only=true``).

    Computes overall and mission-macro MAE/RMSE (percent) over ``cycles``
    (one cycle == one mission). Snapshots every submodule's exact train/eval
    mode (``adapt`` deliberately keeps ``target_extractor``/``discriminator``
    in ``.train()`` while ``source_extractor``/``predictor`` stay ``.eval()``)
    and restores it afterwards, runs under ``torch.no_grad()`` in full eval
    mode, uses a LOCAL DataLoader generator and additionally snapshots and
    restores all global RNG states. Never computes gradients, never updates
    a parameter and never feeds back into training, pseudo-labelling,
    scheduling, early stopping or checkpoint selection."""

    modes = _snapshot_training_modes(model)
    per_cycle: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        with _preserve_global_rng():
            model.eval()
            for cycle in cycles:
                x, y = windows([cycle], tensor_layout)
                if not len(x):
                    continue
                predictions, labels = [], []
                for xb, yb in _diagnostic_loader(x, y, 64):
                    xb = xb.to(device)
                    output = model.forward_source(xb) if branch == "source" else model(xb)
                    predictions.append(output.cpu().numpy())
                    labels.append(yb.numpy())
                per_cycle.append(
                    (np.concatenate(labels).ravel(), np.concatenate(predictions).ravel())
                )
    finally:
        _restore_training_modes(model, modes)
    if not per_cycle:
        raise ValueError("Validation probe received no usable TATN windows")
    truth = np.concatenate([t for t, _ in per_cycle])
    estimate = np.concatenate([e for _, e in per_cycle])
    overall = compute_metrics(truth, estimate)
    macro = [compute_metrics(t, e) for t, e in per_cycle]
    return {
        "overall_mae": overall["mae"] * 100.0,
        "overall_rmse": overall["rmse"] * 100.0,
        "mission_macro_mae": float(np.mean([m["mae"] for m in macro])) * 100.0,
        "mission_macro_rmse": float(np.mean([m["rmse"] for m in macro])) * 100.0,
        "missions": len(per_cycle),
    }


def _validation_metrics(
    model: TATN,
    cycles: list[ShenCycle],
    device: torch.device,
    tensor_layout: str = "paper",
    branch: str = "target",
) -> tuple[float, float]:
    """Backward-compatible ``(mae_percent, rmse_percent)`` view of
    ``_validation_metrics_detailed`` (overall, all timesteps pooled)."""

    detail = _validation_metrics_detailed(model, cycles, device, tensor_layout, branch)
    return detail["overall_mae"], detail["overall_rmse"]


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
) -> tuple[TATN, list[float]] | tuple[TATN, list[float], list[dict[str, float]]]:
    """Source pretraining for TATN, unchanged from before this diagnostic
    addition (same data, optimizer, loss, batch order, early stopping,
    and checkpoint selection).

    ``record_epoch_seconds`` defaults to False, preserving the exact
    original behavior and 2-tuple return for every existing caller
    (``suite.py``'s in-paper runs). When explicitly set True, this ALSO
    returns a third element: a ``list[dict]`` with one entry per completed
    epoch -- ``epoch`` (1-based), ``learning_rate`` (this optimizer's
    current LR), ``runtime_seconds`` (real wall-clock time for that
    epoch's training batch loop only, via ``_EpochTimer`` -- at most one
    CUDA sync pair per epoch, never per batch), and ``loss`` (the same
    per-epoch value already appended to ``history``, duplicated here only
    for convenience when writing a single training-history CSV row).
    Purely diagnostic: never read by the training loop itself, never
    changes what is trained, in what order, or with what hyperparameters."""

    seed_everything(seed)
    x, y = windows([cycles[p] for p in PROFILES if p in cycles], tensor_layout)
    loader = _loader(x, y, batch_size, True)
    model = TATN().to(device)
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
    epoch_diagnostics: list[dict[str, float]] = []
    model.source_extractor.train()
    model.predictor.train()
    for epoch_index in range(epochs):
        timer = _EpochTimer(device) if record_epoch_seconds else None
        if timer is not None:
            timer.__enter__()
        total, samples, steps = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model.forward_source(xb), yb, reduction="sum")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            samples += len(xb)
            steps += 1
        epoch_loss = total / max(samples, 1)
        history.append(epoch_loss)
        if timer is not None:
            timer.__exit__()
            epoch_diagnostics.append(
                {
                    "epoch": epoch_index + 1,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "runtime_seconds": timer.seconds,
                    "loss": epoch_loss,
                    "total_loss": epoch_loss,
                    "source_soc_regression_loss": epoch_loss,
                    "optimizer_steps": steps,
                }
            )
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
        return model, history, epoch_diagnostics
    return model, history


def load_source(checkpoint: str | Path, device: torch.device) -> TATN:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = TATN().to(device)
    model.load_source_state_dict(payload["source_state_dict"])
    return model


def load_released_source(checkpoint: str | Path, device: torch.device) -> TATN:
    """Load a source checkpoint from the authors' released TATN repository."""

    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model = TATN().to(device)
    model.source_extractor.conv.load_state_dict(
        {key.removeprefix("conv."): value for key, value in payload["conv"].items()}
    )
    for name in ("lstm1", "lstm2"):
        prefix = f"{name}.0."
        getattr(model.source_extractor, name).load_state_dict(
            {
                key.removeprefix(prefix): value
                for key, value in payload["lstm"].items()
                if key.startswith(prefix)
            }
        )
    model.predictor.layers[0].load_state_dict(
        {key.removeprefix("fc.0."): value for key, value in payload["fc"].items()}
    )
    model.predictor.layers[2].load_state_dict(
        {key.removeprefix("fc."): value for key, value in payload["regression"].items()}
    )
    model.initialize_target()
    return model


@torch.no_grad()
def _cycle_prediction(
    model: TATN,
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
    source_model: TATN,
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


def adapt(
    source_model: TATN,
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
) -> tuple[TATN, list[dict[str, float]], dict[str, int | str | None]]:
    """TATN adaptation (DANN + MMD), unchanged from before this diagnostic
    addition (same pseudo-labeling, loss terms, optimizers, batch order,
    early stopping, and checkpoint selection).

    ``record_epoch_seconds`` and ``validation_cycles``/``validation_interval``
    are purely diagnostic and default to inert (``False``/``None``/``1``),
    so the default call signature and returned ``history`` schema stay
    backward compatible for every existing caller (``suite.py``'s in-paper
    runs already receive a ``list[dict]`` history; this only adds new keys
    to each epoch's dict, never removes or renames the existing
    ``estimate``/``adversarial``/``mmd``/``total``/``learning_rate`` keys).

    When ``record_epoch_seconds=True``, each epoch's dict also gets
    ``epoch`` (1-based), ``estimate_loss``/``adversarial_loss``/
    ``mmd_loss``/``total_loss`` (identical values to the existing
    ``estimate``/``adversarial``/``mmd``/``total`` keys, added under the
    names asked for the diagnostic training-history CSV), and
    ``runtime_seconds`` (real wall-clock time for that epoch's training
    batch loop only -- ``_EpochTimer``, at most one CUDA sync pair per
    epoch, never per batch; never includes the validation probe below).

    When ``validation_cycles`` is supplied, every ``validation_interval``-th
    epoch (plus epoch 1 and the actual last epoch, handled by the caller
    via its own interval bookkeeping) also runs the read-only diagnostic
    probe in ``_validation_metrics`` against those cycles and records
    ``validation_mae``/``validation_rmse``/``validation_runtime_seconds``/
    ``validation_evaluated`` -- evaluated *after* the epoch's optimizer
    steps, using a LOCAL DataLoader generator (never the global RNG), with
    every submodule's train/eval mode snapshotted and restored around the
    probe. This never feeds back into training, pseudo-labeling, the
    scheduler, early stopping, or checkpoint selection, all of which
    continue to use only the training objective exactly as before."""

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
    # Candidate pseudo-label windows (every non-labelled target window before
    # selection); diagnostic only, kept out of ``target_info`` so its schema is
    # unchanged for existing callers.
    if isinstance(target_cycles, dict):
        _candidates = [
            c for p, c in target_cycles.items() if p != test_profile and p != labeled_profile
        ]
    else:
        _candidates = [c for c in target_cycles if c.profile != labeled_profile]
    candidate_windows = int(sum(len(c.soc) // 90 for c in _candidates))
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
        totals = {
            "estimate": 0.0,
            "adversarial": 0.0,
            "mmd": 0.0,
            "total": 0.0,
            "discriminator": 0.0,
        }
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
                ("discriminator", discriminator_loss),
            ):
                totals[name] += float(value.detach())
            batches += 1
        epoch_metrics = {name: value / max(batches, 1) for name, value in totals.items()}
        epoch_metrics["epoch"] = epoch_index + 1
        epoch_metrics["learning_rate"] = extractor_optimizer.param_groups[0]["lr"]
        epoch_metrics["estimate_loss"] = epoch_metrics["estimate"]
        epoch_metrics["adversarial_loss"] = epoch_metrics["adversarial"]
        epoch_metrics["mmd_loss"] = epoch_metrics["mmd"]
        epoch_metrics["total_loss"] = epoch_metrics["total"]
        # Diagnostic-only extra components (never read by the training loop).
        # ``estimate`` is the regression loss on the SELECTED pseudo-labelled
        # target windows (labels = frozen-source predictions); the frozen
        # source branch has no optimized regression loss during adaptation.
        epoch_metrics["discriminator_loss"] = epoch_metrics["discriminator"]
        epoch_metrics["pseudo_label_loss"] = epoch_metrics["estimate"]
        epoch_metrics["source_soc_regression_loss"] = None
        epoch_metrics["optimizer_steps"] = 2 * batches  # discriminator + extractor step per batch
        accepted = int(target_info.get("pseudo_windows") or 0)
        candidates = candidate_windows
        epoch_metrics["pseudo_accepted_windows"] = accepted
        epoch_metrics["pseudo_candidate_windows"] = candidates
        epoch_metrics["pseudo_accepted_fraction"] = accepted / candidates if candidates else None
        epoch_metrics["pseudo_threshold"] = PSEUDO_LABEL_THRESHOLD
        if timer is not None:
            timer.__exit__()
            epoch_metrics["runtime_seconds"] = timer.seconds
        else:
            epoch_metrics["runtime_seconds"] = None
        # Validation fields default to "not evaluated"; filled in below, AFTER
        # the scheduler/checkpoint-selection/early-stopping decisions for
        # this epoch have already been made from the training objective
        # alone -- validation never influences any of them.
        epoch_metrics["validation_mae"] = None
        epoch_metrics["validation_rmse"] = None
        epoch_metrics["validation_runtime_seconds"] = None
        epoch_metrics["validation_evaluated"] = False
        for key in (
            "validation_overall_mae",
            "validation_overall_rmse",
            "validation_mission_macro_mae",
            "validation_mission_macro_rmse",
        ):
            epoch_metrics[key] = None
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
        will_stop_early = (
            early_stopping_patience is not None
            and len(history) >= min_epochs
            and stale_epochs >= early_stopping_patience
        )
        is_final_epoch = will_stop_early or epoch_index + 1 == epochs
        # Diagnostic-only, read-only validation probe: epoch 1, every
        # validation_interval-th epoch, and the actual last epoch (whether
        # reached by completing `epochs` or by early stopping) -- computed
        # AFTER this epoch's scheduler/checkpoint-selection/early-stopping
        # decisions above, so it can never feed back into any of them.
        if validation_cycles and (
            epoch_index == 0
            or (epoch_index + 1) % validation_interval == 0
            or is_final_epoch
        ):
            with _EpochTimer(device) as validation_timer:
                detail = _validation_metrics_detailed(
                    model, validation_cycles, device, tensor_layout, branch="target"
                )
            epoch_metrics["validation_mae"] = detail["overall_mae"]
            epoch_metrics["validation_rmse"] = detail["overall_rmse"]
            epoch_metrics["validation_overall_mae"] = detail["overall_mae"]
            epoch_metrics["validation_overall_rmse"] = detail["overall_rmse"]
            epoch_metrics["validation_mission_macro_mae"] = detail["mission_macro_mae"]
            epoch_metrics["validation_mission_macro_rmse"] = detail["mission_macro_rmse"]
            epoch_metrics["validation_runtime_seconds"] = validation_timer.seconds
            epoch_metrics["validation_evaluated"] = True
        if will_stop_early:
            break
    assert best_state is not None
    model.target_extractor.load_state_dict(best_state)
    return model, history, target_info


def evaluate_cycle(
    model: TATN,
    cycle: ShenCycle,
    device: torch.device,
    tensor_layout: str = "paper",
) -> dict[str, float]:
    truth, estimate = predict(model, [cycle], device, tensor_layout)
    return compute_metrics(truth, estimate)


def save_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
