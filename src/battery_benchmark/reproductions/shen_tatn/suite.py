"""Table II/IV/V/VI/VII experiment suite for the Shen et al. (2023) TATN reproduction.

Restored verbatim from commit ``973525c^`` (the parent of the SFTNN/MED
refactor). The only changes from that historical version are import paths:
this module's own sibling import (``train``) now points at
``battery_benchmark.reproductions.shen_tatn.train`` instead of
``battery_benchmark.reproductions.shen_2023.train``. The ``data`` import is
unchanged, because ``battery_benchmark.reproductions.shen_2023.data`` is
byte-identical to its pre-refactor version and is shared, CMU-agnostic
data-loading plumbing rather than TATN-specific algorithmic code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from battery_benchmark.reproductions.shen_2023.data import (
    PROFILES,
    TEMPERATURES,
    load_temperature_cycles,
)
from battery_benchmark.reproductions.shen_tatn.train import (
    adapt,
    evaluate_cycle,
    load_source,
    predict,
    save_json,
    train_source,
)
from battery_benchmark.utils.git_info import get_full_commit_hash, is_dirty

PAPER_TABLES = {
    "II": {"MAE": [0.323, 0.280, 0.188, 0.164, 0.515], "RMSE": [0.404, 0.342, 0.226, 0.208, 0.652]},
    "IV": {"MAE": [1.68, 0.466, 1.53, 2.33], "RMSE": [2.14, 0.619, 1.94, 2.98]},
    "V": {"MAE": [1.55, 1.50, 1.83, 1.57, 1.57], "RMSE": [2.04, 1.92, 2.44, 2.03, 1.97]},
    "VI": {"MAE": [8.01, 1.15, 2.26, 4.03], "RMSE": [14.9, 1.50, 2.91, 5.01]},
    "VII": {"MAE": [2.63, 3.86, 4.26, 3.54, 4.31], "RMSE": [3.44, 6.08, 6.60, 5.23, 6.01]},
}


def _label(temperature: int) -> str:
    return f"{temperature}C"


def _mean(records: list[dict], method: str, metric: str) -> float:
    values = [r[metric] for r in records if r["method"] == method]
    return float(np.mean(values)) * 100 if values else float("nan")


class ShenTableSuite:
    def __init__(
        self,
        raw_root: str | Path,
        output_dir: str | Path,
        *,
        epochs: int = 2000,
        transfer_epochs: int = 2000,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        lambdas: tuple[float, float, float] = (1.0, 1.0, 1.0),
        seed: int = 100,
        device: str = "auto",
        early_stopping_patience: int | None = None,
        min_epochs: int = 0,
        min_delta: float = 1e-4,
        tensor_layout: str = "paper",
    ) -> None:
        self.raw_root = Path(raw_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.transfer_epochs = transfer_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambdas = lambdas
        self.seed = seed
        self.early_stopping_patience = early_stopping_patience
        self.min_epochs = min_epochs
        self.min_delta = min_delta
        self.tensor_layout = tensor_layout
        resolved_device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        self.device = torch.device(resolved_device)
        protocol = {
            "epochs": epochs,
            "transfer_epochs": transfer_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lambdas": lambdas,
            "seed": seed,
            "early_stopping_patience": early_stopping_patience,
            "min_epochs": min_epochs,
            "min_delta": min_delta,
        }
        if tensor_layout != "paper":
            protocol["tensor_layout"] = tensor_layout
        run_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
        self.cache_dir = self.output_dir / "cache" / run_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            self.output_dir / "run_metadata.json",
            {
                **protocol,
                "tensor_layout": tensor_layout,
                "raw_root": str(self.raw_root),
                "device": str(self.device),
                "git_commit": get_full_commit_hash(),
                "git_dirty": is_dirty(),
                "cache_id": run_id,
            },
        )
        self._cycles = {}
        self._sources = {}
        self.records: list[dict] = []

    def cycles(self, temperature: int):
        if temperature not in self._cycles:
            self._cycles[temperature] = load_temperature_cycles(self.raw_root, temperature)
        return self._cycles[temperature]

    def source_model(self, temperature: int):
        if temperature in self._sources:
            return self._sources[temperature]
        checkpoint = self.cache_dir / "checkpoints" / f"source_{temperature}C.pt"
        if checkpoint.exists():
            print(f"[source {temperature}C] restored {checkpoint.name}")
            model = load_source(checkpoint, self.device)
        else:
            print(f"[source {temperature}C] training, max {self.epochs} epochs")
            model, _ = train_source(
                self.cycles(temperature),
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                device=self.device,
                seed=self.seed,
                checkpoint=checkpoint,
                early_stopping_patience=self.early_stopping_patience,
                min_epochs=self.min_epochs,
                min_delta=self.min_delta,
                tensor_layout=self.tensor_layout,
            )
        self._sources[temperature] = model
        return model

    def run_table_ii(self, *, profiles: tuple[str, ...] = PROFILES) -> pd.DataFrame:
        rows = {"MAE": [], "RMSE": []}
        for temperature in TEMPERATURES:
            fold_metrics = []
            cycles = self.cycles(temperature)
            for held_out in profiles:
                training_cycles = {p: cycles[p] for p in PROFILES if p != held_out}
                checkpoint = (
                    self.cache_dir
                    / "checkpoints"
                    / "table_ii"
                    / f"{temperature}C_holdout_{held_out}.pt"
                )
                if checkpoint.exists():
                    model = load_source(checkpoint, self.device)
                else:
                    model, _ = train_source(
                        training_cycles,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        learning_rate=self.learning_rate,
                        device=self.device,
                        seed=self.seed,
                        checkpoint=checkpoint,
                        early_stopping_patience=self.early_stopping_patience,
                        min_epochs=self.min_epochs,
                        min_delta=self.min_delta,
                        tensor_layout=self.tensor_layout,
                    )
                truth, estimate = predict(
                    model,
                    list(training_cycles.values()),
                    self.device,
                    self.tensor_layout,
                )
                fold_metrics.append(
                    (
                        float(np.mean(np.abs(truth - estimate))) * 100,
                        float(np.sqrt(np.mean((truth - estimate) ** 2))) * 100,
                    )
                )
            rows["MAE"].append(float(np.mean([metric[0] for metric in fold_metrics])))
            rows["RMSE"].append(float(np.mean([metric[1] for metric in fold_metrics])))
        table = pd.DataFrame(rows, index=[_label(t) for t in TEMPERATURES]).T
        table["Avg"] = table.mean(axis=1)
        table.to_csv(self.output_dir / "table_ii.csv")
        return table

    def source_sanity_table(self) -> pd.DataFrame:
        rows = []
        for temperature in TEMPERATURES:
            truth, estimate = predict(
                self.source_model(temperature),
                list(self.cycles(temperature).values()),
                self.device,
                self.tensor_layout,
            )
            rows.append(
                {
                    "temperature": _label(temperature),
                    "MAE": float(np.mean(np.abs(truth - estimate))) * 100,
                    "RMSE": float(np.sqrt(np.mean((truth - estimate) ** 2))) * 100,
                }
            )
        table = pd.DataFrame(rows).set_index("temperature")
        table.to_csv(self.output_dir / "source_sanity.csv")
        return table

    def run_transfer(
        self,
        source_temperature: int,
        target_temperature: int,
        scenario: str,
        *,
        profiles: tuple[str, ...] = PROFILES,
    ) -> list[dict]:
        source_model = self.source_model(source_temperature)
        source_cycles = self.cycles(source_temperature)
        target_cycles = self.cycles(target_temperature)
        pair_records = []
        for fold, profile in enumerate(profiles):
            result_path = self.cache_dir / "folds" / (
                f"{scenario}_{source_temperature}C_to_{target_temperature}C_"
                f"{profile}_e{self.transfer_epochs}.json"
            )
            model_path = self.cache_dir / "checkpoints" / "transfer" / (
                f"{scenario}_{source_temperature}C_to_{target_temperature}C_{profile}.pt"
            )
            run_label = f"{scenario} {source_temperature}C->{target_temperature}C {profile}"
            if result_path.exists() and model_path.exists():
                print(f"[{run_label}] restored")
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                print(f"[{run_label}] training")
                baseline = evaluate_cycle(
                    source_model,
                    target_cycles[profile],
                    self.device,
                    self.tensor_layout,
                )
                labeled = (
                    next(p for p in PROFILES if p != profile)
                    if scenario == "semisupervised"
                    else None
                )
                transferred, history, target_info = adapt(
                    source_model,
                    source_cycles,
                    target_cycles,
                    test_profile=profile,
                    scenario=scenario,
                    epochs=self.transfer_epochs,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                    lambdas=self.lambdas,
                    device=self.device,
                    seed=self.seed + fold,
                    labeled_profile=labeled,
                    early_stopping_patience=self.early_stopping_patience,
                    min_epochs=self.min_epochs,
                    min_delta=self.min_delta,
                    tensor_layout=self.tensor_layout,
                )
                tatn = evaluate_cycle(
                    transferred,
                    target_cycles[profile],
                    self.device,
                    self.tensor_layout,
                )
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": transferred.state_dict(),
                        "history": history,
                        "target_info": target_info,
                        "source_temperature": source_temperature,
                        "target_temperature": target_temperature,
                        "scenario": scenario,
                        "profile": profile,
                        "tensor_layout": self.tensor_layout,
                    },
                    model_path,
                )
                result = {
                    "source_temperature": source_temperature,
                    "target_temperature": target_temperature,
                    "scenario": scenario,
                    "profile": profile,
                    "tensor_layout": self.tensor_layout,
                    "target_info": target_info,
                    "epochs_completed": len(history),
                    "checkpoint": str(model_path.relative_to(self.output_dir)),
                    "final_training_loss": history[-1] if history else None,
                    "records": [
                        {"method": "Source pretrained", **baseline},
                        {"method": "TATN", **tatn},
                    ],
                }
                save_json(result_path, result)
                print(f"  completed in {len(history)} epoch(s)")
            pair_records.extend(
                {
                    "source_temperature": source_temperature,
                    "target_temperature": target_temperature,
                    "scenario": scenario,
                    "profile": profile,
                    **record,
                }
                for record in result["records"]
            )
        self.records.extend(pair_records)
        return pair_records

    def run_all_transfers(self, *, profiles: tuple[str, ...] = PROFILES) -> None:
        for scenario in ("semisupervised", "unsupervised"):
            for source in TEMPERATURES:
                for target in TEMPERATURES:
                    if target != source:
                        self.run_transfer(source, target, scenario, profiles=profiles)
        pd.DataFrame(self.records).drop_duplicates().to_csv(
            self.output_dir / "fold_results.csv", index=False
        )

    def _records(self, scenario: str, source: int, target: int | None = None) -> list[dict]:
        return [
            r
            for r in self.records
            if r["scenario"] == scenario
            and r["source_temperature"] == source
            and (target is None or r["target_temperature"] == target)
        ]

    def table_for_source(self, source: int, scenario: str) -> pd.DataFrame:
        targets = [t for t in TEMPERATURES if t != source]
        columns = [f"{source}C->{target}C" for target in targets]
        rows = {}
        for method in ("Source pretrained", "TATN"):
            for metric in ("mae", "rmse"):
                values = [
                    _mean(self._records(scenario, source, target), method, metric)
                    for target in targets
                ]
                rows[(method, metric.upper())] = values + [float(np.mean(values))]
        return pd.DataFrame.from_dict(rows, orient="index", columns=columns + ["Avg"])

    def table_for_all_sources(self, scenario: str) -> pd.DataFrame:
        columns = [f"{source}C->R" for source in TEMPERATURES]
        rows = {}
        for method in ("Source pretrained", "TATN"):
            for metric in ("mae", "rmse"):
                values = [
                    _mean(self._records(scenario, source), method, metric)
                    for source in TEMPERATURES
                ]
                rows[(method, metric.upper())] = values + [float(np.mean(values))]
        return pd.DataFrame.from_dict(rows, orient="index", columns=columns + ["Avg"])

    def tables(self, *, profiles: tuple[str, ...] = PROFILES) -> dict[str, pd.DataFrame]:
        tables = {
            "II": self.run_table_ii(profiles=profiles),
            "IV": self.table_for_source(10, "semisupervised"),
            "V": self.table_for_all_sources("semisupervised"),
            "VI": self.table_for_source(10, "unsupervised"),
            "VII": self.table_for_all_sources("unsupervised"),
        }
        for name, table in tables.items():
            table.to_csv(self.output_dir / f"table_{name.lower()}.csv")
        return tables
