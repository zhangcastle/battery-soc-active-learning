"""ExperimentRunner: orchestrates the full benchmark pipeline.

This is the primary coordination point called by both scripts (CLI) and
notebooks (Colab).  All reusable logic lives in modules under src/;
notebooks and scripts only call this runner.

Pipeline:
1. Set random seed
2. Detect device
3. Build source and target datasets
4. Assign splits
5. Build DataModules (normaliser fitted on source train only)
6. Build model
7. Build transfer method
8. Build optimizer + scheduler
9. Build trainer
10. Run source training + target adaptation
11. Evaluate on source test + target test
12. Save results (checkpoint, metrics, predictions, plots, config)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from battery_benchmark.data.registry import DatasetRegistry
from battery_benchmark.data.splits import split_cycles
from battery_benchmark.data.datamodule import BatteryDataModule
from battery_benchmark.evaluation.evaluator import Evaluator
from battery_benchmark.evaluation.plots import plot_soc_trajectory, plot_training_history
from battery_benchmark.experiments.result_schema import ExperimentResult
from battery_benchmark.models.registry import ModelRegistry
from battery_benchmark.training.reproducibility import set_seed
from battery_benchmark.training.schedulers import build_optimizer, build_scheduler
from battery_benchmark.training.trainer import Trainer
from battery_benchmark.transfer.registry import TransferRegistry
from battery_benchmark.utils.config import save_config
from battery_benchmark.utils.device import get_device
from battery_benchmark.utils.git_info import get_commit_hash
from battery_benchmark.utils.logging import get_logger

logger = get_logger("experiments.runner")


class ExperimentRunner:
    """Run a complete benchmark experiment from a resolved config dict.

    Parameters
    ----------
    cfg:
        Merged experiment + paths config.
    output_dir:
        Resolved absolute output directory.
    repo_root:
        Repository root for git info and relative path resolution.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        output_dir: str | Path,
        repo_root: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.repo_root = Path(repo_root) if repo_root else None

    def run(self) -> ExperimentResult:
        """Execute the full experiment pipeline and return results."""
        cfg = self.cfg
        out = self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        # --- 1. Reproducibility ---
        seed = int(cfg.get("random_seed", 42))
        set_seed(seed)
        logger.info("Seed set to %d", seed)

        # --- 2. Device ---
        device = get_device(cfg.get("device", "auto"))
        logger.info("Device: %s", device)

        # --- 3. Build datasets ---
        # Import all adapters so they self-register
        import battery_benchmark.data.synthetic  # noqa: F401
        import battery_benchmark.data.panasonic  # noqa: F401
        import battery_benchmark.data.lg_hg2     # noqa: F401
        import battery_benchmark.data.samsung_48g  # noqa: F401
        import battery_benchmark.data.samsung_30t  # noqa: F401
        import battery_benchmark.datasets.cmu_evtol.adapter    # noqa: F401

        dataset_cfg = cfg.get("dataset", {})
        source_cfg = dataset_cfg.get("source", {})
        target_cfg = dataset_cfg.get("target", {})

        logger.info("Building source dataset: %s", source_cfg.get("name"))
        source_ds = DatasetRegistry.build(source_cfg.get("name", "synthetic"), source_cfg)
        source_cycles = source_ds.generate_cycles() if hasattr(source_ds, "generate_cycles") \
            else self._load_cycles(source_ds, cfg)

        logger.info("Building target dataset: %s", target_cfg.get("name"))
        target_ds = DatasetRegistry.build(target_cfg.get("name", "synthetic"), target_cfg)
        target_cycles = target_ds.generate_cycles() if hasattr(target_ds, "generate_cycles") \
            else self._load_cycles(target_ds, cfg)

        # --- 4. Splits ---
        split_cfg = cfg.get("split", {})
        strategy = split_cfg.get("strategy", "by_cycle")
        # Remove 'strategy' so it isn't passed twice to split_cycles
        split_kwargs = {k: v for k, v in split_cfg.items() if k != "strategy"}
        source_cycles = split_cycles(source_cycles, strategy, seed=seed, **split_kwargs)
        target_cycles = split_cycles(target_cycles, strategy, seed=seed, **split_kwargs)

        # --- 5. DataModules ---
        logger.info("Setting up source DataModule")
        source_dm = BatteryDataModule(source_cycles, cfg)
        source_dm.setup()

        logger.info("Setting up target DataModule")
        target_dm = BatteryDataModule(target_cycles, cfg)
        # Use source normaliser for target — prevents test-set leakage
        if source_dm.normalizer is not None:
            from battery_benchmark.data.splits import filter_by_split
            from battery_benchmark.preprocessing.sequence_builder import SequenceBuilder
            seq_cfg = cfg.get("sequence", {})
            pre_cfg = cfg.get("preprocessing", {})
            feature_cols = pre_cfg.get("feature_cols", ["voltage", "current", "temperature"])
            builder = SequenceBuilder(
                seq_len=seq_cfg.get("seq_len", 50),
                mode=seq_cfg.get("mode", "fixed_non_overlapping"),
                feature_cols=feature_cols,
            )
            for split_name in ("train", "val", "test"):
                cycles_s = filter_by_split(target_cycles, split_name)
                if cycles_s:
                    X, y, meta = builder.build(cycles_s)
                    if X.size > 0:
                        X_norm = source_dm.normalizer.transform(X)
                        # Re-use target_dm's private dataset fields
                        from battery_benchmark.data.cycle_dataset import CycleDataset
                        ds = CycleDataset(X_norm, y if y.size > 0 else None, meta)
                        setattr(target_dm, f"_{split_name}_ds", ds)
            target_dm.normalizer = source_dm.normalizer
        else:
            target_dm.setup()

        # --- 6. Model ---
        import battery_benchmark.reproductions.vidal_2019.model  # noqa: F401
        import battery_benchmark.models.lstm        # noqa: F401
        import battery_benchmark.models.gru         # noqa: F401
        import battery_benchmark.models.cnn_lstm    # noqa: F401
        import battery_benchmark.models.tcn         # noqa: F401
        import battery_benchmark.models.transformer_encoder  # noqa: F401

        model_cfg = cfg.get("model", {})
        model = ModelRegistry.build(model_cfg.get("name", "vidal_lstm"), model_cfg)
        model = model.to(device)
        logger.info("Model: %s (%d params)", model_cfg.get("name"), model.count_parameters())

        # --- 7. Transfer method ---
        import battery_benchmark.transfer.source_only   # noqa: F401
        import battery_benchmark.transfer.scratch       # noqa: F401
        import battery_benchmark.transfer.finetune      # noqa: F401
        import battery_benchmark.transfer.mmd           # noqa: F401
        import battery_benchmark.transfer.coral         # noqa: F401
        import battery_benchmark.transfer.dann          # noqa: F401
        import battery_benchmark.transfer.pseudo_label  # noqa: F401
        import battery_benchmark.transfer.source_free   # noqa: F401

        transfer_name = cfg.get("transfer", {}).get("method", "source_only")
        transfer = TransferRegistry.build(transfer_name, cfg)
        logger.info("Transfer method: %s", transfer_name)

        # --- 8. Optimizer + scheduler ---
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)

        # --- 9. Trainer ---
        ckpt_dir = out / "checkpoints"
        trainer = Trainer(
            optimizer=optimizer,
            epochs=int(cfg.get("epochs", 2)),
            device=device,
            checkpoint_dir=ckpt_dir,
            scheduler=scheduler,
            normalizer=source_dm.normalizer,
            config=cfg,
        )

        # Resume if requested
        resume_path = cfg.get("checkpoint_path")
        if cfg.get("resume") and resume_path:
            trainer.resume_from(resume_path)

        # --- 10. Training + adaptation ---
        t0 = time.time()
        logger.info("Starting source training")
        transfer.fit_source(
            model,
            source_dm.train_dataloader(),
            source_dm.val_dataloader(),
            trainer,
        )
        logger.info("Source training complete (%.1f s)", time.time() - t0)

        # --- 11. Evaluation ---
        evaluator = Evaluator(model, device)

        source_test_results = evaluator.evaluate(source_dm.test_dataloader(), prefix="source_test_")
        target_test_results = evaluator.evaluate(target_dm.test_dataloader(), prefix="target_test_")
        source_train_results = evaluator.evaluate(source_dm.train_dataloader(), prefix="train_")
        source_val_results = evaluator.evaluate(source_dm.val_dataloader(), prefix="val_")

        src_metrics = source_test_results["metrics"]
        tgt_metrics = target_test_results["metrics"]
        trn_metrics = source_train_results["metrics"]
        val_metrics = source_val_results["metrics"]

        result = ExperimentResult(
            experiment_name=cfg.get("experiment_name", "unknown"),
            source_domain=source_cfg.get("domain_id", "unknown"),
            target_domain=target_cfg.get("domain_id", "unknown"),
            transfer_method=transfer_name,
            model_name=model_cfg.get("name", "unknown"),
            random_seed=seed,
            epochs_run=len(trainer.history),
            train_mae=trn_metrics.get("train_mae", float("nan")),
            train_rmse=trn_metrics.get("train_rmse", float("nan")),
            val_mae=val_metrics.get("val_mae", float("nan")),
            val_rmse=val_metrics.get("val_rmse", float("nan")),
            test_mae=src_metrics.get("source_test_mae", float("nan")),
            test_rmse=src_metrics.get("source_test_rmse", float("nan")),
            test_max_error=src_metrics.get("source_test_max_error", float("nan")),
            test_bias=src_metrics.get("source_test_bias", float("nan")),
            test_r2=src_metrics.get("source_test_r2", float("nan")),
            n_parameters=model.count_parameters(),
            extra={
                "target_test_metrics": tgt_metrics,
                "git_commit": get_commit_hash(self.repo_root),
                "training_time_s": round(time.time() - t0, 2),
            },
        )

        # --- 12. Save outputs ---
        self._save_results(result, trainer, source_test_results, target_test_results, cfg)

        logger.info(
            "Experiment complete — source test MAE=%.4f  RMSE=%.4f",
            result.test_mae, result.test_rmse,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_cycles(self, ds, cfg):
        raise NotImplementedError(
            f"Dataset '{ds.name}' does not implement generate_cycles() "
            "and full file-loading is not yet supported through the runner. "
            "See the dataset's notebook for preprocessing instructions."
        )

    def _save_results(
        self,
        result: ExperimentResult,
        trainer: Trainer,
        source_results: dict,
        target_results: dict,
        cfg: dict[str, Any],
    ) -> None:
        out = self.output_dir

        # Resolved config
        save_config(cfg, out / "resolved_config.yaml")

        # Metrics JSON
        metrics_all = {
            **source_results["metrics"],
            **target_results["metrics"],
        }
        metrics_all.update(result.to_dict())
        with (out / "metrics.json").open("w") as f:
            json.dump(metrics_all, f, indent=2, default=str)

        # Predictions CSV
        n = min(len(source_results["y_true"]), len(source_results["y_pred"]))
        pd.DataFrame({
            "y_true": source_results["y_true"][:n],
            "y_pred": source_results["y_pred"][:n],
            "domain": "source_test",
        }).to_csv(out / "predictions_source.csv", index=False)

        n2 = min(len(target_results["y_true"]), len(target_results["y_pred"]))
        pd.DataFrame({
            "y_true": target_results["y_true"][:n2],
            "y_pred": target_results["y_pred"][:n2],
            "domain": "target_test",
        }).to_csv(out / "predictions_target.csv", index=False)

        # Plots
        plots_dir = out / "plots"
        plots_dir.mkdir(exist_ok=True)

        plot_soc_trajectory(
            source_results["y_true"][:500],
            source_results["y_pred"][:500],
            title="Source Test — SOC Trajectory (first 500 samples)",
            output_path=plots_dir / "source_test_trajectory.png",
        )
        plot_training_history(
            trainer.history,
            output_path=plots_dir / "training_history.png",
        )
