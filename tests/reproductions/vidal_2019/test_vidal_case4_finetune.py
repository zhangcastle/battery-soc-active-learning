import json
from pathlib import Path

import torch

from battery_benchmark.reproductions.vidal_2019.case4 import Case4FinetuneConfig, Case4PretrainConfig, run_case4_finetune, run_case4_pretrain
from tests.reproductions.vidal_2019.vidal_case1_test_utils import make_case1_run, make_v2_split


def _make_inputs(root: Path) -> None:
    for dataset in ("panasonic_18650pf", "lg_hg2", "samsung_30t"):
        make_v2_split(root, dataset, "partial_train", [(1, 16, "PASS", "drive_cycle", "Mixed1")])
    make_v2_split(root, "samsung_30t", "full_train", [(1, 16, "PASS", "drive_cycle", "Mixed1"), (2, 16, "PASS", "charge", "Charge7")])
    make_v2_split(root, "samsung_30t", "test", [(1, 16, "PASS", "drive_cycle", "UDDS")])


def test_case4_finetune_loads_only_stage_a_weights_and_refits_target_scaler(tmp_path):
    input_root = tmp_path / "v2"
    _make_inputs(input_root)
    pre_cfg = Case4PretrainConfig(epochs=1, batch_size=2, seq_len=4, stride=4, max_train_windows=2, resample_hz=None)
    pre = run_case4_pretrain(pre_cfg, input_root, tmp_path / "out" / "pretraining", tmp_path, "cpu")
    source = Path(pre["run_dir"]) / "checkpoints" / "last.pt"
    case1_root = tmp_path / "case1"
    make_case1_run(case1_root, "samsung_30t", formal=True)
    cfg = Case4FinetuneConfig(epochs=1, batch_size=2, seq_len=4, stride=4, max_train_windows=2, source_checkpoint=source, resample_hz=None)
    result = run_case4_finetune(cfg, input_root, tmp_path / "out" / "finetune", tmp_path, "cpu", case1_output_root=case1_root)
    run = Path(result["run_dir"])
    metadata = json.loads((run / "run_metadata.json").read_text())
    scaler = json.loads((run / "scaler.json").read_text())
    comparison = json.loads((run / "comparison_to_case1.json").read_text())

    assert metadata["case"] == 4
    assert metadata["stage"] == "target_full_train_finetuning"
    assert metadata["source_checkpoint_stage"] == "multi_source_partial_pretraining"
    assert metadata["weights_loaded_from_source"] is True
    assert metadata["optimizer_reinitialized"] is True
    assert metadata["scheduler_reinitialized"] is True
    assert metadata["source_scaler_used"] is False
    assert metadata["target_scaler_refit"] is True
    assert metadata["target_train_split"] == "full_train"
    assert metadata["optimizer_parameter_count"] == 571
    assert metadata["frozen_parameters"] == 40
    assert metadata["initial_mse_window_count"] == metadata["actual_train_window_count"]
    assert scaler["fit_split"] == "full_train"
    assert comparison["baseline_available"] is True
    assert "case4_rmse_percent" in comparison
    ckpt = torch.load(run / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt["case"] == 4
    assert ckpt["stage"] == "target_full_train_finetuning"


def test_case4_finetune_formal_rejects_smoke_stage_a_checkpoint(tmp_path):
    input_root = tmp_path / "v2"
    _make_inputs(input_root)
    pre_cfg = Case4PretrainConfig(epochs=1, batch_size=2, seq_len=4, stride=4, max_train_windows=2, resample_hz=None)
    pre = run_case4_pretrain(pre_cfg, input_root, tmp_path / "out" / "pretraining", tmp_path, "cpu")
    source = Path(pre["run_dir"]) / "checkpoints" / "last.pt"
    cfg = Case4FinetuneConfig(run_mode="formal", development_only=False, epochs=515, max_train_windows=None, source_checkpoint=source)
    try:
        cfg.validate()
        from battery_benchmark.reproductions.vidal_2019.case4 import load_case4_pretrain_weights_for_finetune
        from battery_benchmark.reproductions.vidal_2019.model import VidalLSTM
        load_case4_pretrain_weights_for_finetune(VidalLSTM(), source, require_formal=True)
        raise AssertionError("Formal Stage B accepted smoke Stage A checkpoint")
    except ValueError as exc:
        assert "formal Stage A" in str(exc) or "Formal Case 4" in str(exc)
