import json
from pathlib import Path

import pandas as pd
import torch

from battery_benchmark.experiments.vidal_case1 import cache_case1_inputs
from battery_benchmark.experiments.vidal_case2 import Case2Config, run_case2
from battery_benchmark.training.checkpointing import load_checkpoint
from vidal_case1_test_utils import make_case1_checkpoint, make_case1_run, make_v2_split


def test_case2_cache_only_needs_target_full_train_and_test(tmp_path):
    source_root = tmp_path / "source_data"
    cache_root = tmp_path / "cache"
    for split in ("full_train", "test"):
        make_v2_split(source_root, "samsung_30t", split, [(1, 12, "PASS", "drive_cycle", "Mixed1")])
    cache_case1_inputs(source_root, cache_root, "samsung_30t")
    checkpoint = make_case1_checkpoint(tmp_path / "case1_lg.pt", "lg_hg2", formal=True)
    case1_root = tmp_path / "case1"
    make_case1_run(case1_root, "samsung_30t", formal=True)
    cfg = Case2Config(
        "lg_hg2",
        "samsung_30t",
        epochs=1,
        batch_size=2,
        seq_len=4,
        stride=4,
        max_train_windows=2,
        source_checkpoint=checkpoint,
        resample_hz=1.0,
    )
    result = run_case2(cfg, cache_root, tmp_path / "out", tmp_path, "cpu", case1_output_root=case1_root)
    run = Path(result["run_dir"])
    assert (run / "predictions" / "test_predictions.csv").is_file()
    assert (run / "figures" / "lg_hg2_to_samsung_30t_case2_test_soc_prediction_and_error.png").is_file()
    assert (run / "figures" / "lg_hg2_to_samsung_30t_case2_training_loss.png").is_file()
    assert (run / "figures" / "lg_hg2_to_samsung_30t_case1_vs_case2.png").is_file()
    metadata = json.loads((run / "run_metadata.json").read_text())
    assert metadata["weights_loaded_from_source"] is True
    assert metadata["target_scaler_refit"] is True
    assert metadata["optimizer_parameter_count"] == 571
    assert metadata["actual_train_window_count"] == 2
    assert metadata["initial_mse_window_count"] == metadata["actual_train_window_count"]
    assert metadata["initial_target_mse_before_training"] >= 0
    assert metadata["final_target_train_mse"] >= 0
    prediction = pd.read_csv(run / "predictions" / "test_predictions.csv")
    assert prediction["soc_predicted"].notna().all()
    model = torch.nn.Module()
    del model
    ckpt = load_checkpoint(run / "checkpoints" / "last.pt", __import__("battery_benchmark.models.vidal_lstm", fromlist=["VidalLSTM"]).VidalLSTM())
    assert "optimizer_state_dict" in ckpt
