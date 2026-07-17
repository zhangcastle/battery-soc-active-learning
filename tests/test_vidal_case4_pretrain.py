import json
from pathlib import Path

from battery_benchmark.experiments.vidal_case4 import Case4PretrainConfig, run_case4_pretrain
from vidal_case1_test_utils import make_v2_split


def _make_three_partial_inputs(root: Path) -> None:
    make_v2_split(root, "panasonic_18650pf", "partial_train", [(1, 16, "PASS", "drive_cycle", "Cycle1")])
    make_v2_split(root, "lg_hg2", "partial_train", [(1, 20, "PASS", "drive_cycle", "Mixed1"), (2, 12, "PASS", "charge", "Charge7")])
    make_v2_split(root, "samsung_30t", "partial_train", [(1, 24, "PASS", "drive_cycle", "Mixed1")])
    # Extreme full/test values must not be read by Stage A.
    make_v2_split(root, "samsung_30t", "full_train", [(9, 10, "PASS", "drive_cycle", "Mixed9")])
    make_v2_split(root, "samsung_30t", "test", [(1, 10, "PASS", "drive_cycle", "UDDS")])


def test_case4_pretrain_uses_three_partial_train_only_and_natural_sampling(tmp_path):
    input_root = tmp_path / "v2"
    _make_three_partial_inputs(input_root)
    cfg = Case4PretrainConfig(epochs=1, batch_size=2, seq_len=4, stride=4, max_train_windows=3, resample_hz=None)
    result = run_case4_pretrain(cfg, input_root, tmp_path / "out" / "pretraining", tmp_path, "cpu")
    run = Path(result["run_dir"])
    metadata = json.loads((run / "run_metadata.json").read_text())
    scaler = json.loads((run / "scaler.json").read_text())

    assert metadata["case"] == 4
    assert metadata["stage"] == "multi_source_partial_pretraining"
    assert metadata["reconstruction_scope"] == "three_battery"
    assert metadata["source_train_split"] == "partial_train"
    assert metadata["scaler_fit_scope"] == "multi_source_partial_train"
    assert metadata["sampling_balance"] == "natural"
    assert set(metadata["source_datasets"]) == {"panasonic_18650pf", "lg_hg2", "samsung_30t"}
    assert metadata["total_train_windows"] == sum(metadata["source_window_count_by_dataset"].values())
    assert metadata["actual_train_window_count"] == 3
    assert metadata["trainable_parameters"] == 571
    assert metadata["frozen_parameters"] == 40
    assert scaler["fit_split"] == "partial_train"
    assert scaler["scaler_fit_scope"] == "multi_source_partial_train"
    assert (run / "checkpoints" / "last.pt").is_file()
    assert (run / "figures" / "training_loss.png").is_file()


def test_case4_pretrain_validation_rejects_formal_shortcuts():
    cfg = Case4PretrainConfig(run_mode="formal", development_only=True, epochs=2, max_train_windows=32)
    try:
        cfg.validate()
        raise AssertionError("Formal Case 4 pretrain accepted development shortcut")
    except ValueError as exc:
        assert "Formal Case 4" in str(exc)
