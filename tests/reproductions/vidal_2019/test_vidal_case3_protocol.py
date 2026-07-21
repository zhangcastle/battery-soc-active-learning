import json
from pathlib import Path

import pandas as pd

from battery_benchmark.reproductions.vidal_2019.case2 import Case2Config, run_case2
from tests.reproductions.vidal_2019.vidal_case1_test_utils import make_case1_checkpoint, make_case1_run, make_v2_split


def _make_case3_inputs(root: Path) -> None:
    full = [
        (1, 16, "PASS", "drive_cycle", "Mixed1"),
        (2, 16, "PASS", "charge", "Charge7"),
        (3, 16, "PASS", "drive_cycle", "Mixed2"),
        (4, 16, "PASS", "charge", "Charge8"),
        (5, 16, "PASS", "drive_cycle", "Mixed3"),
        (6, 16, "PASS", "charge", "Charge10"),
        (7, 16, "PASS", "drive_cycle", "Mixed4"),
        (8, 16, "PASS", "charge", "Charge11"),
    ]
    partial = full[:6]
    test = [
        (1, 16, "PASS", "drive_cycle", "UDDS"),
        (2, 16, "PASS", "charge", "Charge3"),
        (3, 16, "PASS", "drive_cycle", "LA92"),
        (4, 16, "PASS", "charge", "Charge5"),
        (5, 16, "PASS", "drive_cycle", "US06"),
        (6, 16, "PASS", "charge", "Charge6"),
    ]
    make_v2_split(root, "samsung_30t", "full_train", full, timestamp_step=0.1)
    make_v2_split(root, "samsung_30t", "partial_train", partial, timestamp_step=0.1)
    make_v2_split(root, "samsung_30t", "test", test, timestamp_step=0.1)


def _make_case2_baseline(path: Path) -> Path:
    run = path.parent
    run.mkdir(parents=True, exist_ok=True)
    path.write_text('{"mae_percent": 1.0, "rmse_percent": 2.0, "max_error_percent": 5.0}', encoding="utf-8")
    (run / "run_metadata.json").write_text(
        json.dumps({
            "case": 2,
            "source_dataset": "lg_hg2",
            "target_dataset": "samsung_30t",
            "run_mode": "smoke",
            "development_only": True,
        }),
        encoding="utf-8",
    )
    return path


def test_case3_uses_partial_target_split_and_excludes_mix4_plus(tmp_path):
    input_root = tmp_path / "v2"
    _make_case3_inputs(input_root)
    checkpoint = make_case1_checkpoint(tmp_path / "case1_lg.pt", "lg_hg2", formal=True)
    case1_root = tmp_path / "case1"
    make_case1_run(case1_root, "samsung_30t", formal=True)
    case2_metrics = _make_case2_baseline(tmp_path / "case2" / "source_lg_hg2__target_samsung_30t" / "seed_42" / "run_001" / "metrics.json")
    cfg = Case2Config(
        "lg_hg2",
        "samsung_30t",
        epochs=1,
        batch_size=2,
        seq_len=4,
        stride=4,
        max_train_windows=3,
        source_checkpoint=checkpoint,
        target_case2_metrics_path=case2_metrics,
        target_train_split="partial_train",
        case_number=3,
        output_case_name="vidal_case3",
        resample_hz=None,
    )

    result = run_case2(cfg, input_root, tmp_path / "outputs" / "vidal_case3", tmp_path, "cpu", case1_output_root=case1_root)
    run = Path(result["run_dir"])
    metadata = json.loads((run / "run_metadata.json").read_text())
    sampling = pd.read_csv(run / "sampling_inventory.csv")
    scaler = json.loads((run / "scaler.json").read_text())
    comparison = json.loads((run / "comparison_to_case1.json").read_text())

    assert metadata["case"] == 3
    assert metadata["target_train_split"] == "partial_train"
    assert metadata["source_checkpoint_path"] == str(checkpoint)
    assert metadata["source_checkpoint_epoch"] == 514
    assert metadata["optimizer_reinitialized"] is True
    assert metadata["scheduler_reinitialized"] is True
    assert metadata["target_scaler_refit"] is True
    assert metadata["optimizer_parameter_count"] == 571
    assert metadata["frozen_parameters"] == 40
    assert metadata["target_train_window_count"] >= metadata["actual_train_window_count"]
    assert "record_7.mat" not in metadata["target_train_record_ids"]
    assert "record_8.mat" not in metadata["target_train_record_ids"]
    assert "record_7.mat" in metadata["excluded_full_train_record_ids"]
    assert "record_8.mat" in metadata["excluded_full_train_record_ids"]
    assert set(sampling["protocol_split"]) == {"partial_train", "test"}
    assert scaler["fit_split"] == "partial_train"
    assert comparison["baseline_available"] is True
    assert comparison["case2_baseline_available"] is True
    assert "case1_target_rmse_percent" in comparison
    assert "case2_rmse_percent" in comparison
    assert "case3_rmse_percent" in comparison
    assert "vidal_case3" in result["run_dir"]


def test_case3_validation_requires_partial_train_and_case2_still_requires_full_train(tmp_path):
    checkpoint = tmp_path / "x.pt"
    bad_case3 = Case2Config(
        "lg_hg2", "samsung_30t", source_checkpoint=checkpoint, case_number=3, target_train_split="full_train"
    )
    try:
        bad_case3.validate()
        raise AssertionError("Case 3 accepted full_train")
    except ValueError as exc:
        assert "Case 3 target_train_split" in str(exc)

    bad_case2 = Case2Config(
        "lg_hg2", "samsung_30t", source_checkpoint=checkpoint, case_number=2, target_train_split="partial_train"
    )
    try:
        bad_case2.validate()
        raise AssertionError("Case 2 accepted partial_train")
    except ValueError as exc:
        assert "Case 2 target_train_split" in str(exc)
