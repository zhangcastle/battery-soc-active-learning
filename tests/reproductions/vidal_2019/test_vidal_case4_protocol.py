from pathlib import Path

import yaml

from battery_benchmark.reproductions.vidal_2019.case4 import Case4FinetuneConfig, Case4PretrainConfig


def test_case4_configs_match_protocol():
    pre = yaml.safe_load(Path("configs/reproductions/vidal_2019/vidal_case4_pretrain.yaml").read_text())
    fine = yaml.safe_load(Path("configs/reproductions/vidal_2019/vidal_case4_finetune.yaml").read_text())
    assert pre["source_datasets"] == ["panasonic_18650pf", "lg_hg2", "samsung_30t"]
    assert pre["source_train_split"] == "partial_train"
    assert pre["scaler_fit_scope"] == "multi_source_partial_train"
    assert pre["sampling_balance"] == "natural"
    assert pre["epochs"] == 515
    assert pre["max_train_windows"] is None
    assert pre["development_only"] is False
    assert fine["target_dataset"] == "samsung_30t"
    assert fine["target_train_split"] == "full_train"
    assert fine["epochs"] == 515
    assert fine["max_train_windows"] is None
    assert fine["development_only"] is False


def test_case4_dataclass_validation_guards_protocol(tmp_path):
    bad_pre = Case4PretrainConfig(source_train_split="full_train")
    try:
        bad_pre.validate()
        raise AssertionError("Stage A accepted full_train")
    except ValueError as exc:
        assert "partial_train" in str(exc)

    bad_fine = Case4FinetuneConfig(target_train_split="partial_train", source_checkpoint=tmp_path / "x.pt")
    try:
        bad_fine.validate()
        raise AssertionError("Stage B accepted partial target split")
    except ValueError as exc:
        assert "full_train" in str(exc)
