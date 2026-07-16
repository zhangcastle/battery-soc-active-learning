import pytest

from battery_benchmark.experiments.vidal_case2 import Case2Config


def test_case2_requires_distinct_supported_datasets(tmp_path):
    with pytest.raises(ValueError, match="source_dataset != target_dataset"):
        Case2Config("lg_hg2", "lg_hg2", source_checkpoint=tmp_path / "x.pt").validate()
    with pytest.raises(ValueError, match="Unsupported source_dataset"):
        Case2Config("unknown", "lg_hg2", source_checkpoint=tmp_path / "x.pt").validate()


def test_formal_requires_source_checkpoint(tmp_path):
    cfg = Case2Config(
        "lg_hg2",
        "samsung_30t",
        run_mode="formal",
        epochs=515,
        max_train_windows=None,
        development_only=False,
    )
    with pytest.raises(ValueError, match="source Case 1 checkpoint"):
        cfg.validate()
