from pathlib import Path

import yaml


def test_case1_configs_use_protocol_splits_not_profiles():
    root = Path(__file__).resolve().parents[3]
    for name in ("vidal_case1.yaml", "vidal_case1_smoke.yaml"):
        cfg = yaml.safe_load(
            (root / "configs/reproductions/vidal_2019" / name).read_text()
        )
        assert "train_profiles" not in str(cfg)
        assert "partial_train" not in str(cfg)
