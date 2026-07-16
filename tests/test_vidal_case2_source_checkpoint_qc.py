import pytest
import torch

from battery_benchmark.experiments.vidal_case2 import load_case1_weights_for_transfer
from battery_benchmark.models.vidal_lstm import VidalLSTM
from vidal_case1_test_utils import make_case1_checkpoint


def test_rejects_smoke_checkpoint_for_formal_transfer(tmp_path):
    ckpt = make_case1_checkpoint(tmp_path / "smoke.pt", "lg_hg2", formal=False)
    with pytest.raises(ValueError, match="formal Case 1"):
        load_case1_weights_for_transfer(VidalLSTM(), ckpt, "lg_hg2", require_formal=True)


def test_rejects_dataset_mismatch(tmp_path):
    ckpt = make_case1_checkpoint(tmp_path / "source.pt", "lg_hg2", formal=True)
    with pytest.raises(ValueError, match="dataset mismatch"):
        load_case1_weights_for_transfer(VidalLSTM(), ckpt, "samsung_30t", require_formal=True)


def test_rejects_non_515_checkpoint(tmp_path):
    ckpt = make_case1_checkpoint(tmp_path / "source.pt", "lg_hg2", formal=True)
    payload = torch.load(ckpt, weights_only=False)
    payload["config"]["epochs"] = 2
    torch.save(payload, ckpt)
    with pytest.raises(ValueError, match="515"):
        load_case1_weights_for_transfer(VidalLSTM(), ckpt, "lg_hg2", require_formal=True)


def test_rejects_missing_model_state_dict(tmp_path):
    ckpt = tmp_path / "bad.pt"
    torch.save({"config": {"dataset": "lg_hg2", "run_mode": "formal", "development_only": False, "epochs": 515}}, ckpt)
    with pytest.raises(ValueError, match="model_state_dict"):
        load_case1_weights_for_transfer(VidalLSTM(), ckpt, "lg_hg2", require_formal=True)
