import torch

from battery_benchmark.experiments.vidal_case2 import load_case1_weights_for_transfer
from battery_benchmark.models.vidal_lstm import VidalLSTM
from vidal_case1_test_utils import make_case1_checkpoint


def test_case2_optimizer_contains_all_paper_trainable_parameters(tmp_path):
    ckpt = make_case1_checkpoint(tmp_path / "source.pt", "lg_hg2", formal=True)
    model = VidalLSTM()
    load_case1_weights_for_transfer(model, ckpt, "lg_hg2")
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.01)
    assert sum(p.numel() for group in optimizer.param_groups for p in group["params"]) == 571
    assert not model.lstm.bias_hh_l0.requires_grad
    assert torch.count_nonzero(model.lstm.bias_hh_l0) == 0
