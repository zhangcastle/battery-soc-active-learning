import torch

from battery_benchmark.experiments.vidal_case2 import load_case1_weights_for_transfer
from battery_benchmark.models.vidal_lstm import VidalLSTM
from vidal_case1_test_utils import make_case1_checkpoint


def test_transfer_loads_only_weights_and_preserves_trainable_structure(tmp_path):
    ckpt = make_case1_checkpoint(tmp_path / "source.pt", "lg_hg2", formal=True)
    source_payload = torch.load(ckpt, weights_only=False)
    model = VidalLSTM()
    info = load_case1_weights_for_transfer(model, ckpt, "lg_hg2")
    for name, value in model.state_dict().items():
        assert torch.allclose(value, source_payload["model_state_dict"][name])
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.5)
    assert optimizer.state == {}
    assert scheduler.last_epoch == 0
    assert model.count_trainable_parameters() == 571
    assert not model.lstm.bias_hh_l0.requires_grad
    assert torch.count_nonzero(model.lstm.bias_hh_l0) == 0
    assert info["weights_loaded_from_source"] is True
