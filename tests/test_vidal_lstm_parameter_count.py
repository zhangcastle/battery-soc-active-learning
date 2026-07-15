import torch

from battery_benchmark.models.vidal_lstm import VidalLSTM


def test_vidal_trainable_parameter_count_and_frozen_bias(tmp_path):
    model = VidalLSTM()
    assert model.count_trainable_parameters() == 571
    assert sum(p.numel() for p in model.parameters() if not p.requires_grad) == 40
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.01)
    assert all(model.lstm.bias_hh_l0 is not p for group in optimizer.param_groups for p in group["params"])
    path = tmp_path / "model.pt"; torch.save(model.state_dict(), path)
    restored = VidalLSTM(); restored.load_state_dict(torch.load(path, weights_only=True))
    assert not restored.lstm.bias_hh_l0.requires_grad
    assert torch.count_nonzero(restored.lstm.bias_hh_l0) == 0
    assert restored(torch.zeros(2, 5, 3)).shape == (2, 5)
