import torch

from battery_benchmark.reproductions.liu_2025.darm import DARM, multi_kernel_mmd


def test_darm_shapes_and_losses_are_runnable():
    torch.manual_seed(0)
    model = DARM()
    source = torch.rand(4, 40, 3)
    target = torch.rand(4, 40, 3)

    assert model.forward_source(source).shape == (4,)
    latent, skips = model.encode_target(target)
    assert latent.shape == (4, 64, 5)
    assert model.decoder(latent, skips).shape == target.shape

    model.initialize_target()
    loss, parts = model.adaptation_loss(source, target)
    loss.backward()
    assert all(torch.isfinite(torch.tensor(value)) for value in parts.values())
    assert any(parameter.grad is not None for parameter in model.target_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.source_encoder.parameters())


def test_mmd_is_zero_for_identical_batches():
    features = torch.rand(8, 64, 5)
    assert torch.allclose(multi_kernel_mmd(features, features), torch.tensor(0.0), atol=1e-6)
