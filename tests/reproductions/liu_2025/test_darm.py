import torch
import torch.nn.functional as F

from battery_benchmark.reproductions.liu_2025.darm import (
    DARM,
    audit_target_bn_forward,
    flipped_target_adversarial_loss,
    gradient_reverse,
    multi_kernel_mmd,
    parameter_snapshot,
    target_bn_buffer_snapshot,
)


def _batch_norms(model):
    return [
        module
        for module in model.target_encoder.modules()
        if isinstance(module, torch.nn.BatchNorm1d)
    ]


def _state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


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
    loss, parts, metrics = model.adaptation_loss(source, target)
    loss.backward()
    assert all(torch.isfinite(value) for value in parts.values())
    assert torch.isfinite(torch.tensor(metrics["overall_domain_accuracy"]))
    assert any(parameter.grad is not None for parameter in model.target_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.source_encoder.parameters())

    mmd_only, isolated_parts, _ = model.adaptation_loss(
        source,
        target,
        adversarial_weight=0.0,
        alpha=0.0,
        beta=1.0,
    )
    assert torch.allclose(mmd_only, isolated_parts["mmd"])


def test_mmd_is_zero_for_identical_batches():
    features = torch.rand(8, 64, 5)
    assert torch.allclose(multi_kernel_mmd(features, features), torch.tensor(0.0), atol=1e-6)


def test_grl_strength_only_scales_target_encoder_adversarial_gradient():
    torch.manual_seed(5)
    base = DARM()
    base.initialize_target()
    source, target = torch.rand(4, 40, 3), torch.rand(4, 40, 3)
    gradients = {}
    for strength in (1.0, 0.1):
        model = DARM()
        model.load_state_dict(base.state_dict())
        loss, _, _ = model.adaptation_loss(
            source,
            target,
            grl_strength=strength,
            alpha=0.0,
            beta=0.0,
        )
        loss.backward()
        gradients[strength] = {
            "target": [p.grad.clone() for p in model.target_encoder.parameters()],
            "discriminator": [p.grad.clone() for p in model.discriminator.parameters()],
        }

    assert all(
        torch.allclose(weak, strong * 0.1, atol=1e-7)
        for weak, strong in zip(gradients[0.1]["target"], gradients[1.0]["target"])
    )
    assert all(
        torch.allclose(weak, strong, atol=1e-7)
        for weak, strong in zip(
            gradients[0.1]["discriminator"], gradients[1.0]["discriminator"]
        )
    )


def test_frozen_source_parameters_and_buffers_do_not_update_during_adaptation():
    model = DARM()
    model.initialize_target()
    before_encoder = _state(model.source_encoder)
    before_predictor = _state(model.predictor)
    model.train()
    loss, _, _ = model.adaptation_loss(torch.rand(4, 40, 3), torch.rand(4, 40, 3))
    loss.backward()
    torch.optim.SGD(model.adaptation_parameters(), lr=0.01).step()

    assert _state(model.source_encoder).keys() == before_encoder.keys()
    assert all(
        torch.equal(value, before_encoder[name])
        for name, value in model.source_encoder.state_dict().items()
    )
    assert all(
        torch.equal(value, before_predictor[name])
        for name, value in model.predictor.state_dict().items()
    )


def test_frozen_stats_preserves_buffers_but_affine_gets_gradients():
    model = DARM()
    model.initialize_target()
    model.set_target_bn_mode("frozen_stats")
    before = target_bn_buffer_snapshot(model.target_encoder)
    model.train()
    assert all(not module.training for module in _batch_norms(model))

    model.encode_target(torch.rand(4, 40, 3))[0].sum().backward()

    after = target_bn_buffer_snapshot(model.target_encoder)
    assert all(
        torch.equal(before[name][key], after[name][key]) for name in before for key in before[name]
    )
    assert all(
        module.weight.requires_grad and module.bias.requires_grad for module in _batch_norms(model)
    )
    assert all(
        module.weight.grad is not None and module.bias.grad is not None
        for module in _batch_norms(model)
    )


def test_fully_frozen_preserves_bn_buffers_and_affine_parameters():
    model = DARM()
    model.initialize_target()
    model.set_target_bn_mode("fully_frozen")
    buffers_before = target_bn_buffer_snapshot(model.target_encoder)
    affine_before = [
        (module.weight.detach().clone(), module.bias.detach().clone())
        for module in _batch_norms(model)
    ]
    model.train()
    loss = model.encode_target(torch.rand(4, 40, 3))[0].sum()
    loss.backward()
    torch.optim.SGD(model.target_adaptation_parameters(), lr=0.01).step()

    buffers_after = target_bn_buffer_snapshot(model.target_encoder)
    assert all(
        torch.equal(buffers_before[name][key], buffers_after[name][key])
        for name in buffers_before
        for key in buffers_before[name]
    )
    for module, (weight, bias) in zip(_batch_norms(model), affine_before):
        assert not module.weight.requires_grad and not module.bias.requires_grad
        assert torch.equal(module.weight, weight)
        assert torch.equal(module.bias, bias)


def test_bn_forward_audit_has_no_backward_step_or_parameter_drift(monkeypatch):
    model = DARM()
    model.initialize_target()
    monkeypatch.setattr(
        torch.Tensor,
        "backward",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("backward called")),
    )
    monkeypatch.setattr(
        torch.optim.Optimizer,
        "step",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("step called")),
    )

    result = audit_target_bn_forward(model, [torch.rand(4, 40, 3), torch.rand(2, 40, 3)], "legacy")

    assert result["parameter_drift"]["all_unchanged"]
    assert result["parameter_drift"]["max_abs_change"] == 0.0
    assert result["bn_buffer_drift"]["max_num_batches_tracked_change"] > 0


def test_discriminator_only_step_does_not_update_target_encoder():
    model = DARM()
    model.initialize_target()
    model.train()
    before = parameter_snapshot(model.target_encoder)
    optimizer = torch.optim.SGD(model.discriminator.parameters(), lr=0.01)
    loss, _ = model.discriminator_step_loss(torch.rand(4, 40, 3), torch.rand(4, 40, 3))
    loss.backward()
    optimizer.step()

    assert all(
        torch.equal(parameter, before[name])
        for name, parameter in model.target_encoder.named_parameters()
    )


def test_alternating_target_step_freezes_discriminator_but_passes_gradient():
    model = DARM()
    model.initialize_target()
    model.set_target_bn_mode("frozen_stats")
    model.train()
    discriminator_before = parameter_snapshot(model.discriminator)
    optimizer = torch.optim.SGD(model.target_adaptation_parameters(), lr=0.01)
    loss, _ = model.alternating_target_loss(
        torch.rand(4, 40, 3),
        torch.rand(4, 40, 3),
        alpha=0.0,
        beta=0.0,
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.target_encoder.parameters())
    optimizer.step()

    assert all(
        torch.equal(parameter, discriminator_before[name])
        for name, parameter in model.discriminator.named_parameters()
    )
    assert all(parameter.grad is None for parameter in model.discriminator.parameters())


def test_flipped_label_step_increases_target_probability_of_source():
    torch.manual_seed(3)
    discriminator = DARM().discriminator.eval()
    target_features = torch.nn.Parameter(torch.rand(4, 64, 5))
    optimizer = torch.optim.SGD([target_features], lr=0.01)
    with torch.no_grad():
        before = discriminator(target_features).softmax(dim=1)[:, 0].mean()
    loss, _ = flipped_target_adversarial_loss(discriminator, target_features)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        after = discriminator(target_features).softmax(dim=1)[:, 0].mean()
    assert after > before


def test_eq12_and_grl_target_adversarial_gradients_are_opposite():
    torch.manual_seed(4)
    discriminator = DARM().discriminator.eval()
    source_features = torch.rand(4, 64, 5)
    target_features = torch.rand(4, 64, 5, requires_grad=True)
    labels = torch.cat((torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long)))
    source_logits = discriminator(source_features).detach()

    grl_loss = F.cross_entropy(
        torch.cat((source_logits, discriminator(gradient_reverse(target_features)))),
        labels,
    )
    grl_gradient = torch.autograd.grad(grl_loss, target_features)[0]
    eq12_loss = F.cross_entropy(
        torch.cat((source_logits, discriminator(target_features))),
        labels,
    )
    eq12_gradient = torch.autograd.grad(eq12_loss, target_features)[0]

    assert torch.allclose(grl_gradient, -eq12_gradient, atol=1e-7, rtol=1e-5)
