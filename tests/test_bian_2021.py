import torch

from battery_benchmark.reproductions.bian_2021 import (
    DTNN,
    adaptive_multiscale_mmd,
    feature_mmd,
)


def test_dtnn_is_sequence_to_sequence_and_uses_paper_adaptive_weights():
    model = DTNN(input_size=2, cnn_filters=8, bilstm_hidden=4, fc_hidden=6)
    prediction, features = model(torch.randn(3, 20, 2), return_features=True)
    assert prediction.shape == (3, 20)
    assert [feature.shape for feature in features] == [(3, 20, 8), (3, 20, 8), (3, 20, 6)]

    _, losses, weights = adaptive_multiscale_mmd(features, tuple(x + 0.2 for x in features))
    expected = 1.0 - losses.detach() / losses.detach().sum()
    assert torch.allclose(weights, expected)
    assert feature_mmd(features[0], features[0]) == 0

    sequence = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    assert feature_mmd(sequence, sequence.flip(1)) == 0
