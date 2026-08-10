from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scripts.benchmarks.run_bian_cmu import _target_parts

from battery_benchmark.reproductions.bian_2021 import (
    DTNN,
    adaptive_multiscale_mmd,
    feature_mmd,
)


def test_dtnn_is_sequence_to_sequence_and_uses_paper_adaptive_weights():
    torch.manual_seed(2)  # Previously produced an all-zero, gradient-dead ReLU output.
    model = DTNN(input_size=2, cnn_filters=8, bilstm_hidden=4, fc_hidden=6)
    prediction, features = model(torch.randn(3, 20, 2), return_features=True)
    assert prediction.shape == (3, 20)
    assert (prediction > 0).all()
    assert [feature.shape for feature in features] == [(3, 20, 8), (3, 20, 8), (3, 20, 6)]

    _, losses, weights = adaptive_multiscale_mmd(features, tuple(x + 0.2 for x in features))
    expected = 1.0 - losses.detach() / losses.detach().sum()
    assert torch.allclose(weights, expected)
    assert feature_mmd(features[0], features[0]) == 0

    sequence = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    assert feature_mmd(sequence, sequence.flip(1)) == 0


def test_target_protocol_starts_at_initial_soc_then_splits_30_70():
    frame = pd.DataFrame(
        {
            "current": np.arange(101),
            "voltage": np.arange(101),
            "soc": np.linspace(1.0, 0.0, 101),
        }
    )
    record = SimpleNamespace(data=frame, metadata={"mission_cycle_id": "mission"})
    adapt = _target_parts([record], {"mission"}, 0.7, 0.3, "adapt")[0]
    test = _target_parts([record], {"mission"}, 0.7, 0.3, "test")[0]

    assert np.isclose(adapt[1][0], 0.7)
    assert len(adapt[1]) + len(test[1]) == 71
    assert adapt[1][-1] > test[1][0]
