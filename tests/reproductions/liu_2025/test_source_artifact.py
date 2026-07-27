from __future__ import annotations

import torch

from battery_benchmark.preprocessing.normalization import MinMaxNormalizer
from battery_benchmark.reproductions.liu_2025 import (
    DARM,
    load_source_artifact,
    save_source_artifact,
)


def test_source_artifact_round_trip_initializes_identical_target(tmp_path):
    source = DARM()
    with torch.no_grad():
        next(source.source_encoder.parameters()).fill_(0.25)
        next(source.predictor.parameters()).fill_(0.5)
    normalizer = MinMaxNormalizer().fit(torch.tensor([[2.0, -1.0, 10.0], [4.0, 1.0, 20.0]]).numpy())
    path = tmp_path / "source.pt"
    digest = save_source_artifact(path, source, normalizer, {"seed": 42})

    restored = DARM()
    restored_normalizer = MinMaxNormalizer()
    metadata = load_source_artifact(path, restored, restored_normalizer)

    assert len(digest) == 64
    assert metadata == {"seed": 42}
    assert restored_normalizer.state_dict() == normalizer.state_dict()
    for source_parameter, target_parameter in zip(
        restored.source_encoder.parameters(),
        restored.target_encoder.parameters(),
    ):
        assert torch.equal(source_parameter, target_parameter)
        assert not source_parameter.requires_grad
    assert all(not parameter.requires_grad for parameter in restored.predictor.parameters())
