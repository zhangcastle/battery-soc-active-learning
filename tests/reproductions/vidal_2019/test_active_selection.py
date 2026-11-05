import numpy as np
import pytest
import torch

from battery_benchmark.reproductions.vidal_2019.active_selection import (
    committee_uncertainty,
    select_indices,
)
from battery_benchmark.reproductions.vidal_2019.case2 import Case2Config, select_target_windows
from battery_benchmark.reproductions.vidal_2019.model import VidalLSTM


def test_committee_uncertainty_ranks_disagreement() -> None:
    first = np.zeros((3, 2))
    second = np.asarray([[0.0, 0.0], [0.2, 0.2], [0.8, 0.8]])
    scores = committee_uncertainty([first, second])
    assert scores[2] > scores[1] > scores[0]


def test_qbc_selects_highest_uncertainty() -> None:
    result = select_indices("qbc", 2, 4, 7, uncertainty=np.asarray([0.1, 0.9, 0.2, 0.8]))
    assert result.indices.tolist() == [1, 3]


def test_regression_clue_is_deterministic_and_non_redundant() -> None:
    embeddings = np.asarray([[0.0], [0.1], [10.0], [10.1]])
    uncertainty = np.asarray([1.0, 2.0, 1.0, 2.0])
    first = select_indices(
        "regression_clue", 2, 4, 11, uncertainty=uncertainty, embeddings=embeddings
    )
    second = select_indices(
        "regression_clue", 2, 4, 11, uncertainty=uncertainty, embeddings=embeddings
    )
    assert first.indices.tolist() == second.indices.tolist()
    assert len(set(first.indices.tolist())) == 2
    assert any(index < 2 for index in first.indices)
    assert any(index >= 2 for index in first.indices)


def test_qbc_requires_a_committee() -> None:
    with pytest.raises(ValueError, match="at least two"):
        committee_uncertainty([np.zeros((2, 3))])


def test_random_budget_is_applied_to_case2_pool() -> None:
    cfg = Case2Config(
        "lg_hg2",
        "samsung_30t",
        source_checkpoint="unused.pt",
        selection_strategy="random",
        target_budget_fraction=0.25,
    )
    selected, audit = select_target_windows(cfg, list(range(10)), VidalLSTM(), torch.device("cpu"))
    assert len(selected) == 3
    assert int(audit["selected"].sum()) == 3
