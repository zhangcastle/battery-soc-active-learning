from pathlib import Path

import numpy as np
import pandas as pd
import torch

from battery_benchmark.reproductions.shen_2023.data import (
    ShenCycle,
    _profile_paths,
    selected_windows,
)
from battery_benchmark.reproductions.shen_2023.model import TATN, gaussian_mmd
from battery_benchmark.reproductions.shen_2023.suite import ShenTableSuite
from battery_benchmark.reproductions.shen_2023.train import _improved


def test_tatn_shapes_and_freezing():
    model = TATN()
    x = torch.randn(2, 90, 3)
    assert model(x).shape == (2, 90)
    assert model.discriminator(model.target_extractor(x)).shape == (2, 1)

    model.initialize_target()
    assert not any(p.requires_grad for p in model.source_extractor.parameters())
    assert not any(p.requires_grad for p in model.predictor.parameters())
    assert all(p.requires_grad for p in model.target_extractor.parameters())


def test_gaussian_mmd_is_zero_for_identical_features():
    features = torch.randn(4, 90, 128)
    assert torch.isclose(gaussian_mmd(features, features), torch.tensor(0.0), atol=1e-6)


def test_relative_improvement_ignores_noise_below_threshold():
    assert _improved(0.8, float("inf"), 1e-4)
    assert _improved(0.98, 1.0, 1e-2)
    assert not _improved(0.995, 1.0, 1e-2)


def test_table_export_respects_screening_profiles(tmp_path):
    suite = object.__new__(ShenTableSuite)
    suite.output_dir = tmp_path
    observed = []
    table = pd.DataFrame({"Avg": [1.0]})
    suite.run_table_ii = lambda *, profiles: observed.append(profiles) or table
    suite.table_for_source = lambda *args: table
    suite.table_for_all_sources = lambda *args: table
    suite.tables(profiles=("Cycle_4", "UDDS"))
    assert observed == [("Cycle_4", "UDDS")]


def test_pseudo_windows_do_not_cross_rejected_gap():
    features = np.arange(900 * 3, dtype=np.float32).reshape(900, 3)
    pseudo = np.linspace(1, 0.8, 900, dtype=np.float32)
    pseudo[430:] -= 0.2
    cycle = ShenCycle("Cycle_1", 0, features, pseudo.copy(), (Path("dummy.mat"),))
    x, y = selected_windows(cycle, pseudo, threshold=0.08, lookahead=90)
    assert len(x) > 0
    for window in x:
        source_indices = window[:, 0] // 3
        assert np.all(np.diff(source_indices) == 1)


def test_profile_discovery_ignores_combined_files(tmp_path):
    root = tmp_path / "10degC" / "Drive cycles"
    root.mkdir(parents=True)
    (root / "10degC_NN_Pan18650PF.mat").touch()
    (root / "10degC_US06_HWFET_UDDS_LA92_NN_Pan18650PF.mat").touch()
    assert _profile_paths(tmp_path, 10, "NN") == (root / "10degC_NN_Pan18650PF.mat",)
