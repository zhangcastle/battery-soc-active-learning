from pathlib import Path

import numpy as np
import torch

from scripts.benchmarks.run_liu_cmu_darm import _cell_path, _evaluate
from battery_benchmark.reproductions.liu_2025.darm import DARM


def test_cmu_evaluation_reports_overall_and_mission_macro_metrics():
    model = DARM()
    for parameter in model.parameters():
        parameter.data.zero_()
    x = np.zeros((3, 40, 3), dtype=np.float32)
    y = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    metrics = _evaluate(
        model,
        x,
        y,
        [("short", 1), ("long", 2)],
        batch_size=3,
        device=torch.device("cpu"),
    )

    assert np.isclose(metrics["overall_rmse_percent"], np.sqrt(2 / 3) * 100)
    assert np.isclose(metrics["mission_macro_rmse_percent"], 50.0)


def test_cmu_cells_route_to_the_existing_two_artifact_roots():
    base, condition = Path("base"), Path("condition")
    assert _cell_path("VAH01", base, condition) == base / "VAH01_dynamic_soc.parquet"
    assert _cell_path("VAH30", base, condition) == condition / "VAH30_dynamic_soc.parquet"
