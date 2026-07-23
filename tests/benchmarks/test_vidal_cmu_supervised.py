from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/benchmarks/run_cmu_transfer.py"
spec = importlib.util.spec_from_file_location("run_cmu_transfer", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_setting_selection_uses_fixed_splits_without_leakage() -> None:
    rows = []
    for rank in range(1, 11):
        split = "train" if rank <= 6 else "validation" if rank <= 8 else "test"
        rows.append(
            {
                "cell_id": "VAH01",
                "mission_cycle_id": f"m{rank}",
                "split": split,
                "budget_1pct": rank == 1,
                "budget_5pct": rank == 1,
                "budget_10pct": rank == 1,
                "vidal_partial": rank <= 3,
                "vidal_full": rank <= 6,
            }
        )
    manifest = pd.DataFrame(rows)

    partial, validation, test = module.select_mission_ids(
        manifest, "VAH01", "vidal_partial"
    )
    full, validation_again, test_again = module.select_mission_ids(
        manifest, "VAH01", "vidal_full"
    )
    assert partial == {"m1", "m2", "m3"}
    assert full == {"m1", "m2", "m3", "m4", "m5", "m6"}
    assert validation == validation_again == {"m7", "m8"}
    assert test == test_again == {"m9", "m10"}
    assert not partial & validation
    assert not full & test


def test_zero_shot_selects_no_target_training_labels() -> None:
    manifest = pd.DataFrame(
        [
            {
                "cell_id": "VAH01",
                "mission_cycle_id": "train",
                "split": "train",
                "budget_1pct": True,
                "budget_5pct": True,
                "budget_10pct": True,
                "vidal_partial": True,
                "vidal_full": True,
            },
            {
                "cell_id": "VAH01",
                "mission_cycle_id": "validation",
                "split": "validation",
                "budget_1pct": False,
                "budget_5pct": False,
                "budget_10pct": False,
                "vidal_partial": False,
                "vidal_full": False,
            },
            {
                "cell_id": "VAH01",
                "mission_cycle_id": "test",
                "split": "test",
                "budget_1pct": False,
                "budget_5pct": False,
                "budget_10pct": False,
                "vidal_partial": False,
                "vidal_full": False,
            },
        ]
    )
    train, validation, test = module.select_mission_ids(
        manifest, "VAH01", "zero_shot"
    )
    assert train == set()
    assert validation == {"validation"}
    assert test == {"test"}
