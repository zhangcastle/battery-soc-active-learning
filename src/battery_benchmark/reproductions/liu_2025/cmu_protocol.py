"""CMU mission selection for the Liu 2025 DARM extension."""

from __future__ import annotations

import pandas as pd


def select_duration_matched_missions(
    manifest: pd.DataFrame,
    cell_id: str,
    *,
    budget_seconds: float = 8778.0,
) -> tuple[set[str], set[str], set[str], dict[str, float | int | str]]:
    """Select a chronological complete-mission prefix matching target Cycle 1."""
    if budget_seconds <= 0:
        raise ValueError("budget_seconds must be positive")
    rows = manifest.loc[manifest["cell_id"] == cell_id].copy()
    if rows.empty or rows["mission_cycle_id"].duplicated().any():
        raise ValueError(f"{cell_id}: missing or duplicate mission rows")
    required = {"split", "split_rank", "mission_duration_s", "condition"}
    missing = required - set(rows)
    if missing:
        raise ValueError(f"CMU manifest missing columns: {sorted(missing)}")

    train = rows.loc[rows["split"] == "train"].sort_values("split_rank", kind="stable")
    if train.empty or (train["mission_duration_s"] <= 0).any():
        raise ValueError(f"{cell_id}: Train missions require positive durations")
    cumulative = train["mission_duration_s"].cumsum()
    count = min(int((cumulative < budget_seconds).sum()) + 1, len(train))
    selected = train.iloc[:count]
    adaptation_ids = set(selected["mission_cycle_id"].astype(str))
    validation_ids = set(rows.loc[rows["split"] == "validation", "mission_cycle_id"].astype(str))
    test_ids = set(rows.loc[rows["split"] == "test", "mission_cycle_id"].astype(str))
    if adaptation_ids & validation_ids or adaptation_ids & test_ids or validation_ids & test_ids:
        raise ValueError(f"{cell_id}: mission split leakage")
    return (
        adaptation_ids,
        validation_ids,
        test_ids,
        {
            "cell_id": cell_id,
            "condition": str(rows["condition"].iloc[0]),
            "budget_seconds": float(budget_seconds),
            "adaptation_missions": count,
            "adaptation_seconds": float(selected["mission_duration_s"].sum()),
        },
    )
