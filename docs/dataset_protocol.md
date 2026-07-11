# Dataset Protocol

## Required schema

Every dataset adapter must return DataFrames with these columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `timestamp` | float | Yes | Seconds since cycle start |
| `voltage` | float | Yes | V |
| `current` | float | Yes | A (positive = charging) |
| `temperature` | float | Yes | °C |
| `cell_id` | str | Yes | Unique cell identifier |
| `cycle_id` | int | Yes | Cycle number within cell |
| `domain_id` | str | Yes | Dataset domain identifier |
| `profile_name` | str | Yes | Drive-cycle profile name |
| `split` | str | Yes | train / val / test / unsplit |
| `soc` | float | No | [0, 1] (may be NaN) |
| `label_source` | str | No | See LabelSource enum |

## Anti-leakage requirements

1. Never concatenate cycles before split assignment.
2. Normalisers fit on training data only.
3. Per-cycle identity (cell_id, cycle_id) must be preserved.
4. Missions must not be split across train/test.

## Adding a new dataset

1. Create `src/battery_benchmark/data/your_dataset.py`.
2. Subclass `BaseBatteryDataset`.
3. Decorate with `@register_dataset("your_dataset_name")`.
4. Implement all abstract methods.
5. Create `configs/datasets/your_dataset.yaml`.
6. Add preprocessing notebook `notebooks/0X_prepare_your_dataset.ipynb`.
7. Document uncertainties in `docs/known_uncertainties.md`.
