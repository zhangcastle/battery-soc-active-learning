# Benchmark Protocol

See `docs/fairness_protocol.md` for the two comparison modes.

## Running a new experiment

1. Create an experiment config in `configs/experiments/`.
2. Verify the config with `python scripts/validate_dataset.py`.
3. Run: `python scripts/train.py --config ... --paths ...`
4. Inspect outputs in `outputs/<experiment_name>/`.
5. Aggregate results: `python scripts/aggregate_results.py --results-dir outputs/`

## Minimum reporting requirements

- Experiment config (saved as `resolved_config.yaml`)
- Training and validation loss curves
- Source test metrics: MAE, RMSE, MAX error
- Target test metrics: MAE, RMSE, MAX error (if labels available)
- Git commit hash
- Random seed
