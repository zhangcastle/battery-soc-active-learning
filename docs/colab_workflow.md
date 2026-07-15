# Colab Workflow

## Quick start (5 steps)

1. Open `notebooks/00_colab_setup.ipynb` in Colab.
2. Run **Cell 1**: mount Google Drive.
3. Run **Cell 2**: clone or update the repository.
4. Run **Cell 3–4**: install dependencies and package.
5. Run **Cell 10–11**: import test and smoke test.

## Path configuration

Edit `configs/paths/colab_drive.yaml` to match your Drive layout.
**Never edit Python source files** to change paths.

```yaml
drive_root: /content/drive/MyDrive/battery_soc_benchmark
raw_data_root: /content/drive/MyDrive/battery_soc_benchmark/data/raw
output_root: /content/drive/MyDrive/battery_soc_benchmark/outputs
```

## Resume after disconnection

If your Colab session disconnects mid-training, resume with:

```python
from battery_benchmark.experiments.runner import ExperimentRunner
from battery_benchmark.utils.config import load_config

cfg = load_config("configs/experiments/your_experiment.yaml",
                  paths_path="configs/paths/colab_drive.yaml")
cfg["checkpoint_path"] = "outputs/your_experiment/checkpoints/last.pt"
cfg["resume"] = True

runner = ExperimentRunner(cfg, output_dir="outputs/your_experiment")
result = runner.run()
```

## Dataset management

- Raw datasets live in Google Drive: `drive_root/data/raw/`
- Processed datasets live in: `drive_root/data/processed/`
- Never copy datasets into the repository directory.
- Symlinks or config-based paths keep data in Drive.

## Common Colab pitfalls

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: battery_benchmark` | Re-run `pip install -e .` cell |
| `Drive not mounted` | Run the Drive mount cell first |
| `CUDA not available` | Colab free tier: Go to Runtime → Change runtime type → T4 GPU |
| Out of RAM | Reduce `batch_size` or `num_cycles_per_cell` in config |
| Checkpoint not found on resume | Check `output_root` path in colab_drive.yaml |
# Vidal Case 1

Use `notebooks/10_vidal_case1_baseline.ipynb`. It mounts Drive, validates all
nine v2 split directories, performs a non-destructive write/read/delete test,
optionally caches `full_train` and `test` under `/content`, and invokes
`scripts/run_vidal_case1.py`. Smoke outputs are isolated under
`outputs/vidal_case1/_smoke/`; formal runs are never launched automatically.
