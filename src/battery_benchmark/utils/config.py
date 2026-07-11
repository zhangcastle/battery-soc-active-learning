"""YAML configuration loading and merging.

Design rules:
- No hard-coded absolute paths in any config.
- All paths are resolved via ``paths.py`` after loading.
- The resolved config is always saved alongside experiment outputs.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at the top level, got {type(data)}: {path}")
    return data


def load_config(
    config_path: str | Path,
    paths_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load an experiment config and optionally merge a paths config.

    Parameters
    ----------
    config_path:
        Path to the experiment YAML (e.g. ``configs/experiments/smoke_test.yaml``).
    paths_path:
        Path to the paths YAML (e.g. ``configs/paths/local.yaml``).
        When provided, its contents are stored under ``cfg["paths"]``.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    cfg = load_yaml(config_path)
    if paths_path is not None:
        paths = load_yaml(paths_path)
        cfg = merge_paths(cfg, paths)
    return cfg


def merge_paths(cfg: dict[str, Any], paths: dict[str, Any]) -> dict[str, Any]:
    """Inject a paths dictionary under ``cfg["paths"]``.

    Any existing ``cfg["paths"]`` keys are overridden by ``paths``.
    """
    cfg = copy.deepcopy(cfg)
    existing = cfg.get("paths", {}) or {}
    cfg["paths"] = {**existing, **paths}
    return cfg


def override_from_env(cfg: dict[str, Any], prefix: str = "BSB_") -> dict[str, Any]:
    """Override config leaf values from environment variables.

    Environment variable naming convention::

        BSB_EPOCHS=5          → cfg["epochs"] = 5
        BSB_MODEL__HIDDEN_SIZE=32 → cfg["model"]["hidden_size"] = 32

    Values are cast to int/float/bool when possible.
    """
    cfg = copy.deepcopy(cfg)
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        path_parts = key[len(prefix):].lower().split("__")
        node = cfg
        for part in path_parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node is None or not isinstance(node, dict):
            continue
        leaf = path_parts[-1]
        cfg_val = _cast(val)
        if leaf in node or True:  # allow creating new keys from env
            node[leaf] = cfg_val
    return cfg


def _cast(val: str) -> Any:
    """Attempt to cast a string to int, float, or bool."""
    if val.lower() in ("true", "yes"):
        return True
    if val.lower() in ("false", "no"):
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def save_config(cfg: dict[str, Any], output_path: str | Path) -> None:
    """Save a resolved configuration dict to YAML for reproducibility."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
