"""Utility modules: config loading, path resolution, device detection."""

from battery_benchmark.utils.config import load_config, merge_paths
from battery_benchmark.utils.device import get_device
from battery_benchmark.utils.paths import resolve_paths

__all__ = ["load_config", "merge_paths", "get_device", "resolve_paths"]
