"""Runtime environment detection and reporting."""

from __future__ import annotations

import platform
import sys
from typing import Any

import torch


def get_environment_info() -> dict[str, Any]:
    """Collect Python, PyTorch, and hardware environment metadata."""
    info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def is_colab() -> bool:
    """Return True when running inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def is_drive_mounted(mount_point: str = "/content/drive") -> bool:
    """Return True when Google Drive is mounted at *mount_point*."""
    import os
    return os.path.isdir(mount_point) and bool(os.listdir(mount_point))


def print_environment_summary() -> None:
    """Print a concise environment summary to stdout."""
    info = get_environment_info()
    print("=" * 50)
    print("Environment Summary")
    print("=" * 50)
    for key, val in info.items():
        print(f"  {key:25s}: {val}")
    print("=" * 50)
