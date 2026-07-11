"""PyTorch device detection and selection."""

from __future__ import annotations

import torch


def get_device(device_str: str = "auto") -> torch.device:
    """Resolve a device string to a ``torch.device``.

    Parameters
    ----------
    device_str:
        ``"auto"`` selects CUDA if available, then MPS (Apple Silicon),
        then CPU.  Any other string is passed directly to ``torch.device``.

    Returns
    -------
    torch.device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def device_info(device: torch.device) -> dict[str, str]:
    """Return human-readable device metadata."""
    info: dict[str, str] = {"device": str(device)}
    if device.type == "cuda":
        idx = device.index or 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        mem = torch.cuda.get_device_properties(idx).total_memory
        info["gpu_memory_gb"] = f"{mem / 1e9:.1f}"
        info["cuda_version"] = torch.version.cuda or "unknown"
    return info
