"""Model registry: map model names to classes."""

from __future__ import annotations

from typing import Any, Type

import torch.nn as nn

_REGISTRY: dict[str, Type] = {}


def register_model(name: str):
    """Class decorator to register a model."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name: str) -> Type:
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"Unknown model '{name}'. Available: {available}. "
            f"Did you forget to import the model module?"
        )
    return _REGISTRY[name]


class ModelRegistry:
    @staticmethod
    def build(name: str, config: dict[str, Any]) -> nn.Module:
        cls = get_model_class(name)
        return cls(**{k: v for k, v in config.items() if k != "name"})

    @staticmethod
    def list() -> list[str]:
        return sorted(_REGISTRY.keys())
