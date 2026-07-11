"""Transfer method registry."""

from __future__ import annotations

from typing import Any, Type

_REGISTRY: dict[str, Type] = {}


def register_transfer(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_transfer_class(name: str) -> Type:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown transfer method '{name}'. "
            f"Available: {sorted(_REGISTRY.keys())}."
        )
    return _REGISTRY[name]


class TransferRegistry:
    @staticmethod
    def build(name: str, config: dict[str, Any]):
        cls = get_transfer_class(name)
        return cls(config)

    @staticmethod
    def list() -> list[str]:
        return sorted(_REGISTRY.keys())
