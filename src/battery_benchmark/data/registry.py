"""Dataset registry: map dataset names to adapter classes."""

from __future__ import annotations

from typing import Any, Type

_REGISTRY: dict[str, Type] = {}


def register_dataset(name: str):
    """Class decorator to register a dataset adapter."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset_class(name: str) -> Type:
    """Retrieve a registered dataset adapter class by name.

    Raises
    ------
    KeyError
        If no adapter is registered under *name*.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"Unknown dataset '{name}'. "
            f"Available datasets: {available}. "
            f"Did you forget to import the adapter module?"
        )
    return _REGISTRY[name]


class DatasetRegistry:
    """Singleton-style view of the dataset registry."""

    @staticmethod
    def register(name: str, cls: Type) -> None:
        _REGISTRY[name] = cls

    @staticmethod
    def get(name: str) -> Type:
        return get_dataset_class(name)

    @staticmethod
    def list() -> list[str]:
        return sorted(_REGISTRY.keys())

    @staticmethod
    def build(name: str, config: dict[str, Any]) -> Any:
        """Instantiate a dataset adapter by name."""
        cls = get_dataset_class(name)
        return cls(config)
