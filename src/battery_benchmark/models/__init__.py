"""Model definitions and registry."""

from battery_benchmark.models.base import BaseSOCModel
from battery_benchmark.models.registry import ModelRegistry, register_model, get_model_class
from battery_benchmark.models.vidal_lstm import VidalLSTM

__all__ = ["BaseSOCModel", "ModelRegistry", "register_model", "get_model_class", "VidalLSTM"]
