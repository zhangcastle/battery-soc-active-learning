"""Transfer learning methods."""

from battery_benchmark.transfer.base import BaseTransferMethod
from battery_benchmark.transfer.registry import TransferRegistry, register_transfer, get_transfer_class
from battery_benchmark.transfer.source_only import SourceOnly
from battery_benchmark.transfer.scratch import Scratch
from battery_benchmark.transfer.finetune import Finetune

__all__ = [
    "BaseTransferMethod",
    "TransferRegistry",
    "register_transfer",
    "get_transfer_class",
    "SourceOnly",
    "Scratch",
    "Finetune",
]
