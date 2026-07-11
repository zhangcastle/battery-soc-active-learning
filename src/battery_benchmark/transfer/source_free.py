"""Source-free domain adaptation — PLACEHOLDER."""

from battery_benchmark.transfer.base import BaseTransferMethod
from battery_benchmark.transfer.registry import register_transfer


@register_transfer("source_free")
class SourceFreeAdaptation(BaseTransferMethod):
    """Source-free adaptation (no source data at adaptation time). NOT YET IMPLEMENTED."""

    requires_source_data = False
    requires_source_labels = False
    requires_target_data = True
    requires_target_labels = False
    uses_target_test_labels = False

    def fit_source(self, model, train_loader, val_loader, trainer):
        """Source training must be done offline; nothing to do here."""

    def adapt_target(self, model, target_loader, trainer):
        raise NotImplementedError("Source-free adaptation is not yet implemented.")

    def predict(self, model, loader, device):
        raise NotImplementedError("Source-free adaptation is not yet implemented.")
