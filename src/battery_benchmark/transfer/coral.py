"""CORAL domain adaptation — PLACEHOLDER."""

from battery_benchmark.transfer.base import BaseTransferMethod
from battery_benchmark.transfer.registry import register_transfer


@register_transfer("coral")
class CORALAdaptation(BaseTransferMethod):
    """CORAL (CORrelation ALignment) domain adaptation. NOT YET IMPLEMENTED."""

    requires_source_data = True
    requires_source_labels = True
    requires_target_data = True
    requires_target_labels = False
    uses_target_test_labels = False

    def fit_source(self, model, train_loader, val_loader, trainer):
        raise NotImplementedError("CORAL is not yet implemented.")

    def adapt_target(self, model, target_loader, trainer):
        raise NotImplementedError("CORAL is not yet implemented.")

    def predict(self, model, loader, device):
        raise NotImplementedError("CORAL is not yet implemented.")
