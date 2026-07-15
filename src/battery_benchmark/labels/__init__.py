"""SOC label computation, pseudo-label generation, and label quality."""

from battery_benchmark.labels.coulomb_counting import coulomb_count
from battery_benchmark.labels.exceptions import (
    InvalidRecordRoleForSOCError,
    InvalidSOCInputError,
    SOCLabelError,
    UnsupportedSOCFileTypeError,
)
from battery_benchmark.labels.exporter import (
    DEFAULT_DATASET_VERSION,
    export_labelled_dataset,
    label_qc_for_record,
    validate_exported_labelled_dataset,
)
from battery_benchmark.labels.label_qc import SOCLabelQCResult, run_label_qc
from battery_benchmark.labels.soc_calculator import (
    CALCULATOR_NAME,
    LABEL_SOURCE,
    SOCCalculator,
    SOC_FORMULA_VERSION,
    compute_current_integrated_ah,
    compute_vidal_charge_soc,
    compute_vidal_drive_soc,
)
from battery_benchmark.labels.soc_types import SOCType

__all__ = [
    "SOCType",
    "coulomb_count",
    "SOCLabelError",
    "InvalidRecordRoleForSOCError",
    "UnsupportedSOCFileTypeError",
    "InvalidSOCInputError",
    "SOCCalculator",
    "SOCLabelQCResult",
    "run_label_qc",
    "label_qc_for_record",
    "LABEL_SOURCE",
    "CALCULATOR_NAME",
    "SOC_FORMULA_VERSION",
    "compute_current_integrated_ah",
    "compute_vidal_drive_soc",
    "compute_vidal_charge_soc",
    "DEFAULT_DATASET_VERSION",
    "export_labelled_dataset",
    "validate_exported_labelled_dataset",
]
