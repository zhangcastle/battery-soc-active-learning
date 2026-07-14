"""Data loading, schema, and split utilities."""

from battery_benchmark.data.schema import (
    COLUMNS,
    REQUIRED_COLS,
    OPTIONAL_COLS,
    LabelSource,
    Split,
    AdapterSplit,
    FileType,
    REQUIRED_DATA_COLUMNS,
    REQUIRED_METADATA_FIELDS,
    ALLOWED_ADAPTER_SPLITS,
    ALLOWED_FILE_TYPES,
)
from battery_benchmark.data.adapter_qc import (
    QCResult,
    AdapterSchemaError,
    MissingDataColumnError,
    MissingMetadataFieldError,
    InvalidQCStatusError,
    InvalidAdapterSplitError,
    InvalidFileTypeError,
    ALLOWED_QC_STATUS,
)
from battery_benchmark.data.sequence_record import SequenceRecord
from battery_benchmark.data.manifest_loader import (
    ManifestEntry,
    ManifestError,
    MissingManifestColumnError,
    InvalidManifestRowError,
    UnsupportedManifestSplitError,
    SHA256MismatchError,
    load_manifest,
    select_manifest_entries,
    REQUIRED_MANIFEST_COLUMNS,
    SUPPORTED_SPLITS,
)
from battery_benchmark.data.mat_loader import (
    MatLoadResult,
    MatFileError,
    MissingMeasurementStructError,
    UnsupportedMatFormatError,
    load_mat_file,
)
from battery_benchmark.data.registry import DatasetRegistry, register_dataset, get_dataset_class
from battery_benchmark.data.synthetic import SyntheticDataset
from battery_benchmark.data.cycle_dataset import CycleDataset
from battery_benchmark.data.splits import split_cycles

__all__ = [
    # Benchmark-level schema
    "COLUMNS",
    "REQUIRED_COLS",
    "OPTIONAL_COLS",
    "LabelSource",
    "Split",
    # Adapter-level schema
    "AdapterSplit",
    "FileType",
    "REQUIRED_DATA_COLUMNS",
    "REQUIRED_METADATA_FIELDS",
    "ALLOWED_ADAPTER_SPLITS",
    "ALLOWED_FILE_TYPES",
    # QC
    "QCResult",
    "ALLOWED_QC_STATUS",
    # Exceptions
    "AdapterSchemaError",
    "MissingDataColumnError",
    "MissingMetadataFieldError",
    "InvalidQCStatusError",
    "InvalidAdapterSplitError",
    "InvalidFileTypeError",
    # Record
    "SequenceRecord",
    # Manifest loader
    "ManifestEntry",
    "ManifestError",
    "MissingManifestColumnError",
    "InvalidManifestRowError",
    "UnsupportedManifestSplitError",
    "SHA256MismatchError",
    "load_manifest",
    "select_manifest_entries",
    "REQUIRED_MANIFEST_COLUMNS",
    "SUPPORTED_SPLITS",
    # Mat loader
    "MatLoadResult",
    "MatFileError",
    "MissingMeasurementStructError",
    "UnsupportedMatFormatError",
    "load_mat_file",
    # Dataset utilities
    "DatasetRegistry",
    "register_dataset",
    "get_dataset_class",
    "SyntheticDataset",
    "CycleDataset",
    "split_cycles",
]
