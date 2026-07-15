"""Fixed nominal capacity policy for Vidal sequence-aware SOC v2."""

from __future__ import annotations

from dataclasses import dataclass

from battery_benchmark.labels.capacity_lookup import NOMINAL_CAPACITY_AH

CAPACITY_POLICY_VERSION = "vidal_nominal_v1"


@dataclass(frozen=True)
class VidalCapacityPolicy:
    """Resolve reference capacity for sequence SOC labelling."""

    capacity_reference_ah: float
    capacity_source: str
    capacity_policy_version: str = CAPACITY_POLICY_VERSION

    @classmethod
    def resolve(
        cls,
        dataset: str,
        manifest_capacity_hint: float | None = None,
        adapter_capacity_hint: float | None = None,
    ) -> VidalCapacityPolicy:
        if manifest_capacity_hint is not None and manifest_capacity_hint > 0:
            return cls(
                capacity_reference_ah=float(manifest_capacity_hint),
                capacity_source="manifest_capacity_hint_ah",
            )
        if adapter_capacity_hint is not None and adapter_capacity_hint > 0:
            return cls(
                capacity_reference_ah=float(adapter_capacity_hint),
                capacity_source="adapter_capacity_hint_ah",
            )
        if dataset in NOMINAL_CAPACITY_AH:
            return cls(
                capacity_reference_ah=NOMINAL_CAPACITY_AH[dataset],
                capacity_source="dataset_config",
            )
        raise KeyError(f"No capacity policy for dataset {dataset!r}")
