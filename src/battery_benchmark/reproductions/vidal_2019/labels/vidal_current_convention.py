"""Current sign convention inference and audit for Vidal SOC v2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CurrentConvention:
    """Dataset-level current sign convention."""

    positive_is_charge: bool
    convention_name: str
    dataset: str


@dataclass
class CurrentSignAuditRow:
    dataset: str
    filename: str
    file_type: str
    profile: str
    current_mean: float
    current_median: float
    current_positive_fraction: float
    current_negative_fraction: float
    expected_soc_direction: str
    inferred_current_convention: str
    status: str
    reason: str


def expected_soc_direction(file_type: str) -> str:
    if file_type == "charge":
        return "rising"
    if file_type == "drive_cycle":
        return "falling"
    return "unknown"


def infer_convention_from_record(
    file_type: str,
    current_a: np.ndarray,
) -> tuple[bool, str]:
    """Infer whether positive current means charge from median sign."""
    i_arr = np.asarray(current_a, dtype=float)
    finite = i_arr[np.isfinite(i_arr)]
    if len(finite) == 0:
        return True, "positive_is_charge_default"

    median = float(np.median(finite))
    pos_frac = float(np.mean(finite > 0))
    neg_frac = float(np.mean(finite < 0))

    if file_type == "charge":
        if median >= 0 or pos_frac >= neg_frac:
            return True, "positive_is_charge"
        return False, "positive_is_discharge"
    if file_type == "drive_cycle":
        if median <= 0 or neg_frac >= pos_frac:
            return True, "positive_is_charge"
        return False, "positive_is_discharge"
    return True, "positive_is_charge_default"


def audit_record_convention(
    dataset: str,
    filename: str,
    file_type: str,
    profile: str,
    current_a: np.ndarray,
    *,
    dataset_convention: CurrentConvention | None = None,
) -> CurrentSignAuditRow:
    """Audit one record against expected SOC direction."""
    i_arr = np.asarray(current_a, dtype=float)
    finite = i_arr[np.isfinite(i_arr)]
    mean = float(np.mean(finite)) if len(finite) else float("nan")
    median = float(np.median(finite)) if len(finite) else float("nan")
    pos_frac = float(np.mean(finite > 0)) if len(finite) else 0.0
    neg_frac = float(np.mean(finite < 0)) if len(finite) else 0.0

    pos_is_charge, conv_name = infer_convention_from_record(file_type, i_arr)
    exp_dir = expected_soc_direction(file_type)

    status = "PASS"
    reason = conv_name
    if dataset_convention is not None and pos_is_charge != dataset_convention.positive_is_charge:
        status = "FAIL"
        reason = (
            f"record_convention={conv_name} conflicts with "
            f"dataset_convention={dataset_convention.convention_name}"
        )

    return CurrentSignAuditRow(
        dataset=dataset,
        filename=filename,
        file_type=file_type,
        profile=profile,
        current_mean=mean,
        current_median=median,
        current_positive_fraction=pos_frac,
        current_negative_fraction=neg_frac,
        expected_soc_direction=exp_dir,
        inferred_current_convention=conv_name,
        status=status,
        reason=reason,
    )


def resolve_dataset_convention(
    dataset: str,
    audit_rows: list[CurrentSignAuditRow],
) -> CurrentConvention:
    """Pick majority convention across selected drive/charge records."""
    votes: dict[str, int] = {}
    pos_charge_votes = 0
    neg_charge_votes = 0
    for row in audit_rows:
        if row.dataset != dataset:
            continue
        if row.inferred_current_convention == "positive_is_charge":
            pos_charge_votes += 1
        elif row.inferred_current_convention == "positive_is_discharge":
            neg_charge_votes += 1
        votes[row.inferred_current_convention] = votes.get(row.inferred_current_convention, 0) + 1

    positive_is_charge = pos_charge_votes >= neg_charge_votes
    name = "positive_is_charge" if positive_is_charge else "positive_is_discharge"
    return CurrentConvention(
        positive_is_charge=positive_is_charge,
        convention_name=name,
        dataset=dataset,
    )
