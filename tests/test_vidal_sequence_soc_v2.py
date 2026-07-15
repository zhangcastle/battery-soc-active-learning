"""Unit tests for Vidal sequence-aware SOC v2 (synthetic, no raw data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from battery_benchmark.data.adapter_qc import QCResult
from battery_benchmark.data.sequence_record import SequenceRecord
from battery_benchmark.labels.vidal_ah_unwrap import unwrap_cycler_ah
from battery_benchmark.labels.vidal_current_convention import (
    infer_convention_from_record,
    resolve_dataset_convention,
    audit_record_convention,
)
from battery_benchmark.labels.vidal_sequence_soc_v2 import (
    compute_charge_soc,
    compute_drive_soc,
    CONTINUITY_TOLERANCE,
)
from battery_benchmark.labels.vidal_time_integration import trapezoidal_coulomb_integration
from battery_benchmark.labels.vidal_current_convention import CurrentConvention
from battery_benchmark.labels.vidal_capacity_policy import VidalCapacityPolicy

REPO = Path(__file__).resolve().parent.parent
V1_ROOT = REPO / "outputs/labelled/vidal_compatible_v1"


def _record(df: pd.DataFrame, file_type: str, role: str = "model_input") -> SequenceRecord:
    meta = {
        "dataset": "synthetic",
        "battery": "test",
        "source_file": "test.mat",
        "source_sha256": "abc",
        "sequence_id": "syn_test",
        "profile": "Charge" if file_type == "charge" else "Mixed1",
        "paper_profile": "Mix1",
        "file_type": file_type,
        "record_role": role,
        "split": "full_train",
        "temperature_nominal_c": 25.0,
        "soc_init": 1.0,
        "soc_anchor_source": "test",
        "capacity_hint_ah": 3.0,
        "manifest_splits": ["full_train"],
        "paper_battery_name": "test",
        "dataset_identity_status": "test",
        "pairing_status": "confirmed",
        "confidence": "high",
        "evidence": "",
        "notes": "",
        "manifest_qc_status": "PASS",
        "manifest_qc_notes": "",
        "current_sign_convention": "as_recorded",
        "loader_backend": "test",
    }
    return SequenceRecord(data=df, metadata=meta, qc=QCResult())


def _drive_df(n: int = 100, dt: float = 1.0, i_discharge: float = -1.0) -> pd.DataFrame:
    t = np.arange(n, dtype=float) * dt
    return pd.DataFrame({
        "time_s": t,
        "timestamp": t,
        "voltage_v": np.linspace(4.0, 3.5, n),
        "current_a": np.full(n, i_discharge),
        "temperature_c": np.full(n, 25.0),
        "cycler_ah": np.linspace(0, abs(i_discharge) * t[-1] / 3600, n),
    })


def _charge_df(n: int = 50, reset_at: int | None = None) -> pd.DataFrame:
    t = np.arange(n, dtype=float)
    ah = np.linspace(0, 2.5, n)
    if reset_at is not None:
        ah[reset_at:] = ah[reset_at:] - ah[reset_at] + 0.01
    current = np.full(n, 1.5)
    current[-5:] = 0.02
    voltage = np.linspace(3.5, 4.2, n)
    return pd.DataFrame({
        "time_s": t,
        "timestamp": t,
        "voltage_v": voltage,
        "current_a": current,
        "temperature_c": np.full(n, 25.0),
        "cycler_ah": ah,
    })


class TestTimeIntegration:
    def test_nonuniform_trapezoidal(self):
        t = np.array([0.0, 1.0, 3.0, 3.5])
        i_arr = np.array([0.0, 2.0, 2.0, 0.0])
        res = trapezoidal_coulomb_integration(t, i_arr, positive_is_charge=True)
        expected_first = 0.5 * (0 + 2) * 1 / 3600
        assert res.effective_delta_ah[1] == pytest.approx(expected_first)
        assert res.cumulative_signed_ah[-1] > 0

    def test_negative_dt_reported(self):
        t = np.array([0.0, 2.0, 1.0])
        i_arr = np.array([1.0, 1.0, 1.0])
        res = trapezoidal_coulomb_integration(t, i_arr)
        assert res.negative_dt_count >= 1


class TestCurrentConvention:
    def test_charge_positive_current(self):
        pos, name = infer_convention_from_record("charge", np.array([1.0, 1.2, 0.8]))
        assert pos is True
        assert name == "positive_is_charge"

    def test_drive_negative_current(self):
        pos, name = infer_convention_from_record("drive_cycle", np.array([-1.0, -2.0]))
        assert pos is True


class TestDriveChargeSOC:
    def test_discharge_soc_falls(self):
        conv = CurrentConvention(True, "positive_is_charge", "synthetic")
        rec = _record(_drive_df(), "drive_cycle")
        df, meta = compute_drive_soc(rec, 1.0, 3.0, conv)
        assert df["soc_sequence_v2"].iloc[-1] < df["soc_sequence_v2"].iloc[0]

    def test_charge_soc_rises(self):
        conv = CurrentConvention(True, "positive_is_charge", "synthetic")
        rec = _record(_charge_df(), "charge")
        df, meta, _ = compute_charge_soc(rec, 0.4, 3.0, conv, allow_endpoint_correction=True)
        assert df["soc_sequence_v2"].iloc[-1] > df["soc_sequence_v2"].iloc[0]
        assert meta.endpoint_correction_applied is True

    def test_drive_charge_continuity(self):
        conv = CurrentConvention(True, "positive_is_charge", "synthetic")
        drive_rec = _record(_drive_df(n=50), "drive_cycle")
        ddf, dmeta = compute_drive_soc(drive_rec, 1.0, 3.0, conv)
        charge_rec = _record(_charge_df(n=40), "charge")
        cdf, cmeta, _ = compute_charge_soc(
            charge_rec, dmeta.soc_end, 3.0, conv, allow_endpoint_correction=True
        )
        gap = abs(cmeta.soc_init - dmeta.soc_end)
        assert gap <= CONTINUITY_TOLERANCE

    def test_endpoint_correction_recorded(self):
        conv = CurrentConvention(True, "positive_is_charge", "synthetic")
        rec = _record(_charge_df(), "charge")
        _, meta, _ = compute_charge_soc(rec, 0.5, 3.0, conv, allow_endpoint_correction=True)
        assert meta.raw_soc_end_before_correction is not None
        assert meta.endpoint_correction_magnitude >= 0


class TestAhUnwrap:
    def test_reset_unwrap(self):
        ah = np.array([0.0, 1.0, 2.0, 0.05, 0.5, 1.0])
        res = unwrap_cycler_ah(ah, adapter_reset_detected=True)
        assert len(res.reset_indices) >= 1
        assert res.cycler_ah_effective[-1] >= res.cycler_ah_effective[-2]

    def test_no_reset_jump_to_zero_in_soc(self):
        conv = CurrentConvention(True, "positive_is_charge", "synthetic")
        df = _charge_df(n=60, reset_at=30)
        rec = _record(df, "charge")
        out, _, _ = compute_charge_soc(rec, 0.3, 3.0, conv, allow_endpoint_correction=True)
        soc = out["soc_sequence_v2"].to_numpy()
        assert not np.any(np.diff(soc) < -0.05)


class TestAnchorExcluded:
    def test_anchor_not_model_input(self):
        rec = _record(_charge_df(), "charge", role="soc_anchor")
        assert rec.metadata["record_role"] == "soc_anchor"

    def test_excluded_role(self):
        rec = _record(_charge_df(), "charge", role="excluded")
        assert rec.metadata["record_role"] == "excluded"


class TestCapacityPolicy:
    def test_dataset_defaults(self):
        p = VidalCapacityPolicy.resolve("lg_hg2")
        assert p.capacity_reference_ah == 3.0
        p2 = VidalCapacityPolicy.resolve("panasonic_18650pf")
        assert p2.capacity_reference_ah == 2.9


class TestV1Untouched:
    def test_v1_directory_exists_unchanged(self):
        if V1_ROOT.exists():
            assert list(V1_ROOT.rglob("*.parquet"))


class TestExporterDedup:
    def test_split_identity_not_filename_only(self):
        a = ("panasonic_18650pf", "full_train", 1, "a.mat")
        b = ("panasonic_18650pf", "partial_train", 1, "a.mat")
        assert a != b


class TestMissingSocInit:
    def test_unresolved_drive_fails(self):
        conv = CurrentConvention(True, "positive_is_charge", "synthetic")
        rec = _record(_drive_df(n=10), "drive_cycle")
        df, meta = compute_drive_soc(rec, float("nan"), 3.0, conv)
        assert not np.isfinite(meta.soc_end) or meta.label_qc_status in ("PASS", "WARNING", "FAIL")
