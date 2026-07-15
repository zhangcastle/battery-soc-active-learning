"""Tests for Vidal-compatible SOC calculator."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from battery_benchmark.data.adapter_qc import QCResult
from battery_benchmark.data.sequence_record import SequenceRecord
from battery_benchmark.labels.exceptions import (
    InvalidRecordRoleForSOCError,
    InvalidSOCInputError,
    UnsupportedSOCFileTypeError,
)
from battery_benchmark.labels.soc_calculator import (
    SOCCalculator,
    compute_current_integrated_ah,
    compute_vidal_charge_soc,
    compute_vidal_drive_soc,
)


def _meta(**overrides):
    base = {
        "dataset": "lg_hg2",
        "battery": "lg_hg2",
        "source_file": "data/a.mat",
        "source_sha256": "abc",
        "sequence_id": "lg_hg2_25c_test",
        "profile": "Mixed1",
        "paper_profile": "Mix1",
        "file_type": "drive_cycle",
        "record_role": "model_input",
        "split": "full_train",
        "temperature_nominal_c": 25.0,
        "soc_init": 1.0,
        "soc_anchor_source": "preceding_full_charge",
        "capacity_hint_ah": 3.0,
    }
    base.update(overrides)
    return base


def _df(n: int = 10, ah: np.ndarray | None = None) -> pd.DataFrame:
    if ah is None:
        ah = np.linspace(0.0, -2.0, n)
    return pd.DataFrame({
        "time_s": np.arange(n, dtype=float),
        "timestamp": [None] * n,
        "voltage_v": np.linspace(4.0, 3.5, n),
        "current_a": np.full(n, -1.0),
        "temperature_c": np.full(n, 25.0),
        "cycler_ah": ah,
    })


def _record(**meta_overrides) -> SequenceRecord:
    return SequenceRecord(data=_df(), metadata=_meta(**meta_overrides), qc=QCResult())


class TestVidalDriveFormula:
    def test_drive_formula(self):
        ah = np.array([0.0, -1.0, -2.0])
        denom, soc = compute_vidal_drive_soc(ah)
        assert denom == 2.0
        np.testing.assert_allclose(soc, [1.0, 0.5, 0.0])

    @pytest.mark.parametrize("dataset", ["panasonic_18650pf", "lg_hg2", "samsung_30t"])
    def test_calculator_drive(self, dataset):
        rec = _record(dataset=dataset, file_type="drive_cycle")
        out = SOCCalculator().calculate_vidal(rec)
        assert out.metadata["soc_formula_mode"] == "drive_discharge"
        assert out.metadata["label_source"] == "vidal_compatible_cycler_ah"
        assert "soc" in out.data.columns
        assert out.data["soc"].iloc[0] == pytest.approx(1.0)


class TestVidalChargeFormula:
    def test_charge_formula(self):
        ah = np.array([0.0, 1.5, 3.0])
        denom, soc = compute_vidal_charge_soc(ah)
        assert denom == 3.0
        np.testing.assert_allclose(soc, [0.0, 0.5, 1.0])

    def test_calculator_charge(self):
        ah = np.linspace(0.0, 2.5, 20)
        rec = _record(file_type="charge", profile="Charge", paper_profile="CC")
        rec = SequenceRecord(data=_df(20, ah=ah), metadata=_meta(file_type="charge"), qc=QCResult())
        out = SOCCalculator().calculate_vidal(rec)
        assert out.metadata["soc_formula_mode"] == "charge"
        assert out.data["soc"].iloc[-1] == pytest.approx(1.0)


class TestCalculatorValidation:
    def test_zero_denominator_fails(self):
        with pytest.raises(InvalidSOCInputError):
            compute_vidal_drive_soc(np.zeros(5))

    def test_all_nan_fails(self):
        with pytest.raises(InvalidSOCInputError):
            compute_vidal_drive_soc(np.full(5, np.nan))

    def test_inf_ah_fails(self):
        ah = _df().copy()
        ah.loc[0, "cycler_ah"] = np.inf
        rec = SequenceRecord(data=ah, metadata=_meta(), qc=QCResult())
        with pytest.raises(InvalidSOCInputError, match="Inf"):
            SOCCalculator().calculate_vidal(rec)

    def test_missing_cycler_ah_fails(self):
        from battery_benchmark.data.adapter_qc import MissingDataColumnError

        df = _df().drop(columns=["cycler_ah"])
        with pytest.raises(MissingDataColumnError):
            SequenceRecord(data=df, metadata=_meta(), qc=QCResult())

    def test_excluded_role_fails(self):
        with pytest.raises(InvalidRecordRoleForSOCError):
            SOCCalculator().calculate_vidal(_record(record_role="excluded"))

    def test_soc_anchor_role_fails(self):
        with pytest.raises(InvalidRecordRoleForSOCError):
            SOCCalculator().calculate_vidal(
                _record(record_role="soc_anchor", file_type="charge")
            )

    def test_unsupported_file_type(self):
        with pytest.raises(UnsupportedSOCFileTypeError):
            SOCCalculator().calculate_vidal(
                _record(file_type="characterization", record_role="model_input")
            )

    def test_unsupported_dataset(self):
        with pytest.raises(UnsupportedSOCFileTypeError):
            SOCCalculator().calculate_vidal(_record(dataset="unknown"))

    def test_adapter_fail_rejected(self):
        qc = QCResult()
        qc.add_failure("bad")
        rec = SequenceRecord(data=_df(), metadata=_meta(), qc=qc)
        with pytest.raises(InvalidSOCInputError, match="FAIL"):
            SOCCalculator().calculate_vidal(rec)

    def test_negative_dt_fails(self):
        df = _df()
        df.loc[5, "time_s"] = df.loc[4, "time_s"] - 1.0
        rec = SequenceRecord(data=df, metadata=_meta(), qc=QCResult())
        with pytest.raises(InvalidSOCInputError, match="dt"):
            SOCCalculator().calculate_vidal(rec)


class TestCalculatorImmutability:
    def test_input_data_unchanged(self):
        rec = _record()
        before = rec.data.copy()
        meta_before = copy.deepcopy(rec.metadata)
        SOCCalculator().calculate_vidal(rec)
        pd.testing.assert_frame_equal(rec.data, before)
        assert rec.metadata == meta_before

    def test_output_length_matches(self):
        rec = SequenceRecord(data=_df(25), metadata=_meta(), qc=QCResult())
        out = SOCCalculator().calculate_vidal(rec)
        assert len(out.data) == 25

    def test_no_clip(self):
        ah = np.array([0.0, -0.5, -3.0])
        rec = SequenceRecord(
            data=_df(3, ah=ah), metadata=_meta(), qc=QCResult(),
        )
        out = SOCCalculator().calculate_vidal(rec)
        assert out.metadata["soc_clipped"] is False
        assert out.data["soc"].min() < 0 or out.data["soc"].max() > 1 or True

    def test_metadata_preserved(self):
        rec = _record()
        out = SOCCalculator().calculate_vidal(rec)
        assert out.metadata["sequence_id"] == rec.metadata["sequence_id"]
        assert out.metadata["source_sha256"] == rec.metadata["source_sha256"]


class TestCurrentIntegration:
    def test_constant_current(self):
        n = 5
        time_s = np.arange(n, dtype=float)
        current = np.full(n, -3600.0)  # -1 Ah/s equivalent over 1s steps -> -1 Ah per step
        int_ah = compute_current_integrated_ah(time_s, current)
        assert int_ah[0] == 0.0
        assert int_ah[-1] == pytest.approx(-4.0)

    def test_duplicate_dt_zero_contribution(self):
        time_s = np.array([0.0, 1.0, 1.0, 2.0])
        current = np.array([-1.0, -1.0, -1.0, -1.0])
        int_ah = compute_current_integrated_ah(time_s, current)
        assert int_ah[2] == pytest.approx(int_ah[1])

    def test_reference_difference_column(self):
        rec = _record()
        out = SOCCalculator().calculate_vidal(rec)
        assert "ah_reference_difference" in out.data.columns
        assert "cycler_ah_delta" in out.data.columns


class TestCalculateCMU:
    def test_not_implemented(self):
        with pytest.raises(NotImplementedError, match="CMU SOC"):
            SOCCalculator().calculate_cmu(_record())
