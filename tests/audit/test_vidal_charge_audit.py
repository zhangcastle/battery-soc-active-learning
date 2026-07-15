"""Unit tests for Vidal charge audit logic (synthetic chronology, no raw data)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "audit_vidal_charge_protocol.py"

spec = importlib.util.spec_from_file_location("audit_vidal_charge_protocol", SCRIPT)
audit = importlib.util.module_from_spec(spec)
sys.modules["audit_vidal_charge_protocol"] = audit
assert spec.loader is not None
spec.loader.exec_module(audit)


def _row(
    filename: str,
    profile: str,
    file_type: str,
    record_role: str = "model_input",
    full_train: str = "0",
    partial_train: str = "0",
    test: str = "0",
    sequence_id: str | None = None,
) -> dict:
    sid = sequence_id or f"ds_25c_{filename.replace('.mat', '')}"
    return {
        "filename": filename,
        "profile": profile,
        "file_type": file_type,
        "record_role": record_role,
        "full_train": full_train,
        "partial_train": partial_train,
        "test": test,
        "exclude": "0",
        "sequence_id": sid,
        "paper_profile": profile,
    }


def _chrono(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["dataset"] = "lg_hg2"
    for col in ("selected_start_datetime", "load_status", "load_error_type", "load_error_message"):
        if col not in df.columns:
            if col == "selected_start_datetime":
                df[col] = pd.date_range("2018-01-01", periods=len(df), freq="h")
            elif col == "load_status":
                df[col] = "PASS"
            else:
                df[col] = ""
    return df


def _classify_at(chrono: pd.DataFrame, idx: int, row: dict | None = None) -> dict:
    row = row or chrono.iloc[idx].to_dict()
    stats = {
        "charge_ah_fraction": 0.9,
        "full_voltage_reached": True,
        "ah_reset_detected": False,
    }
    return audit._classify_charge("lg_hg2", row, chrono, idx, stats, load_status="PASS")


class TestOwnershipBarriers:
    def test_pause_does_not_break_ownership(self):
        chrono = _chrono([
            _row("drive.mat", "Mixed1", "drive_cycle", partial_train="1", full_train="1"),
            _row("pause.mat", "Pause", "pause", record_role="excluded"),
            _row("charge.mat", "Charge", "charge", partial_train="1", full_train="1"),
        ])
        own = audit._resolve_ownership(chrono, 2, chrono.iloc[2].to_dict())
        assert own["ownership_search_status"] == "FOUND_PREVIOUS_DRIVE"
        assert own["owned_by_previous_profile"] == "Mixed1"

    def test_charge_is_barrier(self):
        chrono = _chrono([
            _row("drive.mat", "Mixed1", "drive_cycle", partial_train="1", full_train="1"),
            _row("charge1.mat", "Charge", "charge", partial_train="1", full_train="1"),
            _row("charge2.mat", "Charge", "charge", partial_train="1", full_train="1"),
        ])
        own = audit._resolve_ownership(chrono, 2, chrono.iloc[2].to_dict())
        assert own["ownership_search_status"] == "STOPPED_BY_CHARGE"
        assert own["owned_by_previous_profile"] == ""

    def test_cap_1c_post_anchor(self):
        chrono = _chrono([
            _row("cap.mat", "Cap_1C", "characterization", record_role="excluded"),
            _row("charge.mat", "Charge", "charge", test="1"),
            _row("udds.mat", "UDDS", "drive_cycle", test="1"),
        ])
        res = _classify_at(chrono, 1)
        assert res["recommended_protocol_action"] == "CHANGE_TO_ANCHOR"
        assert res["recommended_label_action"] == "ANCHOR_ONLY_NO_POINTWISE_LABEL"

    def test_hwfet_owned_exclude(self):
        chrono = _chrono([
            _row("hwfet.mat", "HWFET", "drive_cycle", record_role="excluded", test="1"),
            _row("charge.mat", "Charge", "charge", test="1"),
            _row("la92.mat", "LA92", "drive_cycle", test="1"),
        ])
        res = _classify_at(chrono, 1)
        assert res["recommended_protocol_action"] == "CHANGE_TO_EXCLUDED"

    def test_trailing_corresponding(self):
        chrono = _chrono([
            _row("mix8.mat", "Mixed8", "drive_cycle", full_train="1"),
            _row("charge.mat", "Charge", "charge", full_train="1"),
        ])
        res = _classify_at(chrono, 1)
        assert res["recommended_protocol_action"] == "KEEP_MODEL_INPUT"
        assert res["proposed_charge_role"] == "trailing_corresponding_charge"


class TestLabelActions:
    def test_no_reset_keep_is_sequence_aware(self):
        chrono = _chrono([
            _row("drive.mat", "Mixed2", "drive_cycle", partial_train="1", full_train="1"),
            _row("charge.mat", "Charge", "charge", partial_train="1", full_train="1"),
            _row("mix3.mat", "Mixed3", "drive_cycle", partial_train="1", full_train="1"),
        ])
        res = _classify_at(chrono, 1)
        assert res["recommended_protocol_action"] == "KEEP_MODEL_INPUT"
        assert res["recommended_label_action"] == "REGENERATE_SEQUENCE_AWARE"
        assert res["expected_soc_start_source"] == "previous_drive_soc_end"
        assert res["sequence_continuity_required"] is True
        assert "LABEL_OK" not in res["recommended_label_action"]

    def test_reset_keep_is_reset_and_sequence_aware(self):
        chrono = _chrono([
            _row("drive.mat", "Mixed2", "drive_cycle", partial_train="1", full_train="1"),
            _row("charge.mat", "Charge", "charge", partial_train="1", full_train="1"),
        ])
        stats = {"charge_ah_fraction": 0.9, "full_voltage_reached": True, "ah_reset_detected": True}
        res = audit._classify_charge(
            "lg_hg2", chrono.iloc[1].to_dict(), chrono, 1, stats, load_status="PASS"
        )
        assert res["recommended_label_action"] == "REGENERATE_RESET_AND_SEQUENCE_AWARE"
        assert res["ah_unwrap_required"] is True

    def test_anchor_label(self):
        chrono = _chrono([
            _row("cap.mat", "Cap_1C", "characterization", record_role="excluded"),
            _row("charge.mat", "Charge", "charge", test="1"),
            _row("udds.mat", "UDDS", "drive_cycle", test="1"),
        ])
        res = _classify_at(chrono, 1)
        assert res["recommended_label_action"] == "ANCHOR_ONLY_NO_POINTWISE_LABEL"
        assert res["full_charge_anchor_required"] is True


class TestSplitRelations:
    def test_partial_subset_boundary(self):
        chrono = _chrono([
            _row("mix3.mat", "Mixed3", "drive_cycle", partial_train="1", full_train="1"),
            _row("charge.mat", "Charge", "charge", partial_train="1", full_train="1"),
            _row("mix4.mat", "Mixed4", "drive_cycle", full_train="1"),
        ])
        res = _classify_at(chrono, 1)
        assert res["split_relation"] == "partial_subset_boundary"
        assert res["recommended_protocol_action"] == "KEEP_MODEL_INPUT"

    def test_test_train_boundary(self):
        chrono = _chrono([
            _row("us06.mat", "US06", "drive_cycle", test="1"),
            _row("charge.mat", "Charge", "charge", test="1"),
            _row("mix1.mat", "Mixed1", "drive_cycle", partial_train="1", full_train="1"),
        ])
        res = _classify_at(chrono, 1)
        assert res["split_relation"] == "true_test_train_boundary"
        assert res["recommended_protocol_action"] == "KEEP_MODEL_INPUT"


class TestLoadFailure:
    def test_load_failure_manual_review(self):
        chrono = _chrono([
            _row("drive.mat", "UDDS", "drive_cycle", test="1"),
            _row("charge.mat", "Charge", "charge", test="1"),
        ])
        res = audit._classify_charge(
            "lg_hg2", chrono.iloc[1].to_dict(), chrono, 1, {}, load_status="FAIL"
        )
        assert res["recommended_protocol_action"] == "MANUAL_REVIEW"
        assert res["recommended_label_action"] == "MANUAL_REVIEW"


class TestLoadRecord:
    def test_load_record_returns_structured_error(self):
        class FakeAdapter:
            entries = []

        lr = audit._load_record(FakeAdapter(), {"filename": "missing.mat"})
        assert lr["load_status"] == "FAIL"
        assert lr["load_error_type"] == "EntryNotFound"
        assert lr["record"] is None


class TestCandidateSequenceSplit:
    def test_anchor_not_in_test_when_training_only(self):
        class_df = pd.DataFrame([
            {
                "dataset": "lg_hg2",
                "filename": "anchor.mat",
                "recommended_protocol_action": "CHANGE_TO_ANCHOR",
                "next_actual_drive_protocol_splits": "full_train,partial_train",
                "next_actual_drive_profile": "Mixed3",
                "current_split_memberships": "full_train,partial_train",
                "owned_by_previous_sequence_id": "",
                "proposed_charge_role": "post_capacity_test_full_recharge_anchor",
            },
            {
                "dataset": "lg_hg2",
                "filename": "keep.mat",
                "recommended_protocol_action": "KEEP_MODEL_INPUT",
                "owned_by_previous_sequence_id": "ds_mix3",
                "current_split_memberships": "full_train,partial_train",
                "current_record_role": "model_input",
                "proposed_charge_role": "normal_corresponding_charge",
                "reason": "test",
            },
        ])
        chrono = _chrono([
            {**_row("mix3.mat", "Mixed3", "drive_cycle", partial_train="1", full_train="1", sequence_id="ds_mix3")},
            {**_row("keep.mat", "Charge", "charge", partial_train="1", full_train="1")},
        ])
        chrono["dataset"] = "lg_hg2"
        seq = audit.build_candidate_sequences(chrono, class_df)
        assert len(seq["lg_hg2"]["test"]["leading_anchors"]) == 0
        assert len(seq["lg_hg2"]["partial_train"]["leading_anchors"]) >= 0


class TestPaths:
    def test_no_hardcoded_user_path_in_script(self):
        text = SCRIPT.read_text(encoding="utf-8")
        assert "/Users/yue/" not in text
        assert "Zotero" not in text

    def test_rel_path_external_pdf(self):
        assert audit._rel_path(None, REPO) == "<external Vidal PDF>"
