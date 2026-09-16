"""Tests for the Ns -> mission-phase full-timeline diagnostic labeling."""

from __future__ import annotations

import pandas as pd

from battery_benchmark.datasets.cmu_evtol.ns_phase_labeling_v1 import (
    MISSION_FLIGHT_NS_SEQUENCE,
    MISSION_FLIGHT_PHASE_ROLES,
    NS_MISSION_PHASE_MAP,
    label_full_timeline_phase_role,
)


def test_ns_mission_phase_map_matches_spec_exactly():
    assert NS_MISSION_PHASE_MAP == {
        0: "cc_charge",
        1: "cv_charge",
        3: "rest_2",
        4: "take_off",
        5: "cruise",
        6: "landing",
        7: "rest_1",
    }


def test_mission_flight_constants_match_spec():
    assert MISSION_FLIGHT_NS_SEQUENCE == (4, 5, 6)
    assert MISSION_FLIGHT_PHASE_ROLES == ("take_off", "cruise", "landing")


def test_label_full_timeline_phase_role_all_ns_values_with_repeats():
    # All 9 observed Ns values (0-8) plus a few repeats, in a deliberately
    # non-sorted order to make sure this is a pure elementwise map, not
    # something that depends on row order or grouping.
    ns_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 5, 8, 4]
    data = pd.DataFrame({"Ns": ns_values})

    result = label_full_timeline_phase_role(data)

    expected = pd.Series(
        [
            "cc_charge", "cv_charge", None, "rest_2", "take_off", "cruise",
            "landing", "rest_1", None, "cc_charge", "cruise", None, "take_off",
        ],
        name="Ns",
    )
    pd.testing.assert_series_equal(result.reset_index(drop=True), expected)


def test_label_full_timeline_phase_role_is_read_only():
    data = pd.DataFrame({"Ns": [0, 4, 8], "other": [1, 2, 3]})
    before = data.copy()

    label_full_timeline_phase_role(data)

    pd.testing.assert_frame_equal(data, before)


def test_label_full_timeline_phase_role_never_raises_on_unexpected_ns():
    data = pd.DataFrame({"Ns": [2, 8, -1, 999, float("nan")]})

    result = label_full_timeline_phase_role(data)

    assert result.isna().all()


def test_label_full_timeline_phase_role_requires_ns_column():
    import pytest

    data = pd.DataFrame({"not_ns": [0, 1]})
    with pytest.raises(ValueError, match="Ns"):
        label_full_timeline_phase_role(data)
