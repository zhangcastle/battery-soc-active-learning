"""Ns -> protocol mission-phase labeling for the CMU eVTOL raw timeline (v1).

This module is a brand new, fully independent, diagnostic-only capability.
It labels every row of the raw canonical CMU timeline with the mission
phase implied by the tester-native ``Ns`` column, using a stable mapping
derived empirically from tracing raw voltage/current/duration evidence
against Table 1 of Bills et al. 2023 ("A battery dataset for electric
vertical takeoff and landing aircraft") across all 7 supported cells.

``Ns`` is a per-row, protocol-level constant across the whole dataset (it
does not need to be re-derived per cell, per cycle, or per run): the same
integer always means the same mission phase everywhere it appears.

This module's only consumer is a full-timeline diagnostic export (see
``scripts/datasets/build_cmu_dynamic_soc_dataset.py``). It must NEVER be
wired into ``development_pipeline_v0.build_candidate_rows``,
``dynamic_soc.build_dynamic_soc_rows``, or any training/adaptation/
prediction code path -- those all continue to derive
``development_phase_role`` exclusively from the exact
``take_off``/``cruise``/``landing`` (Ns 4/5/6) mission-candidate detection
in ``development_pipeline_v0.enumerate_diagnostic_candidates``.
"""

from __future__ import annotations

import pandas as pd


# Stable, protocol-level Ns -> mission-phase mapping (see module docstring).
# Ns=2 never appears in the regular cycle. A stray single-row Ns=8 appears
# once per cycle boundary (I=0, V dips to ~2.6V) -- almost certainly a
# periodic deep-discharge/RPT capacity-test artifact, not part of the
# normal 7-phase mission cycle. Any Ns value not present as a key here
# (including 2 and 8) is unclassified and must map to None -- never raise,
# never invent a label.
NS_MISSION_PHASE_MAP: dict[int, str] = {
    0: "cc_charge",   # CC Charge: 1C until V>4.2
    1: "cv_charge",   # CV Charge: V=4.2 until I<C/30
    3: "rest_2",      # Rest 2: I=0 until T<35C, after charge, before flight
    4: "take_off",    # Take-off: 54W for 75s
    5: "cruise",      # Cruise: 16W for 800s baseline; duration varies by cell
    6: "landing",     # Landing: 54W for 105s
    7: "rest_1",      # Rest 1: I=0 until T<27C, after landing, before next charge
}

# Natural cycle order (ascending Ns, wrapping): cc_charge -> cv_charge ->
# rest_2 -> take_off -> cruise -> landing -> rest_1 -> cc_charge...
MISSION_FLIGHT_NS_SEQUENCE: tuple[int, ...] = (4, 5, 6)
MISSION_FLIGHT_PHASE_ROLES: tuple[str, ...] = ("take_off", "cruise", "landing")


def label_full_timeline_phase_role(data: pd.DataFrame) -> pd.Series:
    """Label every row of a raw canonical CMU timeline with its Ns phase.

    Vectorized, read-only mapping of the ``Ns`` column via
    ``NS_MISSION_PHASE_MAP``. Every one of the 7 mapped phases can appear,
    plus ``None``/NaN for any unmapped Ns value (including 2 and 8). Never
    raises on an unexpected Ns value and never mutates ``data``.

    Diagnostic/reporting only: the returned series must never be used as
    model input and must never affect training, adaptation, or prediction.
    """
    if "Ns" not in data.columns:
        raise ValueError("label_full_timeline_phase_role requires an 'Ns' column")
    return data["Ns"].map(NS_MISSION_PHASE_MAP)
