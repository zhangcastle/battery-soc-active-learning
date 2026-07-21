# CMU dynamic-capacity SOC development v1

This development-only label uses the existing CMU fixed 3.0 Ah candidate SOC
integration unchanged. The sole scientific change is replacing the 3.0 Ah
denominator with the capacity assigned at candidate start by causal capacity
lookup v1. Capacity remains constant inside each candidate; no interpolation,
future RPT, impedance input, or within-candidate capacity switch is allowed.

Candidates join lookup rows by `(cell_id, mission_cycle_id)`. The current IDs
remain diagnostic development candidates, not formal mission labels. SOC starts
at 1.0 for every candidate, uses the existing positive-charge/negative-discharge
trapezoidal integration, is not clipped, and retains out-of-range values for QC.
An independent full-row recomputation uses absolute tolerance `1e-12` and zero
relative tolerance.
