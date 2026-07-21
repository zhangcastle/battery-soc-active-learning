# CMU causal capacity lookup v1

This development artifact assigns a capacity reference to each existing
`diagnostic_screening_match` cycle without recomputing SOC or publishing formal
mission labels.

Within each cell, ordering uses raw-file `source_row` provenance. A cycle may
use only the latest accepted, lookup-eligible RPT satisfying
`rpt.end_source_row < mission.start_source_row`. Before the first such RPT the
capacity is the nominal 3.0 Ah. Afterward, capacity is held as a step until a
new RPT has fully ended. `cycleNumber`, interpolation, future observations,
whole-life backfilling, and target SOC are not used.

`time_s` is retained for provenance but is not the cross-block ordering key
because tester time can reset. The current mission identity is the stable
development candidate ID; formal mission segmentation remains deferred.

The independent validator uses `(cell_id, mission_cycle_id)` as the identity
key. For every output row it rebuilds the eligible same-cell RPT set and
selects the maximum `end_source_row` strictly below the mission start. It then
checks the selected ID, all copied RPT provenance, the positive source-row lag,
and nominal fallback validity. Capacity comparisons use zero relative tolerance
and an absolute tolerance of `1e-12 Ah`.
