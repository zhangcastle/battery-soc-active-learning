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
