# Reproduction Protocol

## Status of Vidal et al. reproduction

**Do not claim faithful reproduction** until all items in `docs/known_uncertainties.md`
are resolved.

## Steps to complete the reproduction

1. Obtain datasets: Panasonic NCR18650PF, LG HG2, Samsung 48G, Samsung 30T.
2. Implement dataset adapters (see `docs/dataset_protocol.md`).
3. Verify sequence construction against the paper.
4. Verify normalisation details against the paper.
5. Run `configs/experiments/vidal_case1.yaml` through `vidal_case4.yaml`.
6. Compare metrics against Table X in Vidal et al.
7. Document any discrepancies in `docs/known_uncertainties.md`.

## Reference

Vidal et al. (2022). "Battery SOC estimation using deep learning transfer learning."
*(Full citation to be added once paper details are confirmed.)*
