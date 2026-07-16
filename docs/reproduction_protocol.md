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
# Vidal Case 1 implementation status

Case 1 independently trains Panasonic 18650PF, LG HG2, and Samsung 30T from
random initialisation using `full_train`, then evaluates the matching `test`
split. It does not use transfer learning or `partial_train`. Samsung 48G is not
included because the corresponding public drive-cycle dataset is unavailable.

The LSTM architecture and 571 trainable-parameter count match Vidal et al.
(2019). The main Case 1 protocol uses record-level
`paper_aligned_1hz_reconstructed` sampling because the paper reports 1 Hz while
the public drive and charge files have inconsistent native rates. Fixed
non-overlapping windows (`seq_len=50`, therefore 50 seconds after resampling),
train-only min-max normalisation, and shuffle are documented reconstruction
assumptions because the paper does not fully specify them. Formal settings are
515 epochs, batch size 512, Adam at 0.01, and StepLR(100, 0.5), without early
stopping.

SOC v2 is a benchmark reference label derived from cycler current/Ah and
protocol constraints. It is not independently measured SOC ground truth.
