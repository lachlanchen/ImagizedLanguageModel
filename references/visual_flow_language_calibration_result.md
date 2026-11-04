# Visual Flow Language Calibration: Rejected

Date: 2026-11-04

Decision: reject the whole-cell conditional-flow mechanism

The 1,000-update end-to-end calibration completed in 167.77 seconds on one RTX
4090. It retained V42's reader quality (`0.20898` ordered top-1 versus `0.17969`
with shuffled history), but the visible flow writer did not bind that state to
the generated glyph.

| Generated-cell measurement | Ordered | Shuffled | Unconditioned |
| --- | ---: | ---: | ---: |
| First-sample identity top-1 | 0 | 0 | 0 |
| Selected identity top-1 | 0.00391 | 0.00781 | 0 |
| Selected pixel F1 | 0.42040 | 0.40584 | 0.43041 |
| Target coverage in four samples | 0.00391 | 0.00781 | 0 |
| Mean distinct sampled identities | 3.8867 | 3.8203 | 3.9883 |

The selected ordered identity result is `-0.03906` below V43's `0.04297`, and
ordered pixel F1 is `-0.02467` below V43's `0.44507`. Ordered history is also
worse than shuffled history in selected identity. The writer learned a diverse
glyph-like image prior but not a conditional language distribution.

No longer run or scale this mechanism. V50 replaces whole-cell flow with a
hierarchical patch-autoregressive raster likelihood and an explicit
correct-context-versus-deranged-context constraint.
