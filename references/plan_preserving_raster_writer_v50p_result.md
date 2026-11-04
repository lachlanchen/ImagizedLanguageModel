# Plan-Preserving Raster Writer V50-P: Measured Result

Date: 2026-11-04

Decision: reject the separate conditional writer

## Answer

Bounding the writer around the visual plan is not sufficient. V50-P begins at
the exact direct plan, trains stably, and keeps confident individual pixels
from flipping under the declared residual bound. Joint optimization still
changes enough uncertain pixels to destroy glyph identity, while raster loss
also pulls the causal reader away from its stronger language coordinate.

The separate conditional writer is therefore removed from the active model
path. More writer updates, layers, mixtures, or relaxed residual bounds are not
authorized by this result.

## Calibration Receipt

| Measurement | Result |
| --- | ---: |
| updates | 1,000 |
| training time | 126.57 s |
| total parameters | 20,289,033 |
| reader parameters | 16,277,249 |
| peak allocated VRAM | 0.568 GiB |
| final teacher-forced pixel F1 | 0.81841 |
| final bounded-residual energy | 0.24559 |

The run used one RTX 4090, the registered public-domain Chinese stream, the
V42 reader initialization, a 150-update frozen-reader phase, and the exact
V50-P parameterization and gates frozen before training.

## Development Result

| Measurement | Ordered | Shuffled |
| --- | ---: | ---: |
| direct-plan identity top-1 | 0.12109 | 0.08203 |
| greedy-writer identity top-1 | 0.01563 | 0.01563 |
| first-sample identity top-1 | 0.01172 | 0.01172 |
| direct-plan pixel F1 | 0.46776 | 0.45753 |
| greedy-writer pixel F1 | 0.39660 | 0.38903 |
| first-sample pixel F1 | 0.37724 | 0.36757 |
| four-step mean identity | 0.05078 | 0.03320 |

The writer retains only `0.12903` of ordered direct-plan identity and lowers
pixel F1 by `0.07116`. Ordered coarse target NLL is also slightly worse than
the shuffled control (`-0.00044` bit/pixel gain). Only the image-only boundary,
nonblank output, and feedback-order gates pass.

## Reader Drift

Before V50-P training, the frozen V42-derived direct plan reached `0.18359`
identity and `0.47633` F1 on the same 256-example audit construction. After
joint training, the direct plan reaches `0.12109` identity. The raster
likelihood therefore damages the reader before the local writer is even
applied. Final-batch in-batch language top-1 also falls to `0.28125`, despite
low raster NLL.

This separates two failures:

1. end-to-end raster likelihood distorts a measured predictive visual state;
2. the local writer then removes most of the identity that remains.

High teacher-forced F1 is again an exposure-biased statistic and is not
evidence of autonomous writing.

## Architecture Decision

The active minimal model is the causal image reader followed by its fixed,
invertible visual-field decoder. That path already provides a real image-in,
image-out language nucleus at 16.28M parameters and preserves substantially
more identity than every learned writer tested here.

If a cleaner surface is required, the next mechanism must be independent of
the language optimizer. The mathematically eligible route is a learned
unconditional writing-manifold score combined at inference with the reader's
visual-language energy:

```text
posterior score = writing-prior score + visual-language guidance score.
```

This product-of-experts construction prevents an image prior from ignoring
language and prevents raster reconstruction gradients from rewriting the
language state. It must first beat the direct field under strict plan-identity
retention; otherwise direct field rendering remains the deployed output.

## Claim Boundary

V50-P is a useful negative result. It establishes neither general semantics
nor a complete ILM. It does establish that the repository's direct visual
language path is materially stronger than the tested learned writers and must
be treated as the baseline output mechanism, not merely as conditioning for a
second unconstrained model.
