# Hierarchical Raster Language V50 Spatial-Plan Calibration

Date: 2026-11-04

Decision: reject the unconstrained patch writer; retain the causal visual
reader and its direct spatial plan

## Result

The spatial-plan revision does not qualify as an autonomous raster writer.
It trains stably and learns a weak dependence on ordered context, but its free
patch autoregressor destroys substantially more predictive information than it
adds.

| Development measurement | Ordered | Shuffled |
| --- | ---: | ---: |
| Coarse target lower NLL | 0.57812 | 0.42188 |
| Coarse NLL gain | +0.00121 bit/pixel | -- |
| First-sample identity top-1 | 0.00000 | 0.00391 |
| Greedy identity top-1 | 0.00781 | 0.00781 |
| First-sample pixel F1 | 0.36615 | 0.36401 |
| Four-step mean identity | 0.06250 | 0.04492 |

The model passes the image-only, finite-output, nonblank, positive-likelihood,
and feedback-order checks. It fails the preregistered coarse-accuracy,
first-image identity, ordered identity-margin, and pixel-F1 checks.

## Localized Failure

On the same 256 development examples, directly decoding the reader's
continuous visual field gives:

| Direct spatial-plan measurement | Result |
| --- | ---: |
| visible identity top-1 | 0.18359 |
| pixel F1 | 0.47633 |
| blank rate | 0.00000 |

The learned writer therefore changes a useful image proposal into a generic
box-and-stroke mode. The evidence sheet confirms this visually: ordered and
shuffled outputs share the same coarse enclosure while target-specific
interior structure is lost. High teacher-forced F1 (`0.81458` in the final
training batch) is exposure-biased and does not predict free-generation
quality.

## Mechanism Decision

Do not scale or train the unconstrained writer longer. The next bounded test
must satisfy an architectural invariant: at initialization, its visible output
is exactly the direct spatial plan. Learned output is a bounded residual in
pixel-logit space, gated by plan uncertainty. Confident plan pixels cannot be
flipped; uncertain stroke boundaries may be refined. The writer is thus a
plan-preserving renderer rather than a second model that redraws the glyph.

The completed follow-up is recorded in
[`plan_preserving_raster_writer_v50p_result.md`](plan_preserving_raster_writer_v50p_result.md).
It fails identity retention and removes the separate writer from the active
path. The fixed inverse visual field is the current deployed output mechanism.

## Claim Boundary

This calibration shows that the causal image reader contains a usable visual
language proposal and that a naïve local autoregressor can erase it. It does
not establish general semantics, instruction following, etymology answering,
historical-form generation, or parity with a large token language model.
