# Plan-Preserving Raster Writer V50-P: Calibration Protocol

Date frozen: 2026-11-04

Status: frozen before V50-P calibration

## Question

Can a local image writer refine the successful direct visual proposal without
destroying its language identity?

V50-P changes only the writer's output parameterization. It retains the V42
causal image reader, fixed orthonormal raster field, 32-by-32 visible cells,
four 8-by-8 coarse units, sixteen 8-by-8 detail units, image-only recurrent
boundary, data partition, and evaluator bank used by the rejected V50 spatial
calibration.

## Plan-Preserving Parameterization

For plan probability `p` at one pixel, define

```text
b(p) = clip(logit(p), -4, 4)
u(p) = 0.25 + 0.75 * 4 p (1-p)
r   = 4 tanh(delta)
ell = b(p) + u(p) r.
```

`delta` is emitted by the local autoregressor and `ell` is the Bernoulli
pixel logit. At zero residual, thresholded output is exactly the reader's
direct spatial plan. Near-certain plan pixels receive at most one logit unit
of correction against a base magnitude of four and therefore cannot flip.
Only uncertain stroke boundaries admit a large correction. A `0.01` mean
squared bounded-residual penalty discourages unnecessary redrawing.

This is not a glyph lookup, codebook, OCR path, token target, Unicode target,
or candidate projection. The plan and residual are continuous functions of
input rasters.

## Calibration

Run 1,000 exploratory updates from the same production V42 reader. Freeze the
reader for 150 updates, then train reader and writer jointly. Keep every other
optimizer, batch, context, likelihood, and audit setting equal to the rejected
spatial-plan calibration.

## Retention Gates

The writer is retained only if all of the following hold on the declared
development audit:

1. image-only runtime boundary remains clean;
2. outputs are finite and selected blank rate is below 0.02;
3. ordered coarse likelihood gain over shuffled context is positive;
4. ordered four-step mean identity exceeds shuffled four-step identity;
5. greedy visible identity retains at least 85% of direct-plan identity;
6. greedy pixel F1 is no more than 0.02 below direct-plan pixel F1;
7. ordered greedy identity exceeds shuffled greedy identity; and
8. the first stochastic sample exceeds V43 identity (`0.04297`) and pixel F1
   (`0.44507`).

If retention gates 5 or 6 fail, the separate writer is removed. More updates,
more layers, and more mixture components are not authorized as a response to
that failure.

## Claim Boundary

A passing calibration would establish only that a bounded image residual can
preserve and possibly refine a compact causal visual-language proposal. It
would not establish broad semantics, instruction following, arbitrary script
generation, etymology answering, or parity with a large language model.
