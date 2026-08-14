# Scaled Retinal Glyph Language V46: Research Decision

Date: 2026-08-15

Decision: train one causal reader from initialization in the qualified V45
retinal coordinate, represented as one scaled 1,024-dimensional field. Do not
add a radial token, character head, pair curriculum, writer, extra capacity, or
external teacher.

## Immediate Question

V42 showed that a 24.35M-parameter image-only causal model can learn useful
ordered Chinese continuation, but it failed counterfactual target binding and
readable autonomous generation. V44 found that additional pair supervision on
the frozen V42 coordinate did not repair that failure. V45 then isolated a
representation defect: raw normalized DCT fields shared a dominant common mode,
whereas a fixed partial-whitening transform reduced that mode, enlarged
identity-bearing displacement, increased displacement rank, retained nuisance
continuity, and remained exactly invertible.

V46 asks the causal question authorized by V45:

> If the V42 reader is trained from scratch in the qualified V45 geometry, with
> architecture, data, compute, and evaluator otherwise held fixed, does it learn
> stronger natural continuation, counterfactual binding, and raster output?

This is narrower than the full ILM goal. It is the next causal test needed before
historical forms, instruction data, page layout, or a larger writer are added.

## Why One Full Field, Not Direction Plus A New Channel

V45 emits

```text
z = A(d - mu),       u = z / ||z||,       q = log ||z||,
```

where `d` is the orthonormal DCT of a signed binary raster. The pair `(u,q)` is
exact, but passing it to V42 as 1,025 coordinates would change the input and
output layers and would require a special scalar head. That would confound the
representation test.

Instead define one training-only constant

```text
s = sqrt(sum_i w_i ||z_i||^2)
  = 19.622622215774165
```

from the same frozen 8,000-form V45 fit bank and weights, and use

```text
v = z / s.
```

This is mathematically equivalent to direction plus radius:

```text
u = v / ||v||,       q = log(s ||v||),
d = mu + A^-1(s v).
```

No information is dropped. Direction supports cosine compatibility; vector norm
preserves the radial information needed by the exact raster inverse. The field
remains 1,024-dimensional, so V42's learned-layer shapes and parameter count are
unchanged.

The scale is not tuned on language results. FP64 computation over the frozen
training-only fit bank gives weighted RMS norm exactly `1.0`. Observed scaled
norms range from `0.731955` to `1.259042`, with median `1.090269`; component
standard deviation is `0.033965`. This is close to the numerical energy of V42's
unit DCT field and avoids an unstable extra radial channel.

## Learning Objective

The deterministic anchor must remain a full field rather than a unit vector.
For predicted anchor `a`, target `v`, and stochastic samples `g_r`, V46 retains
V42's five objectives and weights:

```text
L = 1.00 L_nce
  + 0.25 (1 - cos(a, v))
  + 0.20 L_pixel(a, x)
  + 0.50 [2 E_r ||g_r - v|| - E_(r != s) ||g_r - g_s||]
  + 0.25 [1 - max_r cos(g_r, v)].
```

`L_nce` ranks directions because identity compatibility should not depend on a
radial nuisance. The energy score operates on the complete scaled field, and the
pixel loss operates after the exact inverse, so both train radius as well as
direction. This avoids an additional radial regression weight. The energy score
is a strictly proper score for continuous distributions, as used by
[Continuous Visual Autoregressive Generation via Score Maximization](https://arxiv.org/abs/2505.07812).
MAR independently establishes that autoregression over continuous visual values
does not require vector quantization
([Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838)).

V45's symmetric partial whitening follows the statistical motivation that a
whitening orientation can preserve similarity to the original variables
([Optimal Whitening and Decorrelation](https://arxiv.org/abs/1512.00809)).
Whitening has also been used to scatter representation samples and resist
collapse in self-supervised learning
([Whitening for Self-Supervised Representation Learning](https://arxiv.org/abs/2007.06346)).
Those works motivate the components; they do not establish the V46 language
claim.

## Initialization Without A Hidden Capacity Change

Removing output normalization exposes V42's pre-normalization magnitude. With
the original `0.02` final-layer standard deviation, the initial anchor norm is
about `2.6`, while the scaled retinal target is near one. V46 therefore changes
only two output-layer initialization scales:

```text
anchor final weight std       0.0080
generator final weight std    0.0005
all other linear weights      0.0200
all affected biases           0
```

Using the frozen V42 seed and the first eight deterministic training samples,
the update-zero anchor has mean norm `1.07063` and range `0.93371--1.20800`.
The generator residual has mean norm `0.31403`. These checks used no development
targets and no language optimization.

## Exact Positives

Training positives remain exact raster equality. A pre-freeze audit over all
8,000 fit forms found 7,999 distinct binary rasters and one duplicate-raster
pair. The largest cosine between two distinct V45 directions was `0.958143`
(`吕` versus `呂`), leaving a gap of `0.041857` from one. The production loss is
stricter still: it packs each thresholded raster into 32 exact 32-bit words and
constructs the positive mask by word-for-word equality. The cosine audit is a
geometry diagnostic, not the equality mechanism or identity supervision.

## Alternatives Rejected For V46

- **A 1,025-dimensional direction-plus-log-radius vector:** exact, but changes
  learned-layer shapes and mixes spherical and scalar units.
- **A learned radius head:** adds parameters and a second prediction mechanism,
  weakening the representation-only interpretation.
- **Unit-normalizing the V45 field:** discards information and breaks exact
  inversion because V45 radii vary by form.
- **Post-hoc conversion of V42 anchors:** already rejected by V45's fixed
  diagnostic; the old model was optimized in the old geometry.
- **V44 pair supervision:** would confound representation with curriculum. It is
  reserved for a later, separately preregistered experiment if V46 diagnoses a
  remaining binding failure.
- **A diffusion writer or larger model:** unnecessary before the compact causal
  field passes the matched language and output gates.

## Receipts Used To Freeze The Decision

- corpus SHA-256:
  `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`
- V42 checkpoint SHA-256:
  `a5ce2ff20d0fc6d336f489b1e2d29bb96a1d0666399c2e2b95e7ac33a9abe870`
- V42 development report SHA-256:
  `b376c3ae42c147c6e6ae9ed81f0294240bfb3c2da527b52312040e63a36584f7`
- V45 checkpoint SHA-256:
  `0e5947d85a8baeff99d92996ee8434d3aceab39e64042c9b6ec1a142aa174534`
- V45 field-state SHA-256:
  `08b57734ac3ded0c1438cc4bf963d92357ce1f1d31ae49ee6548c56c19db019d`
- V45 fit sequence SHA-256:
  `eb0a4ede44a062eb4107fa857c0cda488734b9ddf5ad56e30a5f375516e61fd8`

No V46 language model was trained before this decision was frozen.
