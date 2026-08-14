# Noise-Limited Retinal Field V45 Protocol

Status: preregistered before V45 implementation or official audit.

Date: 2026-08-14

## Question

Can a fixed, invertible, raster-only transform remove V42/V44's common image
mode and improve held-out identity-bearing displacement geometry without
destroying small-shift or held-font continuity?

V45 is a representation audit. It trains no language model and opens neither
the V43 writer nor the frozen language partition.

## Fixed Inputs

- Corpus: `data/visual_grammar/chinese_wikisource_public_domain.jsonl`.
- Required manifest SHA-256:
  `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`.
- Statistic-fit split: training records only.
- Statistic-fit forms: the 8,000 most frequent supported Han forms under
  `original+simplified`, ordered by frequency and code point exactly as in the
  V42 evaluator.
- Fit raster: Noto Sans CJK Regular, 26 pixels, centered in a `32 x 32` cell.
- Fit weighting: each form's training-corpus count divided by the retained
  count total.
- Identity audit bank: the first 1,024 fit forms.
- Held-font audit: both V25 development fonts at 26 pixels. Frozen fonts are not
  instantiated.
- Shift audit: the canonical 1,024-image bank translated by exactly one pixel
  left, right, up, and down with zero fill.
- Pair audit: the 1,024 untouched V44 training-partition pairs with receipt
  `e41637c5e3846e47d19ade6312205cd30c2c96c65730bce75ee8ab4a0745154c`.
- Pair seed and construction remain those pinned by V44; no consumed V44 pair
  enters this audit.

Strings and corpus counts are host-side fit metadata. Every transform input,
inverse, retrieval query, and pair displacement passed to the audited field is
a floating-point raster or raster-derived continuous tensor.

## Fixed Field

For thresholded raster `x`, let

\[
d(x)=\operatorname{DCT}_2(2\mathbf 1[x\geq0.5]-1)\in\mathbb R^{1024}.
\]

With normalized training-count weights `w_i`, fit

\[
\mu=\sum_iw_i d_i,\qquad
\Sigma=\sum_iw_i(d_i-\mu)(d_i-\mu)^\mathsf T.
\]

Let `Sigma = U diag(lambda) U^T`,
`v_bar = trace(Sigma) / 1024`, and fix

\[
A=U\operatorname{diag}
\left[(\max(\lambda,0)+0.50\bar v)^{-0.10}\right]U^\mathsf T.
\]

The representation and inverse are

\[
z=A(d-\mu),\quad u=z/\max(\lVert z\rVert_2,10^{-8}),\quad
q=\log\max(\lVert z\rVert_2,10^{-8}),
\]

\[
\hat d=\mu+A^{-1}(e^q u),\qquad
\hat x=\mathbf 1[\operatorname{IDCT}_2(\hat d)\geq0].
\]

All eigendecomposition and stored statistics use FP64 on CPU. Encoding and
auditing use FP32 unless the exact-inverse check explicitly uses FP64. Eigenvalue
ordering and signs do not affect the symmetric matrix powers, but the serialized
fit receipt must hash `mu`, `U`, `lambda`, exponent, ridge, font, and form/count
receipts.

No post-audit exponent, ridge, floor, bank size, weighting, font, or threshold
change is allowed.

## Fixed Controls And Metrics

### Exact image path

Encode and invert all 8,000 fit rasters, both 1,024-form held-font banks, and all
four shifted banks. Report maximum DCT reconstruction error, binary pixel
accuracy, ink F1, blank rate, and finite rate.

### Common-mode and covariance geometry

For raw V42 unit fields and V45 directions over all 8,000 weighted fit forms,
report:

- resultant length `||sum_i w_i u_i||`;
- covariance effective rank `exp(H(lambda / sum lambda))`;
- covariance stable rank; and
- minimum, median, and maximum radial value `exp(q)`.

Mean centering without the matrix power and full ZCA with exponent `0.50` are
descriptive controls only. They cannot replace the fixed V45 field.

### Held-font and translation continuity

For each held development font and each one-pixel translation, retrieve among
the 1,024 canonical raster fields by cosine after the field emits its direction.
Report top-1, target cosine, and top-1 change from raw V42 fields.

### Untouched pair displacement

For raw and V45 candidate fields on the 1,024 held pairs, report candidate-pair
cosine, target-delta norm quantiles, target-delta covariance effective rank, and
target-delta stable rank. Candidate character values and pair labels remain
evaluator-side and do not enter the transform.

### Frozen V42 retrofit diagnostic

As a report-only diagnostic, multiply a frozen V42 unit anchor by `32`, apply the
V45 transform, and evaluate the same 2,048 matched development windows used by
V44. This metric cannot qualify or reject V45 because V42 was optimized in the
raw field. It determines whether V46 must train from scratch or whether a
calibration bridge is worth testing. No V42 parameter changes.

## Conjunctive Gates

V45 qualifies only if every condition is true:

1. every audited value is finite and every inverse raster is nonblank;
2. maximum FP64 DCT round-trip absolute error is below `2e-8`;
3. binary pixel accuracy and ink F1 after round trip are exactly `1.0` for every
   audited bank;
4. V45 weighted resultant length is below `0.05` and below `0.10` times the raw
   V42 resultant;
5. V45 covariance effective rank is at least `1.20` times raw V42 effective
   rank;
6. each held-font top-1 is no more than `0.01` below its matched raw-field
   top-1, and mean held-font top-1 is at least the raw mean;
7. each one-pixel-shift top-1 is no more than `0.02` below its matched raw-field
   top-1, and mean shift top-1 is at least the raw mean;
8. held-pair mean candidate cosine decreases by at least `0.25`;
9. held-pair 5th-percentile target-delta norm is at least `1.50` times raw;
10. held-pair target-delta effective rank is at least `1.10` times raw;
11. held-pair target-delta stable rank is at least `1.08` times raw;
12. the fit uses training rasters only, the frozen partition is not instantiated,
    and the image-only boundary contains no text model, OCR, token/Unicode ID,
    character lookup, codebook, quantization, or deployed candidate bank; and
13. the complete official audit peaks below `4 GiB` allocated CUDA memory and
    finishes within 20 minutes on one RTX 4090.

The fixed V42 retrofit diagnostic is reported but is not a gate.

## Decision Rule

If all 13 gates pass, V45 may be called a **qualified retinal field** and a V46
causal reader may be preregistered in the new representation. V45 alone cannot
claim language understanding or generation. V46 must retain the same corpus
partition and must beat V42's matched language controls before the V43 writer is
reopened.

If any gate fails, V45 is rejected or partial. No causal reader is trained in
the field until the failed representation mechanism is diagnosed under a newly
preregistered protocol. The frozen language partition and V43 writer remain
closed in either case.
