# Visual Posterior Language V51: Frozen Calibration Protocol

Date frozen: 2026-11-04

Status: frozen before implementation and calibration

## Core Decision

The language model and the writing model are independent experts.

The measured V42 causal reader supplies a continuous image-derived prediction
`a(H)` from visible history `H`. A separately trained unconditional diffusion
model supplies the score of readable glyph images. At inference, the two are
combined as a posterior rather than learned as another conditional writer:

```text
log p(x | H) = log p_write(x) + kappa * sim(phi(x), a(H)) + constant.
```

Here `phi` is the same fixed orthonormal raster field used by the reader. The
gradient used during reverse diffusion is

```text
score_posterior = score_write + kappa * grad_x sim(phi(x), a(H)).
```

This factorization responds directly to the V50 and V50-P failures. The image
prior cannot ignore language because guidance is applied explicitly at every
sampling step. The language reader cannot be damaged by reconstruction loss
because it is frozen and absent from prior training.

The construction follows the score-prior and guidance principles established
by [score-SDEs](https://research.google/pubs/score-based-generative-modeling-through-stochastic-differential-equations/),
[classifier-guided diffusion](https://arxiv.org/abs/2105.05233), and
[diffusion posterior sampling](https://arxiv.org/abs/2209.14687). The local
use is narrower: a 32-by-32 writing manifold guided by an image-native causal
state.

## Boundary

The deployed path accepts and emits floating or binary raster cells. It has no
text tokenizer, Unicode or character IDs, OCR transcript, glyph classifier,
discrete vocabulary, vector quantizer, codebook, candidate bank, or external
runtime model. Character strings remain offline rendering metadata. The
1,024-image bank is evaluator-only and is applied only after an output image
exists.

## Fixed Components

- causal reader: production V42 checkpoint, frozen;
- reader output: normalized 1,024-dimensional DCT visual field;
- prior training images: canonical training-partition glyph rasters only;
- prior: unconditional 32-by-32 Gaussian diffusion U-Net;
- prior input and output: one-channel signed raster, no labels;
- diffusion steps: 100 cosine-schedule training levels;
- calibration updates: 2,000;
- batch: eight 64-cell training streams, 16 sampled cells per stream;
- optimizer: AdamW, learning rate `2e-4`, weight decay `0.01`;
- precision: BF16 on CUDA:0;
- EMA decay: `0.999`;
- image-to-image start level: 30 of 100;
- reverse evaluations: 16 deterministic DDIM steps;
- normalized visual-guidance strength: `0.15`;
- generation seed: `20265120`.

The prior loss is noise-prediction MSE plus `0.10` foreground-balanced clean
image MSE and `0.10` soft ink Dice. It contains no sequence or character
supervision.

## Calibration Gates

The posterior route is retained only if all conditions hold on the fixed
development construction:

1. image-only runtime boundary is clean;
2. every output is finite and selected blank rate is below 0.02;
3. posterior identity retains at least 90% of direct-plan identity;
4. posterior pixel F1 is no more than 0.01 below direct-plan F1;
5. ordered posterior identity exceeds shuffled posterior identity;
6. ordered four-step feedback identity exceeds shuffled feedback identity;
7. isolated-ink fraction does not exceed twice the target fraction; and
8. no evaluator bank enters prior training, guidance, sampling, or selection.

Identity improvement is desirable but is not required in this first mechanism
test. Failure of gates 3 or 4 removes diffusion from the deployed path; the
direct field remains the output. Longer training or stronger guidance is not a
post-hoc remedy under this protocol.

## Claim Boundary

A passing calibration would show that an unconditional visual prior can clean
a causal image-language proposal without erasing its identity. It would not
show broad semantics, instruction following, arbitrary historical-form
generation, human-equivalent reading, or parity with a large token model.
