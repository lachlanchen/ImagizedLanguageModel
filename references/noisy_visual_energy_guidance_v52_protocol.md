# Noisy Visual Energy Guidance V52: Frozen Calibration Protocol

Date frozen: 2026-11-04

Status: frozen before implementation, training, or development measurement

## Decision Question

Can a compact, noise-aware visual critic bind the frozen V42 language state to
the frozen V51 glyph manifold strongly enough to generate readable,
context-dependent next-glyph images from noise?

V52 changes the coupling mechanism, not the reader or renderer.  V42 already
ranks completed next-glyph images above bigram and shuffled-history controls.
V51 already reconstructs real glyphs and generates glyph-like marginal modes.
Its clean DCT guidance failed because the gradient rewarded scattered pixel
matches rather than a whole-glyph identity.  V52 learns the missing guidance at
the same noise levels where diffusion consumes it.

## Probability Model

Let `H` be an ordered raster history, `X_0` the next glyph image, and `X_t` its
Gaussian-corrupted state.  The frozen V51 prior estimates the marginal score

```text
s_prior(X_t,t) = grad_X log p_t(X_t).
```

A critic `R_psi(H,X_t,t)` is trained contrastively on aligned and mismatched
history--glyph pairs.  With marginal glyph negatives, its ideal row-wise logit
is a conditional density ratio up to a history-dependent constant:

```text
R*(H,X_t,t) = log p_t(X_t | H) - log p_t(X_t) + c(H,t).
```

The conditional score is therefore

```text
s_cond(X_t,H,t) = s_prior(X_t,t) + grad_X R_psi(H,X_t,t).
```

The reverse process starts from Gaussian noise.  It does not start from V42's
speckled inverse-DCT point estimate.  Four independent guided samples are
allowed; selection uses only the critic's own clean-image energy.  No glyph
bank participates in generation or selection.

This construction follows the noisy-classifier score decomposition in
[Diffusion Models Beat GANs](https://arxiv.org/abs/2105.05233) and the
intermediate-energy analysis in
[Contrastive Energy Prediction](https://arxiv.org/abs/2304.12824).  The latter
shows why applying a clean energy to a denoised estimate need not equal the
correct intermediate guidance.  V52 learns the intermediate visual energy
directly.

## Runtime Boundary

The deployed model receives an ordered tensor of floating glyph rasters and
emits floating or binary glyph rasters.  It contains:

- the frozen `16,277,249`-parameter V42 causal raster reader;
- the frozen `3,464,321`-parameter V51 unconditional glyph prior; and
- one trainable noise-aware visual energy critic.

It contains no string, tokenizer, token ID, Unicode value, character ID, OCR
transcript, vocabulary embedding, vocabulary logits, glyph lookup, candidate
bank, quantizer, codebook, or external runtime model.  Strings remain host-side
data-preparation and evaluation metadata.  The 1,024-image identity bank is
evaluator-only and is instantiated only after output images exist.

## Fixed Critic

The V42 reader remains byte-for-byte frozen.  For every causal position it
supplies a 384-dimensional hidden state and a 1,024-dimensional visual anchor.
The critic has two continuous towers:

1. a context MLP maps the concatenated hidden state and anchor to a normalized
   256-dimensional query; and
2. a timestep-conditioned convolutional encoder maps one noisy `32 x 32`
   raster to a normalized 256-dimensional key.

The image tower uses four spatial scales (`32`, `16`, `8`, and `4`), residual
blocks, and sinusoidal timestep conditioning.  A learned positive temperature
converts query--key cosine into a scalar compatibility.  The critic parameter
cap is 6 million.  It may receive gradients only during critic calibration;
reader and prior gradients are prohibited.

## Fixed Data And Loss

- corpus: the existing public-domain Chinese visual stream;
- split: training records only;
- renderer: the fixed V42 canonical `32 x 32` Noto glyph renderer;
- context: 64 visible cells;
- batch: eight rendered streams;
- selected causal positions: 16 per stream, stratified four each from positions
  `0..15`, `16..31`, `32..47`, and `48..63`, including position 63;
- candidates per update: 128 next-glyph images; and
- duplicate positives: exact target-raster equality, never character IDs.

Every update uses one shared diffusion level so all row candidates have the
same noise distribution.  Updates cycle through four fixed strata: clean
`t=0`, low noise `t in [1,30]`, medium noise `t in [31,70]`, and high noise
`t in [71,99]`.  Non-clean levels are sampled uniformly inside their stratum
from the saved training generator.

The primary loss is row-wise multi-positive image InfoNCE.  A symmetric
image-to-context term has weight `0.25`.  A V42-energy transport term has
weight `0.50`: clean V42 candidate scores form stopped-gradient soft labels,
and noisy critic scores predict those labels as in contrastive energy
prediction.  An endpoint ordered-versus-suffix-preserving-shuffled log-mass
margin of `0.05` has weight `0.25`.  All supervision is derived from image
tensors and frozen visual states.

## Fixed Optimization

| Setting | Value |
| --- | ---: |
| updates | 4,000 |
| optimizer | AdamW |
| learning rate | `2e-4` |
| warmup | 200 updates |
| minimum cosine LR ratio | `0.10` |
| weight decay | `0.01` |
| gradient clip | `1.0` |
| EMA decay | `0.999` |
| precision | BF16 |
| seed | `20265210` |
| dataset seed | `20265211` |
| hardware | CUDA device 0, one RTX 4090 |
| peak allocated memory ceiling | 18 GiB |

Checkpoint receipts must include hashes for the reader, prior, protocol,
source files, corpus manifest, render configuration, model boundary, optimizer,
training generator, and EMA state.

## Fixed Sampling

- start level: `99` of 100, effectively the marginal Gaussian;
- deterministic DDIM levels: 25;
- guidance coefficient: `1.0` multiplying the raw critic log-density gradient;
- candidates: four paired-noise samples per context;
- selected output: maximum critic energy at `t=0` among those four;
- primary output: the first sample, reported independently of selection; and
- feedback: the selected visible binary raster is appended to context and
  reread by V42.

No gradient normalization, candidate-bank selection, target-side reranking,
guidance sweep, or development-selected threshold is permitted.

## Fixed Development Audit

- 512 deterministic development windows;
- 1,024 evaluator-only canonical glyph images;
- clean critic ranking for ordered and suffix-preserving shuffled histories;
- 256 one-step generation examples with paired noise for ordered, shuffled,
  and zero-guidance routes;
- first-sample and critic-selected identity top-1;
- pixel F1, ink fraction, isolated-ink fraction, blank rate, and distinct
  evaluator identity count;
- 64 prefixes with four generated feedback steps; and
- deterministic galleries showing target, first guided, selected guided,
  shuffled selected, and zero-guidance selected images.

## Calibration Gates

V52 is retained only if every condition holds:

1. the image-only runtime boundary is clean;
2. the critic has at most 6 million trainable parameters and combined runtime
   has at most 27 million parameters;
3. clean ordered 1,024-way critic top-1 is at least `0.15`;
4. clean ordered top-1 exceeds clean shuffled top-1 by more than `0.01`;
5. ordered first-sample generated identity is at least `0.05`;
6. ordered first-sample identity exceeds paired zero-guidance identity by more
   than `0.02`;
7. ordered critic-selected identity is at least `0.08` and exceeds selected
   shuffled identity by more than `0.015`;
8. ordered four-step feedback identity exceeds shuffled feedback identity;
9. selected blank rate is below `0.02`, isolated-ink fraction is at most
   `0.01`, and generated-to-target ink ratio lies in `[0.70,1.50]`;
10. at least 32 evaluator identities appear among 256 ordered selected outputs;
11. all tensors and gradients remain finite, peak allocated memory is below
    18 GiB, and the frozen reader and prior hashes are unchanged; and
12. evaluator metadata and the identity bank remain absent from critic
    training, guidance, sampling, and selection.

One failed gate rejects V52 as a complete mechanism.  The audit may localize a
failure but may not authorize post-hoc guidance tuning, longer calibration, or
threshold changes under this protocol.

## Claim Boundary

A passing calibration would establish a compact image-only mechanism that uses
ordered visual history to guide a glyph image prior toward readable next-glyph
modes and can reread its own output for four steps.  It would not establish
broad semantics, instruction following, factual question answering,
etymological explanation, arbitrary historical-form control, human-equivalent
reading, or parity with a large token language model.
