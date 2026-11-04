# Visual Posterior Language V51: Result

Date completed: 2026-11-04

Decision: **reject the clean-field posterior as the deployed writer; retain the
unconditional glyph prior and the posterior factorization as measured building
blocks.**

## Question

V51 tested whether a small unconditional diffusion prior could turn the frozen
V42 reader's continuous next-writing field into a cleaner visible glyph without
erasing the field's language identity.  The reader and prior were independent:
the prior learned only the marginal distribution of `32 x 32` glyph rasters,
while reverse diffusion received an explicit gradient from the reader's DCT
field.

The experiment was frozen in
[`visual_posterior_language_v51_protocol.md`](visual_posterior_language_v51_protocol.md)
before calibration.  Passing required all eight development gates.

## Fixed System And Resource Receipt

- frozen V42 causal reader: `16,277,249` parameters;
- unconditional raster prior: `3,464,321` parameters;
- combined runtime: `19,741,570` parameters;
- prior optimization: 2,000 updates, 128 glyph images per update;
- optimization time: `169.423` seconds on one RTX 4090 D;
- peak allocated training memory: `0.768 GiB`;
- prior checkpoint SHA-256:
  `ef607cb67c49da6d9bea32a784cb821a92ea546efbe8bf2226f8065171c00629`;
- frozen reader SHA-256:
  `a5ce2ff20d0fc6d336f489b1e2d29bb96a1d0666399c2e2b95e7ac33a9abe870`;
  and
- no tokenizer, Unicode or character ID, OCR, glyph vocabulary, codebook,
  candidate bank, or external runtime model entered training or generation.

The final prior batch reached noise MSE `0.03573`, foreground clean-image MSE
`0.18640`, and soft-Dice loss `0.28709`.  These are optimization diagnostics,
not evidence of conditional language generation.

## Locked Development Result

The audit used 256 fixed development continuations for one-step generation and
64 fixed prefixes for four-step visible feedback.  Ordered and
suffix-preserving shuffled histories used paired random noise.

| One-step measurement | Ordered | Shuffled |
| --- | ---: | ---: |
| direct-plan identity top-1 | `0.19141` | `0.17188` |
| posterior identity top-1 | `0.16406` | `0.13672` |
| posterior identity retained from plan | `0.85714` | `0.79545` |
| direct-plan pixel F1 | `0.47497` | `0.47140` |
| posterior pixel F1 | `0.47714` | `0.47354` |
| posterior F1 change | `+0.00217` | `+0.00213` |
| posterior isolated-ink fraction | `0.02455` | `0.02462` |
| target isolated-ink fraction | `0.00270` | `0.00270` |
| blank rate | `0` | `0` |

The posterior preserves a conditional signal: ordered identity exceeds
shuffled identity by `0.02734`.  Four-step visible-feedback identity averages
`0.05469` for ordered history and `0.03125` for shuffled history, a `0.02344`
advantage.  Pixel F1 also rises slightly rather than collapsing.

These positive differences do not satisfy the writer decision.  Identity
retention is `85.7%`, below the fixed `90%` minimum, and isolated ink is
`9.08` times the target fraction, above the fixed factor-of-two ceiling.

| Frozen gate | Result |
| --- | :---: |
| image-only runtime boundary | pass |
| finite, nonblank output | pass |
| at least 90% identity retention | **fail** |
| F1 no more than 0.01 below plan | pass |
| ordered identity above shuffled | pass |
| ordered feedback above shuffled | pass |
| isolated ink at most twice target | **fail** |
| evaluator bank excluded from student | pass |

The registered decision is therefore `reject`.

## What The Images Show

The quantitative rejection understates the main visual defect.  The direct
inverse-DCT proposal is a dense speckled field rather than readable writing.
At the fixed image-to-image noise level, the posterior removes about half of
the isolated speckle but remains visually far from a canonical glyph.  A small
F1 gain on this surface is not a readable-writing result.

A post-result motor diagnostic separates the two components:

- when a real target glyph is supplied at diffusion level 30, the prior
  reconstructs a clean, readable glyph;
- at levels 50 and 70, reconstruction degrades progressively but remains
  recognizably glyph-like in many cases;
- from near-pure noise, the prior generates coherent glyph-like modes; and
- guidance from the V42 DCT field at high noise changes those modes but does
  not preserve the intended identity.

Thus the unconditional writer has learned a useful writing manifold.  The
failure lies in the guidance energy: raw DCT cosine ranks completed candidate
glyphs, but its pixel gradient is linear and rewards scattered local matches.
It does not supply a noise-level-dependent basin that moves a diffusion sample
toward the correct whole-glyph mode.

## Mechanistic Consequence

The next experiment should not enlarge the denoiser or add another conditional
decoder.  It should learn a **noise-aware visual language energy** while keeping
both successful components frozen:

1. V42 produces a causal state from ordered glyph images;
2. the unconditional V51 prior supplies the marginal glyph score;
3. a compact critic is trained contrastively on aligned and mismatched
   `(causal state, noised next glyph, diffusion time)` tuples; and
4. the critic gradient is added to the prior score at every reverse step.

At the population optimum, this critic estimates the time-dependent density
ratio between the conditional next-glyph distribution and the marginal glyph
distribution.  This is the image-language analogue of classifier guidance,
but the condition is a learned visual history rather than a symbolic class.
It directly addresses the failure observed here: the guidance network sees
noisy glyphs at the same levels at which its gradient is used.

This direction remains a hypothesis until it passes candidate-ranking,
ordered-versus-shuffled, readable-raster, and closed-loop feedback controls.

## Claim Boundary

V51 does not establish a complete image-native language model, readable
autonomous writing, broad semantics, instruction following, historical-form
generation, or parity with a token LLM.  It establishes three narrower facts:
the compact prior learns a glyph-like image manifold, posterior guidance can
retain measurable ordered-history signal, and a clean-image linear DCT energy
is not sufficient to bind that manifold to the intended language identity.
