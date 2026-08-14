# Image-Conditioned Glyph Motor Bridge V41

Date: 2026-08-14

Status: research decision and reproducible mechanism audit. No language claim is
authorized by this document.

## Research question

Can a pretrained glyph-image generator remove raster corruption from an
image-native language model without reintroducing a text tokenizer, Unicode
lookup, character ID, OCR system, or runtime candidate bank?

The intended deployed loop is:

```text
glyph-image history
  -> continuous visual states
  -> causal language field
  -> continuous next-glyph distribution
  -> canonical glyph raster
  -> optional form projector
  -> visible glyph raster
  -> visual re-encoder
```

The optional form projector may change font, handwriting, script, or historic
surface form. It must not decide which glyph comes next. Language remains the
problem of predicting ordered visual content from ordered visual content.

## Why V39 is not being scaled

V39.1 corrected the first-stop distribution, removed the cyclic order
negative, initialized the visual length head to a realistic prior, increased
order/transition supervision, and used a warm-start EMA. A matched 512-record,
150-update rerun remained negative on held-out semantics:

| Metric | Original pilot EMA | Corrected pilot EMA |
| --- | ---: | ---: |
| answer top-1 | 0.04339 | 0.03562 |
| answer MRR | 0.08302 | 0.07406 |
| answer cosine | 0.18048 | 0.14486 |
| segment top-1 | 0.00523 | 0.00441 |
| segment MRR | 0.01264 | 0.01234 |
| transition-direction cosine | 0.00016 | 0.00648 |
| expected count mean / target | 7.244 / 6.313 | 6.303 / 6.313 |
| count-mode accuracy | 0.00389 | 0.15220 |
| visual-length MAE | 9.025 | 4.915 |

The mechanics improved substantially, but the corrected planner moved farther
below its inherited V38 answer baseline: final answer MRR was `0.01049` below
baseline and the final segment route was worse than its stage-1 route. A
full-bank V39 run would therefore scale an unqualified binding mechanism.

## Relevant published systems

### Pixel language modeling

- [PIXAR](https://arxiv.org/abs/2401.03321) demonstrates autoregressive text
  generation over rendered pixels, while reporting that independent pixel
  likelihood produces noisy writing. It proves that visual autoregression is
  possible, not that a local glyph renderer supplies semantics.
- [GlyphDiffusion](https://arxiv.org/abs/2304.12519) casts conditional text
  generation as glyph-image generation, but its text conditioning and
  text-grounding module do not satisfy this project's deployed image-only
  boundary.
- [MAR](https://arxiv.org/abs/2406.11838) shows that autoregressive generation
  can model a continuous next-state distribution with a diffusion loss instead
  of a vector-quantized codebook. This directly addresses the multimodality of
  the next glyph without forcing a vocabulary classifier.

### Glyph and calligraphy generators

- [FontDiffuser](https://arxiv.org/abs/2312.12142) accepts a content image and a
  style image, then uses multi-scale content aggregation and style contrastive
  refinement. Its [official implementation](https://github.com/yeungchenwa/FontDiffuser)
  is complete but restricted to non-commercial research and requires iterative
  diffusion sampling.
- [MX-Font](https://arxiv.org/abs/2104.00887) factors source glyph images into
  localized content/style experts. Its
  [official repository](https://github.com/clovaai/mxfont) provides MIT-licensed
  source and a 22.76M-parameter checkpoint. Runtime generation accepts only a
  source glyph image and style-reference glyph images.
- [DG-Font](https://arxiv.org/abs/2104.03064) supplies unsupervised font
  transfer but depends on an older deformable-convolution stack and is fixed to
  80-pixel images in the released implementation.
- [UniCalli](https://arxiv.org/abs/2510.13745) jointly recognizes and generates
  calligraphy columns, but the released base is about 23 GB, takes a
  five-character text condition, builds on FLUX and InternVL, and is licensed
  for non-commercial academic use. It is useful evidence for recognition and
  generation synergy, not a lightweight ILM motor.

## External motor selection

MX-Font is selected for the first audit, not as the language model. The reasons
are concrete:

1. its deployed content input is already an image rather than a character ID;
2. it supplies explicit `encode`, `factorize`, `defactorize`, and `decode`
   boundaries;
3. the released generator is 22,761,566 parameters and runs comfortably on one
   RTX 4090;
4. source and checkpoint can be pinned and hashed; and
5. exact glyph generation takes one deterministic pass.

The official checkpoint at revision
`93f3c88517f7c904f16da6333adb2588dcdf3cce` has SHA-256
`dcbcb6438d9b1e3230551bc78fcf64ec5454a01734502bdeac410d2f5c404119`.
An initial ten-glyph audit produced mean ink F1 `0.7110` in the source style
and `0.5786` after transfer to the held style. These are reconstruction/style
measurements, not evidence of language.

## The key nontrivial test

Giving MX-Font the exact next glyph would hide the problem. V41 instead asks
whether it can project an imperfect image-native motor output back onto a clean
glyph manifold. The reproducible audit compares:

1. exact `128 x 128` source glyph images;
2. images reduced to `32 x 32` and enlarged again;
3. images passed through the qualified V34 continuous codec; and
4. V34 states perturbed by Gaussian noise at fixed standard deviations.

For every condition, MX-Font receives only the perturbed source rasters and
four reference rasters. Character strings are host-side evaluation labels and
never model inputs. The audit records target ink F1 before and after the motor,
pixel MAE, nonblank rate, model hashes, font hashes, VRAM, and a contact sheet.

The command is:

```bash
PYTHONPATH=. python scripts/audit_glyph_motor_bridge_v41.py \
  --mxfont-root artifacts/external/mxfont \
  --v34-checkpoint artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt \
  --out artifacts/glyph_motor_bridge_v41_audit_20260814
```

## Measured bridge result

The pinned V34 and MX-Font checkpoints completed the deterministic audit on GPU
0 in `1.650` seconds with `771,669,504` peak allocated CUDA bytes. All states and outputs
were finite, and every glyph remained nonblank.

The tracked receipt SHA-256 is
`4980dc8f1922e76f9f68153c83abd240738e76f2b4d13b55d445f7026865873f`;
the byte-stable deterministic contact-sheet SHA-256 is
`6d8c610fac65806c28253f574f397aa5e62cc8665da633cf364246916aeb93da`.
The receipt records all conjunctive checks as passed and pins every imported
MX-Font source file, both fonts, all four style references, both checkpoints,
the external revision, deterministic settings, resources, and measured values.

| Source supplied to MX-Font | Input-to-target ink F1 | Motor-to-target ink F1 | Motor gain |
| --- | ---: | ---: | ---: |
| exact 128-pixel source | 0.43944 | 0.57863 | +0.13919 |
| coarse 32-pixel source | 0.43968 | 0.56481 | +0.12513 |
| V34 clean projection | 0.43182 | 0.54794 | +0.11612 |
| V34 projection, sigma 0.03 | 0.43124 | 0.54441 | +0.11317 |
| V34 projection, sigma 0.05 | 0.43170 | 0.54610 | +0.11440 |
| V34 projection, sigma 0.10 | 0.42941 | 0.54324 | +0.11383 |

The `sigma=0.03` and `sigma=0.05` routes retain `99.36%` and `99.66%` of
clean V34-projected motor F1, respectively. The visual audit also shows stable
glyph identity through `sigma=0.10`, with some local deformation on difficult
forms. The motor gate therefore passes.

This is a useful but bounded result. MX-Font cleans and restyles an imperfect
source glyph; it does not infer that glyph from language history. The next
experiment must predict the canonical content state autonomously and is not
allowed to receive the exact source glyph from the evaluator.

## Language model after the motor gate

Let `x_t` be one visible canonical glyph cell, `E` the image encoder, and `D`
the canonical raster decoder:

\[
c_t = E(x_t), \qquad h_t = T_\theta(c_{\le t}).
\]

A deterministic regression `M(h_t)` is not sufficient because the next glyph
is multimodal. V41 uses a continuous conditional flow over glyph state. For
target state `c`, Gaussian base `epsilon`, and `tau` sampled uniformly:

\[
u_\tau=(1-\tau)\epsilon+\tau c,
\qquad
v^\star=c-\epsilon,
\]

\[
\mathcal L_{flow}
=\mathbb E\left[\lVert F_\psi(u_\tau,\tau,h_t)-v^\star\rVert_2^2\right].
\]

The sampled state is decoded to a canonical raster, optionally projected into
a requested form from reference images, and then re-encoded from the visible
pixels:

\[
\hat x_{t+1}=M_{form}(D(\hat c_{t+1}),r),
\qquad
c^{feedback}_{t+1}=E(\hat x_{t+1}).
\]

There is no nearest-glyph lookup in this equation. The density head must place
its sample on the learned visual glyph manifold, and the renderer must emit the
pixels directly.

## Training sequence on one RTX 4090

### Stage A: canonical motor qualification

Use one canonical Chinese font. Train or reuse an image encoder and decoder,
then measure exact reconstruction, latent-noise tolerance, and closed-loop
re-encoding. No style augmentation is needed.

### Stage B: real visual continuation

Rasterize public-domain Chinese text into aligned glyph cells. Train a causal
continuous model on next-glyph and short continuation objectives. Report
teacher-forced state likelihood and autonomous raster continuation against
unigram, bigram, shuffled-history, and blank-history controls.

### Stage C: semantic transfer

A local open LLM may generate training answers or supply contextual hidden
targets offline. It may also initialize the causal transform. Token embeddings,
LM heads, tokenizers, and text strings are removed from the deployed artifact.
The independent checkpoint must run from glyph rasters to glyph rasters with no
teacher process.

### Stage D: form invariance

Only after canonical continuation passes, add alternate fonts, handwriting,
MX-Font outputs, and Hanziyuan historic forms as paired views. Content
consistency is trained across views while the form reference controls the
renderer. This tests generalization without making calligraphy a prerequisite
for language.

## Falsification gates

The motor bridge passes only if:

- all outputs and states are finite;
- V34-projected source images remain visibly content preserving;
- the external motor improves mean same-target ink F1 over its imperfect input;
- `sigma=0.03` and `sigma=0.05` retain at least 90% of clean projected-motor
  target F1; and
- every tested glyph remains nonblank.

The first language proof passes only if:

- generated rasters remain readable under autonomous feedback;
- correct history beats shuffled and blank history on held continuations;
- the model beats unigram and bigram visual baselines;
- lexical minimal pairs alter the generated continuation in the correct
  direction;
- exact copy works before factual question answering is claimed; and
- inference contains no tokenizer, Unicode ID, OCR call, codebook, candidate
  bank, or external teacher.

## Claim boundary

Adapting a calligraphy generator can establish a robust visual motor and form
control. It cannot establish language understanding. Language evidence begins
only when the causal model chooses correct unseen glyph continuations and
answers from image prompts under counterfactual controls. This separation keeps
the route simple: one font first, continuous image language next, visual style
last.

## Publication revision record

Revision stage: internal measured-milestone update, not a reviewer response or
submission package.

The pre-edit scientific basis is the research question, pinned audit, measured
bridge result, falsification gates, and claim boundary above. The bounded
revision scope is:

- update `README.md` with V39.1 and V41 as the newest evidence boundary;
- update `publication/ilm-image-native/ilm-image-native.tex` in the abstract,
  contributions, measured-results sequence, conclusion, and bibliography;
- add the deterministic, evidence-pinned V41 result figure and its generator;
- keep the active manuscript filename stable; and
- rebuild and inspect the current PDF.

Explicitly out of scope are a language-capability claim, Qwen/GPT parity,
compute-superiority claims, sealed-data evaluation, autonomous glyph selection,
historic-form generation, and changes to earlier measured receipts. No response
letter, supplement, manuscript baseline, or prior redline exists in this
repository, so this revision does not invent one.

Acceptance requires passing code tests and lint, matching receipt/contact-sheet
hashes, a conjunctive motor gate recorded in the receipt, a clean two-pass LaTeX
build with no undefined citation/reference or overfull box, and visual review of
the compiled V39/V41 pages. The inspected PDF locations are Section 5.20,
Tables 27--28, Figure 28, and pages 59--62 of the August 2026 build.
