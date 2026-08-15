# Visual Future Block Language V48: Frozen Development Protocol

Date frozen: 2026-08-15

Status: production protocol frozen before V48 implementation or training

## Question

Can a compact causal model learn a stronger Chinese visual-language state by
predicting four future glyph rasters at every position, while directly emitting
and rereading a raster without token IDs, Unicode IDs, OCR, a glyph classifier,
a codebook, a candidate bank, or an external language model in the student or
deployed path?

V48 tests only this mechanism. It does not add multiple fonts, page layout,
instruction data, historical forms, Hanziyuan supervision, or a local-LLM
teacher. Those additions remain blocked until the canonical image-language
core qualifies.

## Frozen Predecessor And Motivation

The matched baseline is the production V42 checkpoint:

```text
path   artifacts/canonical_glyph_language_v42_20260814/checkpoint_final.pt
SHA-256 a5ce2ff20d0fc6d336f489b1e2d29bb96a1d0666399c2e2b95e7ac33a9abe870
update 10000
```

V42 reaches 0.199707 development top-1 from full visual history and 0.183594
after a suffix-preserving shuffle. Its 24.35M-parameter model includes an
8.07M-parameter stochastic writer that lowers visible identity relative to
direct output.

The V42--V47 diagnostic fixes the V48 design:

- V47's reconstruction sphere is substantially below V42 before its terminal
  failure;
- V47's final-position-only pair objective produces a hard length-64 cliff;
- V42's direct field output retains 0.165039 visible identity and 0.471588
  pixel F1 on 2,048 fixed development windows; and
- removing V42's learned generator leaves 16,277,249 parameters before adding
  horizon offsets.

V48 therefore changes the temporal prediction objective and removes the weak
writer. It does not change the visual coordinate or enlarge the reader.

## Immutable Data And Rendering

The corpus is:

```text
data/visual_grammar/chinese_wikisource_public_domain.jsonl
SHA-256 76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03
records 7,017
```

It contains 16 public-domain Chinese Wikisource works and inherits the V25
record partition. Training may use only the training partition. Development
may select checkpoints only under this protocol. The frozen partition remains
closed.

Both original and offline simplified script views are eligible. Each cell is
rendered deterministically with:

```text
font /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
font size 26
cell 32 x 32
augmentation disabled
ink-positive floating raster in [0,1]
```

The student collator exposes exactly two tensors:

```text
context       B x 64 x 1 x 32 x 32
future_pixels B x 64 x 4 x 1 x 32 x 32
```

A training segment contains 68 consecutive raster cells. For visible position
`t` in cells 1--64, the four targets are cells `t+1` through `t+4`. Strings,
source identifiers, offsets, script labels, and character identities remain
host metadata and are removed before the student call.

## Exact Visual Coordinate

Let `x` be a glyph raster, `D` the fixed orthonormal 32-point DCT-II matrix,
and

```text
s(x) = 2 * 1[x >= 0.5] - 1
k(x) = vec(D s(x) D^T) / 32.
```

Parseval's identity gives unit norm for every binary target field. The visible
inverse is fixed:

```text
visible(q) = 1[D^T unvec(32 q) D >= 0].
```

The zero threshold is immutable. V48 may use a differentiable sigmoid with
fixed sharpness 6 only inside the pixel training loss; deployed and evaluated
rasters use the zero boundary. There is no learned encoder, decoder, radius,
threshold, quantizer, or inventory lookup.

## Model

The production configuration is fixed:

```text
field dimension       1,024
model dimension         384
causal layers              8
attention heads             6
MLP ratio                 3.0
dropout                  0.05
maximum visible cells      64
future horizons             4
initial temperature       0.07
binary threshold          0.50 for input rasterization
inverse-DCT boundary      0.00
decoder sharpness         6.00 for training probabilities
```

The input projection, causal blocks, RMS normalization, rotary positions,
shared visual head, and logit scale match V42. There is no stochastic
generator. Three learned vectors `e_2`, `e_3`, and `e_4` in `R^384` identify
future distance; horizon one has exactly zero offset:

```text
z_1:T = k(context_1:T)
h_1:T = causal_reader(z_1:T)
u_(t,1) = h_t
u_(t,r) = h_t + e_r                  r in {2,3,4}
q_(t,r) = normalize(head(u_(t,r))).
```

All horizons share the same head and fixed visual coordinate. The complete
model must contain fewer than 17,000,000 trainable parameters. No V42 weights
are loaded; V48 is trained from scratch.

## Loss

For every horizon `r`, flatten all 64 causal positions and encode the matching
future raster fields. Positives are defined only by exact equality of binary
target rasters dynamically present in that minibatch.

```text
L_contrast,r = exact-raster multi-positive contrastive loss
L_angle,r    = mean_t [1 - q_(t,r)^T k(X_(t+r))]
L_pixel,r    = BCE + 0.5 Dice on fixed inverse-DCT probabilities
L_r          = L_contrast,r + 0.25 L_angle,r + 0.20 L_pixel,r
```

The frozen horizon weights are:

```text
alpha_1 = 1.000
alpha_2 = 0.500
alpha_3 = 0.250
alpha_4 = 0.125
L = sum_(r=1)^4 alpha_r L_r.
```

Horizon one retains the full V42 point-prediction pressure. Later horizons are
auxiliary and geometrically decayed. The loss has no energy term, diffusion
term, flow term, pair term, cycle term, latent Gaussian term, teacher term, or
terminal-position term.

The same deterministic position subset is used for all horizons within an
update. At most 512 flattened positions enter each horizon's contrastive and
pixel terms. The angular term uses every position.

## Fixed Optimization

```text
updates                       10,000
batch size                         8
gradient accumulation               2
effective segments/update          16
optimizer                       AdamW
learning rate                    3e-4
warmup updates                    500
minimum LR ratio                 0.10
weight decay                     0.05
Adam betas                  (0.9,0.95)
gradient clip                    1.00
precision                        BF16
training seed                20264800
dataset seed                 20264801
log interval                       50
checkpoint interval             1,000
production device              CUDA:0
```

The cosine learning-rate schedule and checkpoint/resume semantics match V42.
A production checkpoint records the protocol, source hashes, model/data
boundaries, corpus/partition/font receipts, loss weights, optimizer state,
elapsed time, peak allocated VRAM, and all final training metrics.

## Inference Boundary

The ordinary deployed step is deterministic:

1. accept one floating raster stream;
2. map every cell through the fixed DCT field;
3. predict four continuous fields;
4. inverse-transform and zero-threshold horizon one;
5. append the emitted raster; and
6. reread those visible pixels on the next step.

The model also exposes a four-image forecast strip for auditing, but ordinary
autoregressive generation replans after each emitted horizon-one image. It may
not rank, snap, or project its output against the evaluator bank.

The student and deployed path must not receive strings, bytes, token IDs,
Unicode scalars, character IDs, OCR output, a font-to-character map, a glyph
lookup, a vocabulary embedding, a vocabulary classifier, a visual codebook,
quantized latents, an external LLM, or candidate raster images.

## Frozen Development Evidence

### Evaluator bank

The evaluator may use the 1,024 most frequent training Han forms only after a
field or raster has been produced:

```text
ordered-character SHA-256
a88509b3c1d3093e63dd1ceb77dcd86c7ef282c80927284b93c4ef09cd9456ad
```

### Matched next-cell audit

Use the exact V42 algorithm, 2,048 development windows, 16-cell continuations,
and seed `20264220`:

```text
window SHA-256
4665be2d5cf1714a21d6523ca697749456036d7719302171762bc30291f451f3
```

Score horizon one under full 64-cell, suffix-four, last-cell, suffix-preserving
shuffle, and blank histories. Report top-1, top-5, target log probability, and
target-field cosine against the evaluator bank. Report the image-unigram and
symbolic-bigram controls computed only from training text. The evaluator must
also score the frozen V42 checkpoint in the same process and assert its
registered full top-1 `0.19970703125` within numerical tolerance `1e-6`.

### Four-future audit

Reservoir-sample 2,048 development windows whose next four cells all belong to
the evaluator bank. Use seed `20264820`, preserve the ordered reservoir, and
require:

```text
eligible windows 31,555
window SHA-256
9ba79f8f7e196192cd07daa7438daa8a60665b78650dc1b3f4c3d441db561d03
```

For horizons one through four, score teacher-forced top-1, top-5, target log
probability, target cosine, distinct top-1 predictions, and most-common
prediction fraction. Build a host-only offset-conditional control from
training text: condition on the last visible character and count the character
at offset `r`. Additive smoothing is fixed at 0.10.

The evaluator asserts these control top-1 values before scoring V48:

```text
horizon 1  0.1074218750
horizon 2  0.0361328125
horizon 3  0.0268554688
horizon 4  0.0317382812
```

### Counterfactual audit

Use the V42 development-pair construction: 512 pairs, four-cell identical
suffix, different source identifiers, different next targets, seed
`20264221`, and targets restricted to the evaluator bank. Its ordered pair
digest is:

```text
b1d62615aebeda8b48bc9ee526ad5362320bc95dd0a7be612a789a37b6e16e1a
```

Report full, suffix-four, last-cell, and shuffled arm accuracy, both-correct
rate, ties, and margin. No pair is used for training.

### Terminal-position audit

Reuse the exact 2,048 development windows from the V42--V47 predictive-state
diagnostic:

```text
window SHA-256
e68db6c27b75a4d502a0dac7fe08487f1c5e3850d46ea1e561c7f18a5757e630
```

Score horizon one at visible lengths 56 through 64. Report top-1, top-5,
target log probability, target cosine, distinct predictions, and most-common
prediction fraction.

### Direct-raster and closed-loop audit

On all 2,048 matched next-cell windows, inverse-transform horizon one at the
fixed zero boundary and reread the raster. Report proposal and visible-reread
identity, pixel F1, ink-density ratio, blank rate, proposal-to-visible cosine,
and visible-to-target cosine.

On the first 256 four-future windows, generate four cells autoregressively,
replanning and rereading after every visible output. Report per-step and mean
identity, pixel F1, blank rate, and finite-output integrity. Save one
deterministic target/proposal/visible/rollout raster sheet with a declared
generation seed even though generation itself is deterministic.

## Frozen Gates

All gates must pass for V48 to qualify on development.

### Boundary and compute

1. the image-only student and deployed boundary is clean;
2. trainable parameters are below 17,000,000;
3. peak allocated training or evaluation VRAM is below 18 GiB; and
4. 10,000-update production training completes in less than 7,200 seconds on
   one RTX 4090.

### Immediate visual language

5. matched full-history top-1 exceeds frozen V42 by more than 0.005;
6. full-history top-1 exceeds the matched symbolic bigram by more than 0.03;
7. ordered full-history top-1 exceeds shuffled history by more than 0.015; and
8. ordered target log probability exceeds shuffled history by more than 0.10
   nat.

### Predictive state and order

9. every horizon in the four-future audit exceeds its frozen
   offset-conditional top-1 by more than 0.01;
10. every horizon emits at least 128 distinct top-1 predictions and no horizon
    assigns more than 0.15 of examples to one top-1 prediction;
11. same-suffix counterfactual full-history arm accuracy exceeds 0.55; and
12. terminal top-1 at length 64 is no more than 0.01 below length 63, with a
    length-64 most-common prediction fraction below 0.10.

### Visible generation

13. direct visible-reread identity top-1 exceeds 0.15;
14. direct visible pixel F1 exceeds 0.46 and blank rate is below 0.02;
15. visible-reread identity retains at least 0.85 of proposal identity and
    proposal-to-visible cosine exceeds 0.82; and
16. four-step closed-loop mean visible identity exceeds the image-unigram
    top-1 by more than 0.01, with zero non-finite or blank outputs.

Strict comparisons use epsilon `1e-12`. Gate thresholds, window sets, baseline
values, and loss weights cannot change after production starts.

## Integrity Requirements

The development report is invalid unless it proves:

- strict loading of the 10,000-update non-smoke, non-exploratory checkpoint;
- exact corpus, renderer, bank, window, pair, protocol, and source digests;
- no development or frozen record entered training;
- all student inputs and targets were finite floating image tensors;
- evaluator labels and candidate images were excluded from student calls;
- no gradient or optimizer was active during evaluation;
- direct and closed-loop outputs were produced before candidate-bank scoring;
- every recurrent cell was a visible emitted raster, not an unrendered field;
- every metric is finite; and
- the frozen partition remained closed.

## Stop Rules

- Smoke runs may validate shapes, causality, gradients, checkpoint round-trip,
  evaluator hashes, and a short closed loop only. They are never evidence.
- A pre-production integrity run may use at most two optimizer updates. It may
  fix implementation defects but may not tune model, loss, data, or gates.
- Once production begins, do not restart from an earlier checkpoint because a
  metric looks poor. Resume only after an external interruption and preserve
  optimizer, RNG, dataset index, and elapsed-time receipts.
- Evaluate only the final update-10,000 checkpoint. Do not select an earlier
  checkpoint on development.
- Do not open the frozen partition unless all development gates pass and a new
  confirmation protocol is committed first.
- A failed V48 is a measured mechanism failure. Do not add fonts, calligraphy,
  instructions, a local LLM, a learned codec, or more parameters to disguise
  it.

## Claim Boundary

A passing V48 would support one narrow claim: dense multi-horizon image
prediction improves a compact canonical Chinese raster-language model over a
matched next-image baseline, and a fixed continuous image field can serve as
both sensory input and visible output without a deployed symbolic inventory.
It would not establish general semantics, question answering, etymology,
historical glyph generation, arbitrary writing, page-level reading, human-like
vision, Qwen parity, or superior scaling to token language models.
