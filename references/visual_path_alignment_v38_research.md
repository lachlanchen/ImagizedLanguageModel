# Visual Path Alignment V38: Research Decision

Status: pre-implementation research note, not a frozen evidence protocol.

Date: 2026-08-14

## Purpose

V38 is the next bounded attempt toward an independent image-native language
model. It does not open raster generation. It asks a narrower prerequisite
question:

> Can one image-only student map semantically identical writing to stable
> continuous prompt and answer states despite changes in font, raster style,
> and wording?

The deployed route must remain a single raster-and-mask forward pass. Text,
OCR, token IDs, Unicode IDs, candidate banks, Qwen, and BGE are permitted only
before the tensor boundary or in offline supervision and evaluation.

## Evidence from V37

V37 completed 8,000 updates in 61.59 minutes on one RTX 4090 D. Its EMA route
passed 20 of 33 frozen development conditions, but it failed the complete
semantic gate. Four observations determine V38:

1. Prompt retrieval is useful but not absolutely aligned: 47.45 percent top-1
   with only 0.2374 paired cosine.
2. The same prompts rendered in the held Kai font retain only 0.4130 prompt
   cosine and 0.4089 answer-plan cosine relative to the canonical rendering.
3. Original/paraphrase answer-plan cosine is only 0.4558 despite 73.33 percent
   paraphrase top-5 retrieval.
4. The learned answer plan is almost the prompt state itself: their mean cosine
   is 0.9972. V37 constrains the transition to

   `normalize(s + alpha d)`, with `alpha < 0.5`; the learned alpha is 0.07595.

This is not principally a target-availability problem. On the same development
pairs, exact detached BGE prompt targets retrieve answer targets at 82.65
percent top-1, 92.86 percent top-5, and 0.8723 MRR. A train-only orthogonal
Procrustes map raises that target-space ceiling to 86.22 percent top-1 while
preserving effective rank. The missing mechanism is stable image-to-semantic
alignment plus a plan head that can actually depart from the prompt state.

## Relevant external work

External work is useful and must be attributed exactly.

- [PIXEL](https://arxiv.org/abs/2207.06991) establishes that language can be
  modeled from rendered pixels and transfers to unseen scripts without a fixed
  vocabulary.
- [Text Rendering Strategies for Pixel Language
  Models](https://arxiv.org/abs/2311.00522) shows that rendering structure
  changes both quality and anisotropy, and that a 22M-parameter structured
  renderer can match a larger continuous-rendering model.
- [PixelGPT](https://arxiv.org/abs/2404.10710) demonstrates autoregressive
  next-patch pretraining on visual text and uses 16-by-16 visual patches.
- [SPIRAL](https://arxiv.org/abs/2608.02109) identifies cross-path
  inconsistency: identical language presented through text and rendered-image
  paths diverges because the visual encoder retains font and layout cues. Its
  paired-path alignment is training-only and adds no inference-time teacher.

V38 does not claim these contributions. It adapts the shared empirical lesson
to a smaller, stricter setting: both deployable input and deployable output are
continuous image-derived states, and the final student has no text path.

## V38 mechanism

### 1. Deliberately paired visual paths

Every training item produces two renderings of the exact prompt and two of the
exact answer. The two paths must use different font families, not merely
independent random seeds. One path is a clean anchor; the other varies font,
size, origin, blur, contrast, and noise. A third prompt path uses a validated
paraphrase when one exists and otherwise another distinct rendering.

For image encoder `f_theta`, prompt target `u`, and two rendering operators
`r_a`, `r_b`, V38 directly minimizes

```text
L_path = (1 - cos(f_theta(r_a(x)), u))
       + (1 - cos(f_theta(r_b(x)), u))
       + lambda_inv * (1 - cos(f_theta(r_a(x)), f_theta(r_b(x)))).
```

The target term receives materially more weight than in V37. Retrieval alone
is insufficient because relative ranking can improve while the state remains
far from the detached semantic direction.

### 2. Direct answer projection

V38 removes the angle-limited residual planner. Shared visual patch states feed
two normalized heads:

```text
s = normalize(P_prompt(h))
z = normalize(P_answer(h))
```

`s` predicts the detached prompt state and `z` predicts the detached answer
state. This remains candidate-free at runtime. It is intentionally simpler
than a recurrent planner: V38 first has to prove that a visual prompt can bind
to a stable answer-level state before a raster writer or longer state dynamics
are justified.

An optional fixed orthogonal target-space map may initialize the answer head,
but no target matrix is deployed. The model state itself contains the learned
mapping.

### 3. Semantically nearest negatives

V37 samples 512 candidates approximately uniformly, so many updates omit the
nearest wrong answer. V38 computes detached nearest-answer neighbors from the
pinned train target bank and injects them into each training candidate set.
The margin objective therefore distinguishes the correct answer from plausible
semantic alternatives rather than mostly easy random rows.

### 4. Qwen-assisted paraphrase views

The local `qwen3:4b-q8_0` model may prepare up to 1,024 training paraphrases.
It is an Apache-2.0 external data-preparation tool, not part of ILM. Selection
must:

- use only V37-eligible train records;
- exclude every source identifier in the fixed paraphrase audit manifest;
- be deterministic and resumable;
- retain only concise outputs that fit every V38 training font;
- require pinned BGE-M3 cosine of at least 0.82 to the original prompt;
- reject answer-like or exact-copy outputs; and
- record Qwen, BGE, Ollama, source, and output hashes.

The generated strings exist only in the offline renderer. Student batches
contain pixels, masks, target tensors, and numeric bank indices.

### 5. Length and collapse controls

Length remains a visual quantity computed from clean pre-augmentation answer
geometry. V38 increases its loss weight and predicts it from shared visual
features rather than only a normalized semantic direction. Variance and
covariance controls remain, but they cannot dominate direct target alignment.

## Falsifiable boundary

V38 is useful only if one fixed development run improves the complete V37
failure set without opening sealed data:

- canonical prompt and answer retrieval and paired cosine;
- canonical-to-held-font state consistency;
- original-to-paraphrase state consistency;
- correct-versus-shuffled/blank margins;
- answer-plan effective rank and clean visual-length MAE;
- all finite, source-hash, parameter-cap, update, and VRAM conditions.

The exact thresholds, data hashes, update count, model initialization, and
primary EMA/raw route will be frozen in a separate V38 protocol after the data
builder, implementation, and smoke tests are complete but before evidence
training. Failing that protocol keeps sealed evaluation and the raster writer
closed.

## What V38 cannot prove

Even a complete V38 semantic pass would not prove language generation, broad
knowledge, historical-form reasoning, superiority to token models, or Qwen
parity. It would authorize a separately gated image writer experiment. The
long-term ILM succeeds only when independent image input produces correct,
readable answer images, including writing outside ordinary computer encodings.
