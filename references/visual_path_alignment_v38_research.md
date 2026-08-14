# Visual Path Alignment V38: Research Decision

Status: frozen 8,000-update evidence run and EMA/raw development evaluations
complete; decision `not-qualified`; sealed split and raster renderer closed.

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
- reject answer-like or exact-copy outputs;
- pass a separately pinned `qwen3:8b-q8_0` instruction-versus-answer judge that
  decomposes operation preservation, instruction form, condition preservation,
  and task execution into separately recorded decisions; reject unless every
  sub-decision passes; apply a deterministic operation-retention gate before
  that judge for high-risk fill, edit, translation, and calculation tasks;
- pass a second, independently pinned
  `qwen3:30b-a3b-instruct-2507-q4_K_M` adjudicator whose relation-enum protocol
  separately checks operation, quantity and unit, category scope, named input,
  output form and style, request form, and task execution; apply exact
  normalized numeric-constraint preservation before this stage; require a
  second adversarial-falsification prompt and independent seed stream to agree
  before accepting a row; and
- record Qwen, BGE, Ollama, source, and output hashes.

The generated strings exist only in the offline renderer. Student batches
contain pixels, masks, target tensors, and numeric bank indices.

### Prepared paraphrase evidence

The production builder completed on 2026-08-14 with a deterministic prefix of
2,600 eligible non-holdout records:

- 1,857 passed fit, copy, length, BGE-M3 cosine, and answer-proximity filters;
- 1,543 passed the pinned 8B instruction/operation judge;
- 1,322 passed normalized-number equality and the first pinned 30B sparse
  relation adjudication;
- 1,285 passed the independent adversarial confirmation prompt and seed stream;
- the first 1,024 unique source IDs and unique paraphrase strings form the
  bounded training view; and
- all four model-backed stages completed with zero generation, schema, or
  serving errors.

The final manifest is
`data/teacher/visual_path_alignment_paraphrases_v38.jsonl`, SHA-256
`25db6abd4eb266a2ae05b5d8b8e9cf23caa9c523f61e89b08cc52e542fc2a68b`.
Its receipt records all source, font, model, protocol, candidate-journal,
judgment-journal, adjudication-journal, and confirmation-journal hashes. The
audit found 1,024 unique identifiers, 1,024 unique paraphrases, no overlap with
the 188-source fixed holdout, no source-hash mismatch, no exact prompt or answer
copy, and no failed operation, numeric, judgment, adjudication, or confirmation
link. Qwen and BGE remain offline preparation tools with no student runtime
dependency.

### 5. Length and collapse controls

Length remains a visual quantity computed from clean pre-augmentation answer
geometry. V38 increases its loss weight and predicts it from shared visual
features rather than only a normalized semantic direction. Variance and
covariance controls remain, but they cannot dominate direct target alignment.

## Pre-evidence engineering validation

The architecture, schedule, objective, evaluator, and 39 gate conditions were
committed before two explicitly exploratory production-model runs. A 30-update
run established strict V37 initialization, train-only answer-map construction,
full checkpoint and resume state, raw and all-parameter EMA evaluation, and the
checkpoint tensor boundary. A second 500-update run used the production batch
geometry of 64 records, 512 candidates, and 16 deterministic nearest negatives.

The 500-update run completed in 501.05 seconds on GPU 0 with 3,188,168,704 bytes
peak allocated VRAM. Its raw development route improved over V37 in the intended
directions:

- prompt paired cosine: 0.2374 to 0.3298;
- prompt-derived answer paired cosine: 0.1634 to 0.2393;
- transition-direction cosine: 0.3450, while prompt-answer cosine fell to
  0.7493 rather than remaining near identity;
- canonical-to-held prompt/answer cosine: 0.7465/0.7541, compared with
  0.4130/0.4089 in V37;
- original-to-paraphrase prompt/answer cosine: 0.5259/0.5535; and
- clean visual-length MAE: 3.5816.

Answer top-1 remained only 0.2296 after 500 updates, so the pilot is not a
qualification result and cannot justify opening sealed data or a raster writer.
It shows that the full image-only path is trainable and that the principal V37
failure modes move in the intended direction. The frozen 8,000-update run is
required for the actual decision.

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
primary EMA/raw route were frozen in
`references/visual_path_alignment_v38_protocol.md` before evidence training.
Its SHA-256 is
`cc7112e3f04e7fab622652d7da21955a5b632c5008e51f30182f3bcf2094c0c5`.
Failing that protocol keeps sealed evaluation and the raster writer closed.

## Completed frozen result

The production run completed all 8,000 BF16 updates in 5,982.31 seconds
(99.71 minutes) on GPU 0 of one RTX 4090 D. The 90,753,281-parameter model
used 3,188,168,704 bytes (2.969 GiB) peak allocated CUDA memory. All training,
checkpoint, EMA, raw-weight, and evaluation routes were finite. The checkpoint
SHA-256 is
`eea1595107455977bc7ffb96dde3d4cda733186f628f46ab4fb789e83802fcde`.

The primary EMA development route reads prompt images materially better than
V37: top-1/top-5/MRR is 60.71/86.73/72.36 percent and paired cosine is 0.3789.
Canonical-to-held-font prompt consistency rises from V37's 0.4130 to 0.7925.
The learned answer map also ceases to be near identity: prompt-answer cosine
falls from 0.9972 to 0.5769, transition-direction cosine reaches 0.3401, and
answer-state effective rank reaches 49.02.

The answer relation remains below the frozen absolute gate. Prompt-conditioned
answer top-1/top-5/MRR is 21.94/49.49/34.60 percent and paired answer cosine is
0.2441. Held-font answer consistency is 0.7323, original-to-paraphrase answer
consistency is 0.4908, counterfactual assignment is 0.898 against a 0.900
threshold, and visual-length MAE is 3.370 against a maximum of 3. Raw weights
are materially the same, so EMA lag does not explain the result.

The EMA route passes 25/39 conjunctive conditions and is `not-qualified`.
Zero sealed rows were rendered and no raster renderer is authorized. Retrieval
is an evaluator-only probe of continuous states; V38 emits neither answer
pixels nor generated language.

The measured result narrows the next hypothesis. Paired visual paths repair a
large part of the reading and font-invariance problem, but the 5,822-pair
curriculum is seen roughly 88 times and a single answer vector does not
generalize the relation. The next bounded proof should increase deduplicated
instruction and relation diversity, add held-operation and held-composition
tests, and replace one answer state with a small ordered set of conditional
continuous states or recurrent dynamics. External visual and language systems
remain acceptable for initialization or offline data preparation when
provenance, licensing, and deployed absence are exact.

## What V38 cannot prove

Even a complete V38 semantic pass would not prove language generation, broad
knowledge, historical-form reasoning, superiority to token models, or Qwen
parity. It would authorize a separately gated image writer experiment. The
long-term ILM succeeds only when independent image input produces correct,
readable answer images, including writing outside ordinary computer encodings.
