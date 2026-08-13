# Visual Semantic Distillation V37 Protocol

Date frozen: 2026-08-14

Status: preregistered before V37 implementation, smoke training, production
target construction, production optimization, or V37 development measurement

## Decision question

Can a sub-100M-parameter deployable model consume only a rendered Chinese
prompt and clean visual patch mask, then produce a noncollapsed continuous
semantic state and answer plan that pass held-out answer retrieval, visual
reading, prompt-control, counterfactual, font, paraphrase, and resource gates?

V37 is a semantic substrate proof. It is not autonomous raster generation.

## Pinned external artifacts

### Pixel-Linguist-v0 initialization

- repository: `Pixel-Linguist/Pixel-Linguist-v0`;
- Hugging Face revision:
  `086b70818b2241e81b0de131aa5debe982af7a54`;
- weight file SHA-256:
  `84c1bfbeada57e7e70164811a201a116ad18c22df69beb762fdbb853f8e02676`;
- config SHA-256:
  `b61018a997aa030f41571615641204cf40bd7c623f25dfc129a49ffe1f571b97`;
- selected state: all 198 `vit.*` tensors; and
- ignored state: four `pooler.*` training-head tensors, as prescribed by the
  upstream representation-inference route.

The upstream repository and checkpoint state no license. The artifact may be
used only as a pinned local-research initialization. It and derived checkpoints
must not be committed or redistributed.

### BGE-M3 offline semantic teacher

- model tag: `bge-m3:latest`;
- endpoint: exactly `http://127.0.0.1:11434/api/embed`;
- Ollama manifest SHA-256:
  `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`;
- model layer digest and SHA-256:
  `daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c`;
- model layer bytes: `1,157,671,200`;
- expected embedding width: 1,024; and
- official upstream model-card license: MIT.

The target builder must reject a remote endpoint, credentials, query string,
wrong endpoint path, changed manifest, changed model layer, malformed response,
nonfinite embedding, wrong width, or server version below `0.32.6`. The mutable
tag alone is never sufficient provenance.

BGE-M3 may receive source strings only inside the offline target builder. It is
unloaded after target construction and may not enter the student process,
checkpoint, evaluator inference path, or deployed API.

## Pinned data and split

| Purpose | File | SHA-256 | Rights/use |
|---|---|---|---|
| prompt-answer supervision | `data/raw/alpaca_zh.json` | `6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903` | CC BY-NC 4.0; local research |
| wording control | `data/teacher/folio_paraphrases_zh_holdout.jsonl` | `132a8a257d31be3cf607d0313bf042dafaee7f2df08440bf7e012a502ed6c02f` | evaluator-only generated wording; source ID retained |

Use the existing deterministic instruction partition. The raw eligible
population before V37 rendering is 6,231 records: 5,847 train, 197 development,
and 187 sealed. At maximum train font size and origin, the fixed V37 font set
admits 5,822 train records. The development route admits the same 196 records
used by V36; `alpaca-zh:14816` overflows and is excluded. The implementation
must write and hash exact selected identifiers before optimization.

The fixed paraphrase file contains 30 usable nonsealed rows and one sealed row.
As in V36, exclude the sealed-source row before selecting, rendering,
embedding, or scoring it. Because 27 of 30 nonsealed source records belong to
train, paraphrases test wording stability and are not unseen-answer evidence.

The 187 sealed records must not be rendered, embedded, inspected, or scored
unless the complete EMA development gate passes.

## Target construction

Build train targets first. For every selected train record, request raw,
L2-normalized 1,024-dimensional BGE vectors for the exact project-formatted
prompt and answer strings. Let `mu` be the arithmetic mean over all raw train
prompt and answer vectors. Store:

```text
prompt_target = normalize(raw_prompt - mu)
answer_target = normalize(raw_answer - mu)
```

The development target builder must load the immutable train-bank `mu`; it may
not compute a development mean. Store identifiers, transformed vectors, clean
visual answer lengths, `mu`, source hashes, teacher hashes, request counts,
teacher direct ceilings, finite checks, and the transformation receipt in the
offline bank. Store no token IDs.

The train bank and development bank are forbidden checkpoint contents. The
deployed model does not need `mu` because it is trained to emit the transformed
space directly.

The exact nonsealed pre-protocol diagnostic is frozen as a target-space
sanity bound: train-joint-centered development prompt-to-answer top-1 at least
0.80, top-5 at least 0.90, MRR at least 0.85, cyclic margin at least 0.50, and
development answer effective rank at least 70. A production bank outside these
bounds must stop before student training.

## Fixed rendering and clean masks

- canvas: RGB white strip, 16 by 1,024 pixels;
- patch: 16 by 16 pixels, 64 positions;
- layout: contiguous left-to-right visible writing; no character cells,
  segmentation labels, Unicode normalization, crop, or truncation;
- train font size: deterministic uniform integer in `[8,11]`;
- evaluation font size: 10;
- train origin: deterministic uniform integer in `[0,15]`;
- evaluation origin: 0;
- train fonts: Noto Sans CJK Regular, Medium, and Bold; Droid Sans Fallback
  Full; and AR PL SungtiL GB;
- canonical development font: Noto Serif CJK Regular;
- held development font: AR PL KaitiM GB;
- sealed font: Noto Serif CJK Light;
- train pixel augmentation: deterministic mild blur, contrast, and Gaussian
  noise after rendering;
- evaluation augmentation: none; and
- overflow: reject the complete record before target construction.

Patch occupancy must be computed from the clean black-on-white raster before
any blur, contrast, or noise. A train unit test must prove that changing only
augmentation leaves the mask and visual length invariant for the same text,
font, size, and origin. Blank pixels created by augmentation can never activate
patches.

Each train item contains two independently rendered prompt views and two
independently rendered answer views. Strings and metadata exist in the dataset
worker but the tensor boundary presented to the trainable model contains only
the four raster tensors and four clean masks. Identifiers may index detached
targets in trainer orchestration; they are not model inputs.

## Fixed deployable architecture

### Shared visual reader

- ViT patch size: 16;
- input: `[B,3,16,1024]` plus `[B,64]` clean mask;
- hidden width: 768;
- layers: 12;
- heads: 12;
- MLP width: 3,072;
- positions: 64 patches plus CLS;
- initialization: pinned Pixel-Linguist `vit.*` state;
- pooling: clean-mask mean over 64 contextual patch states, excluding CLS; and
- input normalization: clamp to `[0,1]`, then map to `[-1,1]`.

### Semantic and plan heads

- semantic head:
  `LayerNorm(768)-Linear(768,1536)-GELU-Dropout(0.05)-Linear(1536,1024)`;
- semantic state: L2-normalized 1,024-vector;
- residual plan head:
  `LayerNorm(1024)-Linear(1024,512)-SiLU-Linear(512,1024)`;
- residual scale: one learned scalar constrained to `(0,0.5)`, initialized to
  approximately 0.05;
- answer plan: normalized semantic state plus scaled normalized residual; and
- length head: `Linear(1024,256)-SiLU-Linear(256,1)`, sigmoid-scaled to
  `[0,64]` clean patches.

The deployable model must remain below 100,000,000 parameters. It contains no
decoder, writer, vocabulary, tokenizer, embedding table, quantizer, visual
codebook, OCR head, candidate bank, database, answer image, target tensor, BGE
component, or network client.

The deployable `generate_plan(prompt_pixels, prompt_mask)` returns semantic
state, answer plan, and predicted clean visual length. Its result must be
identical with network access disabled.

## Fixed objective

For each record, shared model `F` encodes prompt views `x,x'` and answer views
`y,y'`:

```text
p, p' = semantic(F(x)), semantic(F(x'))
a, a' = semantic(F(y)), semantic(F(y'))
z       = answer_plan(F(x))
u       = detached centered prompt target
v       = detached centered answer target
```

For every physical microbatch, deterministic candidate sampling selects 512
unique train-bank targets including every positive. Candidate tensors are
detached. With temperature `tau=0.05`:

```text
NCE(s, q, C) = cross_entropy((s @ C.T) / tau, positive_index)
D(s, q)      = mean(1 - s dot q)
```

Define:

```text
L_prompt = 0.5 * (NCE(p,u,U) + NCE(p',u,U))
           + 0.25 * (D(p,u) + D(p',u))
L_answer = 0.5 * (NCE(a,v,V) + NCE(a',v,V))
           + 0.25 * (D(a,v) + D(a',v))
L_plan   = NCE(z,v,V) + 0.50 * D(z,v)
```

The hard negative is the highest-scoring answer candidate other than the
positive whose teacher cosine with the positive is below 0.85:

```text
L_margin = mean(relu(0.10 - z dot v + z dot v_hard))
```

Within the physical batch, preserve teacher pair geometry:

```text
L_relation = mse(p @ p.T, u @ u.T) + mse(a @ a.T, v @ v.T)
L_view = 0.5 * mean(1 - p dot p') + 0.5 * mean(1 - a dot a')
```

Apply VICReg-style variance and covariance to `sqrt(1024)` times the normalized
states `p`, `a`, and `z`. Variance penalizes per-dimension standard deviation
below one. Covariance is mean squared off-diagonal covariance. Length uses
smooth L1 against the clean answer-patch count. Residual regularization is the
mean squared scaled residual norm.

The frozen total is:

```text
L_total = 0.70 * L_prompt
          + 0.70 * L_answer
          + 1.00 * L_plan
          + 0.50 * L_margin
          + 0.20 * L_relation
          + 0.20 * L_view
          + 0.05 * L_variance
          + 0.005 * L_covariance
          + 0.02 * L_length
          + 0.01 * L_residual
```

No token, character, Unicode, OCR, next-patch, raw-pixel reconstruction, or
answer-candidate prediction head is permitted.

## Fixed training route

Random seed: `20_263_700`. Device: CUDA device 0 on one RTX 4090. Precision:
BF16. Optimizer: fused AdamW with betas `(0.9,0.95)`, weight decay 0.05,
gradient clipping at 1.0, 200-update warmup per stage, cosine decay to 10
percent of peak, and EMA decay 0.999.

| Stage | Updates | Reader | Head LR | Reader LR | Effective batch |
|---|---:|---|---:|---:|---:|
| projection warmup | 500 | frozen | 3e-4 | 0 | 64 |
| full visual adaptation | 7,500 | all layers trainable | 1e-4 | 1e-5 | 64 |

Default physical batch is 8 records, concatenated as 32 visual views for one
reader forward. Physical batch may be reduced only for memory. Gradient
accumulation must preserve 64 records per optimizer update; each microbatch
computes its own 512-candidate detached contrastive losses so no full-effective-
batch graph is retained.

Save atomically every 500 updates. Exact resume must restore model, optimizer,
scaler, EMA, stage/update counters, deterministic data position, candidate
sampling position, and Python/NumPy/Torch RNG states. Evidence training must
finish all 8,000 updates and remain below 20 GiB peak allocated VRAM.

## Development evaluation

EMA is the primary route. Raw weights are a diagnostic and cannot replace a
failed EMA gate. Infer all student conditions before constructing or loading
evaluator candidate matrices into the scoring step.

### Reading metrics

Compare canonical development prompt semantic states with development prompt
targets:

- paired cosine;
- top-1/top-5 retrieval among all 196 prompt targets; and
- centered effective rank.

This isolates visual reading from prompt-to-answer relation transfer.

### Answer-plan metrics

Compare prompt answer plans with all 196 development answer targets:

- top-1/top-5 retrieval and mean reciprocal rank;
- paired correct cosine;
- correct-minus-cyclic cosine margin;
- fraction where correct beats cyclic; and
- centered effective rank.

### Independent controls

Run fresh inference for:

- correct canonical prompt;
- cyclically shuffled prompt with original labels;
- all-white prompt and all-zero mask;
- final visual quarter only;
- the same prompt in the held AR PL Kaiti font; and
- the fixed 30 nonsealed paraphrases.

Report retrieval, paired cosine, length, output variance/effective rank, held-
font plan cosine, original-paraphrase plan cosine, and paraphrase retrieval
against the 30 original answer targets.

Construct nearest-clean-length counterfactual pairs with different answers and
report the assignment rate for

```text
z_a dot v_a + z_b dot v_b > z_a dot v_b + z_b dot v_a.
```

### Matched baselines and integrity

- exact centered BGE prompt-to-answer ceiling;
- untrained V37 heads on the same pinned Pixel-Linguist reader and seed;
- direct Pixel-Linguist masked-mean prompt-to-answer result;
- cyclic prompt shuffle;
- blank prompt; and
- answer-frequency baseline.

Report checkpoint/source/target hashes, parameter counts, elapsed inference,
peak memory, target and output effective ranks, all finite checks, and a runtime
boundary receipt.

## Frozen semantic gate

`semantic-distillation-qualified` requires every condition below:

- exact protocol, source, data, Pixel-Linguist, BGE manifest, and BGE model
  layer hashes;
- finite targets, model, optimizer, EMA, predictions, and metrics;
- target-space sanity bounds all pass;
- deployable parameter count below 100M and no forbidden boundary field;
- 8,000 completed updates on one RTX 4090;
- peak allocated VRAM below 20 GiB;
- prompt-state paired cosine at least 0.70;
- prompt-target top-1 at least 0.25 and top-5 at least 0.60;
- answer-plan top-1 at least 0.30, top-5 at least 0.60, and MRR at least 0.40;
- answer-plan top-1 at least 10 times untrained top-1 and at least 0.20 higher
  in absolute terms;
- paired answer cosine at least 0.35;
- correct-minus-cyclic margin at least 0.20;
- correct beats cyclic on at least 85 percent of examples;
- correct answer cosine exceeds shuffled by at least 0.20 and blank by at
  least 0.25;
- counterfactual assignment at least 0.85;
- held-font plan cosine at least 0.85;
- paraphrase top-5 at least 0.50;
- original-paraphrase plan cosine at least 0.75;
- answer-plan centered effective rank at least 32 and at least 40 percent of
  the development answer-target effective rank; and
- clean visual-length mean absolute error at most 3 patches.

The exact status vocabulary is:

- `not-qualified`; or
- `semantic-distillation-qualified`.

No average, subset, or favorable control may override a failed conjunctive
condition. Retrieval remains evaluator-only and may not be described as
generated language.

## Sealed and renderer rules

Do not open the sealed split unless the complete EMA development report is
`semantic-distillation-qualified`. If it passes, run exactly one sealed
evaluation with the already selected EMA checkpoint and no tuning. Every
applicable absolute gate must pass and each primary sealed metric/control margin
must retain at least 90 percent of its development value.

V37-R implementation, target preparation, or training is forbidden until both
development and sealed semantic gates pass. A later renderer must be separately
preregistered, initially freeze the qualified V37 planner, accept continuous
plans without candidates or text, emit raster images, and pass generated-pixel
causal and readability controls.

## Required artifacts

- immutable research and protocol documents;
- target-bank receipts with exact teacher, source, split, transform, mean, and
  target-space diagnostics;
- clean-mask invariance tests;
- strict Pixel-Linguist mapping and BGE provenance receipts;
- model/data/runtime boundary receipts;
- atomic resumable checkpoints and standalone EMA model with no targets;
- JSONL training metrics and stage summaries;
- complete EMA and raw development reports;
- baseline, control, counterfactual, font, paraphrase, and rank measurements;
- elapsed time, examples consumed, parameter count, and peak VRAM; and
- a result note that separates teacher ceiling, student evidence, and future
  renderer claims.

## Stop rules

Stop cleanly and preserve evidence if:

- any pinned hash, split count, target sanity bound, or runtime boundary changes;
- BGE is remote, mutable without matching content hashes, malformed, or enters
  student inference;
- clean-mask augmentation invariance fails;
- any loss, gradient, parameter, target, prediction, or metric is nonfinite;
- a checkpoint contains target-bank tensors, teacher mean, BGE, strings,
  candidates, OCR, token/Unicode/character IDs, or an external client;
- allocated VRAM reaches 20 GiB;
- a completed stage has answer-plan effective rank below 8 on a deterministic
  64-record train probe; or
- SIGINT or SIGTERM is received.
