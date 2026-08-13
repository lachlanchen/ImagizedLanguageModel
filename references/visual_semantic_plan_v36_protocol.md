# Visual Semantic Plan V36 Protocol

Date frozen: 2026-08-14

Status: originally preregistered before V36 implementation, smoke training,
development measurement, or sealed evaluation; one data-routing amendment was
recorded after evaluator smoke and before any production target construction or
evidence training

## Pre-evidence amendment

The fixed paraphrase manifest contains 31 rows, but its source identifiers do
not all belong to the deterministic instruction development partition: 27 map
to train, three to development, and one to sealed. This was discovered by the
complete evaluator smoke. The earlier phrase "31 fixed development
paraphrases" was therefore factually incorrect.

V36 evaluates the 30 nonsealed rows as a separate fixed wording-shift
diagnostic. The sealed-source row is excluded before its source prompt or answer
is selected, rendered, embedded, or scored. The evaluator renders the 30
original prompts, paraphrased prompts, and original answers in the development
font. It first infers both prompt routes, then constructs a 30-answer visual
candidate bank with the frozen teacher. It reports paraphrase top-5 retrieval
against those 30 distinct original answers and paired original-paraphrase plan
cosine. This diagnostic tests wording stability; because 27 source
prompt-answer pairs occur in training, it is not evidence of unseen-answer
generalization. The primary 196-record development evaluation remains the
generalization test. No metric threshold, model, training route, primary split,
or sealed rule changed.

## Decision question

Can a sub-100M-parameter model consume only a rendered Chinese prompt and a
visual patch mask at runtime, then predict a continuous answer-level semantic
plan that identifies the correct held-out answer image under prompt shuffle,
blank, counterfactual, wording, and font controls?

V36-P is a semantic planning proof. It is not autonomous answer generation.

## Pinned external foundation

### Pixel-Linguist-v0

- repository: `Pixel-Linguist/Pixel-Linguist-v0`;
- Hugging Face revision:
  `086b70818b2241e81b0de131aa5debe982af7a54`;
- weight file: `pytorch_model.bin`;
- weight SHA-256:
  `84c1bfbeada57e7e70164811a201a116ad18c22df69beb762fdbb853f8e02676`;
- config SHA-256:
  `b61018a997aa030f41571615641204cf40bd7c623f25dfc129a49ffe1f571b97`;
- selected state: all 198 `vit.*` tensors; and
- ignored state: the four `pooler.*` training-head tensors.

The upstream repository and model card state no software or weight license as
of the freeze date. The checkpoint is permitted only as a pinned local research
foundation. It must stay in the Hugging Face cache, must not be committed or
redistributed, and makes V36 checkpoints non-redistributable unless the owner
later supplies compatible permission.

The implementation may not copy upstream source. It maps the released
`vit.*` state into the installed Transformers `ViTModel`, uses white padded
patches, removes CLS before pooling, applies the project mask to active-patch
mean pooling, and L2-normalizes the result. Mapping must be strict after the
documented prefix removal.

### Data

| Purpose | File | SHA-256 | Rights/use |
|---|---|---|---|
| supervised prompt-answer plans | `data/raw/alpaca_zh.json` | `6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903` | CC BY-NC 4.0; research only |
| wording-shift diagnostic | `data/teacher/folio_paraphrases_zh_holdout.jsonl` | `132a8a257d31be3cf607d0313bf042dafaee7f2df08440bf7e012a502ed6c02f` | evaluator-only teacher wording; source ID retained |

The raw eligible instruction population before V36 rendering is 6,231:
5,847 train, 197 development, and 187 sealed under the existing deterministic
instruction partition. The implementation must write exact post-render counts
before optimization. Overflow records are skipped deterministically and may
not be truncated.

The sealed partition must not be rendered, embedded, inspected, or scored
before the complete development report permits opening it.

## Runtime boundary

The deployable `generate_plan` entry point accepts only:

- `prompt_pixels`: floating rasters in `[0,1]`, shape `[B,3,16,1024]`;
- `prompt_mask`: floating visual patch masks, shape `[B,64]`; and
- numerical inference controls.

It returns:

- five normalized continuous plan vectors, shape `[B,5,768]`; and
- one nonnegative visual answer-length estimate per example.

The deployable checkpoint contains the prompt reader and plan head. It must not
contain or call the answer teacher, answer pixels, candidate embeddings,
strings, token IDs, Unicode IDs, character IDs, OCR, a tokenizer, a codebook, a
database, a local LLM, a network service, or a retrieval index.

The evaluator may compare plans with answer-image embeddings after inference.
That candidate bank is evaluator-only and cannot appear in the model boundary
receipt or checkpoint.

## Fixed rendering

- canvas: RGB white strip, 16 by 1,024 pixels;
- patch: 16 by 16 pixels, 64 positions;
- text: black, left-to-right visible rendering with no Unicode normalization;
- font size: 11 pixels;
- train fonts: Noto Sans CJK Regular and Medium;
- development font: Noto Serif CJK Regular;
- sealed font: Noto Serif CJK Light;
- train origin: deterministic uniform integer in `[0,15]`;
- evaluation origin: 0;
- train augmentation: mild deterministic blur, contrast, and Gaussian noise;
- evaluation augmentation: none; and
- overflow: reject the complete record, never crop or truncate.

The answer target is one contiguous strip. Four local target views are made by
splitting its 64 visual patches into fixed consecutive groups of 16, copying
each group to the start of an otherwise white 64-patch canvas, and carrying the
corresponding visual mask. This operation uses pixels and spatial indices only.

Strings and metadata may exist in the data-preparation layer. The student batch
contains only prompt pixels, prompt mask, answer pixels, answer mask, and a
floating visual-length target. No symbolic value enters the model.

## Fixed architecture

### Prompt reader

- ViT patch size: 16;
- hidden width: 768;
- layers: 12;
- heads: 12;
- MLP width: 3,072;
- positions: 64 patches plus CLS;
- initialization: pinned Pixel-Linguist `vit.*` state; and
- pooling: none inside the reader; contextual patch states feed the planner.

### Plan head

- memory projection: `Linear(768,384)` plus LayerNorm;
- learned plan queries: five by 384;
- cross-attention decoder: three layers, width 384, six heads, MLP width 1,536,
  pre-norm GELU, dropout 0.05;
- plan projection: shared `Linear(384,768)` followed by per-slot affine
  LayerNorm and L2 normalization;
- length head: `Linear(384,192)-SiLU-Linear(192,1)` with softplus output; and
- vocabulary, embedding table, quantizer, codebook, OCR head, and raster
  decoder: none.

The instantiated deployable model must contain fewer than 100,000,000
parameters. The answer teacher is a separate frozen training object and is not
counted as a deployable parameter.

## Teacher targets

The frozen answer teacher receives only answer raster views and masks. It
returns detached normalized active-patch means:

```
q_global = normalize(masked_mean(T(answer_pixels)))
q_chunk[k] = normalize(masked_mean(T(answer_chunk[k])))
```

Empty chunks are masked out of every chunk loss. The teacher receives no prompt
and no record identifier. The student receives no answer pixels inside
`generate_plan`.

## Fixed objectives

For normalized predicted global plan `z_i`, target `q_i`, batch size `B`, fixed
temperature `tau=0.07`, and hardest valid negative `h(i)`:

```
S_ij = z_i dot q_j / tau
L_nce = 0.5 * (cross_entropy(S, arange(B))
               + cross_entropy(S.T, arange(B)))
L_cos = mean(1 - z_i dot q_i)
L_margin = mean(relu(0.10 - z_i dot q_i + z_i dot q_h(i)))
```

`h(i)` is the highest-scoring off-diagonal target whose teacher cosine with
`q_i` is below 0.85. If none exists, use the deterministic cyclic target. This
prevents known near-paraphrases from being forced apart while retaining a hard
counterfactual.

Apply the same cosine and one-way contrastive terms to active spatial chunks
at matching slots. For two independently rendered prompt views `a` and `b`,
use `L_view = mean(1 - z_global_a dot z_global_b)`. Visual length is the number
of active answer patches and uses smooth L1 loss.

The total objective is fixed as:

```
L_total = L_nce
          + 0.50 * L_cos
          + 0.50 * L_margin
          + 0.35 * L_chunk_nce
          + 0.20 * L_chunk_cos
          + 0.20 * L_view
          + 0.05 * L_length
```

No raw-pixel, next-patch, token, character, OCR, or reconstruction loss is
permitted in V36-P.

## Fixed training route

Random seed: `20_263_600`. Device: one RTX 4090 on CUDA device 0. Precision:
BF16. Optimizer: fused AdamW with `beta=(0.9,0.95)`, weight decay 0.05,
gradient clipping at 1.0, 200-update linear warmup, cosine decay to 10 percent
of peak, and EMA 0.999.

| Stage | Updates | Reader state | Head LR | Reader LR | Effective batch |
|---|---:|---|---:|---:|---:|
| plan alignment | 2,000 | frozen | 3e-4 | 0 | 128 |
| semantic adaptation | 4,000 | final two blocks plus final norm | 8e-5 | 8e-6 | 64 |

Physical batch size may be reduced only to avoid out-of-memory errors;
gradient accumulation must preserve the effective batch. Exact batch values,
examples consumed, elapsed time, and peak allocated VRAM are part of the run
receipt.

The answer teacher is always frozen and in evaluation mode. Checkpoints must be
saved atomically at least every 500 updates and support exact resume with model,
EMA, optimizer, scheduler counters, data counters, and all RNG states.

Evidence training must remain below 20 GiB peak allocated VRAM. No second heavy
process may be launched for this project while training is active.

## Development evaluation

EMA is the primary weight route. Raw weights are diagnostic. Every metric uses
the complete post-render development set and a candidate bank built only after
student inference.

### Primary retrieval metrics

- top-1 and top-5 answer-image retrieval;
- mean reciprocal rank;
- paired correct-answer cosine;
- correct minus cyclic-answer cosine margin; and
- fraction of pairs where correct cosine exceeds cyclic cosine.

### Prompt controls

Repeat inference with no shared hidden state under:

- correct prompt;
- deterministic cyclically shuffled prompt;
- all-white prompt with an all-zero patch mask;
- final visual quarter of the prompt only; and
- same prompt in the train sans font.

Report retrieval metrics, correct-target cosine, plan cosine to the correct
route, length error, and plan variance. A blank or shuffled prompt may not reuse
the original prompt memory.

### Counterfactual assignment

Construct evaluator-only pairs with different answers and nearest visual answer
lengths. For plans `z_a,z_b` and targets `q_a,q_b`, assignment passes per pair
when

```
z_a dot q_a + z_b dot q_b > z_a dot q_b + z_b dot q_a.
```

Report the mean assignment rate and bootstrap 95 percent interval.

### Wording and font controls

- evaluate the 30 fixed nonsealed paraphrases against a separate bank of their
  original answer-image targets;
- compare original and paraphrase plan cosine;
- rerender every development prompt in the held train font and compare plan
  cosine; and
- rerender every answer in the alternate font to measure teacher-target
  stability.

### Matched baselines

- untrained plan head on the same frozen reader;
- complete cyclic prompt shuffle;
- all-white prompt;
- answer-frequency baseline; and
- direct Pixel-Linguist prompt-to-answer similarity without the learned plan.

The random or untrained head is initialized from the frozen seed and evaluated
once. It is never selected after seeing student results.

## Frozen semantic-plan gate

`semantic-plan-qualified` requires every condition below:

- finite source, model, optimizer, EMA, targets, predictions, and metrics;
- exact external checkpoint and data hashes;
- deployable parameter count below 100M;
- no forbidden runtime or checkpoint boundary field;
- 6,000 completed optimization updates;
- peak allocated VRAM below 20 GiB on one RTX 4090;
- correct-prompt top-1 retrieval at least 0.08 and at least four times the
  untrained-head top-1 value;
- correct-prompt top-5 retrieval at least 0.25;
- mean reciprocal rank at least 0.15;
- correct minus cyclic-answer cosine margin at least 0.05;
- correct beats cyclic answer on at least 70 percent of examples;
- correct-target cosine exceeds shuffled-prompt by at least 0.03 and blank
  prompt by at least 0.05;
- counterfactual assignment rate at least 0.70;
- held-font prompt plan cosine at least 0.85;
- paraphrase top-5 retrieval at least 0.20;
- mean original-paraphrase plan cosine at least 0.75;
- answer-teacher cross-font cosine at least 0.80; and
- visual answer-length mean absolute error at most 4.0 patches.

The exact status vocabulary is:

- `not-qualified`; or
- `semantic-plan-qualified`.

No subset of gates may be described as qualification. Candidate retrieval is
evaluator-only and may not be described as generated language.

## Sealed rule

Do not render, embed, inspect, or score the 187 raw sealed records until the
complete development evaluator has written its report and every semantic-plan
gate passes. After opening, run exactly one sealed evaluation with the already
selected EMA route and no threshold or architecture changes.

Sealed transfer passes when all absolute gates remain satisfied except the
development-only paraphrase gate, and each primary sealed retrieval or control
margin is at least 90 percent of its development value. A failed sealed result
is reported and not tuned away.

## Renderer opening rule

V36-R implementation or training is forbidden until V36-P writes a complete
development report with `semantic-plan-qualified`. Passing permits a new
preregistered renderer protocol; it does not automatically qualify a writer.

The renderer protocol must keep the qualified planner fixed initially, accept
its continuous plans rather than answer candidates, emit raster images, and
test autonomous output under the same prompt controls.

## Required artifacts

- immutable run receipt with protocol, source, data, external revision, and
  checkpoint hashes;
- strict upstream tensor-mapping receipt;
- preflight record counts and boundary receipt;
- atomic checkpoints and final standalone deployable planner checkpoint;
- JSONL training metrics and stage summaries;
- raw and EMA development reports;
- prediction-target similarity matrices or their hashes;
- prompt-control, counterfactual, paraphrase, font, and baseline reports;
- elapsed time, examples consumed, parameter counts, and peak VRAM;
- explicit no-license/no-redistribution notice for derived checkpoints; and
- result note separating external foundation performance from project-trained
  semantic planning evidence.

## Stop rules

Stop cleanly and preserve evidence if:

- a required source, protocol, data, or checkpoint hash differs;
- strict external tensor mapping fails;
- any loss, gradient, parameter, teacher target, or plan is non-finite;
- the answer teacher changes or enters the deployable state;
- the runtime boundary admits symbolic IDs, OCR, candidates, or external calls;
- allocated VRAM reaches 20 GiB;
- a stage completes with plan variance below `1e-4` in every dimension; or
- SIGINT or SIGTERM is received.
