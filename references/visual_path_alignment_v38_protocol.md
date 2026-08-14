# Visual Path Alignment V38 Protocol

Date frozen: 2026-08-14

Status: frozen after implementation, focused tests, smoke execution, and two
explicitly non-evidence production-model pilots; before the 8,000-update
evidence run or any V38 evidence measurement

## Decision question

Can one sub-100M-parameter model consume only a rendered Chinese prompt and a
clean visual patch mask, then produce noncollapsed continuous prompt and answer
states that remain stable across raster style and wording while preserving the
correct prompt-to-answer semantic transition?

V38 is an image-native semantic-path proof. It is not autonomous raster
generation and cannot by itself establish generated language, historical-form
reasoning, broad knowledge, or parity with a text-token language model.

## Freeze chronology and pilot disclosure

The fixed architecture, two-stage schedule, loss weights, evaluator, and all 39
gate conditions were committed before either production-model pilot:

- implementation commit: `627db50`;
- evaluator and gate commit: `d2fc2b1fb625229e09da067a8d303093145313ba`;
- 30-update exploratory checkpoint SHA-256:
  `41c290b0df9c9dbe7cef45b204fd2a1e468f04b702d6bdd45dfda859f7dfa1be`;
- 500-update exploratory checkpoint SHA-256:
  `e41210a94823071ed6ce18791f7be36a68ad6bd34a4d1bbd28e0fa452d7e046a`.

Both pilots used the nonsealed development split and were labeled
`exploratory`. They established executable initialization, optimization,
checkpoint, evaluator, memory, and metric direction. They are not confirmatory
evidence. The 500-update raw route reached prompt cosine 0.3298, answer cosine
0.2393, transition-direction cosine 0.3450, held-font prompt/answer consistency
0.7465/0.7541, and length MAE 3.5816. These observations do not alter any
threshold, update count, learning rate, loss weight, split, or primary route
below. Any later change requires a new experiment identifier and protocol.

The sealed split has not been rendered, embedded, inspected, or scored for V38.

## Exact external attribution

Strong external work is allowed as attributed training scaffolding. It is not
present in the deployed V38 runtime.

### Pixel-Linguist visual initialization

- repository: `Pixel-Linguist/Pixel-Linguist-v0`;
- revision: `086b70818b2241e81b0de131aa5debe982af7a54`;
- weight SHA-256:
  `84c1bfbeada57e7e70164811a201a116ad18c22df69beb762fdbb853f8e02676`;
- config SHA-256:
  `b61018a997aa030f41571615641204cf40bd7c623f25dfc129a49ffe1f571b97`;
- inherited V37 EMA checkpoint SHA-256:
  `367b0dfd5b54c537aaa4cb41305c6c63c08862921228b706bd1dc37c5c2170f8`;
- copied state: the complete V37 visual reader and semantic head, mapped to the
  V38 reader and prompt head; 204 tensors and 88,453,888 elements; and
- runtime role: initialization only, with no upstream service or model object.

The upstream checkpoint states no license. It and derived checkpoints are
restricted to local research and must not be committed or redistributed.

### BGE-M3 detached semantic teacher

- model tag: `bge-m3:latest`;
- Ollama manifest SHA-256:
  `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`;
- model-layer SHA-256:
  `daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c`;
- layer bytes: 1,157,671,200;
- embedding width: 1,024;
- upstream model-card license: MIT; and
- runtime role: none.

V38 reuses immutable V37 train and development target banks. BGE-M3 is not
loaded by the V38 trainer or student inference. Targets are detached tensors
looked up by orchestration and are forbidden checkpoint contents.

### Qwen paraphrase preparation

The following Apache-2.0 local models prepared or audited training-only wording
views:

| Role | Model | Manifest SHA-256 | Model-layer SHA-256 |
|---|---|---|---|
| candidate generation | `qwen3:4b-q8_0` | `6461746fd6b5a2327ba63d5cd1359af119852d82aa8c981efe948d1868a4dc20` | `fb684cd1056921c526f12a9efbad10c4627e151ecc1e28314fae1c2cce0c2c15` |
| instruction judge | `qwen3:8b-q8_0` | `e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130` | `d87f4a5a2f1a6051d9fac010c12f76f3ba2137b137d413ba8f4d3a3d06b3a25b` |
| constraint adjudication and independent confirmation | `qwen3:30b-a3b-instruct-2507-q4_K_M` | `19e422b0231392335cfc49cfd172de7034bb1aeabb08aa307cce745c60b272fe` | `78b329e716e7e9775973d392cd132b1f1ff1c8287a992887caeb6fd6c56ba9cc` |

Qwen receives strings only in offline preparation. It is absent from training
batches, checkpoints, evaluator student inference, and deployment.

## Pinned data and immutable targets

| Purpose | File | SHA-256 |
|---|---|---|
| prompt-answer source | `data/raw/alpaca_zh.json` | `6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903` |
| fixed wording holdout | `data/teacher/folio_paraphrases_zh_holdout.jsonl` | `132a8a257d31be3cf607d0313bf042dafaee7f2df08440bf7e012a502ed6c02f` |
| audited train paraphrases | `data/teacher/visual_path_alignment_paraphrases_v38.jsonl` | `25db6abd4eb266a2ae05b5d8b8e9cf23caa9c523f61e89b08cc52e542fc2a68b` |
| paraphrase receipt | `data/teacher/visual_path_alignment_paraphrases_v38.receipt.json` | `5ea61639ed5e8234a0b461f9fe772ce5f36068bd3742549f1568b3615cb024e4` |
| train target bank | `artifacts/visual_semantic_distillation_v37_targets/train.pt` | `3cd73f0818d65fd45c7700470cd010e292f359eed5aa3e62859bdf50d301711d` |
| development target bank | `artifacts/visual_semantic_distillation_v37_targets/development.pt` | `6dac8ea8df5afbb1fbe032ab6f1dd8b196ea67ef549475025adbe7bb04706b8f` |
| V37 matched report | `artifacts/visual_semantic_distillation_v37_20260814/development_report_ema_v37.json` | `5f7941b7fa9668e9fa61abfb6b689073c3a1891977b40250b63561ce88857c7c` |

The raw source contains 6,231 eligible records under the deterministic existing
partition. V38 selects exactly 5,822 train records. Development selects exactly
196 records and rejects only `alpaca-zh:14816` for overflow. The evaluator-only
wording set contains exactly 30 nonsealed paraphrases; its sealed-source row is
excluded before rendering or scoring.

The training wording manifest contains exactly 1,024 unique source IDs and
1,024 unique paraphrases. It was selected from 2,600 generated candidates after
BGE filtering, an 8B instruction judge, a 30B relation adjudicator, and an
independent adversarial confirmation. It has no overlap with the 188-source
fixed holdout, and each row stores and verifies the exact source-prompt hash.

The V37 target transform remains fixed:

```text
prompt_target = normalize(raw_prompt_embedding - train_mean)
answer_target = normalize(raw_answer_embedding - train_mean)
```

The development bank uses the train-bank mean. The train and development banks
must share exactly equal teacher means. No development target contributes to
optimization or answer-map initialization.

## Fixed rendering and tensor boundary

- canvas: RGB white strip, 16 by 1,024 pixels;
- patch geometry: 64 contiguous 16 by 16 positions;
- layout: visible left-to-right writing without character cells, segmentation,
  Unicode normalization, crop, or truncation;
- augmented train font size: deterministic integer in `[8,11]`;
- augmented train origin: deterministic integer in `[0,15]`;
- clean anchor font size: 10;
- clean anchor origin: 0;
- train augmentation: deterministic mild blur, contrast, and Gaussian noise;
- evaluation augmentation: none; and
- mask: occupancy from the clean pre-augmentation black-on-white raster.

The eight train font paths and hashes are:

| Font path | SHA-256 |
|---|---|
| `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc` | `b76b0433203017ca80401b2ee0dd69350349871c4b19d504c34dbdd80541690a` |
| `/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc` | `197d5e1e019faca33a4d55931c7d68b8056f3b97cb862049f5cb8de9efdfb8ce` |
| `/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc` | `faa5f3656a78b2e2d450d27fe8382c778bc2b6bb5ea29c986664a6a435056ceb` |
| `/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf` | `acb6440a713d880a13a21b468ba7cd43f5a2b2934972e51be791c880730777b8` |
| `/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf` | `705ec2dba81eaee1208e4ed5d3ff23ab259292e8e0e163ea3de297ef1317007a` |
| `/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf` | `61519fb9bdda4a1a3aa02a12cbb76c2ef897ce879775d7ea2265d0a65fe54d16` |
| `/usr/share/fonts/truetype/arphic/ukai.ttc` | `cfc758e69386a21bade8cb3b373fda20d7f2fac6e8904ced2ded98588b63e274` |
| `/usr/share/fonts/truetype/arphic/uming.ttc` | `fe952e55617275142d9cefd4d79eade4df446517b0478b2567d9bc7df49f70e2` |

Each train item deterministically shuffles these paths and uses five distinct
font paths for five raster views:

1. clean exact-prompt anchor;
2. augmented exact-prompt view;
3. augmented paraphrase-prompt view when audited wording exists, otherwise a
   third exact-prompt rendering;
4. clean exact-answer anchor; and
5. augmented exact-answer view.

The canonical development font is Noto Serif CJK Regular. The held development
font is Noto Serif CJK Black. Neither is a training font.

Strings, source metadata, and identifiers may exist in dataset workers.
Immediately before model invocation, the batch is reduced to five pixel tensors
and five numeric clean masks. Identifiers may index detached target rows in the
trainer; they never enter the trainable model. The deployable call accepts only
one prompt raster and one clean mask.

## Fixed deployable model

### Shared visual reader

- input: `[B,3,16,1024]` pixels and `[B,64]` clean mask;
- input normalization: clamp to `[0,1]`, then map to `[-1,1]`;
- ViT hidden width: 768;
- transformer layers: 12;
- attention heads: 12;
- transformer MLP width: 3,072;
- positions: 64 visual patches plus CLS;
- pooling: clean-mask mean over contextual patch states, excluding CLS; and
- initialization: exact V37 EMA reader copied from the pinned checkpoint.

### Prompt, answer, and length paths

For pooled visual state `h`:

```text
f = Linear(1536,1024)(Dropout_0.05(GELU(Linear(768,1536)(LN(h)))))
s = normalize(f)

b = R(s)
c = Linear(512,1024)(SiLU(Linear(1024,512)(LN(f))))
z = normalize(b + c)

ell = 64 * sigmoid(Linear(256,1)(SiLU(Linear(768,256)(LN(h)))))
```

The prompt path uses dropout 0.05 between GELU and its final projection. `R` is
a bias-free 1,024 by 1,024 linear map. The answer adapter final weight and bias
start at zero. The answer path has no residual scale, angular cap, or constraint
that keeps `z` close to `s`.

The model has exactly 90,753,281 trainable parameters before stage freezing and
must remain below 100,000,000 deployable parameters.

The deployable boundary contains no string, token ID, Unicode ID, character ID,
vocabulary logits, OCR, visual codebook, target tensor, teacher, candidate bank,
retrieval database, BGE component, Qwen component, or network client. It returns
continuous prompt state, continuous answer state, and predicted visual length.
It does not yet generate a raster.

## Train-only answer-map initialization

Let `P` and `A` be row-normalized train prompt and answer target matrices. Compute
the full SVD

```text
U, Sigma, Vh = svd(P.T @ A)
R = U @ Vh
```

and copy `R.T` into the linear weight because `nn.Linear` evaluates
`s @ weight.T`. This is the unique train-only orthogonal Procrustes route used
by V38. No development row enters it. The source matrix, target bank, and SVD
factors are not stored in a checkpoint.

The deterministic initialization receipt must report:

- orthogonality maximum error below `2e-3`;
- train top-1 0.7588457465;
- train top-5 0.8795946240;
- train MRR 0.8142514825;
- paired train cosine 0.6200884581;
- the pinned train-bank SHA-256; and
- storage only as the deployable `answer_transform.weight`.

## Fixed detached candidate construction

For every train answer target, precompute the 16 nearest different train answers
whose target-to-target cosine is strictly below 0.85. Each physical microbatch
constructs 512 unique detached candidate rows containing:

- every positive target;
- all available unique rows from the 16-neighbor lists; and
- deterministic coprime-stride fill rows from the train bank.

Candidate sampling is seeded from the fixed seed, global optimizer update, and
microbatch index. Candidates and nearest-neighbor tables are training-only,
never receive gradients, are not stored in checkpoints, and are absent at
runtime.

## Fixed objective

Denote prompt states from the clean prompt, augmented prompt, and semantic prompt
views as `p0,p1,ps`; prompt-derived answer states as `z0,z1,zs`; and prompt-head
states from clean and augmented answer images as `a0,a1`. Let detached prompt
and answer targets be `u,v`.

For state list `S`, target `t`, candidate matrix `C`, positive candidate index
`k`, and temperature 0.05:

```text
NCE(S,t,C)   = mean_s cross_entropy((s @ C.T) / 0.05, k)
Align(S,t)   = mean_s mean(1 - cos(s,t))
```

The nearest-negative margin over `z0,z1,zs` is:

```text
L_margin = mean_z mean(relu(0.10 - cos(z,v) + max_j(z @ v_near_j)))
```

The remaining image-path losses are:

```text
L_exact = mean cosine-distance over (p0,p1), (z0,z1), (a0,a1)
L_semantic = mean cosine-distance over (p0,ps), (z0,zs)
L_binding = mean cosine-distance over (z0,a0), (z1,a1)
L_relation = mse(p0 @ p0.T, u @ u.T) + mse(z0 @ z0.T, v @ v.T)
```

VICReg-style variance and covariance are computed over the concatenated states
`p0,p1,ps,a0,a1,z0,z1,zs` after multiplying normalized states by `sqrt(1024)`.
Length is the mean smooth-L1 loss for `ell0,ell1,ells` against the clean answer
patch count.

The frozen total is:

```text
L = 0.40  * NCE((p0,p1,ps),u,U)
  + 3.00  * Align((p0,p1,ps),u)
  + 0.20  * NCE((a0,a1),v,V)
  + 1.50  * Align((a0,a1),v)
  + 0.60  * NCE((z0,z1,zs),v,V)
  + 4.00  * Align((z0,z1,zs),v)
  + 0.50  * L_margin
  + 0.75  * L_exact
  + 0.50  * L_semantic
  + 0.50  * L_binding
  + 0.10  * L_relation
  + 0.02  * L_variance
  + 0.001 * L_covariance
  + 0.10  * L_length
```

No token, character, Unicode, OCR, next-token, next-patch, raw-pixel
reconstruction, or answer-candidate prediction head is permitted.

## Fixed evidence training

Random seed: `20_263_800`. Hardware: CUDA device 0 on one NVIDIA GeForce RTX
4090 D. Precision: BF16. Optimizer: fused AdamW on CUDA, betas `(0.9,0.95)`,
weight decay 0.05, gradient clipping at 1.0, and all-parameter EMA decay 0.999.

| Stage | Updates | Reader | Head LR | Reader LR | Effective batch |
|---|---:|---|---:|---:|---:|
| head realignment | 500 | frozen | 1e-4 | 0 | 64 |
| full path adaptation | 7,500 | trainable | 5e-5 | 5e-6 | 64 |

Each stage has 200 updates of linear warmup followed by cosine decay to 10
percent of peak. The default physical batch is 8 records and may only be
reduced, never increased. Gradient accumulation must preserve 64 records per
optimizer update. Each physical batch performs one shared-reader forward over
five concatenated views and computes its own 512-row candidate losses.

Save atomically every 500 global updates and at each stage boundary. Exact
resume must restore model, optimizer, gradient scaler, all-parameter EMA, stage
and update position, deterministic data position, candidate position, elapsed
time, peak memory, stage summaries, and Python, NumPy, CPU Torch, and CUDA RNG
states.

Evidence training must complete exactly 8,000 updates. Peak allocated VRAM must
remain below 20 GiB. The primary checkpoint route is the all-parameter EMA;
raw weights are diagnostic only.

## Source and checkpoint integrity

The evidence trainer hashes this exact source set at startup and stores the map
in every resumable and standalone checkpoint:

- `ilm/visual_lm/visual_path_alignment.py`;
- `ilm/visual_lm/visual_path_alignment_data.py`;
- `ilm/visual_lm/visual_path_alignment_evaluation.py`;
- `ilm/visual_lm/visual_path_alignment_training.py`;
- `scripts/eval_visual_path_alignment_v38.py`; and
- `scripts/train_visual_path_alignment_v38.py`.

The evaluator requires every stored source hash to match the current file. It
also records its evaluation source map. The trainer and evaluator both require
the exact SHA-256 of this protocol. Patching that hash into both scripts is the
only allowed post-freeze source change before evidence training; no objective,
threshold, data, architecture, or execution behavior may change.

Checkpoints must declare and prove that they contain no source-language string,
target tensor, teacher model, candidate tensor, or nearest-negative tensor. A
recursive tensor-name audit rejects target, teacher, candidate, nearest, BGE,
or rotation tensors. The standalone EMA contains only deployable model tensors
and numeric provenance.

## Fixed development evaluation

Evaluate the complete EMA route first. Raw weights may be evaluated afterward
but cannot qualify V38. Use exactly 196 canonical development records and the
fixed 30 nonsealed paraphrases.

All student raster inference must finish before either target bank is loaded.
Run these independent image conditions:

1. canonical prompt in Noto Serif CJK Regular;
2. canonical answer image in Noto Serif CJK Regular;
3. cyclic prompt shuffle by one record while retaining original labels;
4. all-white raster with all-zero mask;
5. final visual quarter only;
6. canonical prompt in held Noto Serif CJK Black;
7. each fixed nonsealed paraphrase prompt; and
8. its original source prompt for paired wording consistency.

For canonical prompt states, report paired target cosine, top-1, top-5, MRR,
cyclic margin, cyclic pair win, variance, and effective rank against all 196
prompt targets. For answer-image prompt-head states, report the same reading
metrics against all 196 answer targets. For prompt-derived answer states, report
those metrics against answer targets plus visual-length MAE.

For controls, report retrieval, target cosine, cosine to the canonical output,
conditioned delta, rank, and length. For the held font, report canonical-to-held
prompt and answer cosine plus held retrieval. For wording, retrieve each
paraphrase answer state against the 30 corresponding answer targets and report
original-to-paraphrase prompt and answer cosine.

Construct deterministic nearest-clean-length counterfactual pairs with different
answer labels and report whether the two correct assignments beat the swapped
assignments. Report the prompt-to-answer transition:

```text
cos((z - s), (v - u))
```

over rows with nonzero student and target deltas, as well as prompt-answer
cosine, delta norms, and their ratio.

The matched V37 EMA report is included by its pinned hash. The evaluator writes
a numeric similarity artifact with no identifiers or source-language strings.
No retrieval candidate or target bank enters student inference.

## Frozen 39-condition gate

`visual-path-alignment-qualified` requires every condition below. A failed
condition yields exactly `not-qualified`.

### Integrity and resources

1. primary route is all-parameter EMA;
2. report, targets, model, optimizer, EMA, predictions, and metrics are finite;
3. protocol hash matches;
4. checkpoint source hashes match;
5. all pinned data and target hashes match;
6. V37 initialization hash and eligibility match;
7. BGE manifest and model-layer hashes match;
8. strict V37 tensor mapping and orthogonal answer initialization pass;
9. runtime and checkpoint boundaries contain no forbidden dependency;
10. all student inference precedes target-bank loading;
11. train paraphrases exclude every fixed holdout source;
12. deployable parameter count is below 100,000,000;
13. exactly 8,000 optimizer updates completed; and
14. peak allocated training VRAM is below 20 GiB.

### Reading, answer, and controls

| Condition | Threshold |
|---|---:|
| canonical prompt paired cosine | at least 0.50 |
| canonical prompt top-1 | at least 0.55 |
| canonical prompt top-5 | at least 0.80 |
| canonical answer-image reading cosine | at least 0.45 |
| canonical answer-image reading top-1 | at least 0.45 |
| prompt-derived answer top-1 | at least 0.35 |
| prompt-derived answer top-5 | at least 0.65 |
| prompt-derived answer MRR | at least 0.45 |
| prompt-derived answer paired cosine | at least 0.30 |
| answer correct-minus-cyclic margin | at least 0.15 |
| answer correct-beats-cyclic rate | at least 0.85 |
| answer correct cosine minus shuffled cosine | at least 0.15 |
| answer correct cosine minus blank cosine | at least 0.20 |
| counterfactual assignment rate | at least 0.90 |
| canonical-to-held prompt cosine | at least 0.75 |
| canonical-to-held answer cosine | at least 0.75 |
| held-font answer top-5 | at least 0.50 |
| paraphrase answer top-5 among 30 targets | at least 0.70 |
| original-to-paraphrase prompt cosine | at least 0.70 |
| original-to-paraphrase answer cosine | at least 0.70 |
| transition-direction cosine | at least 0.25 |
| prompt-answer state cosine | at most 0.95 |
| answer-state centered effective rank | at least 32 |
| answer rank relative to target rank | at least 0.40 |
| clean visual-length MAE | at most 3.0 patches |

These 25 metric conditions plus the 14 integrity/resource conditions are
strictly conjunctive. No average, pilot, raw route, favorable font result,
subset, or baseline comparison may override a failure. Retrieval is an
evaluator diagnostic and may not be described as generated output.

## Sealed split and raster-writer prohibition

Do not open the sealed split unless the complete EMA development report is
`visual-path-alignment-qualified`. If it passes, preserve the selected EMA
checkpoint and separately preregister an immutable sealed evaluator without
modifying the development evaluator. Run exactly one sealed evaluation with no
tuning. Every applicable absolute gate must pass, and each primary sealed
metric or control margin must retain at least 90 percent of its development
value.

Do not implement, train, or evaluate a V38 raster writer unless both development
and sealed semantic gates pass. A later writer requires its own protocol, must
accept only continuous image-derived state without text or candidates, must emit
raster pixels, and must pass causal-conditioning, readability, non-memorization,
and historical-form controls.

## Stop rules

Stop cleanly and preserve the current artifacts if:

- any pinned data, target, initialization, protocol, font, BGE, or source hash
  differs;
- a split count or exact selected-identifier hash changes;
- V37 tensor mapping or answer-map orthogonality fails;
- strings, token/Unicode/character IDs, OCR, Qwen, BGE, target banks, candidates,
  or a network client enter model inference or a checkpoint;
- any loss, gradient, parameter, target, prediction, optimizer, EMA, or metric
  becomes nonfinite;
- allocated training VRAM reaches 20 GiB;
- a completed stage has answer-state effective rank below 8 on the fixed
  deterministic 64-record train probe; or
- SIGINT or SIGTERM is received.

## Required evidence artifacts

- this immutable protocol and the V38 research decision;
- exact data, paraphrase, target-bank, V37 initialization, font, external-model,
  and source receipts;
- focused unit and integration tests for raster views, clean masks, target
  lookup, hard-negative injection, loss, optimizer coverage, EMA loading,
  boundary rejection, controls, and conjunctive gating;
- atomic resumable checkpoints and a standalone all-parameter EMA checkpoint;
- JSONL update metrics, stage summaries, rank probes, elapsed time, examples
  consumed, parameter count, and peak VRAM;
- complete fixed-development EMA and raw reports;
- numeric similarity artifacts containing no source-language strings; and
- a result note and paper revision that clearly separate external scaffolding,
  detached teacher ceilings, image-only student evidence, and future raster
  generation claims.
