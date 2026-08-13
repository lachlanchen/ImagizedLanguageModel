# Visual Semantic Raster Transducer V32 Protocol

Date frozen: 2026-08-13

Status: preregistered before V32 implementation, smoke training, or development
measurement

## Decision Question

Can a sub-125M-parameter, pixel-input/pixel-output Chinese student fit on one
RTX 4090, follow a rendered prompt under held-out wording, and autonomously
emit a readable short answer raster without token IDs, Unicode IDs, OCR, a
glyph codebook, or an external language model?

V32 is a bounded proof of visual semantic transduction. It is not a benchmark
claim against general LLMs.

## Frozen External Assets

### Prompt-reader initialization

- repository: `Team-PIXEL/pixel-m4`;
- upstream revision: `56bfcbf71e98f613ee00f8efb7a607bf0058f1e6`;
- license: Apache-2.0;
- weight file: `pytorch_model.bin`;
- weight SHA-256:
  `6aa0642d46fe211727fefc5ac6e0bc28efa8511d1f0d9e5eee1987fa821600bc`;
- config SHA-256:
  `90789708a8b064d848977d256ce4e3e20ba51f57d6803a59295bf401d514f83f`;
- selected state: the complete `vit.*` prefix only, 86,054,400 parameters;
- verified mapping: current `transformers.ViTModel`, no missing or unexpected
  keys; and
- upstream training languages: English, Hindi, Ukrainian, and Simplified
  Chinese, as reported by PIXEL-M4.

The cached file is an input dependency, not committed repository data. The V32
evidence checkpoint must either include the selected reader tensors or include
this immutable hash and fail clearly when the exact file is absent.

### Data

| Purpose | File | SHA-256 | Rights/use |
|---|---|---|---|
| raster and continuation | `data/visual_grammar/chinese_wikisource_public_domain.jsonl` | `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03` | public-domain historical texts; Wikisource contribution layer recorded per row |
| instruction development | `data/raw/alpaca_zh.json` | `6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903` | CC BY-NC 4.0, research only |
| fixed paraphrases | `data/teacher/folio_paraphrases_zh_holdout.jsonl` | `132a8a257d31be3cf607d0313bf042dafaee7f2df08440bf7e012a502ed6c02f` | evaluator-only teacher wording; original record ID retained |

No local PDF with unclear rights enters V32 training. `../Books`, `../ZhJpBook`,
and registered source books remain references for later curation only.

### Reuse ledger

- PIXEL-M4 weights are reused under Apache-2.0 and identified above.
- Current Hugging Face `ViTModel` supplies the implementation; old PIXEL source
  is not vendored.
- PixelGPT and PIXAR are cited baselines; their weights/code are not copied.
- PRA equations motivate decoded feedback. Its unlicensed repository source is
  not copied.
- Qwen or another local LLM may prepare a later dataset, but no teacher call or
  teacher state participates in this fixed V32 run.

## Student Boundary

Allowed arguments to learned model methods:

- prompt raster `float` tensor `[B,3,16,16P]`;
- prompt patch-presence mask `[B,P]`;
- answer raster cells `[B,A,1,24,24]` during training only;
- answer-presence and stop masks;
- continuous noise, latent states, hidden states, positions, and scalar
  timesteps; and
- train/eval route flags that select clean or decoded pixel feedback.

Forbidden arguments, buffers, parameters, and checkpoint values:

- strings or bytes;
- token, Unicode, character, glyph, word, record, answer, or vocabulary IDs;
- one-hot character labels;
- tokenizer output or embedding tables indexed by language symbols;
- OCR text, logits, or hidden states;
- discrete visual-codebook indices;
- source answer selection indices passed to the model;
- external teacher logits or runtime calls; and
- a candidate glyph or answer bank used by inference.

Host-side code may read strings to select records and render pixels. Evaluators
may use source strings, candidate rasters, or OCR after autonomous generation.
Those values cannot influence the generated image.

## Fixed Rendering

### Prompt

| Property | Value |
|---|---:|
| height | 16 pixels |
| patch width | 16 pixels |
| maximum patches | 192 |
| channels | RGB; monochrome ink repeated across channels |
| background | white with bounded intensity jitter |
| ink | black with bounded intensity jitter |
| train fonts | Noto Sans CJK Regular, Noto Sans CJK Medium |
| development font | Noto Serif CJK Regular |
| font size | uniform integer 11-14 pixels |
| horizontal origin | uniform integer 0-15 pixels |
| per-character position jitter | at most 1 pixel |
| blur | radius 0-0.35, probability 0.15 |
| additive noise | standard deviation 0-0.015, probability 0.15 |

At least 25% of examples use a nonzero horizontal origin so patch boundaries
do not consistently equal character boundaries. Text beyond 192 patches is
excluded rather than truncated.

### Answer

| Property | Value |
|---|---:|
| cell shape | `[1,24,24]` |
| maximum cells | 32 |
| train fonts | Noto Sans CJK Regular, Noto Sans CJK Medium |
| development font | Noto Serif CJK Regular |
| font size | uniform integer 18-21 pixels |
| glyph translation | at most 1 pixel |
| target range | `[0,1]`, where 1 is ink |
| primary output | cells concatenated left-to-right into one PNG strip |

Whitespace and line breaks are normalized to one visible blank cell. Responses
longer than 32 normalized cells are excluded. No character is substituted with
an ID when a font cannot render it; that example is skipped and counted.

## Fixed Data Partitions

All partitions are selected by SHA-256 of the source identifier before any
render augmentation.

### Public-domain visual stream

- `train`: stable fraction in `[0.00,0.96)`;
- `development`: `[0.96,0.98)`;
- `sealed`: `[0.98,1.00)`;
- samples: normalized contiguous Chinese segments with 8-192 prompt cells and
  1-32 answer cells;
- continuation boundary: uniformly selected inside a source row; and
- no source row appears in more than one partition.

### Chinese instruction stream

Records must have:

- nonempty instruction and answer;
- prompt length at most 160 normalized characters;
- answer length 1-32 normalized cells; and
- no unsupported control characters.

Partition by record identifier:

- `train`: stable fraction in `[0.00,0.94)`;
- `development`: `[0.94,0.97)`;
- `sealed`: `[0.97,1.00)`.

The fixed paraphrase file is joined by original identifier. Only rows whose
answer fits 32 cells are used. The original wording may train; the paraphrase
wording is evaluator-only and never rendered in a training batch.

### Controlled counterfactual set

The implementation must deterministically construct at least 256 pairs from
instruction records with short categorical, numeric, extraction, translation,
or classification answers. Each pair has the same instruction family and two
different visible inputs/answers. Pair membership and answer identity stay on
the host. If fewer than 256 defensible natural pairs can be constructed without
manually invented labels, V32 records the available count and the
counterfactual gate is not claimed; it cannot silently substitute a synthetic
packet algebra.

## Fixed Architecture

Architecture name: `visual-semantic-raster-transducer-v32`.

### Reader

| Property | Value |
|---|---:|
| base | PIXEL-M4 `vit.*` |
| hidden width | 768 |
| layers | 12 |
| heads | 12 |
| patch size | 16 |
| selected position states | CLS plus first 192 patch states |
| initial trainability | frozen |
| optional unfreeze | final two blocks only, after stage 2 |

Position tensors are sliced, not interpolated, because V32 uses a prefix of the
same one-dimensional 16-pixel patch geometry.

### Planner

| Property | Value |
|---|---:|
| answer-cell retina | `1 -> 64 -> 128 -> 512` convolutional projection |
| width | 512 |
| layers | 6 |
| self-attention heads | 8 |
| cross-attention heads | 8 |
| MLP width | 1536 |
| dropout | 0.05 |
| maximum generated cells | 32 |
| position representation | learned continuous positions |
| start representation | one learned 512-vector |

### Continuous writer

| Property | Value |
|---|---:|
| glyph-state dimension | 32 |
| target-encoder width | 256 |
| target-encoder residual blocks | 3 |
| raster-decoder width | 256 |
| raster-decoder causal layers | 2 |
| raster-decoder heads | 8 |
| pixel output | 576 logits per cell |
| state predictor | diagonal Gaussian mean/log-scale |
| stop head | one scalar per answer position |

The complete parameter count must be below 125,000,000. A smaller exact count
is reported after implementation. No parameter dimension equals the number of
observed characters or answers.

## Fixed Training Objective

Let clean target states be `z`, decoded-feedback cells be `bar_y`, planner
distribution be `(mu, log_sigma)`, reconstructed cells be `y_hat`, and stop
logits be `s`.

\[
\mathcal L_{state}=\operatorname{mean}
\left[
\tfrac12 ((z-\mu)/\sigma)^2 + \log\sigma
\right]
\]

with `log_sigma` clamped to `[-4,2]`.

\[
\mathcal L_{pixel}=BCEWithLogits(y_{hat},y),
\]

\[
\mathcal L_{edge}=\lVert Sobel(\sigma(y_{hat}))-Sobel(y)\rVert_1,
\]

\[
\mathcal L_{ink}=1-Dice(\sigma(y_{hat}),y),
\]

and `L_stop` is binary cross entropy on the first blank position after the
answer. `L_variance` penalizes any active latent dimension whose batch standard
deviation is below `0.20`.

Fixed weights:

| Loss | Weight |
|---|---:|
| state NLL | 1.0 |
| pixel BCE | 1.0 |
| Sobel edge | 0.25 |
| ink Dice | 0.25 |
| stop BCE | 0.2 |
| latent variance | 0.05 |

Target states used by `L_state` are detached. Target encoder and raster decoder
learn through raster and variance losses. The planner learns through state and
stop losses. Once the reader is unfrozen, its gradients come only from planner
losses.

## Fixed Decoded-Feedback Route

For every active target state:

- perturbation probability: `0.90`;
- `t ~ Uniform(0.65,1.0)`;
- noise: standard normal in 32 dimensions;
- decoded cells are detached before the planner's second pass; and
- 5% of active decoded cells are replaced by bounded-noise ground-truth cells
  for stabilization.

The clean teacher-forced route is evaluated as an ablation but cannot supply
the primary autonomous result.

## Training Schedule

One finite development run uses seed `20263200` and at most 12,000 optimizer
updates.

| Stage | Updates | Mixture | Reader |
|---|---:|---|---|
| 1 raster warmup | 2,000 | public-domain answer segments | frozen |
| 2 continuation | 4,000 | public-domain prompt/next-strip pairs | frozen |
| 3 instruction | 6,000 | 75% instruction, 25% continuation replay | frozen for first 3,000; final two blocks may unfreeze for last 3,000 |

Other fixed optimization values:

- AdamW;
- planner/writer peak learning rate `3e-4`;
- unfrozen reader learning rate `2e-5`;
- betas `(0.9,0.95)`;
- weight decay `0.05`, excluding norms and biases;
- 500-update linear warmup followed by cosine decay per stage;
- BF16 autocast;
- gradient clipping at norm `1.0`;
- effective batch size at least 64 through accumulation;
- EMA decay `0.999` for evaluation;
- no validation-selected restart; and
- no more than one active training process.

OOM permits reducing only the physical microbatch and increasing accumulation.
It cannot change effective batch size, model width, or update count.

## Controls

The complete development audit includes:

1. correct prompt pixels;
2. batch-shuffled prompt pixels;
3. blank prompt pixels with the same mask shape;
4. last 25% of prompt patches only;
5. held-out prompt font;
6. non-cell-aligned prompt origin;
7. clean teacher-forced answer prefix;
8. autonomous decoded answer prefix;
9. decoded-feedback model versus clean-prefix ablation; and
10. PIXEL-M4 initialization versus a random-reader probe trained for the same
    first 1,000 stage-3 updates.

No control may replace model output with a target render.

## Evaluation

The primary evaluated artifact is the autonomous answer PNG. The model receives
the prompt raster and generates until stop probability exceeds 0.5 after at
least one cell, or until 32 cells.

An evaluator-only candidate bank renders characters observed in the selected
development answers in the held-out font plus two alternate fonts. Each
generated cell is assigned to its nearest frozen visual embedding. This
assignment supplies character error rate and exact sequence accuracy. The bank
is absent from student inference and its use is visible in the receipt.

Report at least:

- autonomous character accuracy and character error rate;
- autonomous exact answer-strip accuracy;
- predicted-length exact accuracy;
- teacher-forced equivalents;
- target pixel F1, edge F1, ink coverage, blank rate, and overflow rate;
- original-wording and paraphrase metrics;
- counterfactual pair assignment;
- correct-minus-shuffled and correct-minus-blank prompt margins;
- held-out-font degradation;
- clean-prefix versus decoded-feedback degradation;
- reader initialization probe;
- parameter counts, update time, peak allocated/reserved VRAM, and inference
  latency; and
- boundary/checkpoint receipts.

All aggregate metrics include sample counts and bootstrap 95% confidence
intervals where the sample count is at least 100.

## Fixed Gates

### Integrity and resources

All must pass:

1. complete parameter count `<125M`;
2. peak allocated VRAM `<20 GiB` on one RTX 4090;
3. exactly 12,000 finite updates unless a documented hardware fault aborts;
4. no NaN/Inf parameters, gradients, losses, or generated cells;
5. forbidden-boundary scanner finds zero prohibited model inputs/checkpoint
   tensors;
6. autonomous inference runs with no dataset answer access, OCR, tokenizer,
   external model, or candidate bank; and
7. source, checkpoint, config, and evidence hashes are recorded.

### Direct raster

All must pass on the instruction development partition:

1. autonomous nonblank answer rate `>=95%`;
2. autonomous character accuracy `>=60%`;
3. autonomous character error rate `<=0.45`;
4. autonomous exact sequence accuracy `>=15%`;
5. predicted-length exact accuracy `>=60%`; and
6. autonomous character accuracy is no more than 15 percentage points below
   its teacher-forced equivalent.

### Visual language

All applicable gates must pass:

1. original-wording autonomous exact accuracy `>=20%`;
2. paraphrase autonomous character accuracy `>=45%` on at least 24 eligible
   examples;
3. correct prompt improves autonomous target-sequence log similarity by
   `>=0.15` over batch-shuffled prompt;
4. correct prompt improves exact sequence accuracy by `>=10` percentage points
   over blank prompt;
5. counterfactual pair assignment `>=70%` on at least 128 defensible pairs;
6. held-out-font character accuracy drops by no more than 15 percentage points;
7. decoded-feedback training improves autonomous character accuracy by at
   least 5 percentage points over the clean-prefix ablation; and
8. the model beats the evaluator-only answer-frequency baseline in exact
   sequence accuracy.

If the natural counterfactual constructor yields fewer than 128 pairs, gate 5
is reported as unavailable and V32 cannot claim controlled compositional
binding. It may still establish the narrower paraphrase-conditioned raster
result.

## Decision Rule

- **Accepted visual semantic raster proof:** every integrity, direct-raster,
  and applicable visual-language gate passes.
- **Accepted raster transducer only:** integrity and direct-raster gates pass,
  but one or more language gates fail.
- **Rejected writer:** any direct-raster gate fails.
- **Invalid run:** any boundary, finite-state, provenance, or resource gate
  fails.

An accepted result remains limited to short modern-Chinese answer strips under
the specified data. It authorizes a V33 public-domain data replacement and
longer page layout. A rejected result must identify whether failure lies in the
reader, state predictor, raster decoder, stop process, or decoded-feedback
route before scaling parameters or data.

