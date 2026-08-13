# Causal Glyph Flow V35 Protocol

Date frozen: 2026-08-14

Status: preregistered before V35 implementation, smoke training, development
measurement, or sealed evaluation

## Decision Question

Can a sub-150M-parameter student, initialized from a public pixel-language
checkpoint and trained on one RTX 4090, consume only raster patches at runtime
and autonomously emit prompt-dependent Chinese writing through a closed,
continuous, non-quantized visual loop?

V35 is a mechanism proof, not a claim of general-LLM parity.

## Frozen Foundations

### V34 visual codec

- checkpoint:
  `artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt`;
- checkpoint SHA-256:
  `a138c9cb3b0502e43d1227f689c020893d56b468742c32e1840e44d299662f33`;
- selected state: complete EMA shadow, 84 tensors;
- architecture: `continuous-glyph-representation-codec-v34`;
- parameters: 7,423,361; and
- route: frozen and included in the V35 evidence checkpoint.

The V34 decoder is the only runtime actuator. Its development and sealed
qualification is recorded in `references/continuous_glyph_codec_v34_result.md`.

### PIXAR causal initialization

- repository: `https://github.com/april-tools/pixar`;
- audited revision: `810a423336d5fdeb33e4c1695381e357ff32c4bb`;
- source license at that revision: MIT;
- weight file: `artifacts/upstream/pixar/backbone/pytorch_model.bin`;
- weight SHA-256:
  `ae4f899bbbb0bfaa90ee033c6d1dc5aeb3f50b323f726a0df241be8104682eb9`;
- config SHA-256:
  `1bc49b2e8cc59a4865d0f737739e73a55d14a6b849a5cae516de3dbb3f8e2ace`;
- selected state: the complete 12-layer causal transformer core; and
- weight license: not stated in the archive, so V35 is local research evidence
  and no weight redistribution is authorized by this protocol.

The resized PIXAR input projection may supervise interface alignment offline.
It must not be required by the final runtime checkpoint.

### Data

| Purpose | File | SHA-256 | Rights/use |
|---|---|---|---|
| continuation and copy source | `data/visual_grammar/chinese_wikisource_public_domain.jsonl` | `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03` | public-domain texts; row provenance retained |
| short semantic instructions | `data/raw/alpaca_zh.json` | `6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903` | CC BY-NC 4.0; research only |
| wording-shift diagnostic | `data/teacher/folio_paraphrases_zh_holdout.jsonl` | `132a8a257d31be3cf607d0313bf042dafaee7f2df08440bf7e012a502ed6c02f` | evaluator-only teacher wording; source ID retained |

The selected record counts before rendering are:

| Stream | Total | Train | Development | Sealed |
|---|---:|---:|---:|---:|
| public-domain | 7,017 | 6,743 | 135 | 139 |
| short Alpaca Chinese | 6,108 | 5,747 | 159 | 202 |

Public splits use the existing `v33:public-domain:<id>` SHA-256 partition.
Instruction splits use `v33:instruction:<id>`. Synthetic copy records inherit
the split of their public source record; no source text may cross splits.

The 31 fixed paraphrases are a development-only wording-shift diagnostic. They
do not decide sealed access because most share an answer with an instruction
training record.

## Runtime Boundary

The production `generate` entry point accepts only:

- `pixels`: floating raster strips in `[0,1]`;
- `patch_mask`: floating visual-position masks; and
- numerical generation controls and a random seed.

It returns generated raster patches, visual lengths, and stop probabilities.
The model may not accept strings, token IDs, Unicode IDs, character IDs, OCR
text, vocabulary indices, codebook indices, or retrieval keys.

The evidence checkpoint must include every tensor needed for independent
runtime inference. It may not call PIXAR, Qwen, OCR, a tokenizer, a database,
or a network service. OCR is evaluator-only.

## Fixed Architecture

- raster patch: 1 by 32 by 32 binary pixels;
- maximum causal context: 96 patches;
- V34 codec latent: 768 continuous dimensions;
- input adapter: residual `Linear(768,768)-SiLU-Linear(768,768)`;
- causal field: 12 layers, width 768, 12 attention heads, 12 key/value heads,
  SwiGLU width 3,072, RMSNorm, and RoPE;
- deterministic anchor: `Linear(768,768)-SiLU-Linear(768,768)`, followed by
  non-affine layer normalization;
- flow head: width 512, three AdaLN residual MLP blocks, sinusoidal time
  embedding, and 768-dimensional velocity output;
- stop head: one scalar logit per causal position;
- quantizer/codebook/vocabulary: none; and
- total parameters: required to be below 150,000,000 and recorded from the
  instantiated model rather than estimated.

The V34 codec is frozen. The input adapter is trained only during alignment and
then frozen. The causal field, anchor, flow head, and stop head are trainable in
the causal stages.

## Fixed Rendering

- reuse `DirectPatchRenderConfig` with 32-pixel patches;
- train fonts: Noto Sans CJK Regular and Medium;
- development font: Noto Serif CJK Regular;
- sealed font: Noto Serif CJK Light;
- font size: uniform integer in `[24,28]`;
- horizontal origin: uniform integer in `[0,31]`, independently selected for
  prompt and answer regions;
- binary threshold: 0.5 after the existing mild blur/contrast/noise pipeline;
- prompt region: at most 64 patches;
- answer region: at most 31 patches; and
- total active sequence: at most 96 patches.

The random origin prevents a patch position from serving as a hidden character
ID. Render strings and metadata are excluded from the student batch.

### Copy curriculum

For a public source record and deterministic rendering variant, select a
contiguous visible span of 2 to 16 characters. Render the prompt
`照写：<span> 答：` and render `<span>` independently as the answer. Font is
shared, but prompt and answer origins are independent. The student is never
given the source string or character indices.

## Fixed Objectives

For target next-patch latent `z`, normalized anchor `m`, conditional field
state `h`, Gaussian `epsilon`, and `tau ~ Uniform(0,1)`:

```
x_tau = (1 - tau) * epsilon + tau * z
v_target = z - epsilon
L_flow = mean_squared_error(flow(x_tau, tau, h), v_target)
L_anchor = mean(1 - cosine(m, z)) + 0.25 * mean_squared_error(m, z)
L_visual = BCE(decode(m), target_pixels)
           + 0.25 * edge_L1
           + 0.25 * ink_Dice_loss
L_stop = masked_BCE(stop_logits, stop_targets)
L_total = L_flow + L_anchor + 0.25 * L_visual + 0.10 * L_stop
```

`L_anchor` and `L_stop` use all active supervised positions. To bound compute,
sample at most 128 active positions per microbatch without replacement for
`L_flow` and `L_visual`. Sampling is driven by the recorded training RNG.

Alignment uses raw resized PIXAR projection targets `e`:

```
L_align = mean_squared_error(adapter(z), e)
          + 0.25 * mean(1 - cosine(adapter(z), e))
```

The causal field and writer are frozen during alignment. The alignment teacher
is absent from all later student batches and from runtime.

## Fixed Training Route

Random seed: `20_263_500`. Precision: CUDA BF16. Optimizer: fused AdamW with
`beta=(0.9,0.95)`, weight decay 0.05, gradient clipping at 1.0, and EMA 0.999.
Physical batch size is 8 and gradient accumulation is 8, for effective batch
size 64. Each stage uses 500 warmup updates and cosine decay to 10% of peak.

| Stage | Updates | Data | Head LR | Core LR |
|---|---:|---|---:|---:|
| visual interface alignment | 2,000 | public train rasters | 3e-4 | 0 |
| public causal continuation | 8,000 | public train | 1e-4 | 1e-5 |
| instruction and copy | 12,000 | 75% Alpaca, 12.5% copy, 12.5% public replay | 8e-5 | 8e-6 |

“Head LR” applies to the anchor, flow, and stop heads. The input adapter uses
that group only in alignment and is frozen thereafter. The codec is always
frozen. The model must save atomically at least every 1,000 updates and support
exact resume with model, EMA, optimizer, counters, and RNG state.

Evidence training requires one RTX 4090 and must remain below 20 GiB peak
allocated VRAM. No second heavy process may be launched for this project while
the run is active.

## Stage-A Gate

After alignment, evaluate at least 2,048 active patches from public development
records using the held-out serif font. Continue only if all are true:

- every model, latent, and metric value is finite;
- causal-field state hash is unchanged;
- V34 codec state hash is unchanged;
- mean cosine between adapter output and PIXAR projection is at least 0.95;
- mean squared alignment error is at most 0.035; and
- no forbidden runtime input appears in the boundary receipt.

Failure stops the evidence route before causal training. Smoke and explicitly
named exploratory runs may bypass the stop but cannot qualify V35.

## Development Evaluation

Only EMA weights are eligible for the primary decision. Raw weights are a
diagnostic. Evaluation uses fixed render variants and fixed flow noise.

### Teacher-forced diagnostics

- public-development next-patch anchor cosine and decoded ink/edge F1;
- copy-development next-patch anchor cosine and decoded ink/edge F1;
- Alpaca-development next-patch anchor cosine and decoded ink/edge F1;
- flow velocity MSE on fixed noise/times; and
- target-through-V34 OCR ceiling.

### Autonomous routes

Evaluate both writers through the closed raster loop:

- `anchor`: one deterministic anchor pass per patch;
- `flow`: eight Heun steps from fixed Gaussian noise per patch.

The anchor route is primary. Promote the flow route only if its development
OCR character accuracy exceeds the anchor by at least 0.03 while its readable
example rate is no more than 0.02 lower. This selection rule is frozen before
training.

### Prompt controls

For every evaluated instruction, reuse identical generation noise under:

- correct prompt;
- cyclically shuffled prompt from another record;
- all-white prompt with the original visual length; and
- final-quarter prompt only.

Report normalized OCR character accuracy, exact match, readable-example rate,
predicted length, nonblank rate, output-pixel disagreement, and output-latent
cosine for every condition. Correct output must be compared with the target
raster and expected answer; OCR is never fed back to the student.

### Copy counterfactuals

Pair development copy prompts with different source spans of the same rendered
length. With shared generation noise, swapping only the visible span must
change the emitted raster toward the paired target. Report paired OCR accuracy,
pixel disagreement, and target preference.

## Frozen Gates

### Visual-causal qualification

All conditions must pass:

- Stage-A gate passed;
- finite model, generation, checkpoint, and optimizer-state audit;
- closed-loop generation receipt passed;
- copy-development target OCR ceiling at least 0.70;
- autonomous copy OCR retention at least 0.60 of that ceiling;
- autonomous copy correct-prompt character accuracy exceeds shuffled by at
  least 0.20 and blank by at least 0.25;
- copy counterfactual target preference at least 0.75;
- public teacher-forced decoded ink F1 and edge F1 each at least 0.70;
- autonomous public continuation nonblank readable rate at least 0.50; and
- peak allocated VRAM below 20 GiB on one RTX 4090.

### Semantic-raster qualification

This stronger status additionally requires:

- Alpaca target-through-codec OCR character accuracy at least 0.60;
- correct-prompt autonomous character accuracy at least 0.08;
- correct-prompt accuracy exceeds shuffled by at least 0.02 and blank by at
  least 0.03;
- correct-prompt readable-example rate at least 0.35;
- correct-prompt mean output differs from shuffled and blank by at least 0.01
  of binary pixels; and
- at least one fixed wording-shift example is readable and more accurate than
  its blank-prompt control.

The status vocabulary is exactly:

- `not-qualified`;
- `visual-causal-qualified`; or
- `semantic-raster-qualified`.

Passing only the visual-causal gate must not be described as semantic language
understanding.

## Sealed Rule

Do not render, inspect, OCR, or score the 139 public and 202 instruction sealed
records until the complete development evaluator has written its report and at
least `visual-causal-qualified` is true. After opening, run exactly one sealed
evaluation with the already selected EMA route and inference writer.

Sealed transfer passes when every absolute development threshold still passes
and each primary sealed metric is at least 90% of its development value. A
failed sealed result is reported, not tuned away.

## Required Artifacts

- immutable run receipt with protocol, source, data, external-weight, and V34
  hashes;
- stage-A alignment report and gallery;
- periodic atomic checkpoint and final standalone checkpoint;
- JSONL training metrics and stage summaries;
- development report for raw and EMA states;
- autonomous galleries for copy, continuation, Alpaca, and prompt controls;
- one sealed report and gallery only when permitted;
- peak VRAM, elapsed time, examples/patches consumed, and parameter counts;
- boundary receipt proving the absence of token/vocabulary/OCR/runtime-teacher
  interfaces; and
- a result note that distinguishes external foundations from new evidence.

## Stop Rules

Stop cleanly and preserve evidence if:

- Stage A fails;
- any loss, gradient, parameter, latent, or generated raster is non-finite;
- the V34 codec changes;
- the input adapter changes after Stage A;
- the runtime boundary admits symbolic IDs or external calls;
- allocated VRAM reaches 20 GiB;
- a source/hash/protocol receipt differs; or
- SIGINT/SIGTERM is received.

