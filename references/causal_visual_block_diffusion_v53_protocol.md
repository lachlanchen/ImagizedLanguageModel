# Causal Visual Block Diffusion V53: Frozen Mechanism Protocol

Date frozen: 2026-11-04

Status: frozen before implementation, smoke training, or development scoring

## Decision Question

Can a compact image-only model learn the conditional distribution of a visible
four-glyph Chinese future directly, so that paired samples from ordered raster
history recover the intended writing more often than samples from shuffled or
zeroed history?

V53 changes the generated random variable and the score parameterization. It
does not add data, fonts, historical forms, instruction tuning, a character
vocabulary, or a larger causal reader.

## Why V53 Is A New Test

V51 and V52 factorized writing and language:

```text
unconditional glyph prior + history-dependent guidance
```

V52 showed that this factorization can rank completed glyphs while failing to
form the conditional glyph mode from noise. At the terminal diffusion level,
history-dependent score information vanishes; an unconditional trajectory
begins selecting an identity before local guidance can act.

V53 instead trains one conditional field:

\[
v_\theta(X_t,t\mid H),
\]

where `H` is 60 visible glyph rasters and `X` is the next four-glyph raster
strip. The condition enters every denoising scale. The model predicts diffusion
velocity rather than epsilon, so its terminal-noise target retains a direct
clean-image term. Four future cells are generated jointly, allowing local
word/phrase constraints to act before individual glyph modes settle.

This is a bounded application of continuous autoregression without vector
quantization, as in [MAR](https://arxiv.org/abs/2406.11838), and independently
noised causal future cells, as in
[Diffusion Forcing](https://arxiv.org/abs/2407.01392). Clean visual-field
alignment follows the training-efficiency motivation of
[REPA](https://arxiv.org/abs/2410.06940), but uses the repository's frozen
image-only field rather than an external semantic model.

## Runtime Boundary

The student accepts and emits floating raster tensors. Its runtime contains:

- the frozen V42 causal raster reader;
- one fine-tuned conditional strip denoiser; and
- fixed diffusion arithmetic and raster thresholding.

It may not accept or store strings, bytes, tokenizer outputs, Unicode values,
character IDs, OCR transcripts, vocabulary indices, glyph classes, codebook
indices, candidate images, retrieval keys, or external model outputs. It may
not call a font renderer, OCR engine, database, local LLM, or network service
at inference.

Character strings remain host-side data-preparation metadata. A 1,024-image
bank is allowed only in the evaluator after output images have been generated.

## Frozen Foundations

| Foundation | Path | Required SHA-256 | Role |
| --- | --- | --- | --- |
| V42 reader | `artifacts/canonical_glyph_language_v42_20260814/checkpoint_final.pt` | `a5ce2ff20d0fc6d336f489b1e2d29bb96a1d0666399c2e2b95e7ac33a9abe870` | frozen visual history encoder |
| V51 prior | `artifacts/visual_posterior_language_v51_calibration_20261104/checkpoint_final.pt` | `ef607cb67c49da6d9bea32a784cb821a92ea546efbe8bf2226f8065171c00629` | denoiser initialization only |
| public corpus | `data/visual_grammar/chinese_wikisource_public_domain.jsonl` | `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03` | canonical Chinese raster streams |

The V51 U-Net is copied into the new strip denoiser at initialization. Its
checkpoint is not a separate deployed expert. The V42 reader remains
byte-for-byte frozen throughout V53.

## Fixed Visual Sequence

- canonical cell: `1 x 32 x 32` binary ink raster;
- visible history: first 60 cells of a 64-cell canonical training context;
- generated future: the following four cells;
- generated strip: `1 x 32 x 128`, with no inter-cell gutter;
- script views: original and OpenCC simplified when distinct;
- font: the fixed V42 canonical Noto Sans CJK font; and
- augmentation: none in the first mechanism test.

The existing 65-cell V42 training sample already contains the required
sequence. The student receives only `context[:, :60]` and the four target image
cells. Remaining metadata is excluded.

Development uses 512 deterministic windows from the existing development
partition. For each window, the first 60 context cells are visible and the
last four are the target block. A second four-cell block from the registered
continuation is reserved for visible-feedback evaluation. Frozen records and
fonts remain unopened.

## Conditional Strip Denoiser

The denoiser is a three-scale convolutional U-Net operating on the complete
`32 x 128` strip.

- base width: 64;
- spatial scales: `32 x 128`, `16 x 64`, and `8 x 32`;
- residual blocks: one before each down/up transition and two at the
  bottleneck;
- bottleneck block mixer: four pooled cell vectors, one four-head attention
  layer, and one width-512 MLP;
- time width: 128;
- condition: concatenated final V42 hidden state (384) and visual anchor
  (1,024), projected to the time width and injected by FiLM in every residual
  block;
- spatial inputs: noisy strip, first-cell V42 direct plan, normalized
  horizontal coordinate, and per-cell noise-level map; and
- output: one velocity channel over the complete strip.

The noisy-channel convolution, residual blocks, and output convolution are
initialized from the frozen V51 EMA denoiser. New input channels, condition
projections, and block mixer are zero-initialized where that preserves the V51
function. After initialization the complete denoiser is trained jointly.

The denoiser may have at most 8 million parameters. V42 plus the denoiser may
have at most 26 million parameters.

## Independent Future-Cell Corruption

Let signed clean future cells be `x_0,j`, `j=1..4`. Each update assigns the
four fixed noise strata

```text
clean/low:  t in [0, 24]
medium:     t in [25, 59]
high:       t in [60, 89]
terminal:   t in [90, 99]
```

to the four positions by a saved random permutation, then samples uniformly
inside each assigned range. Thus every training strip contains easy, medium,
hard, and nearly pure-noise cells without binding a particular position to a
noise level.

For cumulative signal `alpha_bar_t`, Gaussian noise `epsilon`, and signed
image `x_0`,

\[
x_t=\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\epsilon,
\]

\[
v^*=\sqrt{\bar\alpha_t}\epsilon
-\sqrt{1-\bar\alpha_t}x_0.
\]

The clean estimate is

\[
\hat x_0=\sqrt{\bar\alpha_t}x_t
-\sqrt{1-\bar\alpha_t}v_\theta.
\]

Unlike epsilon prediction, this parameterization presents a full clean-image
target to the terminal-noise model output.

## Fixed Condition Controls

For every strip, construct three conditions without changing the target or
noise:

1. ordered 60-cell history;
2. suffix-preserving shuffle of cells 0--55, leaving cells 56--59 fixed; and
3. zero condition used by classifier-free training and evaluation.

Ten percent of primary training rows use the zero condition. Ordered and
shuffled states are both computed by the frozen V42 reader. Shuffled history is
never labeled as another glyph; it is only a causal control.

## Fixed Loss

For the ordered route:

\[
L_v=\operatorname{MSE}(\hat v,v^*).
\]

From `x_hat_0`, compute:

- foreground-weighted clean-image MSE `L_fg`;
- soft ink Dice loss `L_dice`;
- cosine distance between fixed V42 image fields of predicted and target cells
  `L_field`; and
- a bottleneck representation-alignment loss `L_repa`, where four pooled
  bottleneck vectors predict the stopped-gradient target visual fields.

For high and terminal cells, evaluate the same noisy strip under ordered and
shuffled states. Let `c_order` and `c_shuffle` be target-field cosine. The
condition-use loss is

\[
L_{order}=\operatorname{softplus}
\left(0.02-(c_{order}-c_{shuffle})\right).
\]

The fixed total is

\[
L=L_v+0.10L_{fg}+0.10L_{dice}+0.10L_{field}
+0.10L_{repa}+0.25L_{order}.
\]

No target identity, character label, evaluator index, OCR output, teacher
logit, or candidate bank enters any term.

## Fixed Optimization

| Setting | Value |
| --- | ---: |
| updates | 8,000 |
| physical batch | 8 strips |
| optimizer | AdamW |
| learning rate | `2e-4` new modules, `5e-5` initialized U-Net |
| warmup | 400 updates |
| cosine minimum LR ratio | `0.10` |
| weight decay | `0.01` |
| gradient clip | `1.0` |
| EMA | `0.999` |
| precision | BF16 |
| seed | `20265310` |
| dataset seed | `20265311` |
| hardware | CUDA device 0, one RTX 4090 |
| allocated-memory ceiling | 18 GiB |

The checkpoint must include model and EMA states, optimizer groups, update,
RNG states, source/protocol/data/foundation hashes, initialization receipt,
boundary receipt, elapsed time, and peak allocated memory.

## Fixed Sampling

- start from four independent Gaussian-noise cells at level 99;
- use 32 deterministic DDIM levels;
- apply classifier-free guidance with fixed scale `2.0`;
- generate one primary sample per context;
- do not rerank, reject, repair, or select from multiple samples; and
- threshold the final signed strip at zero to obtain four visible binary cells.

Ordered, shuffled, and zero-condition routes use byte-identical starting noise.
No image bank or OCR participates in sampling.

## Fixed Development Audit

The locked audit uses:

- 512 windows for teacher-forced denoising and condition-use diagnostics;
- 256 paired-noise direct block generations;
- 64 prefixes for two generated four-cell blocks of visible feedback;
- evaluator-only 1,024 canonical glyph images instantiated after generation;
  and
- a deterministic gallery containing target block, ordered sample, shuffled
  sample, zero-condition sample, and the V42 direct first-cell plan.

Report per-position and block-mean identity top-1, pixel F1, ink fraction,
isolated ink, blank rate, connected components, distinct evaluator identities,
ordered/shuffled pixel displacement, and feedback identity. The first sample
is the only autonomous generation result.

## Calibration Gates

V53 is retained only if every condition holds:

1. the runtime boundary is image-only and all frozen hashes match;
2. the denoiser and combined model remain within their parameter caps;
3. all values are finite and peak allocated memory is below 18 GiB;
4. ordered high/terminal denoising target-field cosine exceeds shuffled by
   more than `0.01`;
5. ordered first-position identity is at least `0.05`;
6. ordered first-position identity exceeds both shuffled and zero-condition
   identity by more than `0.02`;
7. ordered four-position mean identity is at least `0.03` and exceeds shuffled
   by more than `0.01`;
8. ordered four-position mean pixel F1 is at least `0.48`;
9. ordered blank rate is below `0.02`, isolated-ink fraction is at most `0.01`,
   and generated-to-target ink ratio lies in `[0.70,1.50]`;
10. at least 64 evaluator identities appear among the 1,024 generated cells;
11. ordered two-block visible-feedback identity exceeds shuffled feedback; and
12. the evaluator bank remains absent from training, denoising, sampling, and
    selection.

One failed gate rejects this complete mechanism. A failed audit may motivate a
new protocol, but it cannot tune guidance scale, sample count, thresholds,
noise strata, or update count retroactively.

## Claim Boundary

A passing V53 would establish a compact, token-free mechanism that reads
Chinese glyph images and directly generates a context-dependent four-glyph
image block on one RTX 4090. It would not establish broad semantics,
instruction following, factual knowledge, etymology explanation, historical
form control, page-scale generation, human-equivalent reading, Qwen parity, or
superior capability-normalized efficiency.
