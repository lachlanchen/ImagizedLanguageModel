# Codec-Spherical Glyph Language V47: Frozen Development Protocol

Status: preregistered before V47 implementation, smoke training, or production
training

Date frozen: 2026-08-15

## Question And Allowed Claim

Can the successful V42 isolated-glyph causal setup learn a better image-native
Chinese language model when its fixed DCT field is replaced by the qualified
V34 learned visual manifold, every field is constrained to a unit sphere,
train-only long-history counterfactuals are consumed without repetition, and
autonomous proposals are selected only after visible decode and reread?

V47 tests bounded next-glyph continuation and raster generation. It does not
test instructions, etymology, page reading, historical-form correspondence,
general semantics, a local-LLM teacher, or parity with a text LLM.

## Immutable Inputs

### Natural corpus

- file: `data/visual_grammar/chinese_wikisource_public_domain.jsonl`;
- SHA-256:
  `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`;
- record partition, original-plus-offline-simplified views, canonical font,
  font size, 64-cell context, and shifted 64-cell targets are exactly V42's;
  and
- student natural batches contain only floating rasters shaped
  `B x 64 x 1 x 32 x 32`.

### Frozen V34 retina/actuator

- checkpoint:
  `artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt`;
- checkpoint SHA-256:
  `a138c9cb3b0502e43d1227f689c020893d56b468742c32e1840e44d299662f33`;
- selected route: complete EMA shadow at update 6,000;
- EMA tensor-state SHA-256:
  `140e6d68d2be3bcbcdb6fb74b27a8f1258caba54bc0f0888ba8acecc48c22edb`;
- architecture: `continuous-glyph-representation-codec-v34`;
- parameters: `7,423,361`, all frozen in V47; and
- the selected state must be embedded in the V47 checkpoint so runtime does
  not depend on a separate process or network service.

### Train-only binding sequence

- construction: existing `build_factorized_suffix_pairs`;
- split: `train`;
- suffix: four raster cells;
- different source identifiers and different next images required;
- seed: `20264702`;
- count: `80,000`;
- unique suffixes: `80,000`;
- each pair is consumed exactly once;
- ordered sequence SHA-256:
  `2f573c4c79deb9e2bf97c2b0af588a438c7a43280bd632c05c0ae477ec6918eb`;
  and
- candidate columns are deterministically permuted per row.

Strings and metadata may construct host-side raster batches and evaluator
labels. They may not enter the V34 field, causal reader, generator, deployed
writer, or recurrent state.

## Fixed Spherical Codec Field

For ink-positive raster `x`, V47 uses the frozen V34 EMA encoder `E` and
decoder `D`:

```text
e(x) = E(1 - 1[x >= 0.5])
c(x) = e(x) / max(||e(x)||_2, 1e-8)

white_logits(c) = D(sqrt(768) * c)
p_ink(c) = sigmoid(-white_logits(c))
x_hat(c) = 1[p_ink(c) >= 0.5].
```

The deployed reread is `R(c)=c(x_hat(c))`. Decoder probabilities may be
reread differentiably during training, but recurrent inference always uses the
thresholded visible raster.

The field has zero trainable parameters. There is no quantizer, codebook,
nearest-neighbor assignment, glyph lookup, character classifier, or learned
radius.

## Mandatory Field Preflight

Before language training, audit the frozen wrapper on the pinned 1,024-image
bank. All must pass:

1. canonical binary reconstruction ink F1 exceeds `0.995`;
2. canonical encode--decode--reread cosine exceeds `0.995`;
3. Noto Sans CJK Bold to canonical retrieval top-1 exceeds `0.95`;
4. Noto Serif CJK Medium to canonical retrieval top-1 exceeds `0.90`;
5. every field, decoder output, and reread field is finite; and
6. field trainable parameters equal zero.

Failure stops production language training.

## Fixed Causal Model

```text
field width                 768
model width                 384
causal layers                 8
attention heads               6
MLP ratio                     3
maximum context              64 raster cells
noise width                 128
generator layers              4
generator samples             4
```

The reader is initialized from scratch. No V42, V43, V44, V46, PIXAR, Qwen,
or other language-model parameter is loaded. Every proposal and anchor is
normalized to the unit sphere. V34 remains frozen.

The model must remain below 32 million total parameters including V34 and
below 25 million trainable parameters. Exact counts are recorded from the
instantiated model.

## Fixed Natural Loss

For anchor `a`, target field `c`, hidden state `h`, and four unit generated
fields `g_r`:

```text
L_nce    = dynamic exact-raster multi-positive contrastive loss
L_anchor = mean(1 - a^T c)
L_pixel  = ink BCE + 0.5 soft Dice after frozen V34 decoding of a
L_energy = 2 E_r ||g_r-c||_2 - E_(r != s) ||g_r-g_s||_2
L_sample = mean(1 - max_r g_r^T c)
L_cycle  = mean(1 - g_r^T soft_reread(g_r))
L_spixel = decoded ink BCE + 0.5 soft Dice for the best-target-cosine sample

L_natural = 1.00 L_nce + 0.25 L_anchor + 0.20 L_pixel
          + 0.50 L_energy + 0.25 L_sample
          + 0.10 L_cycle + 0.10 L_spixel.
```

The differentiable cycle uses frozen decoder probabilities and the frozen
encoder. It does not replace thresholded inference.

Maximum positions per natural microbatch:

```text
contrastive positions       512
energy/cycle positions      128
energy samples                4
```

## Fixed Counterfactual Loss

For pair `b`, two histories `q in {0,1}`, and two permuted candidate raster
fields `k in {0,1}`:

```text
S_bqk = alpha * a_bq^T c(candidate_bk)
L_pair = row_cross_entropy(S, assignment).
```

Each optimizer update consumes one batch of eight previously unseen pairs.
The final objective per update is

```text
L = mean of two natural microbatch losses + 1.00 L_pair.
```

No pair is revisited in the 10,000-update production run.

## Fixed Optimization

```text
updates                         10,000
natural microbatch size              8
natural gradient accumulation        2
natural examples/update             16
pair batch size                       8
unique pair examples/update           8
optimizer                         AdamW
betas                          (0.9, 0.95)
learning rate                     3e-4
warmup updates                      500
minimum LR ratio                   0.10
weight decay                       0.05
gradient clip                       1.0
precision                           BF16
model seed                    20264700
natural dataset seed          20264701
pair seed                     20264702
```

The 80,000 pair rows are consumed in their frozen order. Natural data order is
independent of pair order. Training starts from initialization and runs on CUDA
device 0 only.

## Fixed Autonomous Inference

For each context, generate four latent proposals. Decode and threshold every
proposal, reread every visible raster, and select the proposal whose reread
field has maximum cosine with the predicted anchor. The selected visible
raster is output and is the only state appended for another step.

No evaluator candidate bank is consulted before output selection.

## Fixed Development Audit

Reuse V42/V46 exactly:

- seed `20264220` for the common natural audit;
- 2,048 deterministic development windows;
- 1,024 evaluator-only candidate raster images;
- full, suffix-4, last-only, suffix-preserving shuffled, and blank contexts;
- training-only image-unigram and symbolic-bigram controls;
- 512 same-suffix, different-target development pairs;
- 256 bank-free generated examples with four proposals; and
- visible raster rereading in autoregressive inference.

Additionally report:

- preflight metrics;
- proposal-to-reread and selected-proposal-to-reread cosine;
- anchor and generated finite rates;
- canonical reconstruction ceiling;
- V34 checkpoint and EMA-state digests;
- 80,000-pair sequence digest and consumed-row count;
- trainable/total parameters, elapsed time, and peak VRAM; and
- deterministic target/generated/reread galleries.

The frozen record partition remains unopened.

## Conjunctive Gates

V47 qualifies only if every condition is true:

1. mandatory field preflight passes;
2. full top-1 exceeds image unigram by more than `0.03`;
3. full top-1 exceeds symbolic bigram by more than `0.01`;
4. full target log probability exceeds shuffled by more than `0.05` nat;
5. full top-1 exceeds shuffled by more than `0.015`;
6. full top-1 exceeds V46's `0.20751953125` by more than `0.01`;
7. counterfactual full-history arm accuracy exceeds `0.60`;
8. bank-free generated identity top-1 exceeds image unigram;
9. generated identity top-1 exceeds V46's `0.0859375` by more than `0.01`;
10. generated binary pixel F1 exceeds `0.55`;
11. generated blank rate is below `0.02`;
12. mean selected proposal-to-visible-reread cosine exceeds `0.90`;
13. the image-only student and recurrent boundary audits pass;
14. all 10,000 updates and exactly 80,000 unique pair rows complete;
15. total parameters are below 32 million and trainable parameters below 25
    million; and
16. training plus fixed audit finish within 35 minutes with peak allocated
    CUDA memory below 18 GiB on one RTX 4090.

Strict comparisons use epsilon `1e-12`; metrics are not rounded for decisions.

## Stop And Decision Rules

- Smoke runs validate shapes, gradients, codec loading, pair order, visible
  feedback, checkpoint reload, and evaluator plumbing only.
- Once production begins, no loss, weight, threshold, seed, pair row, update
  count, audit example, or selection rule may change in response to results.
- One failed gate makes V47 non-qualifying. Report it without changing this
  protocol.
- Development failure keeps the frozen record partition closed.
- A passing V47 supports only the bounded allowed claim. Historical writing,
  instruction following, etymology, and general language remain separate
  preregistered stages.

