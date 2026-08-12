# Visual Relation Circuit V23 Protocol

Date preregistered: 2026-08-13

Status: fixed before V23 implementation, smoke training, or evaluation

## Question

Can a compact image-only student compose query-to-label visual equality with a
visible same/other operation, route one of two previously unseen Chinese glyph
images, and emit the requested glyph as a new image under exact visual
counterfactuals?

V22 failed because one soft selector collapsed onto the operation frame. V23
removes that path. It factorizes visual canonicalization and relational binding,
qualifies the writer first, freezes it, and then trains a constrained visual
relation circuit.

## Claims Fixed In Advance

After development success only, the maximum allowed claim is:

- a compact image-only model follows the fixed two-pair same/other visual
  grammar and writes requested unseen Chinese forms on development data.

Even after frozen success, forbidden claims include open-domain language
understanding, arbitrary prompts, factual etymology, historical-form synthesis,
book continuation, page or movie generation, Qwen parity, and efficiency
superiority.

## Visual Stream Grammar

Every prompt has shape `[B,6,1,32,32]` and order:

```text
[label_1, glyph_1, label_2, glyph_2, operation, query_label]
```

The answer stream has shape `[B,1,1,32,32]`. Labels use the visible pairs
`甲/乙`, `天/地`, `上/下`, and `左/右`. Operations are visible `同` and `异`:

- `同`: answer with the glyph bound to the query label;
- `异`: answer with the glyph bound to the other label.

The renderer also creates, from the same episode:

1. a query counterfactual changing only frame 6;
2. an operation counterfactual changing only frame 5;
3. a pair swap exchanging frames `(1,2)` with `(3,4)`; and
4. the corresponding canonical answer images.

Strings and metadata are deleted before every model call. The student receives
only float image tensors.

## Identity And Composition Split

Build the same 1,024-most-frequent supported-Han bank from
`data/visual_grammar/chinese_wikisource_public_domain.jsonl`, excluding every
label and operation marker.

Identity salt: `visual-relation-circuit-v23`.

Partition each character by the first 64 bits of
`sha256(salt + NUL + character)`:

- `<0.80`: training;
- `0.80..0.90`: development; and
- `>=0.90`: frozen.

The identifier-only receipt, computed without rendering development or frozen
images, is fixed as:

- training identities: `817`;
- development identities: `109`;
- development identifier SHA-256:
  `6e89f898a17028125a060deec8249bbf35b4d02f898f716f2f519a29cd314170`;
- frozen identities: `98`; and
- frozen identifier SHA-256:
  `206efd6fa2a0e640368a178c61f2f82ee737260afcaed6e97226bfef1f366d0c`.

The V23 held-out operation/label combinations are `(异, 天/地)` and
`(同, 左/右)`. Every individual operation and label pair occurs in training,
but those two compositions do not.

## Student Boundary

The learned path may receive only prompt/source images, continuous frozen-retina
states of those images, routed continuous source pixels, and generated image
pixels. It may not receive strings, token IDs, Unicode IDs, OCR, character or
operation labels, slot/target indices, a visual codebook, glyph lookup,
candidate identity embeddings, evaluator scores, or an external language
model.

The frozen retina remains:

```text
artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt
SHA-256 90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe
```

No V17--V22 writer or binder weight may enter V23. The canonicalizer and
relation circuit start from their fixed V23 seeds.

## Stage A: Visual Canonicalizer

### Architecture

The canonicalizer maps one noncanonical `32x32` source-glyph image to one
canonical target image. It is a `1,122,081`-parameter convolutional U-Net:

- width `32` at `32x32`, width `64` at `16x16`, width `96` at `8x8`;
- two stride-2 `4x4` down-convolutions;
- bilinear upsampling and skip concatenation;
- residual GroupNorm/SiLU `3x3` blocks, including bounded dilation `2`;
- one output `3x3` convolution; and
- residual input logits `2(x-0.5)` added before the sigmoid.

There are no coordinates, identity embeddings, target font IDs, attention
queries, or global character table. The pixel route is intentionally retained
so topology absent from the frozen global retinal state is not discarded.

### Training

Each episode supplies both source glyphs paired with their canonical targets.
The operation/query renderer chooses the pair offline, but no metadata enters
the canonicalizer.

- AdamW, learning rate `5e-4`, betas `(0.9,0.95)`, weight decay `0.02`;
- 50-step linear warmup and cosine decay to `0.1` of base rate;
- gradient clip `1.0`, BF16, episode batch `64` (effective image batch `128`);
- exactly `1,200` updates;
- validation every `200` updates on `512` deterministic development episodes;
- training seed `20260824`, dataset seed `20260825`; and
- smoke mode at most `20` updates and permanently non-evidentiary.

The loss is weighted BCE plus Dice, `0.5` pixel L1, and `0.2` horizontal/vertical
edge L1. Stroke-positive pixel weight is `4.0` total.

### Selection

Rank eligible checkpoints by pixel F1, global identity top-1, target cosine,
then earlier step. Every gate must pass:

1. pixel F1 `>0.72`;
2. identity top-1 over all 109 development identities and four independent
   source views `>0.80`;
3. generated target cosine `>0.85`;
4. pixel-F1 gain over the raw noncanonical source `>0.12`;
5. pixel-F1 gain over source-shuffled output `>0.25`;
6. identity-top-1 gain over source-shuffled output `>0.70`;
7. mean ink fraction in `[0.03,0.50]`;
8. clean image-only boundary; and
9. frozen images instantiated equals zero.

If no checkpoint selects, Stage B is forbidden. If one selects, copy its exact
selected checkpoint, code/protocol hashes, and development metrics into the
Stage B receipt and freeze every canonicalizer parameter.

## Stage B: Visual Relation Circuit

### Architecture

For normalized global retinal states, compute

\[
c_i=\langle R(q),R(l_i)\rangle,
\qquad
m_i=\operatorname{softmax}_{i\in\{1,2\}}(\tau c_i).
\]

The positive temperature `tau` is one learned scalar initialized to `8.0` and
bounded to `[1,32]`. The operation reader is LayerNorm(`192`), Linear
`192->128`, SiLU, Linear `128->1`, followed by sigmoid:

\[
s=\sigma(U_\theta(R(o))).
\]

Compose

\[
w_i=s m_i+(1-s)(1-m_i),qquad
x_r=w_1g_1+w_2g_2,qquad
\hat y=C_\psi(x_r).
\]

`C_psi` is the selected frozen Stage A canonicalizer. It receives only the
routed source pixels. The trainable relation circuit has exactly `25,602`
parameters: two 192-dimensional continuous null states, the operation reader,
and the temperature scalar. Candidate, query-blind control, and operation-blind
control have identical parameter names and shapes.

### Arms

- `relation_aware`: use every visual state;
- `query_blind`: replace only `R(q)` with the learned normalized null-query
  state; and
- `operation_blind`: replace only `R(o)` with the learned normalized
  null-operation state.

Both null parameters exist in every arm and use seeded zero-mean Gaussian
initialization with standard deviation `0.02`.

### Training

Each batch jointly processes original, query-counterfactual,
operation-counterfactual, and pair-swapped image prompts. Their separately
rendered answer images provide all supervision. No route, operation, query, or
target-index label enters a loss.

- AdamW, learning rate `1e-3`, betas `(0.9,0.95)`, weight decay `0.01`;
- 25-step warmup and cosine decay to `0.1` of base rate;
- gradient clip `1.0`, BF16, episode batch `64`;
- exactly `600` updates per arm, sequentially on one RTX 4090;
- validation every `100` updates on `512` deterministic development episodes;
- relation seed `20260826`, dataset seed `20260827`; and
- smoke mode at most `20` updates and permanently non-evidentiary.

The four output branches use the same Stage A topology loss. Add frozen-retina
generated-target cosine with weight `0.10`. Do not add attention entropy,
operation classification, target-slot supervision, or pair-order labels.

### Structural Tests Before Training

Tests must establish:

1. exact prompt/answer stream shapes and floating tensor types;
2. no integer/string tensor enters any model `forward`;
3. frozen retina and canonicalizer receive no gradients;
4. exact `1,122,081` canonicalizer and `25,602` relation parameter counts;
5. exact candidate/control parameter-name and shape equality;
6. route weights are finite, nonnegative, and sum to one;
7. pair swapping preserves routed pixels and output within `1e-6`;
8. changing only the query has exactly zero effect in `query_blind`;
9. changing only the operation has exactly zero effect in `operation_blind`;
10. candidate output backpropagates to query, labels, operation, and both source
    glyph images;
11. the writer cannot read labels, operation, query, or metadata;
12. checkpoint round-trip preserves configuration and boundary receipt; and
13. development construction does not instantiate frozen images.

### Candidate Selection

Rank eligible candidate checkpoints by the minimum of query and operation
switch accuracy, identity top-1, pixel F1, then earlier step. All gates must
pass on `512` development episodes (`2,048` generated prompt variants):

1. binary choice accuracy over all variants `>0.95`;
2. query-counterfactual paired switch accuracy `>0.90`;
3. operation-counterfactual paired switch accuracy `>0.90`;
4. held-out-combination minimum switch accuracy `>0.85`;
5. pair-swap identity consistency `>0.99`;
6. pair-swap output pixel L1 `<1e-6`;
7. global identity top-1 over at least 96 unseen identities `>0.75`;
8. generated pixel F1 `>0.68`;
9. generated target cosine `>0.82`;
10. query-counterfactual output pixel L1 `>0.12`;
11. operation-counterfactual output pixel L1 `>0.12`;
12. visual query-to-label match accuracy `>0.98`;
13. evaluator-only operation-gate accuracy `>0.98` and mean same/other gate
    separation `>0.80`;
14. clean student boundary; and
15. frozen images instantiated equals zero.

Operation-gate and match labels are evaluator-only diagnostics. They cannot
enter training or inference.

### Control Selection And Fresh Paired Gate

The query-blind control selects by its exact query-invariance receipt, then
pixel F1 and earlier step; the operation-blind control analogously selects by
operation invariance. Neither has a quality minimum.

Only selected checkpoints enter one fresh `1,024`-episode paired development
audit, batch `64`, four identity-bank views, seed `dataset_seed + 2,000,003`.
It must verify:

1. candidate query-switch accuracy exceeds query-blind by `>0.40`;
2. candidate operation-switch accuracy exceeds operation-blind by `>0.40`;
3. candidate identity top-1 exceeds each control by `>0.30`;
4. candidate query-output L1 exceeds query-blind by `>0.10`;
5. candidate operation-output L1 exceeds operation-blind by `>0.10`;
6. candidate retains every arm-specific gate;
7. each control retains exact intervention and sealed-split gates; and
8. all parameter names and shapes remain equal.

Endpoint substitution is forbidden. The paired evaluator must refuse before
data construction if Stage A or any required arm lacks a selected checkpoint,
if a smoke checkpoint is supplied, or if any protocol/source/checkpoint hash
differs.

## Blinded And Frozen Policy

After automatic and paired gates pass, create a fixed 48-episode development
review. Show the six prompt images and generated answer without text
transcription or answer labels. Reviewers choose which of the two visible source
glyphs the answer depicts. Required accuracy is at least `44/48`, including at
least `11/12` held-out-combination episodes.

Only after that numeric gate passes may a separate evaluator instantiate frozen
identity images once. It must verify every prior receipt and rerun the candidate
gates on `1,024` frozen episodes without model selection or threshold changes.

## Stop Rules

- Do not tune V23 after the first non-smoke Stage A or Stage B run.
- Do not promote an endpoint when selection fails.
- Do not inspect blinded or frozen images before their gates authorize it.
- Do not call routed copying or canonicalization general language understanding.
- Preserve negative logs, hashes, checkpoints, figures, and refusal behavior.
