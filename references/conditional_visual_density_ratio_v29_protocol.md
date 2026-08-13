# Conditional Visual Density-Ratio Field V29 Protocol

Date frozen: 2026-08-13

Status: preregistered before V29 implementation, smoke testing, or development
measurement

## Decision Question

Does a candidate-conditioned visual critic, trained on the score increment
from exact suffix to full context, bind earlier image-only Chinese history to
the correct arbitrary next-glyph image?

V29 is a development experiment. It may authorize one later frozen evaluation
and a separately preregistered continuous writer only if every selection gate
below passes. Smoke runs test plumbing only and cannot revise this protocol.

## Fixed Data

- manifest:
  `data/visual_grammar/chinese_wikisource_public_domain.jsonl`;
- required manifest SHA-256:
  `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`;
- identifier and font partitions: unchanged from V25-V28;
- train script views: original and OpenCC simplified when distinct;
- development script views: original and simplified under the fixed evaluator;
- context cells: 64;
- target cells: 1;
- authoritative natural sample shape: `[65,1,32,32]` per font view;
- implementation may load the existing 68-cell V28 record but must expose only
  cells `0..64` to V29;
- exact suffix intervention: final 4 context cells;
- rendering: existing 32-cell renderer, two distinct training fonts, font size
  24-28, at most one-pixel translation, and bounded intensity/blur/noise;
- canonical equality witness: unaugmented Noto Sans CJK Regular pixels; and
- frozen identifiers and frozen fonts: unopened.

Training pairs are the same 32,768 deterministic suffix-4 pair specifications
used by V28. Pair targets differ, record identifiers differ, final four writing
forms match, rendered suffix pixels must match exactly, and candidate order is
independently randomized for each font view.

### Fixed natural candidate bank

The host constructs 1,024 candidate identities from training-partition target
frequency with canonical-pixel deduplication and deterministic lexical
tie-breaking. It renders two unaugmented image views using the first two V25
training fonts. The bank manifest records canonical pixel SHA-256 values, font
paths, image hashes, and ordering.

Natural windows are restricted to targets whose canonical pixels occur in the
bank. The positive column is recovered only by exact canonical-pixel equality.
The student receives candidate images or features produced by its frozen image
encoders, never the bank index or source string.

The bank may be cached in host memory during training. It must not be a model
parameter, persistent model buffer, checkpoint payload, or inference
requirement. Pair candidates remain independently rendered arbitrary images
and are not restricted to the bank.

## Fixed Initialization

V29 initializes from the rejected but mechanically valid V28 final checkpoint:

`artifacts/dense_visual_future_energy_v28_evidence/checkpoint_final.pt`

Required SHA-256:

`22503464cf5f5e8ed2d6adebbd6c794f6bc9b2836f978872027cb51712c7f64f`

The following V28 components load exactly:

- frozen V16 retina;
- online semantic adapter;
- EMA semantic adapter;
- context input projection;
- eight causal context blocks; and
- context RMS normalization.

The V28 horizon embedding, future trunk, raw/semantic future queries, mixture
head, and learned scales are discarded. Both semantic adapters and the retina
remain frozen for all V29 updates. The context input, context blocks, and
context normalization remain trainable at the context learning rate. The new
candidate evidence field initializes randomly from the fixed seed.

The V16 retina embedded in V28 must retain SHA-256 source receipt:

`90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe`.

## Fixed Student Boundary

Allowed learned-path tensors:

- context images `[B,T,1,32,32]`, `1 <= T <= 64`;
- arbitrary candidate images `[N,1,32,32]` or `[B,K,1,32,32]`;
- frozen raw/semantic features computed from those candidate images inside the
  same image path;
- temporary positional assignment columns for the pair loss; and
- floating augmentation/noise values internal to image rendering.

Forbidden learned-path values:

- strings, bytes, or text transcripts;
- token, Unicode, character, vocabulary, or glyph IDs;
- OCR output;
- tokenizer or discrete visual-codebook indices;
- vocabulary embeddings or vocabulary logits;
- glyph lookup or a required inference candidate bank;
- symbolic n-gram probabilities;
- teacher logits or hidden states; and
- external language-model calls.

Host data code may use strings to locate source spans and render images. Host
loss/evaluation code may use canonical pixels to form temporary equality
groups and symbolic controls. Those values never enter a model method or
checkpoint.

## Fixed Representation

The model-facing writing object is

\[
X\in[0,1]^{B\times N\times1\times32\times32}.
\]

`N` is reading time. A 2D page fold or a `32 x 32 x N` description is an
invertible view of the same ordered cells, not an added latent channel. V29
does not train a page fold, 3D convolution, movie model, or character
segmentation module.

## Fixed Architecture

Model name: `conditional-visual-density-ratio-v29`.

| Component | Fixed value |
|---|---:|
| cell size | 32 |
| maximum context | 64 cells |
| exact suffix | 4 cells |
| raw retinal dimension | 192 |
| semantic visual dimension | 192 |
| causal field width | 384 |
| causal blocks | 8 |
| causal attention heads | 6 |
| causal MLP ratio | 3.0 |
| causal dropout | 0.05 |
| candidate query width | 384 |
| candidate-to-context layers | 2 |
| candidate attention heads | 6 |
| candidate MLP ratio | 2.0 |
| candidate dropout | 0.05 |
| relation hidden width | 384 |
| shared-bank score chunk | 128 candidates |
| trainable-parameter cap | 22 million |
| total-parameter cap | 25 million |

For image `x`, fixed V28 perception gives

\[
r(x)=\operatorname{normalize}(R_0(x)),\qquad
z(x)=\operatorname{normalize}(A_{28}(r(x))).
\]

The V28-initialized causal field maps context images to retained per-cell
states

\[
H_T=C_\theta([r(x_1);z(x_1)],\ldots,[r(x_T);z(x_T)])
\in\mathbb R^{T\times384}.
\]

The candidate query is

\[
q_0=W_y[r(y);\bar z(y)].
\]

Each of two pre-normalized evidence layers applies six-head cross-attention
from the candidate query to all context states, followed by a residual MLP:

\[
q_{l+1}=q_l+\operatorname{MHA}_l(\operatorname{LN}(q_l),
\operatorname{LN}(H_T),\operatorname{LN}(H_T)),
\]

followed by

\[
q_{l+1}\leftarrow q_{l+1}+
\operatorname{MLP}_l(\operatorname{LN}(q_{l+1})).
\]

The relation head receives
`[q_2; q_2-q_0; q_2 elementwise-multiplied-by q_0]`, applies layer
normalization, a width-384 SiLU hidden layer, and a scalar linear output:

\[
\rho_\theta(H_T,y)\in\mathbb R.
\]

There is no candidate-ID embedding and no learned bank-sized output layer.
Context states are computed once. Shared candidates are scored in fixed chunks
without changing the result.

Define

\[
F(P,S,y)=\rho_\theta(H(P,S),y),\qquad
B(S,y)=\rho_\theta(H(S),y),
\]

\[
G(P,S,y)=F(P,S,y)-B(S,y).
\]

For a candidate matrix with `K` columns, the assignment delta is row-centered:

\[
\widetilde G_{ij}=G_{ij}-K^{-1}\sum_kG_{ik}.
\]

## Fixed Corruptions

For every full context, the suffix-preserving shuffle independently permutes
cells `0..59` and leaves cells `60..63` and candidate pixels unchanged. The
permutation must not equal the identity; deterministic rotation by one is used
if an RNG draw is identity.

No masking, glyph replacement, OCR corruption, or candidate-image corruption
is added. Existing bounded renderer augmentation remains allowed for training
views.

## Fixed Losses

All natural and pair losses average both cross-font directions.

### Natural full, suffix, and incremental contrast

For each natural row `i`, all 1,024 training-bank images are candidates and
`t_i` is located by exact canonical-pixel equality. Let `CE` denote standard
row-wise cross entropy:

\[
L_{full}=CE(F,t),\qquad L_{suffix}=CE(B,t),
\]

\[
L_{increment}=CE(\widetilde G,t).
\]

Let `G_shuf=F_shuf-B`, row-centered over the same candidates. If `logp` is the
target log probability under a row softmax,

\[
L_{natural-order}=\operatorname{softplus}
\left[0.10-(\log p_{G}(t)-\log p_{G_{shuf}}(t))\right]
\]

averaged per row.

### Exact-suffix pair assignment

For each pair batch, compute two-by-two matrices `F`, `B`, `G`, `F_shuf`, and
`G_shuf`. The two suffix rows must be bitwise equal before scoring. Apply the
existing symmetric row/column assignment loss `A` independently to full score
and row-centered incremental score:

\[
L_{pair-full}=A(F,\pi),\qquad
L_{pair-increment}=A(\widetilde G,\pi).
\]

For every row `i`, let

\[
m_i(M)=M_{i,\pi(i)}-M_{i,1-\pi(i)}.
\]

The per-row positive and order losses are

\[
L_{pair-positive}=\frac12\sum_i
\operatorname{softplus}[0.10-m_i(\widetilde G)],
\]

\[
L_{pair-order}=\frac12\sum_i
\operatorname{softplus}
[0.10-(m_i(\widetilde G)-m_i(\widetilde G_{shuf}))].
\]

Margins are penalized per row before averaging. No pair-level mean may allow
one row to compensate for another.

### Total

\[
L=L_{full}+0.50L_{suffix}+L_{increment}
+0.50L_{natural-order}+L_{pair-full}
+4L_{pair-increment}+L_{pair-positive}+L_{pair-order}.
\]

No identity loss is trained because both validated V28 visual adapters are
frozen. No future-vector regression, energy score, writer, pixel
reconstruction, symbolic classification, distillation, or frozen-partition
loss is permitted.

## Fixed Optimization

| Setting | Value |
|---|---:|
| updates | 8,000 |
| natural examples per microbatch | 8 |
| suffix pairs per microbatch | 8 |
| gradient accumulation | 2 |
| optimizer | AdamW |
| betas | `(0.9,0.95)` |
| evidence-field learning rate | `3e-4` |
| V28-context learning rate | `6e-5` |
| minimum cosine LR ratio | `0.10` |
| warmup updates | 400 |
| weight decay | `0.05` |
| gradient clip | `1.0` |
| seed | `20260920` |
| dataset seed | `20260921` |
| pair seed | `20260922` |
| candidate-bank seed | `20260923` |
| precision | BF16 |
| hardware | one RTX 4090, CUDA device 0 |
| peak allocated CUDA cap | 18 GiB |

The candidate bank is encoded once per evidence run by the frozen visual
encoders and then held as detached floating tensors. It is not serialized.
The evidence command must use these settings exactly. `--smoke`, exploratory
arguments, fewer candidates, changed steps, or a changed checkpoint make an
output ineligible as fixed evidence.

## Fixed Development Audit

- 2,048 deterministic natural windows;
- 512 deterministic cross-record suffix-4 pairs from a new fixed seed;
- 1,024 candidate identities;
- two unseen development fonts;
- natural scoring chunk size 128 and context batch size 16;
- full, suffix-4, and suffix-preserving shuffled scores;
- full-minus-suffix and shuffled-minus-suffix incremental scores;
- image unigram, symbolic bigram, and symbolic trigram controls;
- raw-retina and frozen-semantic cross-font retrieval on the same 1,024 images;
- raw-retina two-candidate pair identity;
- exact candidate-column permutation replay for `F`, `B`, and `G`;
- exact suffix-pixel and suffix-score equality;
- train-bank absence from model state and checkpoint;
- student-boundary receipt and peak allocated CUDA memory; and
- no frozen images instantiated.

Natural metrics report top-1, top-5, and target log probability for full,
suffix, shuffled, increment, and shuffled increment scores. Pair metrics report
tie-aware arm accuracy, both-correct rate, mean per-row margin, balanced row
accuracy, and ordered-minus-shuffled gains for full and incremental scores.

The symbolic controls are built only from training strings by the evaluator.
They never enter the model or checkpoint. Trigram remains diagnostic because
its sparse backoff coverage is not comparable to the dense student
distribution. Unigram and bigram are fixed selection controls.

## Fixed Selection Gates

### Mechanism gates

All must pass:

1. incremental pair arm accuracy `>0.65`;
2. incremental pair both-correct rate `>0.40`;
3. incremental pair arm accuracy minus shuffled-increment arm accuracy
   `>0.10`;
4. incremental mean margin minus shuffled-increment mean margin `>0.05`;
5. full pair arm accuracy `>0.65`;
6. full pair both-correct rate `>0.40`;
7. suffix-4 arm accuracy equals tie-aware chance `0.5` within `1e-6`;
8. suffix pixels and the two suffix score rows are exactly equal;
9. raw-retina two-candidate identity accuracy `>=0.99`;
10. frozen-semantic 1,024-way cross-font identity top-1 `>=0.95`;
11. candidate-permutation maximum score error for each of `F`, `B`, and `G`
    `<1e-5`, with recovered accuracy agreement `1.0`;
12. the student boundary is clean and the training bank is absent from model
    state and checkpoint;
13. peak allocated CUDA memory `<18 GiB`; and
14. frozen images instantiated is false.

### Language gates

All must pass:

1. full top-1 `>=0.15`;
2. full top-1 minus image-unigram top-1 `>0.03`;
3. full top-1 minus symbolic-bigram top-1 `>0.01`;
4. full target log probability minus symbolic-bigram target log probability
   `>0.05` nat;
5. full target log probability minus suffix-4 target log probability
   `>0.03` nat; and
6. full target log probability minus shuffled-prefix target log probability
   `>0.03` nat.

Incremental natural top-1 and its shuffled gain are diagnostic. They cannot
replace the full-score language gates because the deployed critic must provide
a useful complete next-image score.

Threshold comparisons are strict except stated `>=` and exact controls. No
gate can be changed after the fixed run starts.

## Authorization Rule

If any mechanism or language gate fails:

- classify V29 as rejected on development;
- keep the frozen partition sealed;
- do not train a writer as evidence;
- publish every fixed metric and failed gate; and
- localize whether failure lies in conditional binding, natural coverage, or
  the fixed perception basis before proposing V30.

If every gate passes:

- record the selected checkpoint SHA-256;
- run exactly one frozen audit under a separately committed protocol;
- keep all host strings and symbolic controls outside the student; and
- preregister a continuous image writer that renders and rereads its own output.

In neither case may page geometry, a third spatial/depth axis, historical-form
curriculum, external LLM, or writer be added retroactively to V29.
