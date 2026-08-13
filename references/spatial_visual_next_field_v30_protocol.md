# Spatial Visual Next-Field V30 Protocol

Date: 2026-08-13

Status: preregistered before V30 implementation, smoke testing, optimization,
or development measurement

## Decision Question

Does predicting a continuous spatial next-image field let an image-only causal
student bind earlier Chinese writing images to the correct arbitrary next
image, and does that binding exceed a parameter-identical position-blind visual
control with matched candidate visibility?

V30 is a two-arm development experiment. The candidate arm is
`spatial-field`; the control arm is `global-control`. Both must be implemented,
trained from byte-identical initialized parameter tensors, and audited. No
frozen image may be instantiated unless every spatial, matched-control, and
language gate below passes. Writer training requires a later protocol and is
not authorized by this development experiment.

Smoke runs test plumbing only. They cannot revise this protocol, choose a
checkpoint, or provide language evidence.

## Fixed Prior Evidence

V30 initializes from the completed but rejected V29 checkpoint:

`artifacts/conditional_visual_density_ratio_v29_evidence/checkpoint_final.pt`

Required SHA-256:

`a8ec991968b577518d801090f5953406de13c688552107f26ac400fc2d508b8a`

The V29 audit SHA-256 is:

`16645844dd0b9dd4fb1e5157edbdd20d6e13b34b201c1147c177ac52464a5108`

V29 detected order but failed binding: natural top-1 was `0.0234375`, exact-
suffix full arm accuracy was `0.497314453125`, and increment arm accuracy was
`0.507080078125`. A common suffix baseline cancels from aggregate pair margin,
so V30 does not optimize density-ratio subtraction.

Before this protocol, a parameter-free design probe measured the frozen V29
retina on the fixed 1,024 development-bank images in two development fonts:

| Probe representation | Bidirectional cross-font top-1 |
|---|---:|
| pooled raw vector | `0.921875` |
| aligned `4 x 4 x 192` field | `0.96337890625` |
| position-blind normalized-cell mean | `0.14306640625` |
| spatially reversed aligned field | `0.00341796875` |

The V29 frozen global semantic route reaches `0.96435546875` on the same-scope
audit. This supports a local-versus-global matched-visibility experiment. The
probe is not a V30 checkpoint metric and cannot change the gates below.

## Fixed Data

- manifest:
  `data/visual_grammar/chinese_wikisource_public_domain.jsonl`;
- required manifest SHA-256:
  `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`;
- identifier and font partitions: unchanged from V25-V29;
- train script views: original and OpenCC simplified when distinct;
- development script views: original and simplified under the fixed evaluator;
- context images: 64;
- target images: 1;
- authoritative natural sample shape: `[65,1,32,32]` per font view;
- exact suffix intervention: final 4 context images;
- rendering: existing 32-cell renderer, two distinct training fonts, font size
  24-28, at most one-pixel translation, and bounded intensity/blur/noise;
- canonical identity witness: unaugmented Noto Sans CJK Regular pixels;
- frozen identifiers and frozen fonts: unopened; and
- route arms: exactly the same sampled natural windows, suffix-pair
  specifications, rendered image batches, candidate-column permutations, and
  cross-font directions.

Training uses 32,768 deterministic suffix-4 pair specifications. Each pair has
different record identifiers and different targets; the final four writing
forms match; rendered suffix pixels must be bitwise equal; and candidate order
is independently randomized for each font direction.

### Fixed natural candidate bank

The host builds 1,024 candidate identities from training-partition target
frequency, canonical-pixel deduplication, and deterministic lexical
tie-breaking exactly as in V29. It renders two unaugmented training-font views.
The bank receipt records canonical pixels, image hashes, font paths, and order.

For the first-context direction, candidates use bank view 1. For the second-
context direction, candidates use bank view 0. Natural windows are restricted
to targets represented in the bank. Positive columns are located outside the
student by exact canonical-pixel equality.

The bank is a temporary host-side contrast set. It must not be a model
parameter, persistent buffer, checkpoint payload, deployed inference
requirement, or source of population-normalization statistics. Pair candidates
remain independently rendered arbitrary images and need not be bank members.

## Fixed Seeds

| Purpose | Seed |
|---|---:|
| common model initialization | `20261010` |
| natural training dataset | `20261011` |
| suffix-pair training dataset | `20261012` |
| training candidate bank | `20261013` |
| deterministic development audit | `20261014` |

Each arm starts in a fresh process with the same common model seed. The
initialized model state SHA-256, parameter names, shapes, dtypes, and values
must be exactly equal before the first update. Dropout and data-loader random
streams also restart from the same seed. Route identity may affect gradients
only through the fixed candidate representation described below.

## Fixed Student Boundary

Allowed learned-path tensors:

- context images `[B,T,1,32,32]`, `1 <= T <= 64`;
- arbitrary candidate images `[N,1,32,32]` or `[B,K,1,32,32]` during
  contrastive training and audit;
- frozen continuous global or spatial features computed from those images by
  the same student-owned frozen image path;
- predicted continuous fields `[B,16,192]`;
- temporary positional assignment columns used only by the loss; and
- floating augmentation/noise values internal to image rendering.

Forbidden learned-path values:

- strings, bytes, or text transcripts;
- token, Unicode, character, vocabulary, or glyph IDs;
- OCR output;
- tokenizer or discrete visual-codebook indices;
- vocabulary embeddings or vocabulary logits;
- glyph lookup or a required inference candidate bank;
- symbolic unigram, bigram, or trigram probabilities;
- teacher logits or hidden states; and
- external language-model calls.

Host data code may use strings to find source windows and render images. Host
loss and evaluator code may use canonical pixels to locate temporary positive
columns and may compute symbolic controls. Those values never enter a model
method or checkpoint.

## Fixed Representation

The model-facing written-language input is

\[
X\in[0,1]^{B\times T\times1\times32\times32},\qquad1\leq T\leq64.
\]

`T` is reading time. A `32 x 32 x T` volume or invertible 2D page fold is a
view of the same ordered images, not a hidden character channel. V30 does not
train a page fold, 3D convolution, movie model, character segmenter, or visual
tokenizer.

The model's primary prediction is candidate-independent:

\[
P_\theta(X)\in\mathbb R^{B\times16\times192}.
\]

The 16 continuous rows correspond to a `4 x 4` next-image retinal field. The
deployed model can emit this field without a candidate set. Candidate scoring
is a training and evaluation operation, not the definition of the output.

## Fixed Shared Initialization

Both V30 arms load exactly these V29 modules:

- frozen V16 retina;
- frozen online semantic adapter;
- frozen target semantic adapter;
- context input projection;
- eight causal context blocks; and
- context RMS normalization.

The V29 candidate projection, two candidate-evidence blocks, relation
normalization, and scalar relation head are discarded. The retina and both
semantic adapters remain frozen for every V30 update. Context input, causal
blocks, and context normalization remain trainable.

The frozen retina must retain source receipt SHA-256:

`90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe`

## Fixed Architecture

Architecture name: `spatial-visual-next-field-v30`.

| Component | Fixed value |
|---|---:|
| cell size | 32 |
| maximum context | 64 images |
| exact suffix | 4 images |
| raw retinal dimension | 192 |
| semantic visual dimension | 192 |
| causal width | 384 |
| causal blocks | 8 |
| causal heads | 6 |
| causal MLP ratio | 3.0 |
| causal dropout | 0.05 |
| next-field size | `4 x 4` |
| next-field channels | 192 |
| decoder hidden channels | 192 |
| decoder residual blocks | 2 |
| decoder kernel | 3 |
| decoder MLP ratio | 2.0 |
| decoder dropout | 0.05 |
| initial score temperature | 0.08 |
| shared-bank score chunk | 128 candidates |
| total-parameter cap | 20 million |
| trainable-parameter cap | 18.5 million |

For image `x`, fixed V29 perception gives a pooled raw vector, global semantic
vector, and pre-pooling spatial field:

\[
r(x),R(x)=\operatorname{Retina}(x),\qquad
z(x)=\operatorname{normalize}(A_{29}(r(x))),
\]

where `R(x)` has shape `192 x 4 x 4`. The causal field is unchanged:

\[
H_T=C_\theta([r(x_1);z(x_1)],\ldots,[r(x_T);z(x_T)])
\in\mathbb R^{T\times384}.
\]

A shared next-field decoder maps the final causal state to 16 local states:

\[
Q_0=\operatorname{reshape}_{4\times4}
  (W_{seed}\operatorname{LN}(h_T)+b_{seed}),
\]

\[
Q_{l+1}=Q_l+W_{2,l}\,\sigma
\left(W_{1,l}\operatorname{CLN}
\left(Q_l+\operatorname{DWConv}_{3\times3,l}(Q_l)\right)\right),
\]

\[
P_\theta(X)=\operatorname{normalize}_{channel}
\left(W_{out}\operatorname{CLN}(Q_2)\right).
\]

`CLN` applies layer normalization over channels independently at each spatial
cell. Every new linear or convolutional weight uses zero bias and normal
weight initialization with standard deviation `0.02`, except each residual
output is scaled by `1/sqrt(4)`. No route-specific parameter exists.

The learned logit scale is initialized to `1/0.08` and clamped to at most 100.
Shared candidates are scored in chunks without changing values.

## Fixed Candidate Representations And Scores

### Spatial-field arm

The frozen candidate field is channel-normalized independently at every
retinal location:

\[
u_p(y)=\frac{R_p(y)}{\lVert R_p(y)\rVert_2},\qquad p=1,\ldots,16.
\]

The score reduces only after corresponding patches interact:

\[
s_{sp}(X,y)=\tau\frac1{16}\sum_{p=1}^{16}
P_{\theta,p}(X)^\top u_p(y).
\]

No development-bank mean or other population statistic is subtracted.

### Global-control arm

The candidate's frozen target-semantic vector is repeated across the same 16
locations:

\[
u^{gl}_p(y)=z(y),qquad
s_{gl}(X,y)=\tau\frac1{16}\sum_{p=1}^{16}
P_{\theta,p}(X)^\top z(y).
\]

The architecture, parameter tensors, score reduction, candidate count, losses,
and compute path are otherwise identical. The control does not delete field
rows or trainable capacity.

## Fixed Spatial Intervention

The candidate-field permutation is the 180-degree reversal

\[
\pi=(15,14,\ldots,1,0)
\]

under row-major `4 x 4` indexing. For the spatial arm,

\[
s_{sp}^{\pi}(X,y)=\tau\frac1{16}\sum_p
P_{\theta,p}(X)^\top u_{\pi(p)}(y).
\]

This intervention is evaluator-only and never enters training. The global arm
must be exactly invariant because all candidate rows are equal. A separate
candidate-column permutation independently reorders complete candidate images
and must preserve recovered assignments in both arms.

## Fixed Context Corruption

The suffix-preserving shuffle independently permutes context images `0..59`
and leaves images `60..63` and every candidate pixel unchanged. An identity
draw is replaced by a deterministic one-step rotation. The same shuffle is
used for both arms and both cross-font directions.

No context mask, glyph replacement, OCR corruption, or candidate-image
corruption is added. Existing bounded renderer augmentation remains allowed
for training views.

## Fixed Losses

All losses average the two cross-font directions. Let `CE` be row-wise cross
entropy over candidate images, `A` the existing symmetric row/column two-way
assignment loss, and

\[
m_i(S)=S_{i,t_i}-S_{i,1-t_i}
\]

for pair row `i`.

### Positive next-field prediction

For route-specific target cells `u_p(Y)`, define

\[
L_{field}=\frac1{16}\sum_p
\left(1-P_{\theta,p}(X)^\top u_p(Y)\right).
\]

For the global arm, `u_p(Y)=z(Y)` at all positions. Exact target columns are
gathered by host-side positional labels; no identity label enters the model.

### Natural 1,024-image objective

For full context `X`, shuffled context `X_shuf`, target column `t`, and the
same 1,024 candidate images:

\[
L_{natural-full}=CE(s(X,\mathcal Y),t),
\]

\[
L_{natural-order}=\operatorname{softplus}
\left[0.10-left(\log p(t\mid X)-
\log p(t\mid X_{shuf})\right)\right].
\]

The natural route loss is

\[
L_{natural}=L_{natural-full}+L_{field}+0.5L_{natural-order}.
\]

Suffix-only scores are measured but not optimized and are not subtracted.

### Exact-suffix pair objective

For each two-context/two-candidate matrix `S`, its shuffled-context version
`S_shuf`, and assignment `t`:

\[
L_{pair-assign}=A(S,t),
\]

\[
L_{pair-positive}=\frac12\sum_i
\operatorname{softplus}(0.10-m_i(S)),
\]

\[
L_{pair-order}=\frac12\sum_i
\operatorname{softplus}
\left(0.10-[m_i(S)-m_i(S_{shuf})]\right).
\]

The pair field loss gathers the correct candidate field for each context row.
The total pair loss is

\[
L_{pair}=4L_{pair-assign}+L_{field-pair}
+L_{pair-positive}+L_{pair-order}.
\]

### Total microstep loss

\[
L=L_{natural}+L_{pair}.
\]

No symbolic baseline, candidate ID, target string, or teacher value is used in
this loss.

## Fixed Optimization

Each arm runs independently with the following exact settings:

| Setting | Fixed value |
|---|---:|
| optimizer | AdamW |
| updates | 8,000 |
| natural batch | 8 per microstep |
| suffix-pair batch | 8 per microstep |
| gradient accumulation | 2 |
| effective natural rows | 32 per update across views |
| effective pair rows | 64 per update across views and pair arms |
| decoder learning rate | `3e-4` |
| context learning rate | `6e-5` |
| warmup | 400 updates |
| schedule | cosine to 10% of base rate |
| weight decay | 0.05 |
| global gradient clip | 1.0 |
| precision | BF16 autocast |
| checkpoint interval | 1,000 updates |
| checkpoint selection | none; final update only |

The route arms may run sequentially on one physical RTX 4090. They must not
share optimizer state, gradients, learned parameters, or checkpoint selection.
No argument above may change in an evidence run.

## Fixed Development Audit

After both final checkpoints exist, one comparison evaluator audits:

- 2,048 deterministic natural development windows;
- 512 deterministic cross-record suffix-4 development pairs;
- one evaluator-only 1,024-image bank in both development fonts;
- image unigram, symbolic bigram, and symbolic trigram controls formed from
  the training partition outside the student;
- full, suffix-4, suffix-preserving shuffled, and spatially permuted scores;
- both candidate-column orders;
- route-specific candidate visibility;
- checkpoint and boundary receipts;
- parameter and initialization equality;
- model-state finiteness;
- peak allocated CUDA memory and wall time; and
- proof that no frozen image was instantiated.

The two routes must use the same audit windows, pair specifications, rendered
pixels, candidate columns, and batches. The evaluator cannot choose examples
or thresholds from either output.

## Fixed Gate Definitions

All strict inequalities below remain strict. `epsilon=1e-12` is used only to
make strict floating-point comparisons explicit.

### Common mechanism gates for the spatial arm

1. every model state and score is finite;
2. the student boundary receipt is clean;
3. no candidate bank, form string, label, optimizer state, or RNG state occurs
   in the final checkpoint;
4. the deployed context-only method emits `[B,16,192]` without candidates;
5. total parameters are `<20,000,000` and trainable parameters are
   `<18,500,000`;
6. peak allocated CUDA memory is `<18 GiB`;
7. aligned frozen spatial-field cross-font top-1 is `>=0.95`;
8. pair suffix pixels are bitwise equal;
9. pair suffix score rows have maximum absolute error `<1e-6`;
10. candidate-column permutation score error is `<1e-5`;
11. candidate-column permutation accuracy agreement is exactly `1.0`;
12. full exact-suffix arm accuracy is `>0.65`;
13. full exact-suffix both-correct rate is `>0.40`;
14. full minus shuffled pair arm accuracy is `>0.10`;
15. full minus shuffled pair mean margin is `>0.05`;
16. natural full minus shuffled target log probability is `>0.03` nat;
17. natural full minus spatially permuted target log probability is `>0.05`
    nat; and
18. full pair arm accuracy minus spatially permuted pair arm accuracy is
    `>0.05`.

### Global-control integrity gates

1. every model state and score is finite;
2. the same boundary, checkpoint, parameter, and resource gates pass;
3. frozen global-semantic cross-font top-1 is `>=0.95`;
4. suffix pixels and suffix score rows are exact;
5. candidate-column permutation equivariance passes; and
6. candidate spatial-permutation maximum score error is `<1e-6`.

The global arm is a control and need not pass language selection gates.

### Matched-arm gates

1. initialized parameter keys, shapes, dtypes, values, and state SHA-256 are
   exactly equal;
2. final total and trainable parameter counts are exactly equal;
3. source, corpus, candidate-bank, pair, font, and audit-window receipts are
   exactly equal;
4. both arms completed exactly 8,000 finite updates from final checkpoints;
5. spatial full pair arm accuracy minus global-control arm accuracy is
   `>0.05`;
6. spatial full pair both-correct minus global-control both-correct is
   `>0.05`;
7. spatial natural full top-1 minus global-control full top-1 is `>0.01`; and
8. spatial natural target log probability minus global-control target log
   probability is `>0.05` nat.

### Spatial language gates

1. natural full 1,024-way top-1 is `>=0.15`;
2. full top-1 minus suffix-4 top-1 is `>0.03`;
3. full top-1 minus shuffled-prefix top-1 is `>0.03`;
4. full top-1 minus image-unigram top-1 is `>0.03`;
5. full top-1 minus symbolic-bigram top-1 is `>0.01`;
6. full target log probability minus symbolic-bigram target log probability is
   `>0.05` nat;
7. full exact-suffix pair arm accuracy is `>0.65`; and
8. full exact-suffix pair both-correct is `>0.40`.

The spatial mechanism is selected only if every spatial common gate, every
global integrity gate, every matched-arm gate, and every spatial language gate
passes. Otherwise V30 is rejected on development, the frozen split remains
sealed, and writer training remains unauthorized.

## Fixed Artifacts

Evidence outputs:

- spatial arm:
  `artifacts/spatial_visual_next_field_v30_spatial_evidence`;
- global control:
  `artifacts/spatial_visual_next_field_v30_global_control_evidence`; and
- comparison audit:
  `artifacts/spatial_visual_next_field_v30_evidence`.

Each arm writes:

- `checkpoint_step_XXXXXXX.pt` every 1,000 updates;
- `checkpoint_final.pt` atomically after update 8,000;
- `train.jsonl` with losses, route metrics, gradient norm, rates, throughput,
  and CUDA memory; and
- `candidate_bank.json` containing only host-side receipt metadata.

The comparison directory writes:

- `development_audit.json` atomically;
- `comparison_receipt.json` with exact checkpoint, protocol, source, corpus,
  initialization, candidate-bank, and audit hashes; and
- no model weights, candidate images, form strings, or frozen data.

Final checkpoints contain only:

- architecture and route names;
- model config;
- model tensor state;
- update number and fixed optimization arguments;
- source, protocol, corpus, font, pair, retina, V29 source, candidate-bank, and
  common-initialization receipts;
- boundary receipt;
- parameter counts; and
- smoke/exploratory flags.

Final checkpoints must not contain optimizer state, scheduler state, RNG state,
candidate images, candidate forms, canonical target pixels, training examples,
development examples, or frozen examples.

## Fixed Run Order

1. Commit this protocol.
2. Record its SHA-256.
3. Implement model, data boundary, training losses, arm trainer, comparison
   evaluator, and tests without changing this document.
4. Run CPU/unit tests and smoke training for both arms. Smoke artifacts are
   tagged and cannot be promoted.
5. Commit verified implementation.
6. Run the fixed spatial arm on one RTX 4090.
7. Run the fixed global-control arm on one RTX 4090 with the same seed and
   arguments.
8. Run the fixed joint development audit only after both final checkpoints
   exist.
9. Publish the result regardless of pass or fail.
10. Open the frozen partition only if the joint audit explicitly authorizes
    it. A writer still requires a new preregistration.

## Stop And Reject Conditions

Stop and reject the affected evidence run if:

- a fixed argument, seed, source checkpoint, source hash, protocol hash, data
  partition, font partition, or candidate-bank receipt differs;
- the initialized model states differ between arms;
- a route changes parameter count;
- a loss or model method receives forbidden metadata;
- a final checkpoint contains a bank, label, string, optimizer, or RNG state;
- a tensor, loss, gradient, or score becomes non-finite;
- an arm does not complete exactly 8,000 updates;
- the memory cap is exceeded;
- a frozen image is instantiated before authorization; or
- any threshold is changed after a V30 development value is observed.

An interrupted process may resume only from an atomic intermediate checkpoint
that includes optimizer and RNG state in a separately tagged resumable payload;
that payload must be stripped from the final checkpoint. A restarted run from
the common initialization is also valid. Partial or smoke runs provide no
selection evidence.

## Claims Permitted On A Pass

If every gate passes, the project may claim only that, on this fixed
development benchmark:

- a compact image-only causal student predicts a continuous spatial next-image
  field;
- aligned local candidate evidence is causally necessary under the fixed patch
  intervention;
- that spatial mechanism exceeds a parameter-identical position-blind visual
  control; and
- the selected field beats fixed image-frequency and symbolic-bigram controls
  sufficiently to authorize one separately reported frozen evaluation.

A pass does not establish Qwen-8B parity, general instruction following,
historical-glyph etymology, page-scale generation, human-like reading,
efficiency superiority over token LLMs, or a usable writer.

## Claims Required On A Failure

If any gate fails, publish V30 as a completed negative experiment. Report the
failed gates and measured controls. Do not open frozen images, train a writer,
increase model size, add page/3D/movie geometry, or claim language efficiency
from visual identity, low memory, glyph-like output, or order sensitivity.
