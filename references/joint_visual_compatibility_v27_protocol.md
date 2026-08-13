# Joint Visual Compatibility V27: Frozen Protocol

Date: 2026-08-13

Status: preregistered before V27 implementation, optimization, smoke training,
or development-result inspection

Related decision:
[`deterministic_visual_compatibility_v27_research.md`](deterministic_visual_compatibility_v27_research.md)

## Primary Question

Can a compact image-native model jointly learn a causal representation of 64
visible Chinese glyph images and a deterministic compatibility score for an
arbitrary candidate next-glyph image, such that it uses history before a shared
four-glyph suffix and beats natural frequency controls?

V27 is a language-state experiment. It does not train a pixel writer, generate
answer pages, use historical-glyph supervision, call an external language model,
or open the frozen partition unless every fixed development gate passes.

## Fixed Data

The only language corpus is:

`data/visual_grammar/chinese_wikisource_public_domain.jsonl`

Required SHA-256:

`76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`

It contains 7,017 records from 16 public-domain Chinese Wikisource books. V27
inherits the identifier partition and font partition from V25 and V26:

- partition salt: `ilm-v25-natural-chinese-cell-stream-20260813`;
- train: 6,608 identifiers and four train fonts;
- development: 190 identifiers and two unseen development fonts;
- frozen: 219 identifiers and two unseen frozen fonts.

The frozen writing and frozen fonts remain unopened during implementation,
optimization, development selection, debugging, and protocol revision. A
receipt must report `frozen_images_instantiated=false`.

Original and OpenCC simplified renderings are separate visual views where they
differ. Unsupported forms are skipped before rendering rather than replaced by
missing-glyph boxes.

## Visual Unit And Boundary

One visible writing unit is a continuous `1 x 32 x 32` grayscale image. A
context is an ordered `64 x 1 x 32 x 32` visual-time volume. A next-glyph
candidate is another `1 x 32 x 32` image.

The deployed scorer receives only floating image tensors. It must not receive or
contain:

- strings, token IDs, Unicode code points, or character IDs;
- a vocabulary embedding or vocabulary-sized output matrix;
- OCR text, glyph lookup, or a discrete visual codebook;
- an external language-model state or call; or
- a persistent candidate bank.

Strings may exist only in the offline corpus reader, partitioner, and renderer.
Pair alignment and duplicate-positive masks may be prepared offline, but the
student receives only the resulting image tensors and positional assignment
labels. Duplicate positives in natural batches must be established by exact
equality of canonical rendered identity images, not character IDs.

The score for context image stream `X` and arbitrary candidate image `Y` is a
continuous scalar `s(X,Y)`. Candidate banks are evaluator-only collections of
images and are absent from deployed checkpoints.

## Fixed Pair Construction

Training uses 16,384 cross-record suffix-4 pairs constructed with seed
`20260914`. Each pair has:

- two 64-cell contexts from different source identifiers;
- an exactly equal final four-character suffix;
- different next characters in the 1,024-form audit alphabet; and
- two independent train-font renderings.

Every paired rendering uses the same font, size, translation, blur, contrast,
and noise draw for both members, making the four shared suffix image tensors
bitwise equal.

Development uses 512 cross-record suffix-4 pairs with seed `20260915`. A
separate suffix-8 diagnostic may use within-record members if 512 cross-record
pairs are unavailable. Suffix-8 cannot select the model.

For every training pair, candidate order is swapped by a deterministic random
bit independently for each visual view. The scorer receives no row, character,
or source identity. This prevents a fixed diagonal-position shortcut.

## Fixed Natural Windows

Natural training samples a full 64-cell context and one next-cell target. Two
different train fonts and independent augmentation draws render each sequence.
The online context view is paired with the other font's target image in both
directions.

Each target additionally has a canonical, unaugmented identity image rendered
in a fixed train font. Exact canonical pixel equality defines multiple positives
inside a natural batch. The canonical image is training-only supervision and is
not a character ID or deployed memory.

Development fixes 2,048 reservoir-sampled natural windows with seed `20260915`
and a 1,024-form candidate image bank. The evaluator may use strings only to
compute unigram and symbolic-bigram controls and to associate image rows with
ground truth. No evaluator label enters the student.

## Fixed Initialization

The online and EMA image retinas initialize from:

`artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt`

Required SHA-256:

`90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe`

No V25 or V26 language weight initializes V27. The V16 retina is not frozen in
the online language path; it is optimized with a lower learning rate. A separate
unchanged V16 retina may exist evaluator-side only for the cross-font form
control.

## Fixed Model

The trainable model contains:

1. an online convolutional retina initialized from V16;
2. a linear visual input map;
3. eight causal rotary-attention blocks, width 384, six heads, MLP ratio 3;
4. an RMS-normalized final context state and a two-layer query projection;
5. an online residual candidate-image projector; and
6. one learned, bounded contrastive temperature.

The target image route is an exponential-moving-average copy of the online
retina and candidate projector. It receives no gradient. The candidate projector
has a normalized identity residual and a zero-initialized final update, so the
initial target geometry exactly preserves the V16 visual representation.

Let `R_o` and `R_t` be online and EMA retinas, `F` the causal field, and `G_o`
and `G_t` the online and EMA candidate projectors:

\[
q(X)=\operatorname{normalize}\,Q(F(R_o(X))_{64}),
\qquad
k(Y)=\operatorname{normalize}\,G_t(R_t(Y)).
\]

The compatibility score is

\[
s(X,Y)=\exp(\alpha)\,q(X)^\top k(Y),
\qquad \exp(\alpha)\le 100.
\]

There is no stochastic particle head, class table, or pixel decoder. Trainable
parameters must not exceed 20 million; peak allocated CUDA memory must remain
below 18 GiB on one RTX 4090.

## Fixed Losses

### Natural image compatibility

For cross-font natural queries and candidate keys, let `P_i` contain all batch
candidates whose canonical identity images are pixel-equal to query `i`'s
paired target. The multi-positive image contrast is

\[
\mathcal L_{\mathrm{natural}}=-\frac1B\sum_i
\log\frac{\sum_{j\in P_i}\exp s(X_i,Y_j)}
{\sum_j\exp s(X_i,Y_j)}.
\]

### Matched assignment

For a suffix-matched pair, form the two-by-two score matrix `S`, after the fixed
candidate permutation. Row and column cross-entropies both recover the known
image pairing:

\[
\mathcal L_{\mathrm{pair}}=\tfrac12
\left[\operatorname{CE}(S,\pi)+
\operatorname{CE}(S^\top,\pi^{-1})\right].
\]

### Candidate image identity

Online candidate images must retrieve independent-font EMA candidate images,
using the same canonical-image multi-positive mask:

\[
\mathcal L_{\mathrm{identity}}=
\operatorname{MPNCE}(G_o(R_o(Y^{(1)})),G_t(R_t(Y^{(2)})))
\]

in both view directions.

### Variance and covariance

Unnormalized online candidate latents from two fonts receive VICReg-style
invariance, per-dimension standard-deviation, and off-diagonal covariance
penalties:

\[
\mathcal L_{\mathrm{VC}}=
25\mathcal L_{\mathrm{inv}}+25\mathcal L_{\mathrm{var}}+
\mathcal L_{\mathrm{cov}}.
\]

The fixed total loss is

\[
\mathcal L=
\mathcal L_{\mathrm{natural}}+
2\mathcal L_{\mathrm{pair}}+
0.5\mathcal L_{\mathrm{identity}}+
0.05\mathcal L_{\mathrm{VC}}.
\]

No writer, reconstruction, symbolic classification, or teacher-logit loss is
permitted in V27.

## Fixed Optimization

| Setting | Value |
|---|---:|
| updates | 8,000 |
| natural examples per microbatch | 8 |
| suffix pairs per microbatch | 4 |
| gradient accumulation | 4 |
| context/projector learning rate | `3e-4` |
| online-retina learning rate | `3e-5` |
| warmup updates | 400 |
| minimum cosine LR ratio | 0.10 |
| AdamW betas | `(0.9, 0.95)` |
| weight decay | 0.05 |
| gradient clip | 1.0 |
| EMA momentum | 0.996 |
| context corruption maximum | 0.05 |
| model seed | `20260913` |
| natural dataset seed | `20260914` |
| pair seed | `20260914` |
| precision | CUDA BF16 |

The evidence run must start from V16 initialization, not a smoke or exploratory
V27 checkpoint. Checkpoints record model, optimizer, EMA, pair, partition, font,
source-hash, argument, runtime, and peak-memory receipts. Training-only identity
images and optimizer state are absent from the deployed final checkpoint.

## Fixed Development Evaluation

Development evaluation runs once at the fixed 8,000-update endpoint. No
checkpoint selection, threshold revision, or hyperparameter search is allowed.

### Natural metrics

On 2,048 windows and the shared 1,024-image bank, report top-1, top-5, and mean
target log probability for:

- full 64-cell context;
- last cell only;
- final four cells only;
- a deterministic shuffle of cells 1--60 preserving the final four;
- image unigram; and
- evaluator-only symbolic bigram.

Report online-to-EMA and EMA-to-EMA cross-font candidate-bank identity top-1.

### Suffix-pair metrics

On 512 suffix-4 pairs, report:

- exact suffix-pixel equality;
- full-context arm accuracy, both-correct rate, and mean margin;
- last-only and suffix-4 arm accuracy;
- prefix-shuffled arm accuracy and margin;
- full minus shuffled accuracy and margin;
- candidate-permutation equivariance error and accuracy agreement; and
- raw frozen-V16 retina identity accuracy and cosine margin.

Score ties receive one-half accuracy credit and are reported separately. The
appearance-only and suffix-4 controls must be exactly at chance because both
pair rows receive identical image tensors.

### Boundary and resources

Recursively inspect model and student-batch receipts. Report total and trainable
parameters, elapsed training time, throughput, and peak allocated CUDA memory.

## Fixed Gates

All strict inequalities use an epsilon of `1e-12`.

### Mechanism gates

- suffix-4 full-context arm accuracy is greater than `0.65`;
- suffix-4 both-correct rate is greater than `0.40`;
- full-context arm accuracy exceeds suffix-4 by more than `0.15`;
- full-context arm accuracy exceeds prefix-shuffled by more than `0.05`;
- full-context mean margin exceeds shuffled margin by more than `0.02`;
- last-only and suffix-4 controls each equal `0.50` within `1e-6`;
- suffix pixels are exactly equal for every pair;
- candidate-permutation score error is below `1e-5` and accuracy is unchanged;
- raw-retina and learned candidate cross-font identity are each at least `0.99`;
- the recursive image-only boundary audit passes; and
- peak allocated CUDA memory is below 18 GiB.

### Natural-language gates

- full top-1 exceeds image-unigram top-1 by more than `0.03`;
- full top-1 exceeds symbolic-bigram top-1 by more than `0.01`;
- full target log probability exceeds symbolic-bigram by more than `0.05` nat;
- full target log probability exceeds suffix-4 by more than `0.03` nat; and
- full target log probability exceeds prefix-shuffled by more than `0.03` nat.

## Selection, Frozen Split, And Claims

If any mechanism gate fails, V27 is rejected and no frozen evaluation or writer
is authorized.

If every mechanism gate passes but a natural-language gate fails, V27 may claim
only a bounded matched-image relation on development data. It still cannot open
the frozen split or train a writer.

Only if every mechanism and natural-language gate passes may V27 authorize one
fixed frozen evaluation under a separately committed frozen-evaluation command.
That evaluation may verify the accepted language mechanism but may not tune it.

A passing V27 would establish deterministic, vocabulary-free next-image
compatibility for bounded ordinary Chinese. It would not establish free-form
question answering, historical etymology, page generation, Qwen parity, general
multilingual language, or efficiency superiority. A pixel writer remains a
subsequent experiment.
