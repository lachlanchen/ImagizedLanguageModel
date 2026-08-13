# Dense Visual Future Energy V28 Protocol

Date frozen: 2026-08-13

Status: preregistered before implementation, smoke testing, or development
measurement

## Decision Question

Does stratified dense, multi-horizon continuous visual prediction learn a
useful order-conditioned next-writing distribution when perception is fixed
and every student input and candidate is an image?

V28 is a development experiment. It may authorize one later frozen evaluation
and a separate writer stage only if every selection gate below passes. A smoke
run tests plumbing only and cannot change this protocol.

## Fixed Data

- manifest:
  `data/visual_grammar/chinese_wikisource_public_domain.jsonl`;
- required manifest SHA-256:
  `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03`;
- identifier partition and font partition: unchanged from V25-V27;
- train views: original and OpenCC simplified when distinct;
- development views: original and simplified under the fixed evaluator;
- context cells: 64;
- future cells retained per sampled segment: 4;
- authoritative sample shape: `[68,1,32,32]` per font view;
- cell rendering: two distinct training fonts, size 24-28, at most one-pixel
  translation, the existing bounded intensity/blur/noise augmentation;
- canonical equality witness: unaugmented Noto Sans CJK Regular pixels; and
- frozen identifiers and frozen fonts: unopened.

The student-facing natural batch contains only the two floating image streams.
Canonical pixels may be converted by the host loss into temporary within-batch
equality groups. Those groups are never passed to a model method, persisted as
character IDs, or stored in the checkpoint.

Training suffix pairs use 32,768 deterministic suffix-4 pairs with different
targets and, where required by the existing builder, different record
identifiers. Candidate order is independently randomized for each font view.

## Fixed Initialization

The retina is loaded from:

`artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt`

Required checkpoint SHA-256:

`90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe`

All retina parameters remain frozen for the complete run. A trainable semantic
adapter and its EMA copy initialize as exact residual identities over the
192-dimensional normalized retinal vector. The EMA adapter is stop-gradient
and updates only from the online adapter.

## Fixed Student Boundary

Allowed learned-path tensors:

- context images `[B,T,1,32,32]`, `1 <= T <= 64`;
- arbitrary candidate images `[N,1,32,32]` or paired candidates
  `[B,K,1,32,32]`;
- temporary positional assignment indices for the two-by-two pair loss; and
- floating corruption times/noise internal to the image path.

Forbidden learned-path values:

- strings;
- token, Unicode, character, vocabulary, or glyph IDs;
- OCR output or text transcripts;
- tokenizer or discrete visual-codebook indices;
- vocabulary embeddings or vocabulary logits;
- glyph lookup or a deployed candidate bank;
- symbolic n-gram probabilities;
- teacher logits or hidden states; and
- external language-model calls.

The evaluator may use strings to build labels, image banks, and symbolic
controls. Evaluator objects do not enter the model or checkpoint.

## Fixed Architecture

Model name: `dense-visual-future-energy-v28`.

| Component | Fixed value |
|---|---:|
| cell size | 32 |
| maximum context | 64 cells |
| raw retinal dimension | 192 |
| semantic visual dimension | 192 |
| causal field width | 384 |
| causal blocks | 8 |
| attention heads | 6 |
| MLP ratio | 3.0 |
| dropout | 0.05 |
| future horizons | `(1,2,4)` |
| hypotheses per horizon | 4 |
| semantic-adapter hidden width | 384 |
| initial raw temperature | 0.08 |
| initial semantic temperature | 0.08 |
| maximum inverse temperature | 100 |
| trainable-parameter cap | 20 million |
| total-parameter cap | 24 million |

For image `x`, the frozen retina and online adapter produce

\[
r(x)=\operatorname{normalize}(R_0(x)),\qquad
z(x)=\operatorname{normalize}(A_\theta(r(x))).
\]

The context input is a learned projection of `[r(x);z(x)]`. Eight causal
rotary blocks emit one state per visible cell. A learned horizon embedding and
shared future head emit four normalized raw queries, four normalized semantic
queries, and four mixture logits per state and horizon.

Candidate image `y` is encoded by the frozen retina and EMA semantic adapter.
For hypothesis `k`,

\[
e_k(h,y)=a_r\langle q^r_k(h),r(y)\rangle+
a_z\langle q^z_k(h),\bar z(y)\rangle,
\]

where `a_r` and `a_z` are learned positive inverse temperatures clamped at
100. The deterministic candidate score is

\[
s(h,y)=\operatorname{logsumexp}_k
[\operatorname{logsoftmax}(l(h))_k+e_k(h,y)].
\]

No candidate collection is a parameter, buffer, or required inference input.

## Fixed Position Sampling

Each natural microbatch selects exactly 16 shared causal positions:

- four without replacement from indices `0..15`;
- four without replacement from `16..31`;
- four without replacement from `32..47`; and
- index `63` plus three without replacement from `48..62`.

The trainer RNG determines these positions and its state is saved for exact
resume. For selected zero-based position `t`, the unnormalized position weight
is

\[
w_t=0.25+0.75((t+1)/64)^2.
\]

The weight for `t=63` is multiplied by two. Selected weights are normalized to
mean one. Horizon weights are `beta_1=1`, `beta_2=0.5`, and `beta_4=0.25`,
normalized by their sum.

For horizon `d`, a query at position `t` predicts image `x_{t+d}`. The retained
68-cell segment makes every selected position valid for all three horizons.

## Fixed Losses

All natural losses average the first-font-to-second-font and
second-font-to-first-font directions.

### Dense multi-positive image contrast

For each horizon and direction, opposite-font targets from all selected
positions in the microbatch form the temporary candidate set. `P_i` contains
candidates whose canonical pixels exactly equal query `i`'s canonical target.

\[
L_{dense}=-\frac{1}{\sum_i w_i}\sum_i w_i
\log\frac{\sum_{j\in P_i}\exp s_i(y_j)}
{\sum_j\exp s_i(y_j)}.
\]

The three horizon losses are averaged with fixed `beta_d`.

### Continuous visual energy score

For aligned normalized raw target `r_y`, raw hypotheses `q_k`, and mixture
probabilities `pi_k`,

\[
ES(Q,r_y)=\sum_k\pi_k\lVert q_k-r_y\rVert_2-
\frac12\sum_{k,l}\pi_k\pi_l\lVert q_k-q_l\rVert_2.
\]

`L_ES` uses the same position and horizon weighting as `L_dense`.

### Cross-font semantic identity

Online semantic images at the selected horizon-1 targets retrieve EMA semantic
images from the other font. Multi-positive masks again come only from exact
canonical-pixel equality. Both font directions are averaged into `L_id`.

### Natural order margin

At the endpoint, independently permute cells `0..59` and preserve cells
`60..63`. Full and shuffled contexts use identical candidate pixels and exact
positive masks. If `ell_full` and `ell_shuffle` are the log positive masses
under their shared candidate set,

\[
L_{natural-order}=\operatorname{softplus}
[0.10-(\ell_{full}-\ell_{shuffle})].
\]

### Pair assignment and pair order

For each cross-font suffix-4 pair, full-context score matrix `S` receives the
same symmetric row/column assignment loss as V27:

\[
L_{pair}=\tfrac12[CE(S,\pi)+CE(S^T,\pi^{-1})].
\]

Let `m_full`, `m_suffix`, and `m_shuffle` be mean correct-minus-other
assignment margins under full context, the last four cells, and a permutation
of cells `0..59` respectively. Candidate pixels and column permutations are
identical across arms.

\[
L_{pair-order}=\tfrac12\left(
\operatorname{softplus}[0.10-(m_{full}-m_{suffix})]+
\operatorname{softplus}[0.10-(m_{full}-m_{shuffle})]
\right).
\]

### Total

\[
L=L_{dense}+0.25L_{ES}+0.5L_{id}+2L_{pair}+
L_{natural-order}+L_{pair-order}.
\]

No writer, pixel reconstruction, symbolic classification, distillation, or
frozen-partition loss is permitted.

## Fixed Optimization

| Setting | Value |
|---|---:|
| updates | 10,000 |
| natural examples per microbatch | 8 |
| suffix pairs per microbatch | 4 |
| gradient accumulation | 4 |
| optimizer | AdamW |
| betas | `(0.9,0.95)` |
| learning rate | `3e-4` |
| minimum cosine LR ratio | 0.10 |
| warmup updates | 500 |
| weight decay | 0.05 |
| gradient clip | 1.0 |
| EMA momentum | 0.996 |
| context image corruption maximum | 0.03 |
| seed | 20260916 |
| dataset seed | 20260917 |
| pair seed | 20260918 |
| precision | BF16 |
| hardware | one RTX 4090 |
| peak allocated CUDA cap | 18 GiB |

The evidence command must use these values exactly. Exploratory arguments or a
smoke flag make the output ineligible as evidence.

## Fixed Development Audit

- 2,048 deterministic natural windows;
- 512 deterministic cross-record suffix-4 pairs;
- 1,024 candidate identities;
- two unseen development fonts;
- batch size 32;
- full, last-only, suffix-4, and suffix-preserving shuffled contexts;
- image unigram, symbolic bigram, and symbolic trigram controls;
- raw-retina and EMA-semantic cross-font retrieval over the same 1,024-image
  bank;
- exact candidate-column permutation replay;
- exact suffix-pixel equality;
- boundary receipt and peak allocated CUDA memory; and
- no frozen images instantiated.

The trigram is diagnostic because sparse context coverage makes its simple
backoff policy non-comparable to the dense student distribution. Unigram and
bigram remain the fixed language selection controls.

## Fixed Selection Gates

### Mechanism gates

All must pass:

1. full pair arm accuracy `>0.65`;
2. full pair both-correct rate `>0.40`;
3. full minus suffix-4 arm accuracy `>0.15`;
4. full minus shuffled arm accuracy `>0.05`;
5. full minus shuffled mean assignment margin `>0.02`;
6. last-only arm accuracy equals tie-aware chance `0.5` within `1e-6`;
7. suffix-4 arm accuracy equals tie-aware chance `0.5` within `1e-6`;
8. suffix pixels are exactly equal;
9. candidate-permutation maximum score error `<1e-5` and recovered accuracy
   agreement equals `1.0`;
10. EMA-semantic 1,024-bank cross-font identity top-1 `>=0.95`;
11. EMA-semantic identity exceeds raw-retina identity by at least `0.02` on the
    same bank and views;
12. student boundary is clean;
13. peak allocated CUDA memory `<18 GiB`; and
14. frozen images instantiated is false.

### Language gates

All must pass:

1. full top-1 minus image-unigram top-1 `>0.03`;
2. full top-1 minus symbolic-bigram top-1 `>0.01`;
3. full target log probability minus symbolic-bigram target log probability
   `>0.05` nat;
4. full target log probability minus suffix-4 target log probability `>0.03`
   nat;
5. full target log probability minus shuffled-prefix target log probability
   `>0.03` nat; and
6. full top-1 `>=0.15`.

Threshold comparisons are strict except the stated `>=` and exact controls.
Gate definitions cannot be changed after the fixed run.

## Authorization Rule

If any mechanism or language gate fails:

- classify V28 as rejected on development;
- do not open the frozen partition;
- do not train a writer as evidence;
- publish the complete fixed result, including negative controls; and
- localize the failure before proposing V29.

If every gate passes:

- save the selected checkpoint hash;
- run exactly one frozen audit under a separately recorded command;
- keep all symbolic controls evaluator-only; and
- preregister a writer stage that generates continuous glyph pixels and rereads
  its own output.

In neither case may a 2D lattice, longer context, historical-glyph curriculum,
depth channel, or movie be added retroactively to V28.
