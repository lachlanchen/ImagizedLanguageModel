# Visual Answer Trajectory V39: Research and Exploratory Design

Date: 2026-08-14

Status: exploratory mechanism selected; no V39 evidence protocol or result is
claimed

## Decision question

V38 established a bounded but incomplete result. Its image-only reader learned
font-stable Chinese prompt representations, while its single answer vector
reached only 21.94% top-1 and 49.49% top-5 retrieval on 196 development
answers. The final training batch reached 96.35% answer-plan top-1. The gap is
consistent with memorization after roughly 88 effective passes over only 5,822
short-answer pairs and with compressing a whole answer into one state.

V39 asks the next necessary question:

> Can one image-only visual reader map a rendered Chinese prompt to an ordered
> trajectory of continuous answer-span states, including answers never eligible
> for V38, while remaining independent of strings, token IDs, Unicode IDs, OCR,
> answer candidates, and teacher services at runtime?

This is still a semantic-generation mechanism test. A continuous trajectory is
not yet a rendered answer. The V34 raster actuator remains closed until a
separately fixed semantic gate passes.

## Local evidence that fixes the design

### V35 rejects local next-patch learning without an answer plan

V35 trained a complete raster-input/raster-output model for 22,000 updates. It
produced nonblank glyph-like images through a visible decode-threshold-reencode
loop, but autonomous instruction character accuracy was 0.112%, below the
shuffled-prompt control. More next-patch training on the same objective is not
the next experiment.

### V36 rejects low-rank spatial-quarter plans

V36 emitted one global and four ordered answer-image-quarter plans. Its frozen
visual targets had centered effective rank 4.77/768, and development top-1 was
1.02%. Ordered outputs alone are therefore insufficient when their targets do
not carry stable sentence semantics.

### V37 and V38 establish a usable visual reader

V37 distilled detached BGE-M3 geometry into an end-to-end visual reader. V38
then aligned exact, font-varied, and paraphrased visual paths. Its development
prompt retrieval reached 60.71% top-1 and 86.73% top-5. This is the strongest
local reader and is the V39 initialization. Its answer relation remains the
baseline to beat.

### V34 establishes a later continuous raster interface

The 7.42M-parameter V34 codebook-free codec reconstructs unseen CJK fonts and
held-out historical-character families from continuous 768-dimensional
`32 x 32` patch states. It is suitable for a later motor decoder, but its
reconstruction latent is not used as the sole V39 semantic target.

## Corpus audit

The local `data/raw/alpaca_zh.json` contains 48,818 valid Chinese instruction
rows. V38's 32-character answer limit admitted only 6,231 raw rows and 5,822
training rows. With prompts limited to 160 visible characters, 48,775 rows are
eligible before answer segmentation.

A deterministic punctuation-aware segmentation audit produced:

| Maximum visible characters per span | Records with at most 16 spans | Coverage |
|---:|---:|---:|
| 24 | 31,005 | 63.57% |
| 32 | 38,751 | 79.45% |
| 40 | 46,159 | 94.64% |
| 48 | 47,951 | 98.31% |

For 48-character-scale spans, the median, 90th, and 95th percentile answer
lengths are 5, 12, and 13 spans. A 16-state trajectory therefore retains nearly the
whole corpus without truncation. Segmentation is an offline layout operation:
the model receives rendered pixels and visual masks, not characters or segment
strings.

The public-domain Wikisource corpus contributes 7,017 records from 16 Chinese
works and about 3.55M visible characters. It is reserved for a later
source-held continuation curriculum after the instruction mechanism works.
The user-supplied books under `references/source_books/`, `../Books`, and
`../ZhJpBook` remain private reference/evaluation material unless a title-level
rights audit permits training. They must not silently enter a releasable
checkpoint.

## External evidence and exact boundary

External work informs this experiment; none is relabeled as an ILM
contribution.

- [Perceiver IO](https://arxiv.org/abs/2107.14795) shows that learned output
  queries can produce structured outputs while input and output cost scale with
  their respective sizes. V39 uses visual answer-position queries rather than
  a token vocabulary.
- [ColBERTv2](https://arxiv.org/abs/2112.01488) demonstrates that ordered
  multi-vector representations retain information that a single pooled vector
  loses. V39 tests this locally for answer plans rather than retrieval indexes.
- [BLIP-2](https://arxiv.org/abs/2301.12597) uses a compact learned-query bridge
  between frozen vision and language foundations. V39 instead adapts the visual
  reader and emits continuous answer states, but the query bottleneck is a
  relevant efficiency precedent.
- [Large Concept Models](https://arxiv.org/abs/2412.08821) demonstrate
  autoregressive prediction in a sentence-representation space. Their text and
  speech encoder/decoder boundary is not image-native and their scale is far
  beyond this experiment.
- [Continuous Autoregressive Language Models](https://arxiv.org/abs/2510.27688)
  establish continuous next-vector modeling and identify two constraints that
  matter here: reconstructive latents must be smooth under prediction error,
  and naive feedback of compact vectors can underperform decoded feedback.
  V39 therefore predicts robust semantic states first and does not reopen the
  raster feedback loop prematurely.
- [Glyph](https://arxiv.org/abs/2510.17800) and
  [SPIRAL](https://arxiv.org/abs/2608.02109) show that rendered text can compress
  context, but also that cross-path visual/text inconsistency is a major
  bottleneck. V39 retains V38's explicit cross-font and cross-script visual-path
  alignment.

BGE-M3 remains a pinned, detached offline teacher. Its manifest and model-layer
hashes are inherited from V37/V38. OpenCC script conversion is an offline
rendering augmentation. Neither tool may be imported or called by the deployed
student.

## Selected representation

Let the prompt raster and clean patch occupancy be

\[
x \in [0,1]^{3\times16\times1024}, \qquad
m \in \{0,1\}^{64}.
\]

The shared visual reader produces contextual patch states and a normalized
semantic read state:

\[
H = R_\theta(x), \qquad
h = \frac{\sum_j m_j H_j}{\sum_j m_j}, \qquad
r = \operatorname{norm}(E_\theta(h)).
\]

The V38 answer map supplies an initialized global baseline

\[
b = \operatorname{norm}(A r + G(E_\theta(h))).
\]

where `A` and `G` are copied from the V38 EMA route. This makes an untrained
V39 planner reproduce the measured V38 answer direction rather than discard a
working mechanism.

## Two-pass causal visual-query planner

V39 has one global query and `K=16` ordered span queries. The visual patch
states are projected to planner width. A shared causal cross-attention decoder
first maps learned position queries and the V38 baseline into preliminary
states:

\[
U^{(1)} = D_\theta(Q + P(b),\; P_H(H),\; m).
\]

The second pass receives shifted preliminary answer states so later positions
can use earlier predicted content without target-side teacher forcing:

\[
\tilde Q_0=P(U^{(1)}_0), \qquad
\tilde Q_k=P(z^{(1)}_{k-1}) + q_k, \quad k>0,
\]

\[
U^{(2)} = D_\theta(\tilde Q,\; P_H(H),\; m).
\]

The same decoder weights are reused in both passes to control parameters. Final
global and span states are residual corrections around the V38 answer baseline:

\[
z_0=\operatorname{norm}(b+C_0(U^{(2)}_0)), \qquad
z_k=\operatorname{norm}(b+C_k(U^{(2)}_k)).
\]

The correction projections are zero-initialized. The planner therefore starts
as V38 plus distinct visual position queries and must earn every deviation.

Each span also predicts a stop hazard and occupied visual length. Stop hazards
produce a monotone survival probability rather than 16 unrelated binary
activity decisions:

\[
a_1=1, \qquad a_k=\prod_{j<k}(1-\sigma(s_j)).
\]

## Detached targets

For each record, the pinned BGE teacher embeds the prompt, whole answer, and
each punctuation-aware answer span before training. The fixed V37 training
mean `mu` preserves the coordinate system learned by V38:

\[
u=\operatorname{norm}(b_{prompt}-\mu),\quad
v=\operatorname{norm}(b_{answer}-\mu),\quad
v_k=\operatorname{norm}(b_{span,k}-\mu).
\]

Targets are stored as FP16 detached tensors in an ignored local artifact. Span
states use a flattened bank plus record offsets; a dense
`records x 16 x 1024` tensor would waste roughly 1.6 GB. No target, source
string, candidate, or teacher tensor enters the deployable checkpoint.

## Training paths and objective

Each item renders four visual paths with distinct fonts:

1. canonical prompt anchor;
2. augmented prompt view, optionally converted between simplified and
   traditional Chinese;
3. one deterministically sampled answer-span anchor; and
4. an independently rendered view of the same sampled span.

All four paths pass through the shared reader. Only the two prompt paths invoke
the planner. The sampled answer paths train direct visual reading of the same
span geometry that the planner must predict without multiplying reader compute
by all 16 spans.

The exploratory loss is

\[
\begin{aligned}
\mathcal L ={}&
\lambda_r\mathcal L_{read}
+\lambda_g\mathcal L_{global}
+\lambda_s\mathcal L_{span}
+\lambda_p\mathcal L_{path} \\
&+\lambda_o\mathcal L_{order}
+\lambda_t\mathcal L_{transition}
+\lambda_h\mathcal L_{stop}
+\lambda_\ell\mathcal L_{length}
+\lambda_v\mathcal L_{VICReg}.
\end{aligned}
\]

`read`, `global`, and `span` combine candidate contrastive loss and cosine
alignment. Global candidates are detached records sampled from the train bank;
span candidates are the active target spans in the effective batch. `path`
aligns canonical, font-varied, and script-varied visual routes. `order` requires
each predicted state to prefer its paired span over adjacent or permuted spans.
`transition` aligns consecutive state differences, testing progression rather
than only unordered content. Stop uses masked binary cross entropy through the
true final span. Length uses smooth L1 against clean pre-augmentation patch
occupancy. Mild variance/covariance terms expose collapse.

A stochastic energy head is deferred from the first pilot. CALM motivates that
ablation for genuinely multimodal continuations, but adding it before the
deterministic trajectory beats V38 would confound whether sequence structure or
sampling repaired the mechanism.

## Evaluation before any renderer

The exploratory evaluator uses only development data and reports:

- V38-compatible short-answer global retrieval;
- long-answer global retrieval on records that V38 could never load;
- per-span top-1/top-5/MRR and cosine, with inactive padding excluded;
- exact-position versus adjacent/permuted span preference;
- transition-direction cosine;
- predicted span-count accuracy and stop-position MAE;
- visual-length MAE;
- canonical versus held-font and simplified/traditional consistency;
- shuffled-prompt and blank-prompt drops;
- operation-stratified results without claiming operations unseen by the V38
  initialization;
- centered effective rank and finite-state controls; and
- parameters, throughput, wall time, and peak allocated VRAM.

The first pilot is explicitly exploratory. No sealed row, sealed font, renderer,
or publication claim is opened. A later evidence protocol is frozen only if all
of the following are credible on development:

1. long-answer global retrieval materially exceeds an untrained V39 head;
2. span retrieval exceeds position-only and shuffled-prompt controls;
3. ordered and transition metrics show content progression;
4. held-font/script consistency does not collapse;
5. the model remains finite and under the one-4090 resource ceiling; and
6. the trajectory improves the V38 relation rather than merely memorizing the
   enlarged training set.

## Runtime boundary

The intended deployable method is exactly:

```text
prompt pixels + visual occupancy
  -> shared visual reader
  -> V38-initialized global answer relation
  -> two-pass causal visual-query planner
  -> global state + ordered span states + stop/length fields
```

Forbidden at runtime and in the standalone checkpoint:

- strings, tokenizer calls, token/byte/Unicode/character IDs;
- OCR, transliteration, OpenCC, BGE, Qwen, or another external service;
- answer candidates, retrieval indexes, target banks, or nearest-neighbor
  selection;
- hidden source text or target answer rasters; and
- raster output claims before a separately evaluated writer exists.

## Falsification and next action

V39 is rejected if it only raises training-batch retrieval, predicts position
priors, ignores prompt interventions, loses visual invariance, or fails on the
long-answer development stratum. In that case, more updates are not evidence.
The failure determines whether to change the visual reader, continuous target
geometry, or conditional generative objective.

If the trajectory passes a later frozen semantic gate, the next experiment can
condition a V34 latent motor decoder on each `z_k`, emit visible raster spans,
and feed only decoded/re-read images into later generation. Semantic and raster
qualification remain separate until both are measured.
