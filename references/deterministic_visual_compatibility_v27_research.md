# V27 Research Decision: Joint Visual Compatibility Learning

Date: 2026-08-13

Status: research direction after V26 and its frozen-state diagnostic; fixed
protocol preregistered in
[`joint_visual_compatibility_v27_protocol.md`](joint_visual_compatibility_v27_protocol.md)

## Decision

V27 will test a deterministic image-conditioned compatibility model trained
jointly with its visual context representation. It will not inherit V26's
language encoder as a frozen foundation, generate stochastic visual particles,
train a pixel writer, or open the frozen split before a separate fixed protocol
authorizes it.

The narrow hypothesis is:

> A compact causal image encoder can learn a transferable relation between a
> 64-cell written context and an arbitrary candidate next-glyph image when
> matched alternatives are the supervised object of every relevant update.

## Why V26 Requires This Change

V26's stochastic proposal changes only slightly when suffix-matched context
changes, even though its fused state changes materially. A post-hoc probe then
froze the entire 8,000-update checkpoint and trained separate nonlinear visual
candidate scorers over its appearance, history, and fused states. On 512
development pairs and 2,048 cross-font decisions, history reached `0.506836`
and fused state `0.503418`. The same retina matched candidate identity at
`0.999512` with a `0.747237` cosine margin.

This rules out the simple V27 plan of retaining V26's context state and changing
only the output distribution. The learning objective must shape the context
representation itself around visual candidate relations.

## Proposed Core

Let `X` be a sequence of 64 glyph images and `Y` an arbitrary candidate image.
A trainable image retina `R`, causal context field `F`, and candidate projector
`G` produce

\[
q(X)=\operatorname{normalize}(F(R(X))),\qquad
k(Y)=\operatorname{normalize}(G(R(Y))).
\]

The deterministic visual compatibility is

\[
s(X,Y)=\tau^{-1}q(X)^\top W k(Y),
\]

where `W` is low-rank or identity-initialized. This is an image-image energy,
not a class logit: the model can score a candidate form never assigned an ID.

For suffix-matched pairs `(X_A,Y_A),(X_B,Y_B)`, the primary local objective is
the symmetric assignment loss

\[
\mathcal L_{2\times2}=\tfrac12\left[
\operatorname{CE}(S,I)+\operatorname{CE}(S^\top,I)
\right],
\quad S_{ij}=s(X_i,Y_j).
\]

A larger in-batch or queued image set supplies frequency and hard visual
negatives. Duplicates must be detected by independent image views rather than
character IDs. Appearance-only, suffix-only, and history-shuffled branches are
optimized or evaluated under the same candidate set so gains are attributable.

## Representation Constraint

V16's frozen retina is an excellent form recognizer but has effective rank near
19 in 192 dimensions on the audited bank. V27 should not rely on it as the sole
language key. Two image-only routes are preferable:

1. a trainable language retina shaped by cross-font candidate compatibility;
2. a frozen appearance retina retained only for exact form preservation and
   evaluator positive controls.

The trainable route should use independent font/augmentation views and explicit
variance/covariance regularization so its dimensions do not collapse. A
stop-gradient target retina with exponential moving average is permitted if its
inputs remain images and the deployed model contains no character table.

## Relevant Learning Principles

- [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) makes future
  compatibility, rather than future point reconstruction, the predictive task.
- [SimCLR](https://arxiv.org/abs/2002.05709) shows the importance of normalized
  projections, temperature, independent views, and sufficiently informative
  negatives.
- [VICReg](https://arxiv.org/abs/2105.04906) supplies direct variance and
  covariance constraints without a discrete vocabulary.
- [data2vec 2.0](https://arxiv.org/abs/2212.07525) supports efficient contextual
  latent targets and a teacher/student route without making an external model a
  deployed dependency.
- [Continuous Visual Autoregressive Generation via Score
  Maximization](https://arxiv.org/abs/2505.07812) remains relevant for a later
  writer, but V26 shows that stochastic form modeling should follow rather than
  substitute for identifiable language compatibility.

## Offline Teacher Boundary

Qwen 4B/8B or BGE-M3 may later generate curricula, paraphrases, relation labels,
or difficulty schedules offline. Such outputs must be rendered into images
before reaching the student, provenance must be recorded, and no teacher call or
hidden text channel may exist at student inference. V27's first bounded test
should avoid a semantic teacher so the natural next-glyph mechanism is isolated.

## Required Controls Before Protocol Freeze

- exact pixel equality for every matched suffix;
- appearance-only pair accuracy exactly at chance;
- raw-retina cross-font identity at least `0.99`;
- context candidate accuracy above `0.65` on cross-record suffix-4 pairs;
- ordered-history gain over a suffix-preserving prefix shuffle;
- natural next-image retrieval above image unigram and symbolic bigram;
- a candidate-set permutation check showing no row or position shortcut;
- a recursive boundary receipt proving image-only student inputs and outputs;
- no V27 frozen images before development selection and protocol freeze.

## What Passing Would Mean

Passing would establish one bounded, vocabulary-free image-to-image language
relation on ordinary Chinese: context images rank their associated next-glyph
image above matched alternatives. It would not yet establish free-form answers,
historical etymology, page generation, Qwen parity, or superior efficiency.

Only after this mechanism passes should a conditional visual writer sample the
form favored by the learned compatibility energy. Ordered glyph planes,
serpentine page lattices, geometric depth, and movies remain compatible internal
or interface representations, but they are not the immediate bottleneck.
