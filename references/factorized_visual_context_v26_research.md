# V26 Research Decision: Factorized Visual Context

Date: 2026-08-13

Status: architecture decision written before V26 implementation, optimization,
or result inspection

## Decision

V26 will test whether a compact image-native model can use visible Chinese
history beyond a shared four-cell suffix to predict a distribution over the
next glyph image. It will not train a page renderer, open the frozen split, or
claim general language understanding.

The model will preserve three exactly related views of writing:

1. one glyph is a continuous `1x32x32` retinal image;
2. writing is an ordered visual-time volume `T x 1 x 32 x 32`; and
3. that volume can be folded reversibly into a serpentine 2D page lattice.

The first two are the V26 learning interface. The third is storage and display
geometry. No view introduces a character ID.

## What V25 Actually Established

V25 was a valid negative result. Its frozen V16 retina recognizes the correct
row in the 1,024-form development image bank perfectly (`2048/2048`), so the
sensor and evaluator are not the observed bottleneck. The learned language
proposal nevertheless reached only `0.01123` top-1, below the image-unigram
baseline (`0.01611`) and far below the symbolic training-text bigram
(`0.12158`).

A post-result diagnostic used the fixed V25 checkpoint and the same 2,048
development windows. It changed only the visible suffix length and never
instantiated frozen images:

| Visible cells | Top-1 | Target log probability | Target cosine |
|---:|---:|---:|---:|
| 1 | 0.00146 | -7.01288 | 0.24018 |
| 2 | 0.00244 | -6.99326 | 0.25205 |
| 4 | 0.00830 | -6.97218 | 0.26855 |
| 8 | 0.00977 | -6.96818 | 0.27025 |
| 16 | 0.01025 | -6.96688 | 0.27225 |
| 32 | 0.01025 | -6.96608 | 0.27373 |
| 64 | 0.01123 | -6.96406 | 0.27513 |

Most of V25's measurable gain appears by four cells. Cells 17 through 64 add
only about `0.00282` nat relative to 16 cells. This does not support a claim
that the model learned distributed visual context.

## Diagnosed Objective Mismatch

V25 generated 64 predictions from every 65-cell sample and averaged its cosine
loss uniformly over all positions. Its in-batch contrastive loss flattened the
same 64 transitions and subsampled at most 512 of them. The development gate,
however, judged only the final transition after a full 64-cell context.

Consequently, short-prefix and locally predictable transitions supplied most
of the gradient while the selected behavior was a long-context endpoint. A
single normalized proposal also represented a generally multimodal next-glyph
distribution by one point. Mean-seeking in the continuous retina can raise
cosine without ranking the correct visible form.

V26 changes the supervised unit rather than merely adding parameters:

- every natural training example has a full 64-cell context;
- the last glyph appearance and preceding history are separate branches;
- natural pairs share their last four glyphs but have different next glyphs;
- the target is a continuous conditional particle distribution; and
- future-image heads at offsets 2, 4, and 8 provide larger predictive targets.

## Natural Pair Feasibility

The host-side audit used only the existing train and development partitions.
It did not render or inspect the frozen partition.

- Training contains `2,635,312` transitions whose previous and next forms are
  both in the 1,024-form audit bank. All 1,024 previous forms have at least two
  observed next forms.
- Development contains `76,021` such transitions; 1,019 previous forms have
  at least two observed continuations.
- Exact four-cell suffixes yield 5,406 ambiguous development suffixes and
  12,790 balanced pair examples. Requiring the two contexts to come from
  different source-record identifiers leaves 990 suffixes, enough for a fixed
  512-pair causal audit.
- Exact eight-cell suffixes yield 1,712 ambiguous suffixes and 3,622 balanced
  examples, but only 59 cross-record suffixes. V26 therefore uses suffix-4
  cross-record pairs as its selection test and suffix-8 within-record pairs as
  a stricter diagnostic.

Pair construction may inspect writing strings offline. Student batches contain
only rendered floating-point image tensors. Paired contexts use the same font,
size, and augmentation draw so their shared suffix is pixel-identical.

## Mathematical Basis

### Continuous conditional distributions

[Continuous Visual Autoregressive Generation via Score
Maximization](https://arxiv.org/abs/2505.07812) formulates continuous visual
autoregression through strictly proper scoring rules and studies the energy
score without vector quantization. This directly addresses V25's one-point
regression problem. V26 uses a small empirical conditional distribution of
continuous retina particles, not a finite glyph vocabulary.

For unit particles `q_k` and target retina `y`, the empirical energy score is

\[
\mathcal E(Q,y)=\frac1K\sum_{k=1}^{K}\lVert q_k-y\rVert_2
-\frac{1}{2K^2}\sum_{k,j=1}^{K}\lVert q_k-q_j\rVert_2.
\]

The first term rewards fidelity. The second prevents a multimodal conditional
distribution from being represented only by its mean. Particles are produced
from continuous Gaussian noise and context; they are not persistent class
prototypes.

### Predicting sufficiently large futures

[I-JEPA](https://arxiv.org/abs/2301.08243) reports that semantic
representations depend on predicting sufficiently large targets from
distributed context. [Multi-token
Prediction](https://arxiv.org/abs/2404.19737) finds that auxiliary future heads
can improve sample efficiency and induction behavior in language models. V26
therefore predicts image observations at offsets 1, 2, 4, and 8 from one
64-cell context. Only offset 1 is used for autoregressive inference; the other
heads are training-time predictive constraints.

### Continuous entities and noisy history

[xAR](https://arxiv.org/abs/2502.20388) treats an autoregressive unit as a
continuous entity and trains with noisy context to reduce teacher-forcing
exposure bias. V26 keeps the entity human-inspectable: one complete glyph
image. It uses mild continuous corruption on natural contexts, but leaves
causal pair suffixes clean so the intervention remains exact.

## Chosen Factorization

Let the context end in visible cell `x_t` and let the earlier cells be
`X_{<t}`. A frozen image retina gives

\[
a_t=R(x_t),\qquad H_t=F(R(X_{<t})).
\]

The appearance branch can see only `a_t`. The history branch can see only the
preceding 63 image states. Their fused state is

\[
s_t=\operatorname{RMSNorm}\left(A(a_t)+g_t\odot C(H_t)\right).
\]

Setting the history residual to zero is therefore a real last-cell
intervention. Swapping history residuals between pixel-matched pair members is
a real causal intervention: the last appearance remains fixed while only the
earlier visual context changes.

For future horizon `h` and Gaussian samples `epsilon_k`, one shared conditional
head emits

\[
q_{h,k}=\operatorname{normalize}
P(s_t+e_h+N(\epsilon_k)),\qquad k=1,\ldots,8.
\]

`e_h` describes visual time distance, not glyph identity. The output contains
no vocabulary-sized matrix.

## Why A Training-Only Visual Queue Is Allowed

The 1,024-row audit bank proves that the frozen retina can separate the visible
forms, but a microbatch supplies too few negatives to train that separation.
V26 may maintain a FIFO queue of detached target-retina observations. It is a
training loss buffer derived from images, not a model parameter, character
table, or deployed retrieval bank. Near-identical target observations are
treated as multiple positives by retinal cosine, not by labels. The queue is
absent from checkpoints used for deployed generation.

## Alternatives Rejected

- **Increase V25 width or steps unchanged:** preserves the endpoint/objective
  mismatch and gives no reason to expect long-context dependence.
- **Classify 1,024 character IDs:** would likely improve top-1 while abandoning
  the image-native output distribution and unseen-form interface.
- **Use an `8x8` glyph as the sole record:** destroys distinctions in complex
  Han forms. Coarse views may become auxiliary signals only after `32x32`
  writing works.
- **Flatten every pixel into an autoregressive token:** makes one glyph require
  1,024 serial decisions and confounds language with stroke rendering.
- **Train a writer before language selection:** can produce attractive glyphs
  while hiding a failed conditional language state.
- **Open the V25 frozen split:** V26 was designed after V25 development results;
  only a new preregistered V26 gate may authorize one frozen evaluation.
- **Use suffix-8 cross-record pairs as the sole gate:** 59 available pair keys
  are too small for a stable 512-pair selection audit.

## Falsification Standard

V26 fails its core hypothesis if it cannot prefer the correct next-glyph image
when two held-out contexts have pixel-identical four-cell suffixes and
different targets, or if zeroing/shuffling the earlier visual history leaves
its scores unchanged. Better reconstruction, smooth particles, or higher
target cosine cannot substitute for that causal result.

Passing this bounded test would justify attaching a continuous pixel writer
and then extending the same stream to rendered prompts and short answers. It
would not establish book-scale knowledge, historical etymology, arbitrary
script generation, Qwen parity, or superior compute efficiency.
