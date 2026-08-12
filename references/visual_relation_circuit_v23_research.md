# V23 Research Decision: Relation Before Generation

Date: 2026-08-13

Status: architecture decision before V23 implementation or training

## Decision

V23 will not scale or tune the rejected V22 Transformer selector. It will test a
smaller and more constrained hypothesis:

> A visual prompt can control a generated Chinese answer image when the model
> first computes explicit continuous relations among visible frames, then
> routes the resulting source image through an independently qualified writer.

The bounded task is unchanged in spirit. Two visible label images bind two
previously unseen glyph images. A visible operation image means same or other.
A final visible query-label image determines which glyph should appear in the
answer image. Strings and symbolic values exist only in the offline renderer
and evaluator.

## Why V22 Failed

V22 represented all six prompt frames with a visual Transformer, then collapsed
them through one softmax selector. Its endpoint writer and unseen-identity
metrics rose, but prompt behavior did not. The query-aware endpoint reached
`0.00781` counterfactual switch accuracy and only `0.00890` paired-output pixel
L1. A reproducible audit found:

- operation-frame selector argmax: `1,024/1,024` prompts;
- mean operation attention: `1.0`;
- mean query-label attention: `1.37e-13`; and
- glyph-frame argmaxes: `0/1,024`.

The answer is a relation, not a frame. It depends jointly on query-to-label
equality, label-to-glyph pairing, and operation semantics. The V22 entropy term
made the easiest stable marker an even stronger shortcut. More parameters or
steps would not repair the missing computation.

## Primary Work And Transferable Lessons

### Explicit relation modules

[Santoro et al., *A simple neural network module for relational
reasoning*](https://arxiv.org/abs/1706.01427) showed that strong convolutional
features alone did not solve relational questions in Sort-of-CLEVR, while an
explicit Relation Network did. The relevant lesson is architectural: if the
target depends on relations among perceptual entities, expose pairwise
composition in the hypothesis class.

### Exchangeable visual slots

[Slot Attention](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html)
maps perceptual features to exchangeable object slots and formalizes
permutation equivariance over slot order. V23 does not need unsupervised object
discovery because its six visual frames are already separated by the sensor,
but it adopts the exchangeability requirement: swapping the two label/glyph
pairs must leave the routed answer unchanged.

[Slot Abstractors](https://proceedings.mlr.press/v235/mondal24a.html) combine
object slots with a strong relational inductive bias and report systematic
generalization across abstract visual reasoning tasks. This supports separating
object/form representation from relation computation rather than asking one
generic attention stack to discover both.

[Parallelized Spatiotemporal Slot Binding](https://proceedings.mlr.press/v235/singh24g.html)
shows that slot binding can be computed across sequential visual inputs without
an inherently serial recurrent bottleneck. V23 remains a six-frame proof, but
its relation circuit is compatible with later parallel Visual Language Stream
windows.

### Same/different visual generalization

[Tartaglini et al., *Deep Neural Networks Can Learn Generalizable
Same-Different Visual Relations*](https://arxiv.org/abs/2310.09612) report that
the same/different relation can generalize out of distribution under suitable
visual pretraining and inductive bias. V23 uses a frozen image-trained retina
and a shared metric comparator; it does not infer equality from answer identity
or a character table.

### Permutation structure

[Deep Sets](https://arxiv.org/abs/1703.06114) characterizes permutation-
invariant set functions and motivates building the required symmetry into the
model. V23's two-pair exchange is small enough to enforce algebraically and test
at floating-point tolerance rather than hope it appears from augmentation.

## Measured Design Diagnostics

These diagnostics use the already-open V22 development renderer and are not
V23 evidence. No V23 development or frozen image was rendered.

### Retinal visual equality

Across `4,096` V22 development prompts with independent font/augmentation views,
cosine similarity between the frozen V16 retinal states matches the query image
to the correct label image with:

| Diagnostic | Measured |
|---|---:|
| Correct-label match accuracy | `0.99829` |
| Correct-label cosine | `0.85276` |
| Wrong-label cosine | `0.14468` |
| Mean cosine margin | `+0.70808` |
| Fifth-percentile margin | `+0.21746` |

This supports a shared metric comparator. The comparator is image-native because
the retina learned its state from pixels; no label identity is available at
inference.

### Operation image state

A deterministic train/test linear probe on the same old-split retinal states
separates the visible `同` and `异` operation images at `1.0` held-out accuracy.
The probe is a design diagnostic only. V23 resets and learns its operation gate
from answer-image loss, without operation labels entering the student.

### Pixel-preserving writer

A disposable `1,122,081`-parameter convolutional canonicalizer was trained on
V22 training identities and evaluated on V22 development identities. It maps a
noncanonical glyph image to a canonical answer image and preserves a direct
local pixel route. Development F1 progressed:

| Updates | Canonicalizer F1 | Raw source-to-target F1 |
|---:|---:|---:|
| 200 | `0.70030` | `0.52790` |
| 400 | `0.73205` | `0.52790` |
| 600 | `0.74429` | `0.52790` |
| 800 | `0.76336` | `0.52790` |
| 1,000 | `0.75690` | `0.52790` |
| 1,200 | `0.76601` | `0.52790` |

This is not a checkpoint or result and was not saved. It justifies
preregistering a fresh-split canonicalizer instead of reusing V22's lossy
state-only writer (`0.60196` oracle F1).

## Chosen Mathematical Circuit

Let the six prompt images be

\[
X=(l_1,g_1,l_2,g_2,o,q).
\]

The frozen retina emits normalized continuous visual states. Equality is the
shared retinal metric

\[
c_i=\langle R(q),R(l_i)\rangle,
\qquad
m_i=\operatorname{softmax}_{i\in\{1,2\}}(\tau c_i),
\]

where the positive temperature `tau` is learned. A small operation-image reader
emits

\[
s=\sigma(U_\theta(R(o))).
\]

No same/other label supervises `s`; answer-image loss does. The final source
weights are

\[
w_i=s m_i+(1-s)(1-m_i).
\]

For two normalized match weights, they sum to one. `s=1` routes the visually
matching glyph and `s=0` routes the other glyph. The routed source image is

\[
x_r=w_1g_1+w_2g_2,
\qquad
\hat y=C_\psi(x_r),
\]

where `C_psi` is the independently selected and frozen visual canonicalizer.
The writer cannot inspect labels, operation, query, or Transformer context. The
relation circuit cannot synthesize an average character from an operation
state; it can only choose a continuous mixture of the two visible source
images.

## Exact Symmetry And Counterfactuals

Let `P` swap `(l_1,g_1)` and `(l_2,g_2)`. Shared comparison and commutative
weighted summation require

\[
f(PX)=f(X).
\]

Changing only the query image swaps `m_1,m_2`, so the output identity must
switch. Changing only the operation image maps `s` to the other semantic branch
and must also switch the answer. These are distinct causal tests:

- query-blind control: replace only `R(q)` with a learned continuous null;
- operation-blind control: replace only `R(o)` with a learned continuous null;
- pair swap: exchange both label/glyph pairs;
- query counterfactual: change only `q`;
- operation counterfactual: change only `o`.

All arms retain the same modules and parameter shapes. The controls remove one
visual cause rather than reducing capacity.

## Why This Is Not A Hidden Token Model

The student receives six float image tensors and emits a float image tensor.
The retina, metric, operation gate, routed source, and writer are continuous.
There is no finite answer table, character embedding, Unicode index, token
distribution, OCR transcript, glyph lookup, target index, or external model.
The two-pair algebra is a fixed inductive bias for a bounded visual grammar,
analogous to convolution encoding translation structure. It is not claimed to
be general Chinese syntax.

## Alternatives Rejected For V23

- **Wider generic Transformer:** does not remove the V22 shortcut path.
- **Supervised target-slot attention:** would make the renderer's symbolic
  answer index part of training and weaken the image-only proof.
- **End-to-end binder and writer:** permits the writer to absorb operation or
  query biases and hide failed routing.
- **Vector-symbolic identity keys:** useful for larger binding systems, but a
  fixed discrete key inventory is unnecessary and risks recreating a codebook.
- **Full-page diffusion:** spends compute on surface appearance before the
  one-glyph causal relation is established.
- **Historical-form generation now:** factual historical forms require source
  provenance and copy/synthesis labeling. The first relation proof should use
  controlled modern Chinese images, then extend the same source-image route to
  oracle, bronze, seal, manuscript, and unencoded forms.

## Efficiency Hypothesis

The relation circuit has `25,602` trainable parameters. The independently
trained writer has `1,122,081`; it is frozen during binding. The deployed learned
student therefore adds about `1.15M` parameters around the already frozen
retina. The V22 prototype evidence used only about `0.315 GiB`, and the writer
prototype fits comfortably on one RTX 4090. V23 tests structural sample and
compute efficiency only; it cannot establish superiority over an LLM until the
same useful task and quality are compared.

## Scope Of A Positive Result

A positive V23 result would establish only that a compact image-only circuit can
compose this visible same/other binding grammar and generate the requested
unseen Chinese form under causal counterfactuals. It would justify moving to a
second answer frame and then a short image-language stream. It would not prove
open-domain language understanding, free-form questions, etymology knowledge,
arbitrary historical synthesis, page/movie generation, Qwen parity, or
end-to-end efficiency.
