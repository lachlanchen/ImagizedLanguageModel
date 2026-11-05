# V72: image semantics and a causal continuous writing model

Registered before cache creation, optimization or evaluation. V71 retains
bounded visual reading in the imported contextual body through numeric task
controls. V68's deterministic code regression remains a failed full-answer
model. V72 separates a frozen image semantic memory from a small causal
writing model with a joint conditional code distribution.

## Complete deployed computation

1. Receive a physical prompt image. Apply the declared V71-style native-budget
   Pillow bilinear observer, preserving original input pixels separately.
2. Run the authenticated source-derived vision/contextual body with the fixed
   numeric controls. Its vocabulary module remains absent. Retain all final
   task-frame contextual states, stored and used in BF16.
3. A causal writing transformer conditions on these continuous states and the
   image-derived codes of completed answer blocks. A joint flow predicts the
   next two 768-dimensional writing vectors, without a character dictionary.
4. The fixed V34 decoder emits a physical 32-by-64 block. Threshold the actual
   output pixels and encode those emitted pixels again before the next step.
   Do not recycle predicted states as if they were reread images.
5. A learned structural stop decision ends the answer, with a uniform
   128-block capacity. No reference length or code enters generation.

This is an independent pixel-to-pixel candidate. It has no tokenizer, OCR,
source output head, Unicode IDs, persistent glyph bank or teacher call in
inference. The inherited weights and numeric control initialization retain
their external-pretraining attribution. Successful reader diagnostics do not
establish successful writing.

## Frozen source memory and actual training data

Use all 12,976 admitted CMRC/Dolly complete training images, from the original
source-bound cache. No V70/V71 development response may enter training. Remove
only the cache's two explicit white termination blocks; preserve every real
answer block and its ordinary padding. Cache prompt-only semantic states and
detached target codes on physical GPU 0. Use the qualified original V34 codec
and the pure numeric `structural_vectors.pt` from V71, checking their hashes.
No symbolic compatibility helper is loaded during caching or training.

Shuffle the full admitted file list once with seed 20267201 for cache order.
Every source appears exactly once. Save source file hashes, input/transformed
pixel hashes, geometry and every cache payload hash. Support exact safe cache
resume. A first 128-record cache phase is an execution preflight, not a
language-training subset or a quality result. Complete the full corpus before
production optimization.

## Candidate decoder and objective

Use a writing width of 512, eight causal transformer layers, eight attention
heads and a 2,048-wide feed-forward path, with no dropout. Project each
4096-dimensional semantic-memory state to width 512. A learned structural
start vector and projected previous image codes follow the memory prefix.
Fixed sinusoidal positions and a causal mask prevent access to future blocks.
Predict a stop logit from every writing state, including the terminal one.

Use a joint code-flow head with width 768 and three residual MLP blocks. For
target code z=E_w(Y), Gaussian noise e, and uniform time t, train

    z_t = (1-t)e + t z
    L_flow = E ||v_theta(z_t,t,h) - (z-e)||^2.

This is conditional flow matching, not a categorical vocabulary or a
normalized text likelihood. Average uniformly sampled block losses within
each example, then over examples. Sample up to 16 blocks per example while
retaining complete causal histories. Add the existing balanced stop/continue
logistic loss with equal weight. The fixed codec supplies detached image
targets and does not train. Use 32 explicit midpoint flow steps in generation,
then apply the codec's non-affine LayerNorm to each 768-vector before
physical decoding. This preserves its training coordinates (mean zero,
variance one); unit Euclidean normalization would be incorrect.

## Budget and qualification boundary

The planned production run uses 40,000 AdamW updates, batch size 16, learning
rate 1e-4, 500-update warmup, cosine decay to one tenth, betas (0.9,0.95),
weight decay 0.01 and gradient clipping at one. Use BF16 decoder autocast with
FP32 parameters/optimizer states; no EMA. Complete fresh seeded full-corpus
permutations, never choosing checkpoints by development scores. This exposure
and objective differ from V68; it is a new candidate, not a matched claim
about the isolated contribution of one component.

Before the full cache and training job, run real cached-example checks of
causality, target exclusion, finite gradients, learned stopping, rendered
feedback and exact optimizer/RNG resume. Do not label that execution check a
language pass. Keep one heavy project job on GPU 0, preserve other projects,
measure memory and phase-specific time, and reuse existing weights.

Evaluate complete autonomous answers on the unchanged 192-case V68 audit,
including all matched blank controls and original Chinese quality gates.
Preserve every raw output and score it independently. Failed outputs cannot
be repaired or replaced. The sealed final split stays unopened unless the
original development requirements pass. The overall goal remains useful,
scalable image-language generation on one consumer GPU, not merely a fitting
or decoder execution result.
