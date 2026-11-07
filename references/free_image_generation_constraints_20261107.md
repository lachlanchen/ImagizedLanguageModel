# Free image generation: support and semantic assignment

The current contextual adaptation comparison tests answer selection with an
exact source-pixel motor. A complete ILM must also compose readable answers
that are absent from the input. This requirement remains unachieved.

## The released native generator is already an attempted baseline

The pinned local MIXAR-477M stage-2 model card identifies its refinement as
GRPO, consistent with the current
[author release](https://huggingface.co/TYTTYTTYT/llama-pixar-477m-p32-stage2).
It is not an untried replacement for an older GAN checkpoint. The card reports
34.6% English and 1.4% Chinese bAbI task-1 free-generation accuracy, with
external pretraining. These author-reported tasks are not a matched comparison
with this project's models.

The local [native-font demonstration](mixar477_baseline_result.md) retains
readable English spans as well as wrong continuations and damaged CJK.
The [full native adaptation](native_pixel_instruction_v61_result.md) trains all
477M parameters for 1,000 updates but gives 0/256 exact development answers.
It uses a disclosed rendering shift, so it neither isolates an intrinsic
foundation limitation nor supplies a working free writer. Repeating that
unchanged recipe or relabelling GRPO as a new remedy is not justified.

## Coherent pixels do not determine the correct answer

For a block with conditionally independent Bernoulli pixels,

\[
q(Y\mid H)=\prod_j q_j(Y_j\mid H),
\]

cross-entropy fits each conditional pixel marginal. When the next block is
ambiguous, thresholding those marginals can combine parts of different
possible writings. The MIXAR paper discusses this failure and its
free-running refinement in [Section 3.3](https://arxiv.org/html/2604.11575v2#S3.SS3).
It does not establish that an arbitrary joint image density learns language.

A latent joint law can couple pixels,

\[
q(Y\mid X)=\int q(Y\mid Z)q(Z\mid X)\,dZ,
\]

but the contextual distribution over coherent outcomes still has to be
learned. This project's [correct-history and supplied-code controls](decoder_metric_v82_result.md)
separate accurate decoding from failed conditional prediction. The supplied
code can produce the correct image while the prompt-driven model predicts
the wrong image. Correlation, reconstruction and lower likelihood losses
therefore cannot substitute for autonomous conditional-answer tests. The
observations do not identify one unique internal cause.

The [retained-history audit](history_prefix_alignment_result.md) also finds
that only 2/75 intermediate image-history boundaries have a matching saved
source-language state. Most 64-column prefixes do not equal a complete
reference-text prefix raster. A state comparison by ordinal step alone is
therefore invalid here. Causal history alignment remains an untested
intervention, not an established explanation or repair.

## Copying has a strict support limit

Let \(\Omega(X)\) contain all admissible complete source crops of image \(X\).
The current component satisfies

\[
p_{\mathrm{copy}}(Y\mid X)=0\qquad\text{for }Y\notin\Omega(X).
\]

No amount of optimizing this distribution can make it compose an absent
answer. An ordinary mixture with a free-writing distribution would enlarge
the support, but would not establish that the free branch learns the correct
language or remains legible. This is a support constraint, not a new algorithm
or a validated repair.

The running [matched comparison](contextual_pixel_adaptation_protocol.md)
tests one unresolved component: contextual semantic assignment while copying
is physically accurate. Its fixed controls, training budget and final
assessment remain unchanged. If it improves, its learned reader may be a
useful component for later writing experiments; if it fails, more unchanged
optimization is not established as the solution. Either outcome still requires
free answers on prompts whose references are absent from their inputs, raw
image inspection, held-out semantic evaluation and complete cost accounting.
There is no new model run or download associated with this review.

The companion [visual-reader and glyph-source review](visual_reader_and_glyph_source_review_20261107.md)
checks newer source papers and the already attempted PIXEL-M4 initialization.
An encoder, a character classifier and a supplied-character font generator
each leave the conditional free-writing requirement unresolved.
