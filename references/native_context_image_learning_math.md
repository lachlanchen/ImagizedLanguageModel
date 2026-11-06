# Conditional language learning through continuous writing images

This note describes the implemented native-context candidate and its unresolved
learning problem. It does not establish a new learning theorem or a successful
image-language model. The completed experiments are bound in
[native-context evidence](evidence/native_context_v80.json); the current run is
defined by the [Chinese training protocol](native_chinese_v80_protocol.md).

## Physical observations and the deployed state

Let X be an observed prompt raster, and let Y=(Y₁,…,Yₙ) be an answer stream of
binary 32×64 image blocks. These are physical coordinates, not members of a
language vocabulary. The frozen codec E maps each block to two continuous
768-dimensional tile vectors. Write Zⱼ=E(Yⱼ). Its decoder D maps predicted
coordinates back to soft image pixels. The final physical emission operator is

\[
  B(z)=\operatorname{threshold}\big[D(\operatorname{LN}(z))\big].
\]

LN acts separately on each 768-dimensional tile. Thresholding produces the
actual signed pixels retained in the output. During inference the next input
is E(B(z)), never an invisible unrendered coordinate substituted for that image.
The complete deployable algorithm needs neither a Unicode table nor a text
decoder. These interface properties do not establish semantic competence.

The inherited frozen visual reader prepares full layerwise attention keys and
values K(X), an initial final-layer state h₀(X), and native rotary offsets.
For j completed image blocks, the learned tile encoder Aφ and frozen native
contextual body Fψ compute

\[
  h_j=F_\psi\left(K(X), A_\phi(Z_1),\ldots,A_\phi(Z_j)\right)_{2j}.
\]

There are two native input positions per image block. No character boundaries
or text tokens are inferred by this equation. The continuation mask allows
all owning prompt keys and earlier image positions, excludes later image
positions and right-padding, and keeps actual source rotary coordinates
separate from padded cache indices. Each hⱼ conditions the next block.

The first block is conditioned on h₀ alone. Its clean predictor does not make
an additional query into the full KV. This is a remaining bottleneck: a source
decoder's correct first symbolic piece does not prove that h₀ supports a full
multi-character image block. Later image decisions do query the full KV through
Fψ. First-block image quality therefore needs explicit measurement; retaining
the cache is not by itself a remedy for every output failure.

## What the writing loss actually learns

For a target coordinate z, draw ε∼N(0,I) and t∼Uniform[0,1], then construct

\[
  z_t=(1-t)\epsilon+tz.
\]

The clean predictor Cθ(zₜ,t,h) has a full 1,536-dimensional output. For each
target block the implemented objective is

\[
  \mathcal L_{\mathrm{write}}=
  \mathbb E\left[
  \frac{\|C_\theta(z_t,t,h)-z\|_2^2/1536}
       {\max(1-t,0.05)^2}\right].
\]

Blocks are averaged within each example, then examples are equally weighted.
At a fixed t and h, the unrestricted squared-error optimum is the conditional
mean E[z|zₜ,t,h]. This elementary regression fact does not mean that one direct
mean prediction is a readable answer. At high noise, an inadequate condition
can average incompatible glyphs or answers. A flow can represent a multimodal
conditional distribution only if the learned field and numerical sampler
preserve the required structure. A low average loss does not verify that.

The associated flow velocity estimate is

\[
  v_\theta(z_t,t,h)=\frac{C_\theta(z_t,t,h)-z_t}{1-t}.
\]

With T=32 uniform steps, tₖ=k/T and r=T−k, the implemented update is

\[
  z_{k+1}=\frac{r-1}{r}z_k+\frac1rC_\theta(z_k,t_k,h).
\]

This contracts linear noise exactly when the clean estimate is constant over
the interval; a learned varying predictor still introduces approximation
error. The last step eliminates the explicit previous-noise coefficient. It
does not eliminate prediction error, guarantee a valid glyph or imply that
the answer matches the question. This loss is not reported as an exact image
likelihood or as a proof of language learning.

The stop head sω(h) supplies a balanced logistic loss: continuing positions are
averaged within an example, and the terminal position gets equal aggregate
weight. Inference emits a block while sigmoid(sω(h))<0.5 and stops otherwise,
with a declared 128-block cap. Calibration of this balanced objective as a
true length hazard is not assumed. Retained stop events and actual output
lengths are measured directly.

## Rendering, semantic conditioning and rollout are distinct requirements

Three different errors can each yield unreadable or irrelevant output:

1. D(E(Y)) can corrupt marks even when its coordinates are supplied correctly.
2. Cθ can fail to select the owning answer or to predict correct coordinates,
   even under true answer history and supplied answer length.
3. Free inference conditions on E(B(ẑ)), whereas training uses E(Y). Errors
   in early emissions can therefore create histories not covered by training.

The previous V75 results already contradict a rollout-only explanation:
correct history and length still give zero exact answers on the 32 training
diagnostics, while supplied-code reconstruction reaches 99.9953% F1. The source
visibility experiment also identifies a contextual-access difference: masking
older keys reduces correct helper answers from 19/32 to 2/32 after identical
prefill. Neither experiment isolates a unique cause of the entire failure.

The next measurements must include first-block output, full free output,
owning-question dependence, physical feedback and stopping. Reconstructing a
fixed glyph, fitting selected training cases and passing numerical execution
tests are necessary component checks with narrower scope than useful language.

## What recent work supports, and what it does not

[TextLDM, May 2026](https://arxiv.org/html/2605.07748) uses token-derived continuous
latents and reports that alignment to pretrained language features is important
for generation. It supports investigating prediction-relevant representations;
its tokenizer and text-decoder formulation does not satisfy an image-only
deployment contract. Its reconstruction results are not evidence for this ILM.

[MIXAR v2, October 2026](https://arxiv.org/html/2604.11575v2) evaluates multilingual
pixel generation and introduces an own-rollout reinforcement stage. It is
relevant evidence that rendered-language understanding and free writing require
separate evaluation. Its foundation-training budget is external to local ILM
training, and its results do not establish parity with a large instruction model.

[Multi-Block Diffusion Language Models, June 2026](https://arxiv.org/html/2606.29215)
studies training/inference alignment and prefix-cache semantics for discrete
language diffusion. That scope differs from continuous glyph generation. Its
analysis motivates checking the actual feedback distribution; it supplies no
guarantee for this codec, body or one-GPU budget.

Training-only teacher assistance is allowed by this project: a future objective
could align continuous image features to useful frozen language states while
omitting all teacher services and symbolic heads at deployment. This is a
possible later intervention, not part of the frozen V80 experiment. The current
candidate already inherits externally pretrained numeric contextual weights;
its complete parameter count and external training remain explicit.

No reviewed result makes image language inherently easier than token language
or guarantees frontier-level knowledge from single-card training. The practical
claim to test is narrower and measurable: locally learn a reliable visual
writing interface that uses available contextual language information, produces
correct new answers and remains an independent image-input/image-output actor.
The full general language and historical-script objective remains open.
