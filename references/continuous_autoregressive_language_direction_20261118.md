# A native visual sequence model with a continuous joint writing density

This is an implemented research candidate, not a working language result or
an efficiency claim. The complete language, stopping, held-out source, original
success and consumer-GPU criteria in the engineering goal remain unchanged.

## Failure and selected change

The completed correction motor lowers validation F1 and introduces ink at all
3,206 true spaces. It is rejected. The binary plan also loses detail that the
continuous spatial codec retains. Do not extend those fits, choose intermediate
weights or admit either failed actor for language generation.

Earlier native V46/V48 models learn some ordered Chinese continuation but draw
conditional averages. V54's normalized pixel density preserves a supplied
image while giving almost identical ordered/shuffled first-image results. The
later pixel head fits all 512 training blocks and none of the development
answers. The inherited-body V68 coordinate predictor and full-context flow
comparison also fail. These are distinct failures; no universal explanation is
established. A better renderer alone cannot provide language understanding.

The selected route uses the already qualified V34 continuous neural codec,
a new native causal reader and a normalized autoregressive density over
**continuous writing coordinates**. It avoids another lossy binary codec and
does not regress a single complete future vector. All visual-coordinate choices
are used during generation. Both prospective arms have the same new causal
reader and local autoregressor; change only the group density from one Gaussian
to eight shared-component Gaussians. Additional output parameters and work are
reported, not hidden as an equal-capacity result. No old failed control is rerun.

This differs from V62's eight whole-pixel Bernoulli experts on an externally
pretrained 477M backbone: continuous codec coordinates have a separate accurate
motor, and 48 successive conditional groups can select different components.
It differs from flow matching: there is a tractable normalized latent density,
no time schedule, velocity target or numerical diffusion sampler. These changes
define a new comparison, not proof that either prior mechanism was impossible.

## Probability and actual image history

Let I_t be the actual binary ink image and z_t=E(1-I_t) its 768-dimensional V34
code. The original encoder expects white-positive pixels. A trainable causal
reader computes h_t=F(z_1,...,z_t), without inherited language weights, teacher
states, glyph IDs or lexical heads. Split the future code into 48 ordered groups
of 16 coordinates. For each group j,

\[
 p(z_{t+1}\mid h_t)=\prod_{j=1}^{48}\sum_{k=1}^{K}
 \pi_{jk}(h_t,z_{t+1,<j})
 \prod_{d=1}^{16}{\cal N}(z_{t+1,jd};\mu_{jkd},\sigma_{jkd}^2),
 \quad K\in\{1,8\},\quad \sigma=.02+\operatorname{softplus}(s).
\]

The component is shared across the sixteen coordinates in one group. This is
a normalized joint code density, with positive scales and log-sum-exp scoring.
Group j never receives its own or a later target. Teacher-forced groups allow
parallel training; generation uses cached attention to the emitted prefix.
Gaussian code likelihood can be negative in nats and is **not pixel NLL**.
The encoder/decoder is not invertible and no Jacobian identity is asserted.
Thresholded decoding induces an image probability through the code law, but
its exact marginal probability is not calculated here.

At generation, either sample a component and its Gaussian, or use the fixed
greedy component mean. The latter is a deterministic heuristic, not joint MAP
and not a weighted mean of all components. Decode it with the frozen neural
motor, threshold to an image, then **encode those actual emitted pixels** before
the next causal step. A learned Bernoulli stop head decides before drawing the
next image. It receives neither target length nor future span boundaries.
Whitespaces remain ordinary images and must be learned, not removed by a rule.

The objective is mean next-code NLL per coordinate plus mean stop BCE over
answer/continuation positions including the final stop. Both arms use identical
weights for these two terms. Target-derived codes supervise training only.
The independent generation method takes prompt images and a finite cap, with
optional stochastic sampling; it cannot access a corpus or future answer.

## Research basis and claim limits

[Graves, 2013](https://arxiv.org/abs/1308.0850) uses autoregressive mixture
densities for continuous handwriting trajectories. [PixelCNN++](https://arxiv.org/abs/1701.05517)
studies logistic mixture image likelihood; its quantized-pixel law differs from
the continuous Gaussian groups here. [Masked Autoregressive Flow](https://arxiv.org/abs/1705.07057)
is further precedent for neural autoregressive density estimation. The present
model is a mixture density, not a normalizing flow. These established tools do
not constitute a new physics theorem or a first pixel-language-model claim.

The V34 codec has 7,423,361 parameters and external local training costs that
must remain in whole-system accounting. Its historical evaluation contained
shared training rasters, disclosed in the original split audit. It establishes
an image interface, not semantics. This new body starts from random weights;
offline rendering and teacher-written portions of the existing corpus still
count as preparation. No 1,000-fold gain or Qwen parity is established.

## Bounded execution before a full fit

Use the same three complete authentic training records selected by the original
full-context preflight, including the complete 513-image book. Load only their
pixel shards and the fixed codec. Do not import the old language body, teacher
states, source embeddings, development answers or reserved DRCD data. Compare
K=1 and K=8 with identical common initial tensors, source order and precision.
The longest record has 1,042 images; the causal reader limit is fixed at 2,048,
with no truncation. For book pretraining the first source image is the input
anchor and its remaining 512 images are targets. QA has 5 and 35 answer targets,
for 552 next-image targets and three additional final-stop decisions per arm.

For each arm and each complete record: save the original state, score every
answer/continuation image and final stop, backpropagate, take one actual AdamW
step, score again and reset every changed tensor exactly. Require finite losses,
positive gradients through the input, first/last reader blocks, local writer
and stop head; strictly lower same-record total objective after the step;
unchanged codec and exact reset. All six trial updates are discarded. They are
numerical/source support, not copying, complete language or generalization.
Record actual decoded initial/updated next-image outputs for all target
positions under **correct supplied source history**. These are assisted motor/
prediction diagnostics, not autonomous answers. Do not select thresholds or a
checkpoint from their appearance.

Bind code, source pixels, original case identities, codec, package versions,
initialization and this protocol before GPU execution. Use the shared GPU0
mutex, full-reader admission (48 GiB entry/24 GiB runtime host, 20 GiB free GPU,
22 GiB maximum allocation), at most 8 GiB process RSS, 48 GiB disk reserve,
two CPU threads, 900 seconds including at most 180 pressure wait. Preserve a
failure without retry. A pass only permits separately registering a complete
training/evaluation comparison; it does not authorize an unregistered full fit.
That registration must retain the original complete QA/book, erased-context,
actual emitted-pixel feedback, stopping, confirmation and matched-quality cost
criteria. A source-only NLL decrease cannot qualify this ILM.
