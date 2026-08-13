# Conditional Visual Field Flow V31: Research Direction

Date: 2026-08-13

Status: design analysis completed before V31 protocol, implementation, smoke
training, or development measurement

## Research Question

Can a compact image-only causal student learn a **conditional distribution**
over the next written image, rather than one average field or a fixed set of
global prototypes, and can samples from that distribution bind earlier visual
history to the correct arbitrary next glyph?

V31 is deliberately narrower than the full ILM claim. It tests the missing
probabilistic mechanism on the existing Chinese image-stream benchmark before
more data, parameters, page geometry, instruction tuning, or a pixel writer
can obscure the cause of success or failure.

## What V28-V30 Actually Ruled Out

The three most recent controlled experiments constrain the next design:

| Experiment | Learned future object | Natural top-1 | Exact-suffix pair assignment | Conclusion |
|---|---|---:|---:|---|
| V28 | four global continuous prototypes | `1.4160%` | `49.5605%` | fixed prototypes did not bind context to alternatives |
| V29 | candidate-conditioned scalar density ratio | `2.3438%` | `49.7314%` | the critic detected order but not the correct candidate |
| V30 spatial | one deterministic `4x4x192` field | `1.2695%` | `50.0488%` | aligned local prediction collapsed conditional alternatives |
| V30 global | one deterministic tiled field | `2.4902%` | `50.5859%` | the stronger control still remained at chance on binding |

V30 also measured a `96.3379%` bidirectional cross-font identity rate for the
frozen aligned retinal field. Perception is therefore sufficiently visible for
this benchmark. The unresolved failure is conditional distribution learning:
the same exact four-image suffix must support different next glyphs when the
earlier image history differs.

Simply increasing V28's number of heads is not a new hypothesis. A finite set
of context-only prototypes can collapse to similar directions, and independent
per-patch mixtures can combine the top-left patch of one glyph with the
bottom-right patch of another. V31 instead makes one random draw select one
coherent full-field future.

## Model-Facing Object

The reader still consumes only ordered writing images:

\[
X\in[0,1]^{B\times T\times1\times32\times32},\qquad 1\leq T\leq64.
\]

The target is a continuous field extracted from an arbitrary next image:

\[
Y\in\mathbb R^{B\times16\times192}.
\]

The spatial arm uses 16 channel-normalized cells from the frozen `4x4`
retinal field. The matched global control tiles the frozen 192-dimensional
semantic vector over those same 16 rows. No string, token, Unicode scalar,
character ID, OCR result, vocabulary embedding, or visual codebook enters the
student.

## Why Conditional Flow Matching

Flow Matching trains a continuous normalizing flow by regressing a vector
field on fixed conditional probability paths. Its conditional optimal-
transport path is simulation-free during basic training and has straight
per-example trajectories, making it a practical fit for one RTX 4090
([Lipman et al., 2023](https://arxiv.org/abs/2210.02747)). Conditional flow
matching has also been generalized with minibatch optimal transport
([Tong et al., 2024](https://arxiv.org/abs/2302.00482)), while recent work
shows that continuous visual autoregression need not quantize its output
([Shao et al., 2025](https://proceedings.mlr.press/v267/shao25b.html)).

For context state `h`, normalized target field `Y`, and base field `E`, V31
uses the straight interpolant

\[
Z_t=(1-t)E+tY,\qquad U^*=Y-E,\qquad t\sim U(0,1),
\]

and learns

\[
L_{CFM}=\mathbb E\left[
  \frac1{16}\sum_p\lVert v_\theta(h,Z_t,t)_p-U^*_p\rVert_2^2
\right].
\]

The optimum is a time-dependent conditional vector field. Unlike V30's single
endpoint, it can route different base draws toward different plausible next
fields. Generation integrates

\[
\frac{dZ_t}{dt}=v_\theta(h,Z_t,t),\qquad Z_0=E,
\]

and normalizes each final retinal cell. The generated field exists without a
candidate bank and is therefore a genuine continuous output object.

## One Choice Variable, One Coherent Glyph

Independent noise at all `16x192` coordinates gives maximum entropy but makes
patchwise incoherence easy. V31 instead draws

\[
e\sim\mathcal N(0,I_{192}),\qquad
E=(e,e,\ldots,e)\in\mathbb R^{16\times192},
\]

with per-vector normalization. The 192-dimensional base draw is a global
choice variable. Learned positions and local convolution unfold that choice
into a spatial field. This is a low-dimensional conditional manifold, not a
discrete glyph class and not a vocabulary index.

The same base law is used in both arms. In the global control, both target and
base are tiled, so permuting candidate rows is exactly a no-op. In the spatial
arm, the target rows differ, so the same intervention destroys local
correspondence. This preserves the clean local-versus-global causal comparison
from V30 while adding multimodality.

The 192-dimensional bottleneck is intentional. Written glyph variation on the
fixed 32-pixel benchmark is far below the ambient 3,072-dimensional field; a
small coherent latent also keeps generation cheap. V31 does not claim that 192
dimensions suffice for unrestricted pages or handwriting.

## Candidate Scoring Without A Vocabulary

### Why energy distance is not the primary ranker

An empirical energy score is a strictly proper score for a forecast
distribution. It is useful for distribution training and evaluation, and
continuous visual autoregression has demonstrated this route. However, for a
fixed predicted sample set, its sample-diversity term is constant across
candidate `y`. Candidate ranking then reduces to average distance from `y` to
the samples, which can prefer a geometric center between modes rather than a
plausible glyph mode.

Properness also does not guarantee useful discrimination at finite sample
sizes; this limitation has been measured systematically for multivariate
forecasts
([Marcotte et al., 2023](https://proceedings.mlr.press/v202/marcotte23a.html)).
V31 therefore keeps generation metrics and controlled interventions separate
from its primary arbitrary-candidate ranker.

### Conditional path score

Diffusion models can be used as generative classifiers by comparing
conditional denoising errors
([Li et al., 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.html)).
The analogous V31 score evaluates whether the learned conditional flow explains
a candidate's low-to-mid-noise paths:

\[
s(h,y)=-\frac1M\sum_{m=1}^{M}\frac1{16}\sum_p
\left\lVert
v_\theta(h,(1-t_m)E_m+t_m y,t_m)_p-(y_p-E_{m,p})
\right\rVert_2^2.
\]

All candidate columns share the same fixed probe noises and times. The score
accepts any image that the frozen retina accepts; it has no vocabulary-sized
parameter or persistent image bank. Low and intermediate times limit direct
candidate leakage and make the context responsible for explaining the
destination.

This score is not treated as exact log likelihood. Conditional diffusion
likelihood can be unexpectedly insensitive to its condition
([Cross and Ragni, 2024](https://proceedings.mlr.press/v255/cross24a.html)),
and first-order score matching is not generally identical to maximum-
likelihood optimization
([Lu et al., 2022](https://proceedings.mlr.press/v162/lu22f.html)). V31 therefore
trains and gates direct condition interventions: exact-suffix assignment,
suffix-preserving prefix shuffle, and matched spatial/global routes.

### Generated-sample score

The evaluator separately draws eight full-field samples with an eight-step
Heun solver. It ranks an arbitrary candidate by a fixed radial-kernel sample
score

\[
s_{sample}(h,y)=\frac1K\sum_k
\exp\left(\kappa\frac1{16}\sum_p
\hat Y_{k,p}^{\top}y_p\right).
\]

This asks whether actual candidate-independent generations place mass near the
correct visual form. It cannot pass merely because a candidate-conditioned
path is easy to denoise. Kernel and path rankings, sample diversity, and
condition sensitivity are all reported rather than collapsed into one number.

## Compact Velocity Field

V31 retains the V30 global-control retina, frozen semantic adapters, context
projection, eight causal blocks, and final normalization. The rejected V30
field decoder and scalar temperature are discarded. Both V31 arms load this
same backbone and initialize the same new velocity decoder byte-for-byte.

The decoder receives the noisy `4x4x192` field, learned retinal positions, a
sinusoidal time embedding, and the final 384-dimensional causal state. Four
small spatial residual blocks use depthwise `3x3` mixing and condition-
modulated channel MLPs. It outputs an unconstrained velocity field of the same
shape. This keeps the complete student below 20 million parameters; generation
requires only eight or sixteen small decoder evaluations after one causal
context pass.

Self-conditioning, learned schedules, reflow, consistency distillation, and
larger U-Nets are intentionally excluded from V31. Recent continuous language
work reports gains from information-aware schedules and self-conditioning
([Chen et al., 2026](https://arxiv.org/abs/2604.11748)), and consistency models
can later reduce sampling to one or a few steps
([Song et al., 2023](https://arxiv.org/abs/2303.01469)). Those are follow-up
optimizations only if the basic conditional visual field first demonstrates
binding.

## What Would Count As Evidence

V31 is useful only if all of these are true on the fixed development data:

1. the path score binds different earlier histories to different candidates
   despite pixel-identical four-image suffixes;
2. actual candidate-independent flow samples recover the correct next-image
   neighborhood above fixed controls;
3. aligned local fields outperform a parameter-identical tiled-global control;
4. permuting retinal positions hurts only the spatial route;
5. arbitrary candidate-column permutation is exactly equivariant;
6. the checkpoint and inference path remain image-only and bank-free; and
7. the gains beat image frequency and symbolic bigram controls, rather than
   only reconstruction or visual identity probes.

Passing visual reconstruction, cross-font identity, low memory, smooth loss,
or glyph-like samples is not language evidence by itself.

## Failure Interpretations

- **Flow loss falls, path assignment stays at chance:** the denoiser ignores
  context and reconstructs candidate paths from leakage.
- **Path score passes, generated samples fail:** the candidate-conditioned
  surrogate is discriminative but the autonomous flow is not a writer state.
- **Both arms pass equally:** multimodal global semantics help, but local
  retinal topology is not the cause.
- **Spatial arm passes only before patch permutation:** aligned local structure
  is causally involved.
- **Samples collapse:** the coherent latent is ignored; a later design needs
  stronger distributional or mutual-information training.
- **Samples are diverse but unrelated to context:** the flow models visual
  variation without language.

Every outcome is publishable. A failure keeps the frozen partition sealed and
does not authorize a pixel writer.

## Scope

V31 is a mechanism experiment for one-step Chinese visual continuation. It
does not establish a general ILM, human-like reading, etymology knowledge,
instruction following, page-scale generation, Qwen parity, or efficiency
superiority over token language models. Its purpose is to decide whether a
coherent continuous conditional flow is a viable next-writing primitive before
the project spends a larger training budget.
