# Next learning decision after the native writer controls

This is a research decision record, not a successful model result or an
executed training protocol. The goal remains independent image-question to
image-answer language generation, including forms outside a language font
inventory. None of the component controls qualifies that goal.

## What the existing evidence already establishes

The [reference-removal control](native_reference_latent_ablation_result.md)
does not repair the native DeepGen writing path. Source writing changes from
1/4 to 0/4 exact pages, and adapted writing remains 0/4, despite the supplied
correct text. Reader inputs are preserved. More unchanged reader-query fitting
or another unmotivated sampling change is not the next learning experiment.

Several attractive alternatives have already been tested:

| Existing observation | Consequence for a new proposal |
| --- | --- |
| [V54/V55 exact raster density](pixel_language_research_refresh_20261105.md) models pixel dependence with row attention and a masked within-row likelihood | Do not describe the earlier density as independent pixels. Its selected language positions receive complete image NLL, not isolated sampled pixel labels. |
| [V68](codec_language_v68_training_result.md) trains a 7.539B actor with a faithful image codec and 9.352M contextual/query/head parameters on 12,976 examples | A pretrained contextual body plus a faithful continuous codec is already tested; neither that combination nor reconstruction alone is a new solution. |
| [V68 exact-history diagnosis](codec_language_v68_diagnosis_result.md) still fails first blocks and correct-history writing | Generated-history drift is not the only unresolved issue. |
| [V69](conditional_readout_v69_result.md) fits calibrated shared/joint ridge heads on 512 actual contextual states | Linear readout and centering have already been investigated. More overlap is not precise writing. These selected regularized heads are not an interpolation-capacity bound. |
| [V70](native_visual_baseline_v70_result.md) obtains 13/16 exact Chinese image answers through the authenticated source's native text-output interface | Source visual-language capability exists on these cases. The source text decoder is an offline diagnostic, not ILM inference. |
| [V81](adaptive_readout_v81_result.md) improves first blocks but still gives no exact complete training answers | A larger adaptive readout alone is already tested and does not qualify the full path. |

The retained V68 reader is Qwen3-VL 8B from the authenticated Q4 archive,
not the smaller MIXAR backbone. Its implementation passes DeepStack visual
features and four-plane rotary positions. Do not diagnose omitted DeepStack
or substitute a smaller reader as a repair without a controlled test.
Native-source geometry, state alignment and later decoder-metric experiments
also remain part of the record and must be read before registering another
adaptation of those same interfaces.

## A remaining candidate: native semantic context with a direct image density

Investigate a normalized, autoregressive distribution over physical image
pieces conditioned on the existing native semantic context. This combines
the source's demonstrated reading capability with a direct image likelihood;
it avoids requiring a squared-error head to choose an arbitrary continuous
codec coordinate before drawing the image. The combination remains untested.
It does not establish that the earlier regression objective uniquely caused
failure, or that another conditional density will succeed.

One simple representation is a lossless packing of a binary 32-by-64 physical
block into 256 adjacent 2-by-4 pixel pieces. Each piece has 256 possible binary
patterns. Let b(t,j) be its eight-bit pattern. A causal contextual state h(t)
depends only on the prompt and completed image blocks. Train

\[
  -\sum_{t,j}\log p_\theta\bigl(b(t,j)\mid h(t),b(t,<j)\bigr).
\]

The within-block decoder receives preceding physical patterns under teacher
forcing and its own patterns in generation. The slow contextual body reads
completed output images before predicting the next block. Include a learned
structural stop decision. No current/future answer image may condition its
own contextual query. Packing and unpacking must be exact inverses; the
unpacking operation contains no glyph inventory, font or answer lookup.

These are discrete pixel-pattern outcomes, not word or character identities.
This must not be described as eliminating every discrete internal variable.
All binary block images are representable, including previously unseen forms;
that representational fact does not establish semantic understanding of them.
The initial binary resolution also does not qualify arbitrary handwriting,
historical scripts or grayscale reproduction.

An exact discrete likelihood can preserve multiple sharp alternatives where
squared-error prediction gives a conditional mean. V54 already supplies a
correlated exact image likelihood, however. The proposed test is the specific
combination with native semantic context and a different within-image
factorization, not a claim to invent autoregressive pixel modeling.
[MIXAR v2](https://arxiv.org/html/2604.11575v2) also makes free-running generation
central and uses an additional rollout-training stage. Its reported success
does not qualify this proposed head or imply comparable local pretraining.
[Glyph-ByT5-v2](https://arxiv.org/abs/2406.10208) provides related evidence for
dedicated glyph/text alignment in writing models; its lexical conditioning
would not be admitted as the final independent ILM interface.

## What must happen before a larger run

First inspect the source-bound V69 cache and the later native-context records.
Determine whether cached input-only query states and original first-block
pixels permit a bounded, matched head-learning comparison without another
backbone load. Preparation must bind the exact image geometry, context source,
feature statistics, training/calibration/test split and every prior control.
Any feature normalization must be fitted on training states only.

Implement and register the full likelihood, true pixel-prefix causality,
cached/uncached generation equality and a real update/replay check. Use all
pixels of each selected target block. A head test must compare complete
autonomous first blocks, supplied-prefix likelihood and erased-input controls,
including held-out question pairs. It is not a complete-answer demonstration,
and it must not be reported as a new end-to-end model until the PNG-input
runtime actually invokes the contextual body and generates full answers.

Only after that control is qualified should the full actor train on complete
answer streams and receive a fresh autonomous answer/dependence assessment.
Do not infer success from code execution, likelihood reduction, a few good
images, supplied targets or first-block fidelity. Compare the new proposal
against the measured failures rather than asserting that a new head alone
solves them. Memory, inference cost and any efficiency advantage require
measurement at useful language quality on the actual single-4090 runtime.

The current manuscript already records the earlier complete native-flow
failure. The query-fit, supplied-content and reference-removal results now
require one bounded manuscript/figure revision with preserved baselines,
matching redlines and actual PDF review. Their result notes and original
evidence remain the authoritative records until that revision is built.
