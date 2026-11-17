# Generate a global visual plan, then a lossless pixel correction

Status: implemented architecture with six passing CPU numerical checks;
**untrained and unqualified for language**. No source preparation, neural trial
or full language fit is registered by this note.

## Measured problem and bounded change

The [whole-form auxiliary trial](global_form_preflight_result_20261118.md)
improves its temporary classifier to 79/552 but leaves actual drawing at F1
.455463432, zero exact images and four false-ink spaces. Even its 4/5 correctly
classified short-answer positions produce no exact masks. Classification is a
different computation from the physical generator. This does not prove a
unique cause, but gives no basis for another unchanged auxiliary fit.

Change the output factorization: a generated global visual plan directly
controls a learned drawing function. A normalized physical correction retains
support for every binary image. This is a composite writer architecture; its
effect cannot be attributed separately to quantization, drawing or correction
without further controlled comparisons. Keep the reader, contextual source,
emitted-pixel feedback contract and stopping criteria separate.

## Computation and normalized image law

Let Y be a complete 32-by-32 binary glyph image and H its pre-target image
history state. A convolutional encoder produces 64 hard bits B=E(Y). A
convolutional decoder draws a binary base image D(B). Neither has a per-character
embedding, class-to-image table or nearest-image lookup. Bit positions are
measurements learned from pixels. Quantization is explicit: this candidate is
not wholly continuous or free of discrete variables. Its binary states are
not assumed to encode meanings.

Define R=Y XOR D(B). A causal prior generates B as eight eight-bit patterns,
each ranging over all 256 physical bit combinations. A conditional occupancy
tree generates R using H and the actually generated B:

\[
 B\sim p_\theta(B\mid H),\qquad
 R\sim p_\psi(R\mid H,B),\qquad
 Y=D(B)\mathbin{\mathrm{XOR}}R.
\]

Inference never invokes the encoder on a future target. The correction may be
empty and its topology comes from its own previous decisions. Different plans
can produce the same image; the image law is the marginal

\[
 P(Y\mid H)=\sum_B p_\theta(B\mid H)
 p_\psi(Y\mathbin{\mathrm{XOR}}D(B)\mid H,B).
\]

For each B, XOR with D(B) is a bijection on all 2^1024 binary images. Thus
the sum over Y is one when both component laws are normalized. Finite real
logits with positive legal branch probabilities retain full image support;
machine-precision probabilities and actual sampling still need qualification.
The correction prevents a finite plan from closing the output alphabet.
Sixty-four bits alone cannot losslessly represent every 1,024-bit image. No
universal compression or information-creation claim follows.

After qualifying and freezing the codec, supervised training uses the fixed
posterior branch B=E(Y):

\[
 \mathcal L_{\rm branch}=-\log p_\theta(E(Y)\mid H)
 -\log p_\psi(Y\mathbin{\mathrm{XOR}}D(E(Y))\mid H,E(Y)).
\]

It upper-bounds marginal image NLL. Its gap is -log P(B=E(Y)|Y,H), not zero
in general. Do not report it as exact marginalized image likelihood. Target-
derived plans are training supervision, never inference input. The codec's
straight-through sign estimator is a biased reconstruction-training gradient;
the later fixed-plan density objective needs no derivative through that sign.

The intended benefit is fewer global choices followed by sparse corrections.
Eight plan events are an architectural count. Eight plus **measured** correction
events must improve on the old source tree's 97.302 average before claiming
fewer serial events. These are not entropy bits, GPU throughput or whole-model
efficiency. Wrong plans can make corrections dense, and unused binary states
can decode poorly. Readable irrelevant forms would still fail language.

## Prior work and completed controls

[Finite Scalar Quantization](https://arxiv.org/abs/2309.15505) uses a product of
fixed scalar levels instead of learned vector quantization. It is precedent
for implicit discrete visual states, not evidence for this global glyph plan.
[MAGVIT-v2](https://arxiv.org/abs/2310.05737) studies learned visual tokenization
for image/video generation; its results do not qualify visual language or this
runtime boundary. [L3C](https://arxiv.org/abs/1811.12817) models learned auxiliary
representations jointly with images for lossless compression. These are
established mechanisms, not a new theorem or physics breakthrough.
[PIXAR](https://arxiv.org/abs/2401.03321) already studies autoregressive language
generation in pixel space; no first-pixel-language-model claim is made.

The [continuous codec](continuous_glyph_codec_v34_result.md) reconstructs supplied
writing accurately. Do not repeat that result or call it language. Its
768-dimensional continuous latent and later flow/metric predictors have not
qualified conditional language. This candidate generates a short joint binary
plan and exact physical correction; it does not add another coordinate
regression. Quantization can still lose useful geometry and must be measured.

The [spatial-plan writer](hierarchical_raster_language_v50_spatial_result.md)
and its bounded residual follow-up lose identity while refining an uncertain
DCT field. Here the global latent is explicitly generated before correction
and has a joint training law. The failed
[prior/guidance](visual_posterior_language_v51_result.md) and
[conditional diffusion](causal_visual_block_diffusion_v53_result.md) results
remain protected; this adds neither score guidance nor a denoising schedule.
The [old exact-pixel head](pixel_density_long_fit_result.md) already fits all
512 training blocks while giving zero exact development blocks. A clearer
motor cannot replace that unresolved generalization problem. No old control
needs to be re-executed.

## Qualification before a new fit

The [implementation](../ilm/visual_lm/binary_plan_glyph_density.py) has 6,536,211
writer parameters: 2,030,465 in the codec, 2,232,256 in the plan prior and
2,273,490 in the correction. This excludes the contextual actor and proves
neither a small language foundation nor efficiency. Six CPU checks exercise
code packing, exact corrections at every pixel, hard values, actual codec
update/reset, causal/cached plan logits, joint/context gradients, target-free
generation and reproducible stochastic sampling. Numerical fixtures do not
qualify source-image fidelity or language. A first check execution's incorrect
generator argument is preserved; the established physical generator is unchanged.

Next register source-only preparation and a codec comparison on the existing
complete training-only physical support. Group identical masks together and
reserve a deterministic mask-hash partition for physical reconstruction only.
Preserve source bytes; do not open reserved language confirmation or repeat
source-alias audits. Compare fresh continuous and hard-binary 64-dimensional
codecs with identical tensors, data order, optimizer and budget, changing only
quantization. Fix reconstruction, empty-image, correction-event and resource
thresholds before fitting. XOR exactness alone is not a measured quality gain.

If the binary plan loses fidelity or does not reduce refinement work, reject
the route before a language fit. If it qualifies, separately register the
conditional prior-plus-correction comparison against the unchanged physical
tree. Require actual generated images, not supplied-plan reconstruction.
Retain all complete-language, original-success, stopping, emitted-history,
untouched-source and whole-model resource criteria. No full fit, deployment,
autonomous score or goal completion follows from this implementation.
