# Causal Glyph Flow V35 Research Note

Date: 2026-08-14

Status: design evidence collected before V35 implementation or training

## Question

V34 established that a 7.4M-parameter continuous codec can preserve modern and
historic Chinese glyph rasters without a vocabulary, character IDs, Unicode
IDs, OCR, or vector quantization. It did not establish language modeling. V35
asks the next irreducible question:

> Can a causal model consume only rendered patch images, use continuous visual
> states internally, and autonomously write a prompt-dependent Chinese answer
> as an image on one RTX 4090?

This is deliberately narrower than claiming parity with a general 8B LLM. A
positive result must demonstrate closed-loop visual generation and prompt
dependence on held-out records. Reconstruction, retrieval, teacher forcing, or
an OCR-only score cannot answer the question by themselves.

## What External Work Already Establishes

V35 may use external work when it is the strongest available engineering
foundation. Origin and license are recorded; outside results are not relabeled
as this project's contribution.

### PIXAR

[PIXAR](https://arxiv.org/abs/2401.03321) established autoregressive language
modeling over rendered binary patches. Its 113M decoder uses a 12-layer,
768-wide causal transformer, but predicts pixels independently with a sigmoid
head. The paper reports that maximum likelihood produces noisy text and uses a
short adversarial stage to recover readability. Its released checkpoint gives
V35 a causal visual prior instead of requiring million-step pretraining on one
workstation.

### MIXAR

[MIXAR](https://arxiv.org/abs/2604.11575) extends PIXAR to eight languages and
shows that 32 by 32 patches are necessary to preserve CJK stroke detail. It
also reports three constraints relevant here:

1. its 116M and 477M models still use conditionally independent Bernoulli
   pixels;
2. the 32 by 32 model needs more capacity and remains substantially weaker on
   CJK than token baselines; and
3. the authors identify diffusion-based output modeling as future work because
   the GAN objective has diminishing returns.

MIXAR used 138B rendered patches and many V100 GPUs. V35 therefore does not
pretend that a small local corpus can recreate MIXAR pretraining. It tests
whether a qualified visual representation plus transferred causal structure
can make a useful, auditable single-GPU proof much cheaper.

### MAR and representation autoencoders

[MAR](https://arxiv.org/abs/2406.11838) shows that autoregressive models do not
require categorical visual tokens: a small conditional diffusion model can
represent the distribution of each continuous latent. Its
[official implementation](https://github.com/LTH14/mar) supplies the reference
AdaLN-conditioned density-head design.

[Representation Autoencoders](https://arxiv.org/abs/2510.11690) show why the
quality and geometry of the continuous representation matter to latent
generation. V34 supplies the reconstructive half of that requirement for
writing. It is not yet a semantic encoder, so V35 measures semantic structure
in causal hidden states rather than assuming that the local glyph latent is
already semantic.

## Failed Local Alternatives

- V32 joined a multilingual visual reader to a weak 32-dimensional answer
  codec. Its target reconstruction ceiling was only about 13.5%, so semantic
  generation could not be interpreted.
- V33 and V33.1 resized PIXAR's linear projections to 32 by 32. After 8,000
  calibration updates, raw reconstruction reached only 0.892 ink F1, 0.881
  edge F1, and 0.638 OCR retention. The run correctly stopped before causal
  claims.
- V34 replaced the linear interface with a continuous convolutional codec and
  passed clean, noisy, historic, held-font, and sealed reconstruction gates.

The failure sequence isolates the remaining uncertainty: causal language
learning and continuous next-patch generation.

## Alignment Probe

Before preregistration, an uncommitted diagnostic fitted a residual two-layer
map from frozen V34 EMA latents to the frozen resized PIXAR input projection.
It used public-domain training rasters only and did not save weights.

| Measurement | Value |
|---|---:|
| optimization updates | 500 |
| development cosine before alignment | 0.0097 |
| development cosine after alignment | 0.9026 |
| development MSE after alignment | 0.0391 |
| GPU | RTX 4090, CUDA device 0 |

This supports a short, explicit visual-interface stage. It does not support a
semantic claim and is not counted as V35 evidence.

## Selected Model

V35 is a causal continuous visual model with four components:

1. **Retina/actuator:** the frozen V34 EMA encoder and decoder, included in the
   final checkpoint.
2. **Interface adapter:** a residual MLP maps V34 latents into the released
   PIXAR transformer's input coordinate system. The PIXAR pixel projection is
   an offline alignment teacher and is removed from runtime.
3. **Causal field:** the PIXAR-initialized 12-layer Llama transformer operates
   on a sequence of continuous visual states. There is no embedding table.
4. **Writer:** a deterministic latent anchor and a conditional rectified-flow
   head predict the next V34 latent; the frozen V34 decoder emits the raster.

For raster patch \(p_t\), frozen codec \(E,D\), adapter \(A\), and causal field
\(T\):

\[
z_t = E(p_t),\qquad a_t=A(z_t),\qquad
h_t=T(a_{\le t}).
\]

The fast deterministic writer predicts

\[
\hat z_{t+1}=\operatorname{LN}(M(h_t)).
\]

The density writer uses conditional flow matching. For target \(z\), Gaussian
base \(\epsilon\), and \(\tau\sim U(0,1)\):

\[
x_\tau=(1-\tau)\epsilon+\tau z,\qquad
v^\star=z-\epsilon,
\]

\[
\mathcal L_{flow}=
\mathbb E\left[\lVert F_\theta(x_\tau,\tau,h_t)-v^\star\rVert_2^2\right].
\]

The anchor prevents the density head from being the only learning signal and
provides a one-pass inference route. The flow head tests whether a continuous
conditional distribution improves over a conditional mean.

## Closed Visual Loop

Generation must not feed an invisible latent side channel back to the causal
model. Every predicted latent is decoded, thresholded into the emitted binary
raster, and re-encoded before it becomes the next context state:

\[
\tilde p_{t+1}=\mathbf 1[\sigma(D(\hat z_{t+1}))\ge 0.5],
\qquad z^{feedback}_{t+1}=E(\tilde p_{t+1}).
\]

This makes the visible image the recurrent state boundary. A generated raster
can be inspected directly, and no unrendered symbol can be carried between
steps.

## Efficient Training Strategy

- Freeze the qualified codec throughout V35.
- Align the small input adapter while the causal field is frozen.
- Freeze the aligned adapter, then train the causal field and writer.
- Apply the inexpensive anchor loss to every active next-patch position.
- Subsample valid positions for the heavier flow and decoded-visual losses.
- Use BF16, fused AdamW, gradient accumulation, EMA, and a context of at most 96
  patches.
- Mix public continuation, deterministic visual-copy instructions, and short
  Chinese Alpaca instructions. Copy data proves visual binding; Alpaca remains
  the semantic test.

The final evidence checkpoint contains the codec, adapter, causal field, and
writer. It calls no external model at runtime. PIXAR and any local LLM are
permitted only as immutable initialization or offline data-preparation tools.

## Claims This Experiment Can and Cannot Support

A successful V35 can support:

- causal generation in a continuous, non-quantized visual space;
- a closed raster-input/raster-output feedback loop;
- held-out visual copying and continuation;
- prompt-dependent short Chinese raster answers if the semantic gate passes;
- practical training and inference on one RTX 4090.

It cannot by itself support:

- parity with Qwen 8B or any general assistant;
- broad factual reliability;
- learning from raw books without preprocessing;
- unrestricted historic-glyph generation; or
- a claim that token models are universally less compute-efficient.

Those become scale-up questions only after the causal mechanism passes.

