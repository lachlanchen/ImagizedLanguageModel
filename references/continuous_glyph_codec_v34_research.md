# Continuous Glyph Representation Codec V34

Date: 2026-08-14

## Problem Exposed by V33.1

V33/V33.1 proved that a pretrained causal transformer can be connected directly
to `32 x 32` writing patches and trained cheaply on one RTX 4090. The route is
not yet reliable enough for language training. After 8,000 adapter-only updates,
the raw model reached ink F1 `0.8919` and edge F1 `0.8807`, but only `0.6553` of
the OCR accuracy available on untouched targets. The output is readable at line
scale while small stroke substitutions can change character identity.

The observed failure is consistent with the interface geometry. V33 compresses
1,024 binary pixel values into 768 hidden coordinates with one linear map,
passes them through a frozen contextual transform, and asks another linear map
to recover all 1,024 pixels. A linear decoder has no explicit locality,
multiscale stroke model, or mechanism that prefers coherent character
components over plausible pixel averages.

## Relevant External Evidence

### Pixel language models

- [PIXAR](https://arxiv.org/abs/2401.03321) demonstrates autoregressive
  rendered-text generation without a conventional text tokenizer. Its released
  checkpoint supplies the V33 transformer initialization, but its exact small
  patch renderer is brittle when transferred to `32 x 32` CJK patches.
- [MIXAR](https://arxiv.org/abs/2604.11575) scales pixel autoregression across
  scripts and reports that `32 x 32` patches are important for CJK. It supports
  the patch scale selected here, but its April 2026 code and weights were not
  publicly available during this audit.

### Continuous visual latents

- [Autoregressive Image Generation without Vector
  Quantization](https://arxiv.org/abs/2406.11838) establishes that an
  autoregressive model can predict continuous visual representations by
  learning a per-position continuous distribution; vector quantization is not
  required. Its diffusion loss is a later option for genuinely multimodal
  glyph futures, not a prerequisite for codec reconstruction.
- [Diffusion Transformers with Representation
  Autoencoders](https://arxiv.org/abs/2510.11690) shows that a strong continuous
  representation encoder paired with a trained decoder can preserve visual
  detail and improve downstream generative learning. Its official
  [RAE implementation](https://github.com/bytetriper/RAE) also uses latent
  normalization and decoder noise augmentation. Natural-image RAEs are much
  larger than this task requires, but the separation of representation and
  rendering is directly relevant.
- [GlyphMastero](https://arxiv.org/abs/2505.04915) argues that text-image
  generation needs hierarchical stroke, interaction, and whole-character
  structure. V34 uses a small multiscale convolutional codec to represent those
  spatial regularities without importing an OCR encoder.

These works support a continuous visual interface. None by itself proves the
project's stronger language claim, and V34 must not be presented as language
understanding.

## Selected Model

Let a binary raster patch be

\[
x \in \{0,1\}^{1\times32\times32}.
\]

A deterministic convolutional encoder maps the patch into a 768-dimensional
continuous representation:

\[
z = \operatorname{LN}(E_\phi(x)) \in \mathbb{R}^{768}.
\]

`LN` is non-affine per-sample layer normalization. There is no quantizer,
nearest-neighbor assignment, codebook, vocabulary, token ID, Unicode value, or
character classifier.

A spatial decoder produces direct pixel logits:

\[
\hat{x} = \sigma(D_\psi(z + \sigma_n\epsilon)),
\qquad \epsilon\sim\mathcal{N}(0,I).
\]

Small latent noise during training makes the decoder tolerant of imperfect
future states predicted by a language transformer. At clean evaluation,
`sigma_n = 0`.

### Encoder

The encoder uses a `3 x 3` stem and four residual scales:

```text
1x32x32
 -> 32x32x32
 -> 64x16x16
 -> 96x8x8
 -> 192x4x4
 -> flatten 3072
 -> linear 768
 -> non-affine layer normalization
```

Each scale has a GroupNorm-SiLU residual block. Downsampling uses `4 x 4`,
stride-2 convolutions.

### Decoder

The decoder mirrors the hierarchy:

```text
continuous 768
 -> linear 3072
 -> 192x4x4
 -> 96x8x8
 -> 64x16x16
 -> 32x32x32
 -> 1x32x32 pixel logits
```

Upsampling uses `4 x 4`, stride-2 transposed convolutions. This spatial
inductive bias is the single intended architectural change relative to the V33
linear visual interface.

## Training Objective

For target `x`, define a boundary-weighted pixel loss, Sobel loss, and ink Dice
loss:

\[
\mathcal{L}_{\mathrm{codec}}
= \mathcal{L}_{\mathrm{BCE,boundary}}
+ 0.5\mathcal{L}_{\mathrm{Sobel}}
+ 0.5\mathcal{L}_{\mathrm{Dice,ink}}.
\]

Boundary weighting gives stroke edges three times the base BCE weight. Dice
prevents the white background from dominating. No perceptual network or OCR
loss enters training.

## Data Geometry

V34 combines two visual streams:

1. arbitrary `32 x 32` chunks from rendered public-domain Chinese lines,
   retaining random subpatch horizontal offsets and train/dev font separation;
2. isolated historical forms from the existing Hanziyuan-derived research
   corpus under `/home/lachlan/ProjectsLFS/incoder/data/historic/glyphs`.

The historical SQLite index currently contains 84,642 records, of which 84,626
are SVG glyphs: 32,216 oracle, 23,500 bronze, 6,996 seal, and 21,923
Liushutong records, plus seven unknown-stage rows. The files occupy about 434 MB
and are reused in place under the workstation storage policy.

Historical train/development/sealed assignment is by modern-character key, not
individual glyph file. All forms associated with a held-out modern character
therefore remain outside training. The corpus is a local research dependency;
its images are not redistributed with model source.

## Future Language Interface

If and only if the codec passes its reconstruction gate, the next causal model
will use:

\[
h_t = T_\theta(A_{in}z_{\le t}), \qquad
\hat z_{t+1}=A_{out}h_t, \qquad
\hat x_{t+1}=D_\psi(\hat z_{t+1}).
\]

Generated patches are fed back through the same encoder:

\[
z_{t+1}=\operatorname{LN}(E_\phi(mathbf{1}[hat x_{t+1}\ge0.5])).
\]

This keeps deployed input and output as images while allowing the causal core
to learn in a smooth continuous visual space. A small conditional diffusion or
flow head is reserved for a later ablation only if deterministic latent
regression produces averaged or unstable glyphs.

## Falsification Logic

V34 answers only whether the visual manifold can preserve unseen writing forms
at low compute. If it cannot reconstruct unseen fonts and held-out historical
character families with high stroke fidelity, it is not a sound language
interface and no causal run should follow. If it passes, semantic learning must
still be demonstrated independently with autonomous prompt, shuffle, blank,
paraphrase, and counterfactual controls.
