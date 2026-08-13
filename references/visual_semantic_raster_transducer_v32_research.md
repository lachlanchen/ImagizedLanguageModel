# Visual Semantic Raster Transducer V32: Research Direction

Date: 2026-08-13

Status: research decision completed before V32 implementation or measurement

## Decision

V32 will stop trying to prove language by ranking the next isolated glyph.
It will train the first bounded prompt-image to answer-image system in this
repository:

```text
rendered Chinese prompt -> visual reader -> causal continuous answer plan
                        -> generated glyph rasters -> answer strip image
```

The student receives writing pixels and emits writing pixels. Strings are used
only by the offline renderer and evaluator. No tokenizer, Unicode value, OCR
result, character ID, vocabulary logit, answer lookup, or language-model call
enters a student method.

This is a proof experiment, not a claim of general intelligence. Its decision
question is whether a compact student on one RTX 4090 can learn a real,
prompt-dependent Chinese mapping and autonomously render a short multi-glyph
answer that survives held-out wording and font changes.

## Prior Art Changes The Claim

The broad idea "language from pixels" is established prior art. The project
must build on that work accurately rather than claim it as new.

| Work | Relevant result | What V32 reuses | Remaining gap |
|---|---|---|---|
| [PIXEL](https://arxiv.org/abs/2207.06991) | masked rendered-text reconstruction and cross-script transfer | visual patch encoder and renderer discipline | primarily an encoder; not prompt-to-answer pixel generation |
| [PIXAR](https://arxiv.org/abs/2401.03321) | autoregressive pixel-input/pixel-output text generation | direct binary patch prediction, next-patch baseline, readability audit | expensive pretraining; noisy maximum-likelihood output; limited instruction semantics |
| [PixelGPT](https://aclanthology.org/2024.emnlp-main.182/) | 317M autoregressive pixel-only model trained on more than 400M English document images | causal next-patch baseline and exact public implementation reference | English-centric, 240B-patch scale, generation acknowledged as difficult |
| [PIXEL-M4](https://arxiv.org/abs/2505.21265) | multilingual pixel encoder including Simplified Chinese | Apache-2.0 visual-reader initialization | encoder only; no answer writer |
| [UniModel](https://arxiv.org/abs/2511.16917) | visual-only unified understanding and generation with painted text | confirms that image-only input/output is technically viable | large VAE/MMDiT recipe, qualitative text output, glyph distortion, no one-GPU proof |
| [Parallel Rollout Approximation](https://arxiv.org/abs/2606.27978) | low-dimensional continuous intermediate states and decoded pixel feedback reduce pixel-AR error accumulation | clean-room decoded-feedback mechanism | ImageNet class generation, not visual written-language transduction |

V32 therefore does **not** claim the first pixel language model or the first
visual-only multimodal generator. Its narrower contribution is a reproducible,
single-4090 **written-language raster transducer** that combines:

1. a Chinese-aware pixel reader;
2. an answer planner with no symbolic vocabulary;
3. a continuous, non-quantized glyph state;
4. direct generated raster cells fed back during training and inference; and
5. semantic counterfactual, paraphrase, font, and boundary audits on the same
   autonomous output.

## Reuse Is A Strength When The Boundary Is Explicit

The user explicitly permits existing work that performs well. V32 uses that
permission conservatively.

### Selected external initialization

`Team-PIXEL/pixel-m4` is selected as the default prompt reader because it is:

- trained on English, Hindi, Ukrainian, and Simplified Chinese;
- only about 86.05M encoder parameters;
- released under Apache-2.0;
- exactly representable by the current `transformers.ViTModel`; and
- loadable without vendoring the old PIXEL package.

The locally verified encoder mapping has no missing or unexpected weights. The
checkpoint's `vit.*` state maps exactly to a current 12-layer ViT with width
768, 12 heads, 16-pixel patches, and 530 positional states.

The final V32 checkpoint is still an independent student. Pretraining history
does not make a model dependent on a runtime teacher. The released receipt must
identify the upstream commit and weight hash, and a random-reader control must
measure how much initialization contributes.

### Baselines, not dependencies

PixelGPT is a serious direct-pixel baseline, but its 350,277,632-parameter
public weight file is F32, its pretraining is English-centric, and its model
card has no explicit weight-license field. The GitHub code is MIT. V32 will
cite and, if practical, evaluate PixelGPT; it will not copy its checkpoint into
the student or imply a license that the model card does not state.

The PRA repository has no visible license file at the audited revision. V32
will not copy its source. The implementation will be a clean-room realization
of equations and algorithms published in the paper.

## Why V31 Failed And V32 Is Different

V31 learned a conditional flow over a frozen next-glyph retinal field. It
detected ordered context and local structure, but exact-suffix assignment
remained at chance and the model never emitted pixels. That experiment asked a
hard and indirect question: infer the next arbitrary character in a natural
book stream, represent it in another model's latent space, and then recover it
through an evaluator bank.

V32 changes all three weak links:

- **Task:** a rendered natural-language prompt explicitly specifies a short
  answer instead of relying on weak next-character mutual information.
- **Target:** the target is a sequence of answer rasters, not a frozen semantic
  vector or candidate score.
- **Training interface:** the planner is exposed to decoded, imperfect answer
  prefixes, matching the pixels it will receive during autonomous inference.

The model can still fail. The new experiment simply makes success and failure
about language transduction rather than candidate-bank geometry.

## Native Geometry

The first proof uses two continuous visual scales.

### Prompt strip

A Chinese prompt is rendered into a one-line image:

\[
X\in[0,1]^{B\times3\times16\times(16P)},\qquad P\leq192.
\]

Each 16-by-16 region is a retinal patch. Character placement is jittered, and
some examples receive a non-cell-aligned horizontal offset, so a patch is a
sensor region rather than a character ID. The prompt reader never receives the
source string or its length in characters. A continuous patch-presence mask is
allowed to suppress blank canvas.

### Answer strip

The answer is generated as up to 32 monochrome 24-by-24 raster cells:

\[
Y=(y_1,\ldots,y_A),\qquad
y_i\in[0,1]^{1\times24\times24},\quad A\leq32.
\]

Cells are concatenated into the primary answer image after generation. A cell
can contain a modern character, punctuation, a handwritten mark, or later an
unencoded historical crop. It is never represented by an integer class. The
first proof renders modern Chinese because semantic learning, not archaeology,
is the immediate bottleneck.

## Model

V32 has three learned parts inside one checkpoint.

### 1. Visual prompt reader

Let `E_phi` be the PIXEL-M4-initialized ViT. It maps prompt patches to memory:

\[
M=E_\phi(X)\in\mathbb R^{B\times(P+1)\times768}.
\]

The first development arm freezes `E_phi`; the selected run may unfreeze only
its final two blocks after the writer has stabilized. A random-initialized,
otherwise identical reader is the initialization control.

### 2. Causal visual planner

Previously visible answer cells are embedded by a small convolutional retina.
A six-layer causal transformer with cross-attention to `M` computes

\[
h_i=F_\theta(\bar y_{<i},M),\qquad h_i\in\mathbb R^{512}.
\]

The start state is one learned continuous vector. The end decision is a scalar
Bernoulli head, not a text token. The planner has no embedding table indexed by
characters and no vocabulary-sized output layer.

### 3. Continuous glyph-state writer

A target encoder maps the current raster and causal state to a compact
continuous state:

\[
z_i=G_\psi(y_i,\operatorname{sg}(h_i))\in\mathbb R^{32}.
\]

Layer normalization fixes scale while retaining direction. A causal raster
decoder reconstructs the cell:

\[
\tilde y_i=D_\omega(z_{\leq i})\in[0,1]^{24\times24}.
\]

The planner predicts a diagonal continuous distribution over the next state:

\[
(\mu_i,\log\sigma_i)=Q_\eta(h_i),\qquad
\mathcal L_{state}=
\frac12\sum_k\left[
\frac{(z_{ik}-\mu_{ik})^2}{\sigma_{ik}^2}+2\log\sigma_{ik}
\right].
\]

Inference uses `mu_i` for the deterministic evidence run. Optional sampling is
reported separately. There is no vector quantizer, nearest code, or latent
class table.

## Parallel Decoded Feedback

Teacher forcing is particularly damaging for pixel autoregression: clean
ground-truth cells at training time do not resemble the model's own imperfect
cells at inference. V32 adopts the central PRA insight without copying its
code.

For each target state, sample noise and interpolation strength:

\[
\epsilon_i\sim\mathcal N(0,I),\qquad
\hat z_i=t_i z_i+(1-t_i)\epsilon_i,
\qquad t_i\sim U(0.65,1).
\]

Decode all perturbed states in parallel:

\[
\bar y_i=D_\omega(\hat z_{\leq i}).
\]

The second planner pass consumes `sg(bar_y_<i)` and predicts the clean,
detached target states. This approximates generated-prefix training while
remaining parallel. At inference, the exact same route is sequential:

```text
planner state -> continuous glyph state -> raster cell -> planner retina
```

An ablation uses clean teacher-forced cells. Decoded feedback is useful only if
it improves autonomous sequence accuracy rather than reconstruction alone.

## Loss

The fixed objective is

\[
\mathcal L=
\mathcal L_{state}
+\lambda_{pix}\mathcal L_{BCE}
+\lambda_{edge}\mathcal L_{edge}
+\lambda_{ink}\mathcal L_{Dice}
+\lambda_{stop}\mathcal L_{stop}
+\lambda_{var}\mathcal L_{variance}.
\]

- `BCE` supervises antialiased monochrome pixels.
- `edge` is an L1 loss on fixed Sobel responses.
- `Dice` prevents the white background from dominating.
- `stop` trains output length without an EOS token.
- `variance` keeps the 32-dimensional state from collapsing.

The reader is supervised only through visual language and raster losses. No
contrastive target contains text labels. A later optional adversarial
readability stage may be investigated only after the fixed V32 result; PIXAR's
PCAA result motivates it but it is not needed to interpret the main test.

## Curriculum

V32 deliberately mixes a releasable visual-language source with a private
research instruction source.

1. **Raster warmup.** Encode and reconstruct 1-to-32-cell segments sampled
   from the public-domain/Wikisource Chinese corpus. This teaches the writer to
   draw diverse observed forms.
2. **Book continuation.** Render a preceding strip as the prompt and the next
   short strip as the answer. This connects the reader, planner, and writer on
   public-domain Chinese without promising broad next-page understanding.
3. **Instruction transduction.** Render Chinese Alpaca instructions whose
   answers fit 32 cells. The source is CC BY-NC 4.0 and therefore the resulting
   V32 research checkpoint is non-commercial and not the final releasable ILM.
4. **Held-out wording.** Evaluate fixed locally generated paraphrases whose
   underlying answer records were seen only under the original wording.

The next data revision should replace the non-commercial instruction layer
with teacher-audited questions derived from public-domain books. V32 does not
delay the mechanism proof for that larger curation job.

## Evidence Hierarchy

The result is interpreted in this order:

1. **Boundary:** model methods and saved batches contain pixels, masks, and
   continuous states only.
2. **Raster:** autonomous output contains recognizable multi-glyph writing.
3. **Condition:** changing or shuffling prompt pixels changes the generated
   answer in the correct direction.
4. **Language:** held-out paraphrases and controlled counterfactuals preserve
   the intended answer better than visual-frequency controls.
5. **Generalization:** held-out fonts and layout offsets do not destroy the
   result.

Autoencoding, teacher-forced reconstruction, low training loss, or a beautiful
single cherry-picked strip cannot establish language.

## Efficiency Hypothesis

The selected configuration is approximately:

- 86.05M prompt-reader parameters;
- 25-35M planner, target encoder, decoder, and heads;
- fewer than 125M total parameters;
- at most 192 prompt patches and 32 answer steps; and
- BF16 training with the reader frozen for most updates.

This should fit comfortably on one 24 GiB RTX 4090. It does not yet establish
that visual language is more compute-efficient than token language. The
experiment reports update time, peak allocated VRAM, total training pixels,
and autonomous latency. Efficiency is a measured comparison, not an argument
from parameter count.

## Falsification

- **Raster passes, semantics fail:** V32 is a visual copier/writer, not a
  language model.
- **Teacher-forced passes, autonomous fails:** exposure bias remains; decoded
  feedback or state geometry is inadequate.
- **Original prompts pass, paraphrases fail:** the reader memorizes visual
  templates rather than meaning.
- **Prompt shuffle has little effect:** answer priors dominate.
- **Pretrained passes, random reader fails:** useful external visual-language
  knowledge transferred; the final model is still independent, but the
  initialization contribution must be stated.
- **Both readers fail:** the writer or curriculum is the bottleneck, not proof
  against image-native language.
- **Font holdout fails:** the system learned typography rather than stable
  written-language structure.

## Scope After A Pass

A V32 pass authorizes expansion to longer answer strips, two-dimensional page
layout, public-domain instruction curation, and historical glyph cells. It
does not authorize claims of Qwen parity, unrestricted conversation, verified
etymology, arbitrary book reading, human neuromorphic equivalence, or superior
training efficiency.

The first historical extension should ask about one modern Chinese character
and generate a short explanation plus clearly marked synthesized and attested
image regions. Source retrieval and provenance remain separate from generation
so a plausible ancient-looking glyph is never misrepresented as evidence.

