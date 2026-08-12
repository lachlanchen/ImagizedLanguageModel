# The First Imagized Language Model: Engineering Goal

Date: 2026-08-12

## North star

Build an independent language model whose native computational object is
visible writing:

```text
typed text -> deterministic rasterizer --+
                                         +-> writing pixels -> ILM -> answer pixels
uploaded page / glyph / handwriting -----+
```

The rasterizer is a boundary adapter, not a tokenizer. After this boundary the
student receives only image tensors and continuous visual states. It must never
receive strings, token IDs, Unicode IDs, character labels, OCR transcripts, a
discrete glyph codebook, or an external language-model call. Its primary answer
is an image. OCR may create an optional searchable sidecar after inference.

This goal includes modern simplified and traditional Chinese, historical
writing, regional variants, handwriting, damaged print, and forms absent from
Unicode. It eventually includes image questions in Chinese or English and
image answers that combine readable explanation with provenance-linked oracle,
bronze, seal, clerical, manuscript, traditional, simplified, and Kanji forms.

## Current verdict

The repository now contains a complete 11.69M-parameter image-only student,
training loop, model-induced rollout objective, calibrated fixed-bank evaluator,
and autonomous inference loop. V7 added independent image-anchor context
contrast and differentiable sampled-image identity. It improved the strongest
fixed-bank top-1 from 1.197% to 2.311%, crossed the unigram and generated-signal
gates, and reduced the normalized context deficit from -0.9066 to -0.2155 nats.
It still failed the language-model gate.

It proved:

- a strict pixels-to-pixels student boundary;
- a cross-font visual alphabet;
- a recurrent state that changes candidate-image energies;
- direct continuous ink generation by conditional rectified flow;
- context-sensitive generated pixels;
- autonomous rereading and feedback without OCR or symbolic decoding;
- improved long-rollout ink stability after training on its own sampled images;
- useful gradients through a deployed two-step image sampler; and
- improved calibrated visual prediction from independent image anchors.

It did not prove:

- prediction better than a symbolic bigram baseline;
- positive normalized full-history target probability;
- readable autonomous continuation, despite stable V6/V7 ink occupancy;
- historical-form question answering;
- lower end-to-end cost than a text LLM; or
- parity with Qwen 8B.

The present MVP is therefore **rejected but informative**. V7 tested the direct
context and sampled-identity correction rather than only proposing it. The
result localizes a deeper architectural problem: one conditional pixel writer
is being asked to infer linguistic identity and render its strokes at once. The
next milestone separates continuous next-visual-state modeling from pixel
rendering. It is not a larger model.

## New paradigm: predictive visual fields

The next architecture is the **Predictive Visual Field (PVF)**:

```text
writing image -> retinal state -> causal visual field
              -> distribution over next retinal state
              -> visual actuator -> writing image -> reread
```

It makes the language distribution explicit in an image-derived continuous
space before asking a decoder to draw pixels. This is not a discrete bottleneck:
the retinal state is produced by a convolutional image encoder, has no inverse
ID table, and may represent handwriting, damaged print, mixtures, and forms
absent from Unicode.

For visible fixation \(x_t\), let a slowly moving target retina provide

\[
z_t=R_{\bar\theta}(x_t),\qquad z_t\in\mathbb S^{d-1},
\]

and let a causal field integrate fast glyph-scale and slower line/page evidence,

\[
h_t=C_\phi(h_{t-1},R_\theta(x_t)).
\]

Instead of using only an energy score, PVF learns the multimodal conditional
distribution of the next visual state. For Gaussian \(\epsilon\) and
\(\tau\in[0,1]\),

\[
q_\tau=(1-\tau)z_{t+1}+\tau\epsilon,
\qquad v^*=\epsilon-z_{t+1},
\]

\[
\mathcal L_{\mathrm{state\text{-}flow}}=
\left\|P_\eta(q_\tau,\tau,h_t)-v^*\right\|_2^2.
\]

Sampling this flow yields an intended next retinal state
\(\hat z_{t+1}\), not a character index. A separate visual actuator writes it:

\[
\hat x_{t+1}=W_\omega(\epsilon_x,h_t,\hat z_{t+1},R_\theta(x_t)),
\]

and is trained both in pixel flow space and through rereading,

\[
\mathcal L_{\mathrm{act}}=
\mathcal L_{\mathrm{pixel\text{-}flow}}+
\lambda\left[1-\cos\left(R_{\bar\theta}(\hat x_{t+1}),z_{t+1}\right)\right].
\]

At deployment there is no nearest-glyph projection, classifier table, Unicode
lookup, or token unembedding. The actual generated pixels are reread and become
the next observation. The writer is therefore a visual actuator for a planned
continuous state, rather than the sole place where language and typography
must emerge together.

![Predictive Visual Field](../publication/ilm-image-native/figures/predictive_visual_field_paradigm.png)

PVF is the V8 hypothesis. It has not yet passed a language gate. The first
implementation must prove the state distribution alone before coupling it to
autonomous rendering.

## Implemented precursor: retinal flow language modeling

The current architecture is the **Retinal Flow Language Model (RFLM)**. It is a
causal probability model over a sequence of continuous image fixations:

\[
p_\Theta(x_{1:T}) = p(x_1)\prod_{t=1}^{T-1}
p_\Theta(x_{t+1}\mid x_{\leq t}),
\qquad x_t\in[0,1]^{H\times W}.
\]

The random variables are image regions. A fixation is no more a language token
than a camera frame is an object label: it is a continuous sensor observation.
The fixed grid in the MVP is an engineering scaffold that can later be replaced
by learned saccades without changing the model contract.

### 1. Read: continuous foveal retina

A convolutional retina maps a `32x32` ink image to a unit visual vector:

\[
z_t = \frac{R_\theta(x_t)}{\lVert R_\theta(x_t)\rVert_2},
\qquad z_t\in\mathbb{S}^{191}.
\]

Two renderings of the same visible content are paired only at the offline image
construction boundary. Cross-font contrastive and variance losses force the
retina to preserve stroke identity while suppressing font-specific nuisance.
Unsupported-font characters are excluded by actual font cmap coverage, so tofu
boxes cannot become an accidental visual class.

### 2. Remember: recurrent visual field

A three-layer GRU integrates ordered visual observations:

\[
h_t = G_\phi(h_{t-1}, z_t),
\qquad h_t\in\mathbb{R}^{384}.
\]

This state is not supervised with words or semantic IDs. It is trained only by
the compatibility and writing consequences of the next image. Dense causal
positions turn every 48-fixation page into multiple prediction problems.

### 3. Predict: energy over arbitrary images

The model does not classify into a finite character table. It scores any
candidate image through its retinal embedding:

\[
s_{t,j}=E_\psi(h_t,z_t,R_{\bar\theta}(c_j)).
\]

Multi-positive visual NCE treats near-identical candidate views as additional
positives, preventing the model from being penalized when two image renderings
carry the same visible form. The target retina is an exponential moving average
of the online retina.

### 4. Write: conditional pixel-space rectified flow

For a clean next fixation (x_{t+1}), noise \(\epsilon\), and time
\(\tau\in[0,1]\), training follows the straight path

\[
y_\tau=(1-\tau)x_{t+1}+\tau\epsilon,
\qquad u_\tau=\epsilon-x_{t+1}.
\]

The writer predicts this velocity from pixels and recurrent context:

\[
\hat u_\omega = F_\omega(y_\tau,\tau,h_t,z_t),
\qquad
\mathcal L_{\mathrm{flow}}=
\lVert\hat u_\omega-u_\tau\rVert_2^2.
\]

Time is biased toward high noise, where the target cannot be copied from the
input. Stroke weighting and endpoint supervision preserve sparse ink. Sampling
integrates the learned field from noise toward data in a small number of steps.

### 5. Reread: visual write-read consistency

The predicted clean endpoint at training time is

\[
\hat x_0 = y_\tau-\tau\hat u_\omega.
\]

It is passed through the frozen target retina and contrasted with independent
target renderings:

\[
\mathcal L_{\mathrm{cycle}}=
\operatorname{NCE}
\left(R_{\bar\theta}(\hat x_0),
      R_{\bar\theta}(x_{t+1}^{(1)}),
      R_{\bar\theta}(x_{t+1}^{(2)})\right).
\]

This connects writing quality to visual identity without asking OCR to read the
output.

### 6. Act: autonomous visual feedback

At inference the writer samples several candidate images, the retina rereads
them, and the energy model selects one:

\[
\hat x_{t+1}=\arg\max_{c\in\mathcal C_t}
E_\psi(h_t,z_t,R_{\bar\theta}(c)).
\]

The selected pixels are pasted into the page and become the next retinal input.
The model therefore lives under the image distribution that it creates. V6
uses this exact path during training. V7 additionally differentiates through a
short image endpoint. These changes remove the earlier monotonic ink drift and
strengthen held-out target signal, but the selected images remain unreadable.

## Why this is a different paradigm

RFLM and PVF are not OCR plus an LLM plus a renderer. OCR and text embeddings
are absent from the student path. They are not byte, character, or subword
models because their observed random variables are continuous image fields.
PVF's hidden random variable is also continuous and image-derived. Neither is a
visual VQ language model because neither has a finite visual-code vocabulary or
an inverse ID table. They are not generic page diffusion because they preserve
causal visual state and reread every generated mark. They are not merely
retrieval because the actuator synthesizes new pixels.

The design combines five ideas that are individually established but not a
proof of visual language when isolated:

- foveated observation makes high-resolution writing locally affordable;
- joint-embedding prediction makes visual identity learnable across views;
- energy scoring supplies an evaluator without a fixed output alphabet;
- state flow supplies a continuous multimodal language distribution; and
- pixel flow supplies a continuous visual actuator.

The scientific novelty claim must concern their strict image-only composition
and measured behavior. It must not claim that humans literally implement this
network or that the current run is a useful language model.

## Measured proof receipt

V6 and V7 use the same public-domain Chinese book manifest, eight Noto CJK font
files, model width, and fixed evaluation bank. Font bytes and SHA-256 hashes are
stored in checkpoints and receipts. V7 resumed V6 for 800 updates. Its training
image anchors use disjoint render seeds from the evaluator pixels.

| Property | V6 closed loop | V7 step 5,800 | V7 step 6,000 |
|---|---:|---:|---:|
| Parameters | 11,690,244 | 11,690,244 | 11,690,244 |
| Training peak VRAM | 2.576 GiB | about 3.0 GiB | about 3.0 GiB |
| Held-out candidate bank | 512 x 4 views | same bank and SHA-256 | same |
| Eligible held-out contexts | 2,423 | 2,423 | 2,423 |
| RFLM full-context top-1 | 1.197% | **2.311%** | 2.022% |
| RFLM full-context top-5 | 3.219% | **5.613%** | 5.324% |
| Last-fixation-only top-1 | 1.692% | 2.022% | 1.940% |
| Unigram top-1 | 1.857% | 1.857% | 1.857% |
| Symbolic bigram top-1 | 13.578% | 13.578% | 13.578% |
| Retina oracle top-1 | 98.184% | **98.267%** | 98.225% |
| Raw target-score gain | +2.806 | +2.463 | +2.510 |
| Normalized target-log-probability gain | **-0.9066** | **-0.2155** | -0.2195 |
| Generated context cosine gain | +0.0077 | +0.0303 | **+0.0318** |
| Autonomous late/early ink | 1.168 | **1.050** | not selected |
| Autonomous sparse cells | 18.75% | **15.63%** | not selected |
| Autonomous human readability | false | false | not evaluated |
| Acceptance | false | false | false |

![Matched autonomous V6 and V7 evidence](../publication/ilm-image-native/figures/anchor_identity_v7_result.png)

The 98.27% oracle localizes the bottleneck away from basic cross-font
perception. V7 is a real improvement: it more than doubles V6 top-1, beats
last-only and unigram at the selected checkpoint, restores held-out generated
context signal, and closes about 76% of the calibrated context deficit. It is
still 5.9 times below the bigram, has negative mean normalized context gain,
and writes dense pseudo-glyphs rather than Chinese.

The exact V7 command, intermediate checkpoints, bank hash, and autonomous
receipt are preserved in
[`retinal-flow-v7-anchor-identity-result.md`](retinal-flow-v7-anchor-identity-result.md).
The V6 precursor remains in
[`retinal-flow-v6-closed-loop-result.md`](retinal-flow-v6-closed-loop-result.md).

## Implemented V6 and V7 algorithms

### V6: induced visual trajectories

For a clean image prefix, V6 runs the deployed two-step flow sampler, generates
two candidate bitmaps, rereads them with the target retina, and selects by the
deployed visual energy. The selected bitmap is detached, reread by the online
retina, and fed into the GRU. It adds

\[
\mathcal L_{\mathrm{roll}} =
0.15\mathcal L_{\mathrm{traj}} +
0.35\mathcal L_{\mathrm{energy}}^{\mathrm{roll}} +
0.30\mathcal L_{\mathrm{recovery}}.
\]

This is online model-induced sampling, not a replay queue. It trains the state,
energy, and recovery writer on the exact pixels the current model visits while
keeping memory bounded. The experiment validates the stability effect and
rejects the assumption that trajectory consistency alone yields semantics.

### V7: calibrated context and sampled identity

V7 replaces the raw target-energy margin with normalized visual likelihood. For
candidate image set \(\mathcal C\),

\[
\ell(y\mid h,\mathcal C)=s(y\mid h)-
\log\sum_{c\in\mathcal C}\exp s(c\mid h),
\]

\[
\mathcal L_{\mathrm{ctx}}=
\max\left(0,m-\ell(y\mid h_{\mathrm{full}},\mathcal C)
+\ell(y\mid h_{\mathrm{last}},\mathcal C)\right).
\]

The last branch is detached. The training-only candidate set is formed from
independently rendered images; positives are inferred by retinal similarity.
Anchor labels and target indices do not enter the student, and the bank is not
deployed. This is a finite visual curriculum, not a hidden output codebook.

V7 also differentiates through a truncated two-step flow endpoint:

\[
\mathcal L_{\mathrm{sample}}=
\operatorname{NCE}\left(
R(\hat x_0^{K=2}),\{R(y^{(v)})\}_{v=1}^{V}\right).
\]

This supervises images sampled by the numerical inference path rather than only
a one-step denoising estimate. V6's trajectory and recovery terms remain as
stability regularizers. V7 validates both corrections but rejects the monolithic
language-plus-rendering writer.

## Next proof: state flow before pixel flow

V8 must be implemented and evaluated in four ordered stages:

1. Freeze or slowly update the proven retina and train a conditional flow over
   the next retinal state. Do not train the pixel writer in this first ablation.
2. Compare sampled next-state likelihood and retrieval against full-history,
   last-only, unigram, and bigram branches on the unchanged bank.
3. Condition a separate pixel actuator on a sampled target state and require
   its reread state to match the plan without nearest-neighbor decoding.
4. Close the pixel feedback loop only after both state prediction and isolated
   actuation pass their gates.

Start with one causal fast state. Add line and page timescales one at a time only
if the state-flow ablation establishes positive normalized context. This keeps
the experiment attributable and preserves the single-4090 constraint.

## Acceptance gates for the next checkpoint

All gates are evaluated on a frozen manifest and glyph bank:

1. State-flow full-context top-1 must exceed last-fixation-only and unigram;
   beating the symbolic bigram remains the causal-language acceptance gate.
2. Full-history normalized target log probability must exceed last-only. Raw
   target energy is diagnostic only and can never satisfy this gate.
3. Sampled state context cosine gain must exceed `0.02` and the random branch by
   at least `0.01` on held-out fonts.
4. The isolated actuator must produce nontrivial ink whose reread state matches
   its sampled plan and remains readable under a blinded rubric.
5. The closed model must retain readability over a 32-cell autonomous rollout.
6. A visual identity evaluator may score output, but no evaluator signal may
   enter inference.
7. Student-boundary receipts must continue to report false for token IDs,
   Unicode IDs, OCR, visual codebooks, and external language models.
8. Scaling beyond the current width is allowed only after state flow and the
   isolated actuator each pass their own gate.

## Dataset path

Large language datasets remain useful at the offline boundary:

1. Read an openly licensed text or instruction record.
2. Render it into multiple image trajectories with different font, form,
   spacing, direction, paper, blur, and scan damage.
3. Filter fonts by real glyph coverage.
4. Keep source rights, artifact path, transform, and hash in the manifest.
5. Delete strings before the student batch is formed.
6. Mix scanned books, handwriting, Kanji vectors, and historical glyph images
   without forcing unencoded forms through Unicode.

The local historical snapshot currently contains 9,055 characters and 84,642
glyph records across oracle, bronze, seal, and Liushutong stages. These assets
are evidence, not generic style targets. A factual etymology answer must copy or
cite attested source pixels; a synthesized form must be labeled synthesized.

## Capability stages

### P0: boundary and visual alphabet

Implemented. The model trains and infers with only continuous image tensors;
the fixed-bank oracle establishes cross-font visual identity.

### P1: causal visual language

Not achieved. V7 remains visually populated under autonomous feedback and now
beats unigram top-1, but its marks are unreadable, its normalized context gain
is negative, and it remains far below the symbolic bigram. PVF state flow is
the next bounded P1 experiment.

### P2: bounded visual instruction following

Rasterize openly licensed Chinese and English instruction pairs, train image
question to image answer trajectories, and evaluate task correctness plus
readability. Historical panels are provenance-gated source images.

### P3: bounded Qwen-8B comparison

Define a fixed benchmark for visual instruction following, bilingual questions,
historical-form recognition, and glyph-origin answers. Compare correctness,
legibility, provenance, latency, VRAM, parameters, and throughput. No general
parity claim is allowed.

### P4: broad image-native model

Scale only after P1 and P2. Add public-domain multilingual books, handwriting,
damaged manuscripts, learned saccades, multiscale page memory, and multi-page
dialogue.

## Efficiency claim

Image-native representation is a hypothesis, not automatically more efficient
than tokens. The current receipt shows that a complete 11.69M model fits easily
on one 4090 at about 3.0 GiB peak allocation, but its accuracy is not
competitive.
Future comparisons must report quality at equal tasks alongside VRAM, training
energy, latency, throughput, storage, and rendering cost. Parameter count alone
is not efficiency.

## Non-negotiable rules

- Do not call a smoke run trained.
- Do not call reconstruction language generation.
- Do not infer language ability from retina-oracle accuracy.
- Do not hide a symbolic vocabulary inside a visual codebook.
- Do not let OCR, text labels, or a teacher model enter the deployed student.
- Do not present a generated historical-looking glyph as attested evidence.
- Do not claim Qwen parity without a named benchmark and measurements.
- Preserve negative runs, manifests, fonts, hashes, metrics, and inference
  receipts as scientific evidence.
