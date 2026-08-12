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

The repository now contains two complete experimental systems:

- RFLM V7 is an 11.69M-parameter pixels-to-pixels precursor with autonomous
  write-reread feedback. It produces stable but unreadable pseudo-glyphs.
- PVF V15 is a 10.47M-parameter image-to-continuous-image-state model. It
  factorizes next-state prediction into a deterministic visual proposal and a
  stochastic hyperspherical field. It deliberately has no pixel actuator yet.

On the frozen 512-form Chinese bank, V15's proposal reaches **5.872%** top-1,
versus **4.418%** last-only, **1.734%** unigram, **0.168%** random dynamics, and
**13.143%** symbolic bigram. Its normalized full-history gain is `+0.07069`
nats. The state flow reaches **3.412%**, versus **2.685%** last-only, with
sampled context cosine gain `+0.08053`. Both continuous branches pass their
state gates; neither passes the bigram language gate.

It proved:

- a strict image-only student boundary;
- a cross-font visual alphabet;
- a recurrent state whose prediction improves with more than the last image;
- a continuous visual proposal that beats random, unigram, and last-only
  branches without a character output table;
- a hyperspherical stochastic field with target-related samples and positive
  full-history probability gain;
- measurable visual-language learning with 10.47M parameters and 1.181 GiB peak
  allocated CUDA memory on one RTX 4090;
- direct continuous ink generation by conditional rectified flow;
- context-sensitive generated pixels;
- autonomous rereading and feedback without OCR or symbolic decoding;
- improved long-rollout ink stability after training on its own sampled images;
- useful gradients through a deployed two-step image sampler; and
- improved calibrated visual prediction from independent image anchors.

It did not prove:

- prediction better than a symbolic bigram baseline;
- readable PVF pixel output, because the actuator is not yet coupled;
- readable autonomous continuation, despite stable V6/V7 ink occupancy;
- historical-form question answering;
- lower end-to-end cost than a text LLM; or
- parity with Qwen 8B.

The present result is therefore an **accepted visual-state proof and a rejected
language-system proof**. It breaks the narrower assumption that image-native
training on one consumer GPU cannot acquire causal language structure. It does
not justify claiming that language is solved by image generation. The next
milestone strengthens causal visual memory until it beats the fixed bigram,
then audits a separate state-conditioned pixel actuator.

## Implemented paradigm: predictive visual fields

The current architecture is the **Predictive Visual Field (PVF)**:

```text
writing image -> retinal state -> causal visual field
              -> continuous proposal + distribution over next retinal state
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

PVF first predicts a low-variance continuous mode,

\[
\mu_t=\frac{P_\psi([h_t,z_t])}{\lVert P_\psi([h_t,z_t])\rVert_2}
\in\mathbb S^{d-1}.
\]

This is an image-derived state, not an index. In parallel, PVF learns a
multimodal conditional distribution on the unit sphere. Let \(q_\tau\) follow
the geodesic from target \(z_{t+1}\) to random unit source \(\epsilon\), with
intrinsic tangent velocity \(u_\tau\). Then

\[
\mathcal L_{\mathrm{state\text{-}flow}}=
\left\|v_\eta(q_\tau,\tau,[h_t,z_t])-u_\tau\right\|_2^2.
\]

Sampling integrates the tangent field backward with the spherical exponential
map and yields intended next retinal states \(\hat z_{t+1}\), never character
indices. Image-only multiview anchors train object separation and full-history
advantage. Their labels and target indices do not enter the student and the bank
is not deployed.

A later separate visual actuator writes the selected state:

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

![Predictive Visual Field V15 result](../publication/ilm-image-native/figures/predictive_visual_field_v15_result.png)

V15 proves the continuous state distribution uses causal visual history and
beats random, last-only, and unigram controls. It still falls 2.24 times below
the symbolic bigram. The actuator remains isolated by design so a good retina
score cannot be mistaken for readable writing.

## Earlier precursor: retinal flow language modeling

The earlier architecture is the **Retinal Flow Language Model (RFLM)**. It is a
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

The design combines six ideas that are individually established but not a
proof of visual language when isolated:

- foveated observation makes high-resolution writing locally affordable;
- joint-embedding prediction makes visual identity learnable across views;
- deterministic continuous proposal supplies a low-variance language mode
  without a symbolic vocabulary;
- energy scoring supplies an evaluator without a fixed output alphabet;
- state flow supplies a continuous multimodal language distribution; and
- pixel flow supplies a continuous visual actuator.

The scientific novelty claim must concern their strict image-only composition
and measured behavior. It must not claim that humans literally implement this
network or that the current run is a useful language model.

## Measured proof receipt

The selected PVF V15 step 2,000 checkpoint uses the same fixed 512-form,
four-view bank as V14. It was chosen on the development image bank before the
external evaluation was run.

| Property | V14 step 1,100 | V15 step 2,000 |
|---|---:|---:|
| Parameters | 10,470,273 | 10,470,273 |
| Trainable parameters | 9,137,345 | 9,137,345 |
| Proposal parameters | 3,547,968 | 3,547,968 |
| Classifier / pixel actuator parameters | 0 / 0 | 0 / 0 |
| Peak allocated VRAM | 1.181 GiB | 1.181 GiB |
| Eligible frozen contexts | 1,788 | 1,788 |
| Proposal full-context top-1 | 2.517% | **5.872%** |
| Proposal last-only top-1 | 2.237% | 4.418% |
| Proposal normalized context gain | +0.02853 | **+0.07069** |
| State-flow full-context top-1 | 2.349% | **3.412%** |
| State-flow last-only top-1 | 1.734% | 2.685% |
| State-flow normalized context gain | +0.02759 | **+0.03032** |
| Unigram / bigram top-1 | 1.734% / 13.143% | same |
| Retina oracle top-1 | 98.546% | 98.546% |
| State/proposal acceptance | true / true | true / true |
| Bigram language acceptance | false | false |

The complete receipt and V8-V15 ablations are in
[`predictive-visual-field-v15-result.md`](predictive-visual-field-v15-result.md).

### Retinal-flow precursor receipt

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

## Next proof: stronger causal vision before pixel actuation

V14 and V15 complete the bounded state-flow ablation. The next proof has three
ordered stages:

1. Replace the GRU bottleneck with one compact multiscale causal visual memory
   while freezing the proven retina, bank, proposal, and evaluator protocol.
   Require full-history performance above the `13.143%` symbolic bigram.
2. Condition a separate pixel actuator on the continuous proposal and sampled
   state. Require generated pixels to reread to the intended state across held-
   out fonts, including image forms that are not represented by an output ID.
3. Close the write-reread loop only after causal state and isolated actuation
   independently pass. Require readable 32-cell continuation before adding
   instruction tuning or historical-answer composition.

The likely bottleneck is causal compression, not image rasterization: the
retina oracle is 98.546%, proposal history helps, and more V15 optimization
plateaus well below bigram. A compact local/global causal memory is therefore a
more attributable next intervention than a larger diffusion model.

## Acceptance gates for the next checkpoint

All gates are evaluated on a frozen manifest and glyph bank:

1. Proposal and state-flow full-context top-1 must exceed last-fixation-only,
   unigram, and ultimately the symbolic bigram.
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
8. Scaling beyond the current width is allowed only after a causal-memory
   ablation or the isolated actuator passes its own gate.

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

Partially achieved. V15's proposal and stochastic field both beat random,
last-only, and unigram while obtaining positive normalized context gain. The
proposal reaches 5.872%, but the symbolic bigram remains 13.143%. P1 is not
complete until the fixed bigram is beaten without labels entering the student.

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
than tokens. V15 shows that a complete 10.47M visual-state model trains on one
4090 at 1.181 GiB peak allocated memory and learns real causal signal, but its
accuracy remains below a symbolic bigram and it has no pixel actuator.
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
