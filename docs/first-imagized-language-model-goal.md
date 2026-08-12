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
training loop, fixed-bank evaluator, and autonomous inference loop. The first
one-RTX-4090 run **did not pass the language-model gate**.

It proved:

- a strict pixels-to-pixels student boundary;
- a cross-font visual alphabet;
- a recurrent state that changes candidate-image energies;
- direct continuous ink generation by conditional rectified flow;
- context-sensitive generated pixels; and
- autonomous rereading and feedback without OCR or symbolic decoding.

It did not prove:

- prediction better than elementary symbolic language baselines;
- useful long-context visual language state;
- stable readable autonomous continuation;
- historical-form question answering;
- lower end-to-end cost than a text LLM; or
- parity with Qwen 8B.

The present MVP is therefore **rejected but informative**. The next milestone
is closed-loop visual trajectory learning, not a larger model.

## New paradigm: retinal flow language modeling

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
The model therefore lives under the image distribution that it creates. This
last property is essential and exposes the present failure: one-step training
states are cleaner than autonomous model states, so small visual errors compound
into ink drift.

## Why this is a different paradigm

RFLM is not OCR plus an LLM plus a renderer. OCR and text embeddings are absent
from the student path. It is not a byte, character, or subword model because its
random variable is a continuous image field. It is not a visual VQ language
model because it has no finite visual-code vocabulary. It is not generic page
diffusion because it preserves causal retinal state and rereads every generated
mark. It is not merely retrieval because the writer synthesizes new pixels.

The design combines four ideas that are individually established but not a
proof of visual language when isolated:

- foveated observation makes high-resolution writing locally affordable;
- joint-embedding prediction makes visual identity learnable across views;
- energy scoring avoids a fixed output alphabet; and
- flow matching supplies a continuous multimodal image distribution.

The scientific novelty claim must concern their strict image-only composition
and measured behavior. It must not claim that humans literally implement this
network or that the current run is a useful language model.

## Measured proof receipt

The clean V5 run used public-domain Chinese book text rendered through eight
Noto CJK font files. Font bytes and SHA-256 hashes are stored in every
checkpoint and evaluation receipt.

| Property | Measured value |
|---|---:|
| Parameters | 11,690,244 |
| Training peak VRAM | 2.56 GiB |
| Autonomous throughput | 11.3 generated cells/s |
| Held-out candidate bank | 512 characters x 4 image views |
| Eligible held-out contexts | 2,423 |
| Random full-context top-1 | 0.041% |
| RFLM full-context top-1 | 0.908% |
| RFLM full-context top-5 | 2.229% |
| Last-fixation-only top-1 | 1.362% |
| Unigram top-1 | 1.857% |
| Symbolic bigram top-1 | 13.578% |
| Retina oracle top-1 | 97.648% |
| Retina oracle top-5 | 100% |
| Generated context cosine gain | +0.0211 |
| Generated best-of-four target hit | 1.5625% |
| Acceptance | false |

The 97.65% oracle result localizes the main bottleneck: visual perception is
already strong on this bank. The recurrent language distribution and its
closed-loop stability are weak. Full-context exact prediction being worse than
last-fixation-only is direct evidence that the current recurrent state does not
yet use its history effectively.

## Next learning algorithm: induced visual trajectories

The next stage changes the training distribution before changing model size.
It extends the same RFLM rather than adding a symbolic teacher at runtime.

### Rollout aggregation

For each clean image sequence, periodically run the current model for `K`
steps. Store the generated fixation images and recurrent states in a bounded
replay queue. Subsequent updates mix clean prefixes, damaged-font prefixes, and
model-induced prefixes. This is the image-generation analogue of dataset
aggregation: the student learns on states it actually visits.

### Trajectory consistency

For the same underlying clean sequence, compare a clean recurrent trajectory
with one that rereads generated or perturbed images:

\[
\mathcal L_{\mathrm{traj}} =
\sum_{k=1}^{K} w_k
\left(1-cos(h_{t+k}^{\mathrm{clean}},
             h_{t+k}^{\mathrm{rollout}})\right).
\]

The target is a continuous clean visual trajectory, not a character ID. Initial
rollouts use stopped gradients for memory efficiency. Later short unrolls can
differentiate through the writer and retina if the receipt shows a benefit.

### Distributional recurrent state

The current deterministic GRU is asked to represent a multimodal next-image
distribution in one vector. After rollout training is validated, replace or
augment it with a compact stochastic state or recurrent latent flow. Do not add
this complexity until the simpler clean-versus-rollout ablation is measured.

### Multiscale reading

The fixed fovea sees local form efficiently but has no explicit line or page
summary. Add a slow page state updated every row alongside the fast fixation
state. This remains continuous image memory and is accepted only if it improves
full-context over last-fixation evaluation.

## Acceptance gates for the next checkpoint

All gates are evaluated on a frozen manifest and glyph bank:

1. Full-context top-1 must exceed random, last-fixation-only, unigram, and the
   symbolic bigram baseline.
2. Full-context target energy and top-1 must exceed last-fixation ablations.
3. Generated target signal must remain positive on held-out fonts.
4. A 32-cell autonomous rollout must preserve nontrivial ink and remain human
   readable under a blinded rubric.
5. A visual identity evaluator may score output, but no evaluator signal may
   enter inference.
6. Student-boundary receipts must continue to report false for token IDs,
   Unicode IDs, OCR, visual codebooks, and external language models.
7. Scaling beyond the current width is allowed only after the trajectory
   ablation improves at least two failed gates.

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

Not achieved. The model must beat elementary language baselines and remain
stable under autonomous visual feedback.

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
on one 4090, but its accuracy is not competitive. Future comparisons must report
quality at equal tasks alongside VRAM, training energy, latency, throughput,
storage, and rendering cost. Parameter count alone is not efficiency.

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
