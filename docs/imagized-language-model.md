# Imagized Language Model

Version: 0.6, predictive visual field plus isolated visual actuation

Date: 2026-08-12

## Thesis

An Imagized Language Model (ILM) should treat visible writing as its native
language substrate, not as decoration around a symbolic language model:

```text
image of writing -> continuous visual language state -> image of writing
```

Typed prompts remain convenient, but a deterministic renderer converts them to
images before the learned model boundary. Uploaded pages, handwriting,
historical characters, damaged print, and forms absent from Unicode can enter
the same boundary directly. The primary output is an image. OCR is an optional
post-process, never the model's hidden language channel.

## Strict model boundary

The independent student may receive:

- grayscale or RGB writing images;
- continuous convolutional feature fields;
- continuous recurrent states;
- noise fields and continuous flow time; and
- source image regions selected by an explicit provenance gate.

It may not receive:

- text strings after rasterization;
- BPE, word, byte, character, or Unicode IDs;
- OCR transcripts;
- character-class labels as model inputs;
- a finite visual or glyph codebook;
- an inverse table that decodes hidden IDs into writing; or
- calls to an external language model at inference.

A fixed image grid is a sensor layout, not a vocabulary. A continuous retinal
state is a visual observation, not a renamed token.

## Current implementation: predictive visual field

V7 shows that direct pixel flow can learn stable, context-sensitive ink while
still failing to become a useful language distribution. V14 through V17
therefore separate **imagining the next visual state** from **drawing that
state**:

```text
writing pixels -> frozen retina -> causal visual field
               -> continuous proposal + hyperspherical state flow
               -> isolated visual actuator -> writing pixels -> reread
```

![Predictive Visual Field](../publication/ilm-image-native/figures/predictive_visual_field_paradigm.png)

The deterministic branch predicts a unit visual proposal directly from image
history,

\[
\mu_t=\operatorname{normalize}(P_\psi([h_t,z_t])).
\]

The stochastic branch learns intrinsic tangent velocity along a geodesic
between image-derived target state \(z_{t+1}\) and random unit source
\(\epsilon\):

\[
\mathcal L_{\mathrm{sphere}}=
\mathbb E\lVert v_\eta(q_\tau,\tau,[h_t,z_t])-u_\tau\rVert_2^2.
\]

It samples intended continuous visual states, never character indices. A
separate actuator renders a plan and is checked by rereading. No nearest-glyph
lookup, token unembedding, OCR path, or output vocabulary is allowed.

![Measured V16 visual-memory proof](../publication/ilm-image-native/figures/predictive_visual_field_v16_result.png)

V16 adds three residual causal memory blocks with dilated local visual fields
and global attention. It passes the frozen proposal and stochastic state gates.
The proposal reaches 6.264% top-1 over 512 forms, versus 3.971% last-only and
1.734% unigram; the state flow reaches 3.691%, versus 3.244% last-only. Both use
full history with positive normalized target-probability gain. The 13.143%
symbolic bigram still wins.

![Measured V17 visual-state actuator](../publication/ilm-image-native/figures/visual_state_actuator_v17_result.png)

V17 independently tests the drawing half. Its `5.73M` trainable parameters turn
a continuous state from a different-font target view plus a separate style
image into ink. On one untouched frozen split, correct-state visual identity is
58.59% versus 0.98% after shuffling only the intended state. The target-cosine
gain is `+0.6269`. Pixel F1 is only 0.4385 and human review rejects the output as
mostly pseudo-characters. V17 therefore accepts causal visual control and
rejects readable actuation. It is not yet coupled to V16's predicted states.

## Earlier precursor: read, predict, write, reread

The implemented **Retinal Flow Language Model (RFLM)** is a causal probability
model over image fixations:

\[
p_\Theta(x_{1:T}) = p(x_1)\prod_{t=1}^{T-1}
p_\Theta(x_{t+1}\mid x_{\le t}),
\qquad x_t\in[0,1]^{H\times W}.
\]

![Retinal Flow Language Model](../publication/ilm-image-native/figures/retinal_flow_paradigm.png)

### Read

A compact convolutional retina maps each `32x32` image fixation to a normalized
continuous visual state:

\[
z_t=R_\theta(x_t)/\lVert R_\theta(x_t)\rVert_2.
\]

Independent font views train the retina to preserve visible identity across
typographic nuisance. A target retina updated by exponential moving average
provides stable image-space targets.

### Remember

A recurrent visual field integrates the ordered history:

\[
h_t=G_\phi(h_{t-1},z_t).
\]

It is supervised only through consequences for future image prediction and
writing. There is no text-semantic teacher in the current checkpoint.

### Predict

A compatibility energy scores an arbitrary candidate image after the retina
reads it:

\[
s(c\mid x_{\le t}) = E_\psi(h_t,z_t,R_{\bar\theta}(c)).
\]

This avoids a finite character output layer. Multi-positive visual contrastive
training admits several renderings of the same visible form.

### Write

A conditional rectified flow learns a continuous path between target ink and
noise:

\[
y_\tau=(1-\tau)x_{t+1}+\tau\epsilon,
\qquad
\hat u=F_\omega(y_\tau,\tau,h_t,z_t).
\]

The model writes directly in pixel space. High-noise training forces it to use
context instead of copying visible target pixels.

### Reread

Generated endpoints are passed back through the target retina. A write-read
cycle loss aligns generated and real visual identity without OCR. During
autonomous inference, the model samples candidate ink images, rereads each,
selects by visual energy, pastes the selected pixels, and treats them as its next
observation.

This final feedback step is part of the language model, not a UI feature.

### Learn under its own visual distribution

V6 uses the deployed sampler inside training. From a clean image prefix it
generates two short-flow candidates, selects one by visual energy, detaches the
selected bitmap, rereads it through the online retina, and advances the
recurrent state. The added objective is

\[
\mathcal L_{\mathrm{roll}}=
0.15\mathcal L_{\mathrm{state}}+
0.35\mathcal L_{\mathrm{energy}}^{\mathrm{roll}}+
0.30\mathcal L_{\mathrm{recovery}}.
\]

The terms align induced and clean states, predict the real next image from an
induced state, and recover the clean next pixels after generated feedback. The
selected image remains a continuous bitmap; there is no character target ID.
This closes the train-inference state-distribution gap without relaxing the
student boundary.

V7 retains this closed-loop objective, adds a normalized full-history advantage
against independent image anchors, and backpropagates visual identity through a
two-step sampled endpoint. The anchor labels and target indices remain outside
the student; the anchor pixels are not deployed.

## Why the paradigm is useful

Visible writing carries information that symbolic normalization deletes:

- stroke topology and spatial composition;
- historical and regional form;
- handwriting, emphasis, correction, and damage;
- layout, reading order, annotation, and marginalia;
- continuity between pictorial and conventional writing; and
- forms that have no stable computer encoding.

A successful ILM could place modern explanation and ancient visual evidence in
one native answer object. For example, a question about `言` could return an
image page with readable Chinese or English prose and attested oracle, bronze,
seal, manuscript, and modern forms.

This does not imply that images are automatically more compute-efficient than
tokens or that human neural processing is identical to RFLM. Both remain
empirical questions.

## What the current experiments prove

The isolated V17 actuator trained for 1,600 updates in 339.67 seconds and used
1.588 GiB peak allocated CUDA memory on one RTX 4090:

| Frozen actuator measurement | Correct state | Shuffled state |
|---|---:|---:|
| Global visual identity top-1 | **58.594%** | 0.977% |
| Target cosine | **0.71301** | 0.08612 |
| Pixel F1 | **0.43849** | 0.28001 |
| Ink fraction | 0.15947 | 0.15967 |

This proves that a small non-token continuous visual state causally controls
generated writing pixels. It does not prove readable writing: V17 fails the
preregistered `0.50` pixel-F1 gate and human review. The next actuator must make
stroke topology explicit with a learned spatial visual motor plan.

### Predictive visual field language core

PVF V16 has 16,471,809 parameters, of which 15,138,881 are trainable. Its peak
allocated CUDA memory was 1.479 GiB on one RTX 4090. The frozen evaluator uses
512 common Han forms, four independent font views, and 1,788 eligible contexts:

| Measurement | V15 step 2,000 | V16 step 2,200 |
|---|---:|---:|
| Retina-oracle top-1 | 98.546% | 98.546% |
| Proposal full-context top-1 | 5.872% | **6.264%** |
| Proposal last-only top-1 | 4.418% | **3.971%** |
| Proposal normalized context gain | +0.07069 | **+0.07732** |
| State-flow full-context top-1 | 3.412% | **3.691%** |
| State-flow last-only top-1 | 2.685% | 3.244% |
| State-flow normalized context gain | +0.03032 | **+0.03579** |
| State-flow sampled context cosine gain | **+0.08053** | +0.07947 |
| Unigram / symbolic bigram top-1 | 1.734% / 13.143% | same |

This proves a small student can learn causal language signal from images of
writing without token IDs or a character classifier. It does not prove useful
general language, readable image output, or efficiency over a text model.

### Retinal-flow precursor

The complete RFLM has 11,690,244 parameters. V7 resumed V6 for 800 updates on
one RTX 4090 and peaked near 3.0 GiB. The same frozen 512-character Chinese
visual bank gives:

| Measurement | V6 | V7 step 5,800 |
|---|---:|---:|
| Retina-oracle top-1 | 98.18% | 98.27% |
| Recurrent full-context top-1 | 1.20% | **2.31%** |
| Last-fixation-only top-1 | 1.69% | 2.02% |
| Unigram top-1 | 1.86% | 1.86% |
| Symbolic bigram top-1 | 13.58% | 13.58% |
| Normalized target-log-probability gain | -0.9066 | **-0.2155** |
| Generated context cosine gain | +0.0077 | **+0.0303** |
| Autonomous late/early ink | 1.168 | **1.050** |
| Autonomous sparse cells | 18.75% | **15.63%** |

![Measured V6 and V7 result](../publication/ilm-image-native/figures/anchor_identity_v7_result.png)

V7 more than doubles full-context top-1, beats last-only and unigram, restores
generated context signal, and closes about 76% of V6's calibrated deficit. It
does **not** solve language: normalized context gain is still negative, top-1
is far below the bigram, and the output remains unreadable. V7 is rejected as a
useful language model. Positive raw energy alongside negative normalized
probability also falsifies raw score gain as an acceptance measure.

## What comes next

The next intervention is again not immediate scale:

1. Keep the proven retina and external bank fixed.
2. Run an attributable V16 memory ablation on a newly preregistered bank, with
   paired per-context predictions and correction-norm receipts; require
   proposal/state performance above the symbolic bigram.
3. Replace V17's blank spatial condition with a visual-state-decoded motor plan,
   supervise topology directly, and require readable output under a shuffled-
   state control without lookup.
4. Close the autonomous feedback loop only after state prediction and isolated
   actuation pass independently.
5. Require the complete system to remain readable for 32 cells before widening
   it or adding instruction tuning.

After this causal visual gate passes, add image-to-image instruction tuning and
provenance-gated historical glyph composition. A local LLM may help create or
critique offline curricula, but its strings and weights must not enter the
deployed ILM.

## Research contract

- Preserve negative results and compare against simple baselines.
- Separate retina identity from language prediction.
- Separate generated style from attested historical evidence.
- Record dataset rights, transforms, fonts, hashes, checkpoints, VRAM, and
  throughput.
- Claim efficiency only at matched quality on a named benchmark.
- Claim Qwen-8B parity only on a fixed published benchmark, never generally.

The detailed equations, measured receipt, acceptance gates, and staged
capabilities are in
[`first-imagized-language-model-goal.md`](first-imagized-language-model-goal.md).
The V16 memory receipt is in
[`predictive-visual-field-v16-memory-result.md`](predictive-visual-field-v16-memory-result.md).
The isolated V17 actuator receipt is in
[`visual-state-actuator-v17-result.md`](visual-state-actuator-v17-result.md).
The V8-V15 visual-state record is in
[`predictive-visual-field-v15-result.md`](predictive-visual-field-v15-result.md).
The exact V7 experiment and frozen receipts are in
[`retinal-flow-v7-anchor-identity-result.md`](retinal-flow-v7-anchor-identity-result.md).
The V6 precursor is in
[`retinal-flow-v6-closed-loop-result.md`](retinal-flow-v6-closed-loop-result.md).
The literature dossier is in
[`references/image-native-language-model-research.md`](../references/image-native-language-model-research.md),
and the compiled paper is in
[`publication/ilm-image-native/ilm-image-native.pdf`](../publication/ilm-image-native/ilm-image-native.pdf).
