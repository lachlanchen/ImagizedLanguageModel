# Conditional Binary Glyph Diffusion V49: Research Direction

Date: 2026-08-15

Status: mechanism research completed before a V49 protocol, metric selection,
implementation, optimization, development measurement, or frozen audit

## Decision

V49 should preserve the V42 causal visual reader and replace point-valued
raster prediction with a compact **conditional distribution over the raw binary
ink image**. The first candidate is a 32-by-32 Bernoulli/D3PM writer whose
reverse network is conditioned by the reader's continuous state. It samples a
coherent glyph image and feeds that visible sample back to the same reader.

This is a possible path, not a result. V42 supplied evidence that an ordered
image-only reader can learn Chinese continuation. V48 supplied evidence that a
deterministic future-field head is not an adequate writer. V49 asks only
whether the missing conditional-distribution mechanism can turn the useful
reader state into crisp, context-dependent visible writing on one RTX 4090.

The student boundary remains strict:

- input context is a sequence of raster cells;
- output is a sampled raster cell;
- the 1,024 binary pixels are the generative state, not 1,024 token IDs;
- there is no tokenizer, Unicode value, OCR transcript, character ID, glyph
  vocabulary, nearest-code lookup, or candidate image bank in training or
  deployment;
- strings and character labels remain offline rendering and audit metadata;
- the V42 checkpoint is part of the independent student, not an external
  runtime teacher.

## The Core Mathematical Failure

Let visible history be $H$, and let the next binary glyph raster be
$X\in\{0,1\}^{32\times32}$. If a deterministic writer is trained with squared
error, its population optimum is

\[
f^*(H)=\mathbb E[X\mid H].
\]

If it instead predicts independent pixel logits with binary cross entropy, the
population optimum at pixel $p$ is

\[
\sigma(f_p^*(H))=\Pr(X_p=1\mid H).
\]

Neither result is generally a sample from the joint distribution
$p(X\mid H)$. When several next glyphs are plausible, thresholding the
per-pixel marginals can combine strokes belonging to different modes. V48's
inverse-DCT sign boundary made this failure especially visible: a field could
move closer in background-dominated cosine while its decoded identity became
worse.

The required object is therefore not a sharper conditional mean. It is a
sample from a learned joint image distribution:

\[
X_{t+1}\sim p_\theta(X_{t+1}\mid X_{t-63:t}).
\]

The writer may factorize each reverse transition over pixels, but its logits
are computed from the whole noisy glyph and the whole causal context. Repeated
globally coupled reverse transitions therefore define a non-factorized joint
distribution over final glyph images.

## What Existing Experiments Already Rule Out

| Route | Local evidence | Why V49 is different |
|---|---|---|
| Independent 8-pixel InkStream logits | teacher-forced F1 reached `0.639`, but every autonomous stream collapsed to repetitive grey columns | V49 samples one globally coupled glyph distribution instead of locally advancing independent strip marginals |
| V31 conditional continuous flow | order sensitivity was measurable, but exact-suffix assignment stayed near chance and the target lived in a frozen retinal field | V49 writes the observed binary image itself and conditions the denoiser strongly at high corruption, where target leakage is unavailable |
| V42 stochastic residual field | the reader reached useful visual continuation, while its writer did not preserve identity | V49 keeps the reader evidence and replaces only the rejected output distribution |
| V43/V46 deterministic actuators | direct rendering could become sharp when supplied an exact target field, but autonomous language binding remained weak | V49 represents uncertainty explicitly instead of demanding that one point field simultaneously choose and render a mode |
| V48 four-horizon point fields | `11/16` gates passed, but direct visible identity, retention, binding, all-horizon advantage, and V42 improvement failed | V49 returns to the strongest single-next reader and tests the writer mechanism in isolation |

Scaling the transformer, adding fonts, adding etymology labels, or using a
larger image generator would not distinguish these mechanisms. Those changes
remain deferred.

## Why Binary Discrete Diffusion Is The First Candidate

[D3PM](https://arxiv.org/abs/2107.03006) defines diffusion directly in a
discrete state space using tractable transition matrices and combines a
variational objective with an auxiliary denoising cross entropy. It does not
require relaxing a binary raster into a learned codebook. [BerDiff](https://arxiv.org/abs/2304.04429)
shows that Bernoulli corruption and stochastic reverse sampling can generate
multiple coherent binary masks, including accelerated subsequences. These are
not language results; they establish a mathematically appropriate writer for
binary spatial objects.

The main alternatives have worse first-proof trade-offs:

- PixelCNN gives an exact raw-pixel density, but a 32-by-32 glyph requires
  1,024 serial pixel decisions. The original [PixelRNN/PixelCNN](https://proceedings.mlr.press/v48/oord16.html)
  result is a useful density baseline, not the selected one-GPU writer.
- Parallel multiscale autoregression reduces serial depth to logarithmic scale
  ([Reed et al., 2017](https://proceedings.mlr.press/v70/reed17a.html)), but it
  introduces a hierarchy whose alignment and boundary effects would become a
  second hypothesis.
- Gaussian diffusion or continuous flow can model unquantized images, and
  [MAR](https://arxiv.org/abs/2406.11838) establishes that visual
  autoregression does not require vector quantization. V31 and V48 nevertheless
  show that this repository first needs a foreground-faithful terminal state,
  not another continuous field whose final threshold can hide the failure.
- A categorical glyph codebook would make generation easy by reintroducing a
  finite character vocabulary. It violates the desired ability to emit unseen
  written forms and is therefore excluded.

## Binary Forward And Reverse Process

Let $x_0\in\{0,1\}^{1024}$ denote one flattened glyph, where one means ink.
For diffusion step $s$, use the two-state transition matrix

\[
Q_s=\alpha_s I+(1-\alpha_s)
\begin{bmatrix}
1-\pi_{ink} & \pi_{ink}\\
1-\pi_{ink} & \pi_{ink}
\end{bmatrix}.
\]

The scalar $\pi_{ink}$ is the training-partition ink rate, frozen before
development evaluation. The stationary prior is therefore sparse Bernoulli
ink rather than a dense grey image. With
$\bar\alpha_s=\prod_{j=1}^{s}\alpha_j$, the marginal is

\[
q(x_s=1\mid x_0)=
\bar\alpha_s x_0+(1-\bar\alpha_s)\pi_{ink}.
\]

Arbitrary $x_s$ can be sampled in one operation. The exact D3PM posterior is
computed from the two-by-two cumulative transition matrices. A denoiser
$D_\theta$ predicts clean-ink logits from noisy ink, reader state, spatial
condition, and diffusion time:

\[
\ell_0=D_\theta(x_s,s,h,a),\qquad
\tilde p_\theta(x_0=1\mid x_s,h,s)=\sigma(\ell_0).
\]

Here $h$ is the final V42 causal hidden state and $a$ is its full
image-derived anchor decoded to a spatial map. The predicted clean
distribution is combined with the exact posterior to parameterize
$p_\theta(x_{s-1}\mid x_s,h)$. Every reverse step samples binary states; the
terminal output is therefore binary by construction rather than sharpened by
an evaluator-side threshold.

The initial training objective should retain probabilistic calibration:

\[
\mathcal L_{writer}=\mathcal L_{D3PM-VLB}
+\lambda_{ce}\mathbb E_s\operatorname{BCE}(x_0,\ell_0)
+\lambda_{fg}\mathcal L_{foreground}.
\]

The first two terms follow the D3PM clean-image parameterization. The last is a
small, fixed auxiliary selected by the training-only foreground diagnostic
below; it cannot replace the probabilistic objective. Class weighting may be
used only in the auxiliary term, because weighting the primary clean-image BCE
would distort the sampled Bernoulli probabilities.

## Foreground Geometry Before Writer Training

V48 proved that raw signed-DCT cosine is not a trustworthy visible identity
metric. Its invertible DCT remains a valid carrier, but identity selection and
reporting need a separate foreground-sensitive chart derived from the raster
alone.

The diagnostic should compare fixed candidates on a deterministic
training-partition reservoir:

1. signed-DCT cosine, retained as the known baseline;
2. ink-only cosine on zero-background rasters;
3. soft Dice/Tanimoto overlap;
4. a multiscale foreground chart containing normalized ink, pooled ink, and
   Sobel-magnitude blocks.

One concrete chart is

\[
\phi(x)=\operatorname{norm}\left[
\operatorname{norm}(x),
\operatorname{norm}(P_2x),
\operatorname{norm}(P_4x),
\operatorname{norm}(|S_x*x|+|S_y*x|)
\right],
\]

where blank blocks are mapped to zero and each nonblank block has equal norm.
All operations are fixed convolutions, pooling, and normalization. No symbol
label enters the chart.

A chart is eligible for the later protocol only if, on training data, it:

- retrieves the same rendered form across the held-out training fonts better
  than the signed-DCT baseline;
- assigns blank or background-only images low similarity to inked targets;
- makes increasing similarity positively associated with image-bank identity
  for the already frozen V48 predictions; and
- remains stable under one-pixel translation, antialiasing, and mild stroke
  dilation without collapsing visibly different glyphs.

Exact thresholds, sample count, seeds, and tie-breaking must be preregistered
before the diagnostic is run. Development and frozen partitions stay closed
during chart selection.

## Compact Writer Architecture

The first writer should be intentionally smaller than the reader:

- a three-scale convolutional denoiser at `32x32`, `16x16`, and `8x8`;
- base width near 64 with depthwise or ordinary `3x3` residual blocks;
- fixed sinusoidal diffusion-time embedding;
- FiLM modulation from the 384-dimensional V42 hidden state at every scale;
- the noisy binary raster and decoded spatial anchor as input channels;
- one self-attention block only at `8x8`, if the convolutional receptive field
  is insufficient;
- approximately 3--6 million new parameters;
- 16 or 32 reverse steps for the evidence run, with a 10-step accelerated arm
  only after the full sampler works.

The V42 reader is frozen in the first arm. This isolates whether the writer can
use an already measured predictive state. A later end-to-end arm is warranted
only if the frozen-reader writer passes direct raster and condition-binding
gates. The combined deployed checkpoint remains far below one billion
parameters; parameter count is not evidence of language quality, but it makes
the mechanism affordable to falsify on one 4090.

## Preventing Condition Ignorance

At low noise, the corrupted target reveals most of itself, so a denoiser can
ignore language context exactly as V31's candidate path did. V49 therefore
needs condition pressure that does not introduce symbolic supervision:

1. sample diffusion times with a fixed high-noise quota, including terminal
   states where the next glyph cannot be recovered from $x_s$ alone;
2. drop the context condition on a small fixed fraction of examples, allowing
   a measured conditional-versus-unconditional denoising gap and optional
   classifier-free guidance;
3. include a suffix-preserving context swap during training diagnostics and
   require the correct visual history to lower high-noise denoising loss;
4. report sample changes under shuffled and zeroed history rather than relying
   on reconstruction loss.

The swapped context is a negative diagnostic, not a character label. The
writer is rejected if its samples remain invariant to history even when its
denoising objective falls.

## Sampling And Selection Without A Glyph Bank

The primary evidence sample is the first draw from a published seed. It is not
selected by an evaluator bank. Four draws may additionally measure conditional
diversity and best-achievable coverage, but best-of-four is never reported as
the autonomous primary score.

If later deployment needs deterministic choice among samples, the only
eligible selector is a reader-side foreground query trained on target rasters:

\[
q_\psi(h)\in\mathbb R^d,\qquad
s(x,h)=q_\psi(h)^\top\phi(x).
\]

This query can rank newly sampled arbitrary images because it has no persistent
candidate bank. It must be trained and audited separately, and the first V49
mechanism result should not depend on it.

## Learning From Visible Errors

One-step writing is necessary but not sufficient. The model must eventually
condition on the marks it actually emitted. Blind scheduled sampling is also
unsafe: if a generated glyph changes identity, pairing the altered history
with the original continuation creates a false training sequence.

The recovery phase should begin only after the one-step writer passes. For a
short two-step training segment:

1. sample the first visible glyph from the current writer with gradients
   stopped;
2. compare it with the intended first target using the frozen foreground
   chart;
3. if it lies inside a preregistered identity-preserving neighborhood, insert
   the emitted glyph into the reader history and train the second target;
4. otherwise insert a partially denoised or morphologically corrupted version
   of the intended raster and record the rejection rather than silently using
   clean teacher forcing.

This is generated-prefix recovery with an image-only safety gate. It trains on
the model's real surface defects while limiting semantic relabeling errors.
An ablation uses only forward-corrupted target images. A later sequence-wide
extension can borrow the independent per-cell noise principle of
[Diffusion Forcing](https://arxiv.org/abs/2407.01392), which optimizes causal
sequence denoising under different noise levels and reports stable long
continuous rollouts. V49 should not implement full sequence diffusion before
the one-glyph writer works.

## Staged Falsification Path

### Stage A: training-only geometry diagnostic

Freeze a foreground chart without viewing development or frozen outcomes. If
no chart removes the V48 similarity/identity contradiction, stop and research
geometry rather than training a writer.

### Stage B: isolated binary writer

Freeze V42, train the conditional binary writer on real next-glyph rasters,
and compare it with:

- the V42/V48 point writer;
- an unconditional binary diffusion writer of identical architecture; and
- a parameter-matched independent Bernoulli one-pass head.

The decisive evidence is direct sampled-raster identity, foreground structure,
condition sensitivity, diversity without speckle, and finite one-step
generation. Training loss alone cannot select V49.

### Stage C: visible-error recovery

Add the gated generated-prefix phase and measure two-, four-, and eight-cell
rollouts. The recovery arm must outperform the same writer trained only on
clean contexts while preserving one-step quality.

### Stage D: end-to-end reader/writer adaptation

Only after Stages A--C pass, unfreeze a bounded reader suffix or add
Diffusion-Forcing-style independent context corruption. Page layout,
instruction tuning, historical forms, multilingual data, and external teacher
data remain later experiments.

## Required Evidence In The V49 Protocol

The protocol should preregister at least these independent gates:

1. exact image-only boundary and checkpoint inspection;
2. foreground chart selection on training data only;
3. direct first-sample identity above V48 and unconditional controls;
4. nonblank binary output with plausible training-relative ink density;
5. connected-stroke and edge statistics that reject salt-and-pepper speckle;
6. correct-context advantage over shuffled/zeroed context at high noise;
7. sample diversity above zero without context-invariant mode spread;
8. no evaluator bank in training, sampling, or primary selection;
9. one-step output reread by the actual V42 image path;
10. recovery improvement in finite two-, four-, and eight-cell rollouts;
11. fixed one-4090 memory, time, parameter, and sampler-call receipts; and
12. exact resume, data partition, checkpoint, and frozen-boundary integrity.

The visual audit bank remains permissible for evaluator-side identity
measurement because it does not enter the student. Results must separately
report first-sample, all-sample, and any selector-assisted numbers.

## Failure Interpretations

- **Low diffusion loss, context invariance:** the denoiser is an image prior,
  not a language writer.
- **Diverse but unrelated samples:** stochasticity is present, conditional
  binding is absent.
- **Correct average ink, speckled samples:** the reverse process or architecture
  has not learned joint stroke geometry.
- **Crisp glyphs, poor target identity:** visual writing works but the V42
  predictive state is insufficient for generation.
- **Good one-step identity, collapsing rollout:** exposure/recovery is the
  remaining bottleneck.
- **Recovery improves only evaluator-selected best-of-four:** the autonomous
  generation claim fails.
- **Foreground metric passes only same-font tests:** the chart memorizes
  rendering details rather than stable written identity.

Each outcome narrows the mechanism. None authorizes a claim of Qwen parity or
general language understanding.

## Why This Is The Possibly Correct Path

The current evidence supports a division of labor:

\[
\text{ordered visible history}
\xrightarrow{\text{V42 reader}}
\text{predictive visual state}
\xrightarrow{\text{binary distribution writer}}
\text{sampled visible glyph}
\xrightarrow{\text{same retina}}
\text{next state}.
\]

The reader need not represent a Unicode character, and the writer need not
choose a vocabulary entry. The continuous state carries predictive context;
the binary diffusion carries uncertainty and spatial stroke coherence; the
visible feedback loop makes the generated mark, rather than a hidden token,
the recurrent language state.

That combination directly addresses the measured failures without abandoning
the measured success. It is more credible than another end-to-end redesign,
but it remains a hypothesis until a preregistered V49 run produces readable,
context-sensitive closed-loop samples.

## Claim Boundary

V49 research does not establish a complete ILM, human-like visual reading,
semantic understanding, instruction following, etymology answering,
historical-script generation, OCR, Qwen-8B parity, or efficiency superiority
over token language models. It identifies the smallest next experiment that
could connect the repository's strongest causal image reader to a real sampled
image writer while remaining trainable on one RTX 4090.
