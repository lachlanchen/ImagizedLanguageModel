# One-forward learning for the visual-semantic candidate

The next learning path uses supervised conditional flow matching with an
image answer. The implementation removes the need to differentiate through
an entire sampling trajectory for each example. It does not establish that
the candidate learns useful language, and no training-speed ratio has been
measured. The current registered QA experiment continues unchanged.

## Objective and information path

Let \(x\) be a complete question/passage page and \(y\) its answer image. A
frozen image codec gives the clean answer latent \(z_y\). The model receives
an image-derived semantic condition \(s(x)\) and reference-image latents
\(r(x)\), with the generic task fixed across examples. Neither condition may
contain the answer image, its transcription or a question-specific text
embedding. The pretrained semantic body is retained and must be counted;
this is adaptation of an existing model, not foundation training from scratch.

For Gaussian noise \(\epsilon\) and a native scheduler noise level \(\sigma\),

\[
z_\sigma=(1-\sigma)z_y+\sigma\epsilon,
\qquad v^*=\epsilon-z_y.
\]

The unweighted training objective is

\[
\mathcal L(\theta)=\mathbb E\left[
\left\|v_\theta(z_\sigma,t_\sigma,s(x),r(x))-v^*\right\|_2^2
\right],
\]

with the mean taken over latent elements and examples. The clean answer is
supervision in this training objective. Inference must instead start from
independent noise, use only the input page and fixed task, and generate actual
answer pixels. Training-time denoising of a noisy target is not a free-running
answering demonstration.

This is the simulation-free regression principle of
[flow matching](https://arxiv.org/abs/2210.02747), with a linear noise/data
path related to [rectified flow](https://arxiv.org/abs/2209.03003).
It requires one logical denoiser forward per training example, with any
activation-checkpoint recomputation accounted for separately. Sampling can
still require many integration steps.

Under squared error, an unconstrained population optimum predicts the
conditional mean velocity given \((z_\sigma,\sigma,s(x),r(x))\). That statement
does not guarantee semantic understanding, a faithful image codec, successful
finite-data optimization or readable writing. If the conditioning path loses
the distinction between two questions with different answers, the generator
cannot recover it from a fixed task instruction. Paired questions, erasure
controls and actual held-out answer images remain necessary evidence.

## Native scheduler agreement

The [pinned DeepGen training implementation](https://github.com/deepgenteam/deepgen/blob/fbaaaf4ae1c79450d6a34b30beb4f59840568c69/src/models/sd3_kontext/qwen2_5_vl_sd3_hf_dynamic_fusion.py)
uses this interpolation and velocity target. Its default unweighted loss
upcasts predictions and targets to float32 before the squared-error reduction.
The released scheduler uses 1000 training timesteps, fixed shift 3 and
deterministic reverse Euler updates. The helper samples a uniform index from
the fresh native training grid and takes both \(t_\sigma\) and \(\sigma\) from
that grid. It rejects a scheduler already mutated to an inference schedule.
It does not substitute a continuous uniform noise level for the shifted grid.

The reverse update is

\[
z_{\sigma'}=z_\sigma+(\sigma'-\sigma)\widehat v,
\qquad \sigma'<\sigma.
\]

The velocity sign and model timestep scale must therefore remain consistent.
Explicit time and Gaussian-noise generators isolate these draws from reader
and codec RNG use. Reference-image conditioning always comes from \(x\),
independently of \(y\).

## Implemented and checked

`ilm/image_edit/visual_flow_learning.py` implements the native shifted-grid
sampling, interpolation, velocity loss and single-call denoiser loss helper.
`scripts/check_visual_flow_math.py` executes the exact pinned source ASTs
for interpolation, target construction, sigma lookup and loss. The
[numerical receipt](evidence/visual_flow_math.json) records exact agreement
of noisy inputs, targets, scalar losses and prediction gradients in float32
and bfloat16, with reproducible explicit RNG draws.

The real 50-step reverse scheduler, supplied an oracle velocity in float32,
returns the numerical test data with maximum absolute error
\(1.5497207641601562\times10^{-6}\). Reversing that velocity's sign produces
error 11.375895500183105. These are algebra checks using actual existing RGB
answer pages as numerical arrays. They are not VAE reconstructions, neural
predictions or evidence of language learning. The helper's actual model
forward/backward and its GPU memory requirements have not yet been exercised.

## Image data

`scripts/prepare_visual_flow_pages.py` prepares all 448 existing image pairs:
384 training pairs and 64 validation pairs. Training contains 256 QA pairs
and 128 copying pairs; validation contains 32 of each. QA articles remain
disjoint, with 128 training articles and 16 validation articles. These reused
development examples are not a new blinded test set.

The [image-preparation receipt](evidence/visual_flow_pages.json) verifies
that every input remains byte-identical. Each 512 × 256 answer strip is placed
at its original coordinates on a white 1024 × 1024 canvas; the original
answer region remains pixel-identical, with no resizing or rerendering. The
manifest contains image paths, hashes and grouping metadata, with no reference
strings. The offline preparer has access to the original accepted answers;
all QA targets are checked against that source.

The candidate's native/compiled/vocabulary-removal replay must finish before
admitting its learning stage. Its codec must also reconstruct the actual
answer glyphs at this geometry, followed by a real differentiated training
step and autonomous generation. Only subsequent complete training and
unrepaired held-out answer pages can establish whether this path works.
