# Direct supervision of the actual autonomous writing trajectory

The completed recognition-loss treatment still fails copying. Its full
64-output reader diagnostic records much larger target NLL on autonomous
answers than on the earlier noisy-target-assisted updates. The same reader
also ignores some repeated glyphs and duplicate lines. Recognition alone
therefore cannot enforce exact requested content on these observations.
Neither observation identifies a unique cause of the writer's failure.

Test a direct, differentiable final-image objective. This is an application of
[DRaFT, Clark et al., ICLR 2024](https://arxiv.org/abs/2309.17400), not a new
backpropagation algorithm or a demonstrated language-model breakthrough.

## Exact autonomous sampler before optimization

Use the already pinned undistilled FLUX.2-klein-base-4B and VAE, fixed generic
condition and the completed recognition-treatment adapters, checkpoint SHA256
`23332544624497e12b1c9b18f21b65a0543bea025a44eb18fa933caa5a5723af`.
Input is the native image-reference latent, that same generic condition and
independent Gaussian noise. No target image, target code, answer length,
recognizer label or question-specific text enters the sampling function.
Use exactly 50 native scheduler steps, guidance 1, 512x256 answer canvas and
the native BF16/FP32 arithmetic. The scheduler is deterministic. Its mutable
step index must not be advanced inside a recomputed checkpoint closure.

Before any optimizer, compare the differentiable sampler to a real native
pipeline call on the first Chinese and English training inputs, using their
original inference seeds. Every saved step and final pixel must be bit-exact;
also compare to the previously retained final tensors. A mismatch blocks this
training implementation rather than relaxing the numeric bound. Record the
source pipeline, scheduler and image-processor implementation hashes.

## Final-image objective and gradient preflight

Let G_theta(x,epsilon) be the actual full 50-step sampling procedure followed
by native VAE decoding. Let p be its unclipped RGB-valued image and c its
native clipped display pixels. For the training target page y and string t,

    L = CE(R(c), t + EOS) + 10 B(p,y),
    B(p,y) = .5 mean_ink(|p-y|) + .5 mean_background(|p-y|).

The target ink mask is mean RGB < .9; both classes must be nonempty. Average
each class over its own pixels and RGB channels. The unclipped decoder branch
keeps a gradient when white output is saturated. Native clipping remains in
the recognizer and deployed output branches. The target enters only these
losses after autonomous sampling. The pixel term tests a canonical writing
style; it does not qualify arbitrary-font or historical-script generalization.
No target/noise interpolation or assisted velocity objective is used in this
test. No OCR correction or rendering is allowed after generation.

Checkpoint the transformer evaluations and VAE decode, retaining the full
trajectory gradient. Verify finite nonzero loss gradients on predictions at
steps 0, 24 and 49; input-reference gradients; changed rank-16 adapters; and
frozen base, VAE and recognizer parameters without gradients. Run two real
optimizer updates on the first Chinese and English examples. Use FP32 AdamW
masters, learning rate 1e-5, decay .01 and global clipping 1. Record both actual
before/after pages. A decreased loss is an execution observation, not a
language or writing qualification.

Use GPU0 alone with the existing project lock. The allocated-memory limit
is 22 GiB. Abort on nonfinite gradients or exceeding that limit. Reuse the
existing source weights and environment. No second heavy job is permitted.

## Bounded learning pilot and complete evaluation

Conditional on numeric and gradient preflight, start again from the declared
checkpoint and perform 32 new updates on the same first 32 copying pairs.
Sample with replacement from CPU RNG seed 20268402; sample each independent
noise seed from CPU RNG 20268403, in [0,2^31-1). Record every ID and noise seed.
Model seed is 20268401. The fixed budget is a learning pilot, not permission
to declare the full ILM objective complete or to select a favourable checkpoint.

Generate the entire existing 64-case input-only cohort from the final pilot
checkpoint in a fresh native pipeline process, with the original seeds and
settings. Score every raw output with the original independent CPU evaluator
and unchanged reference calibration. Review all raw images. Compare to the
starting checkpoint's 6/32 seen and 9/32 held-out OCR observations, while
retaining their evaluator limitations. Do not replace this score with the
training reader's more permissive transcripts.

Writing admission remains at least 30/32 exact legible requested strings alone
per split. At least three additional held-out OCR-exact outputs and no seen
regression would justify considering a separately budgeted longer learning
run; this is an exploratory progress rule, not writing admission. Otherwise
this pilot does not earn unchanged scale-up. No natural QA is optimized in
this pilot; meaningful unseen instruction answering, wider forms and efficient
language learning remain explicit subsequent requirements of the ILM goal.
