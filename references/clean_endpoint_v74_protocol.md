# Clean writing prediction with analytic noise contraction

The full-dimensional velocity control improves rendering but still fails its
strict noise-repeatability requirement. The 32-step control gives 63/64 exact
known-image renders. A post-hoc precision/integration diagnosis selects BF16
with 64 steps on the original draws, then fails its one fresh-noise
confirmation: 116/128 exact, worst foreground F1 0.876068. Those failures
remain unchanged.

This control replaces learned velocity with a prediction of clean writing
codes. It tests the same original Chinese block, with no prompt reader or
language claim. Clean-data prediction has prior foundations in
[Li and He, 2025](https://arxiv.org/abs/2511.13720); conditional probability
paths are developed in [Lipman et al., 2023](https://arxiv.org/abs/2210.02747).
The application, arithmetic choices and measurements here remain separate
experiments.

## Model and objective

Let y be the two concatenated frozen writing codes, epsilon independent
standard Gaussian noise, and z(t) = (1-t) epsilon + t y. The network
D(z(t),t,h) directly predicts y. Use width 1,536, three residual MLP blocks,
an identity-initialized noisy-input projection and an initially zero final
projection. Body contractions use BF16; the final projection actually
executes in FP32, preserving writing-coordinate precision.

The fixed training objective is

$$
 L = \mathbb{E}\frac{\|D(z(t),t,h)-y\|^2}
                       {\max(1-t,0.05)^2}.
$$

Time is uniform in [0,1]. The denominator is clipped only in the loss.
This is clipped velocity-space weighting with clean-code parameterization.
For each interior time, its positive time-only weight leaves the ideal
conditional mean estimator unchanged. This statement is not an equivalence
of finite-model training behavior.

The complete causal writer retains image-derived context, continuous
image history, balanced learned stopping and re-encoding of actual emitted
pixels. Its configuration identifies clean-endpoint prediction so that an
old velocity-only loader cannot silently misinterpret its weights.

## Endpoint update

Freeze the clean estimate over each integration interval. For t < s,

$$
 z(s)=\frac{1-s}{1-t}z(t)+\frac{s-t}{1-t}D(z(t),t,h).
$$

For a constant correct clean estimate this transports the Gaussian component
exactly. On the 32-step uniform grid, implement coefficients using the integer
number of remaining intervals. The last update has zero coefficient on the
old noisy value. No division is evaluated at t=1. This is an endpoint
prediction update, distinct from the earlier midpoint velocity integrator.
For a learned nonconstant estimator, finite-step error remains to be tested.

## Fixed known-image control

Keep the same first CMRC cached record and its first original 32-by-64 block
selected by the V73 protocol. Use zero 512-vector conditioning. The target
code appears only in supervised training and offline scoring, never as a
sampler input or an output correction.

Train from scratch for 4,000 updates, batch 64, model seed 20267400,
independent training generator seed 20267401, AdamW betas (0.9,0.95), weight
decay 0.01 and gradient clipping at one. The learning rate is 5e-4 times a
100-update warmup and cosine factor decaying to 0.1, as in V73. Save every
500 updates and resume only the identical source/configuration.

Evaluate once on all 256 fresh Gaussian draws from seed 20267402, using
32 endpoint updates and the frozen writing decoder. Apply the unchanged
image requirements: mean original F1 >= 0.995, every original F1 >= 0.99,
and at least 90% exact supplied-code renders. Retain every binary output,
initial noise, endpoint and normalized code. Recompute physical scores and
fixed-decoder outputs independently on CPU.

Passing establishes only repeatable generation of one learned image.
Prompt-conditioned language, previously unseen glyphs, semantic correctness
and complete autonomous answers remain unproved.
