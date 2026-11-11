# One bounded intervention: train against decoded glyph geometry

The completed conditional-writing diagnosis finds residual physical errors
even on training states, especially later states: candidate median foreground
F1 is 0.974518, only 23/32 reach 0.95, and none is pixel-exact. Direct supplied
reconstruction is exact for all 32. First training forms are largely
recognizable, with median 0.986408 and all 32 above 0.95; the failure is not
an absence of learning. The large development gap, rare conditional targets,
native-reading differences and stopping errors remain separate problems.

Test one change to the current head's training objective: add a late-time
decoded-image endpoint loss. Preserve architecture, actual state cache,
normalization, data exposure, initialization, optimizer schedule, EMA,
sampler and deployment boundary. Freeze the inherited reader, adapters,
encoder, decoder and stop predictor. This is a physical-writing hypothesis,
not a claim that glyph sharpening alone will repair language.

For normalized motor target z, Gaussian noise e and uniform time t,

\[
x_t=(1-t)e+tz,\qquad
v^*=z-e,\qquad
\widehat z_1=x_t+(1-t)v_\theta(x_t,t,h).
\]

The control minimizes mean squared velocity error. The proposed objective is

\[
\mathcal L=\mathbb E\|v_\theta-v^*\|^2+
\mathbb E\!\left[\ell_{\mathrm{balanced\ BCE}}
  \big(D(\mu+s\widehat z_1), I\big)\mid t\geq0.75\right].
\]

Use coefficient 1.0 for the second term, fixed before training. D returns
ink logits. Balance foreground/background per image, then average over the
eligible images in the batch; an empty eligible set contributes zero. Keep
gradients through the frozen decoder to the motor, without updating decoder
weights. Do not introduce new random draws beyond the control's batch, noise
and time draws. Log both unweighted losses, eligible counts and gradient
norms. The endpoint is a training estimator, not a replacement sampling rule.

The local Bernoulli likelihood curvature in motor coordinates contains
J_D transpose times diag[p(1−p)] times J_D, with affine scale factors and
foreground weights. This follows by differentiating the decoded likelihood.
Euclidean motor error treats all coordinates equally; decoded loss can weight
directions by their visible effect. This motivates an experiment, not an
identified cause of the observed errors. Late-time restriction reduces, but
does not eliminate, the risk of encouraging averaged endpoints at uncertain
early times. The modified objective need not retain the original flow-matching
optimum or improve complete answers.

The baseline probability-path construction follows
[Lipman et al., Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
[Weijler et al., Latent Riemannian Flow Matching](https://arxiv.org/abs/2607.19120),
revised 5 November 2026, provides a distinct 3D example where the frozen
decoder's latent geometry matters. Its normalized-sphere construction is not
assumed here: this motor ends in GELU features and does not justify imposing
that manifold. Neither paper establishes the proposed ILM intervention or a
language-efficiency claim. This is an inference from our measurements and a
local derivative calculation, not a physics breakthrough or a novelty claim.

Train exactly 12,000 updates with the same 64 first and 64 later examples per
batch, seed 202611171, final EMA only, and the original learning-rate schedule.
Keep the raw loss history and the final/previous checkpoints. Register the
new source and input identities before execution. Use one GPU 0 worker,
48 GiB entry and 24 GiB runtime host reserves, at most 8 GiB allocation,
1,800 seconds per fit attempt and at most 600 seconds of pressure waiting
within that deadline. Preserve failed attempts and original controls.

Repeat all 128 positions from the completed physical diagnosis, with exactly
the same training seeds and supplied states. For each training pool, require
median F1 at least 0.99 and at least 30/32 scores at least 0.95. If either
training gate fails, reject this recipe as a sufficient physical repair;
do not tune it or sample alternatives from this development feedback.

If both gates pass, run the same complete 216-stream language comparison,
including both input layouts, erasures and books, with unchanged per-case
seeds and stopping. Freeze every whole reading before reference scoring.
The development language pilot requires at least 10/32 proportional exact
answers, 2/16 correct pairs and 16/32 readable outputs; retain the union of
the original six and the current seven exact proportional answers (eight
distinct cases), plus both original fixed exact answers. Report every book
failure; this short-answer intervention does not qualify broad book language.
A pilot pass still requires untouched source-group confirmation and the
full engineering acceptance criteria. A failure requires a different tested
mechanism, not indefinite continuation of this recipe.

This decision is written after the complete negative language comparison and
all 128 physical positions are reviewed. The new objective is not implemented,
registered or trained yet. Maintain the existing verified manuscript while
preparing one bounded results/figure revision from these completed comparisons.
