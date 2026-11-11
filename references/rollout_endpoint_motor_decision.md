# Train the endpoint actually reached by the glyph sampler

## Evidence and the one new test

The direct-context flow gives 7/32 proportional exact answers, retains only
five of the earlier six, and fails its language pilot. Its first/later
training medians are 0.986408/0.974518. The late-time decoded objective also
fails both physical gates. It supervises a one-step endpoint estimate on a
noise/target interpolant, rather than the endpoint reached during generation.
This difference is observable in the code; it is not an established sole
cause of the failed writing or language.

The earlier balanced deterministic writer already passes training fidelity
without improving new-context answers. Another deterministic writer or better
reconstruction alone is therefore insufficient evidence. The next test keeps
the existing conditional distribution and its entire inference sampler. It
adds supervision on its actual sampled endpoint, with gradients through all
32 integration intervals. No correct target is supplied to that rollout.

Two arms start from the **same completed original flow EMA**, not from the
failed decoded-loss model. Both receive 2,000 additional updates with identical
training positions, noise, times, optimizer and EMA. The control uses the
original velocity loss. The candidate adds one balanced decoded BCE term on
the full realized endpoint. This paired continuation distinguishes the added
objective from extra updates. It does not claim a new foundation model or
establish that writing repair will transfer to language.

## Math and deployment

Let h be an actual frozen-reader state, z a normalized training motor target,
e Gaussian noise and t uniform on [0,1]. Let S_theta(e,h) be exactly the
existing 32-interval Heun sampler with its final Euler step. The control is

    x_t = (1-t)e + tz
    L_FM = mean[(v_theta(x_t,t,h) - (z-e))^2].

The candidate minimizes

    L = L_FM + balanced_BCE(D(mu+s*S_theta(e,h)), I).

The BCE coefficient is 1. Foreground/background terms are normalized per
image and averaged equally, then over all batch images. D is the unchanged
frozen decoder returning ink logits. The sampled endpoint receives h and e
only, never I, z, source IDs, native answers or reference lengths. The FM
term alone receives its ordinary training target. Gradients pass through D
and **all** sampler intervals. There is no detached sampling prefix, truncated
gradient, straight-through binary estimator or custom adjoint approximation.
Binary pixels remain an evaluation/inference threshold.

For an integrator update x_(k+1)=F_theta,k(x_k,h), its parameter sensitivity
obeys A_(k+1)=d_x F*A_k+d_theta F with A_0=0. Reverse-mode differentiation of
the finite Heun computation evaluates this discrete chain rule. Recomputing
each deterministic integration interval during backward saves activation
memory without changing this derivative. This tests the finite deployed
solver, not the exact continuous ODE. The BCE remains a surrogate for writing
and cannot certify semantic correctness.

The probability-path construction follows
[Lipman et al.](https://arxiv.org/abs/2210.02747v2). Differentiating a full
generative sampling chain is demonstrated by
[Clark et al., DRaFT](https://arxiv.org/abs/2309.17400v2), ICLR 2024. Its
diffusion results do not establish success for this conditional motor.
Applying that chain-rule idea here is an experimental inference, not a
physics breakthrough or novelty claim.

Deployment remains the existing image-input, sampled-image-output actor.
It contains no new head, lexical table, answer bank, OCR or font renderer.
The reader, contextual adapters, image encoder, decoder, normalization and
stopping remain unchanged. Only the existing 24,477,696 flow parameters train.
Actual emitted pixels alone supply generated history. The inherited 7.522B
visual/contextual body remains a separate acquisition and runtime cost.

## Fixed data, training and resources

Reuse the authenticated 3,176 first and 20,086 later training states and
original motor targets. Development/native-helper answers never train either
arm. Starting checkpoint SHA256 is
`79160d69ecd8559a939cba144566610e61f3582b92e9ff3e8e141eb39bcdd976`.
The source-bound registration must bind every input and both initial weight
maps before either arm fits.

Both arms use seed 202611181, batch 32 (16 first plus 16 later), 2,000 updates,
AdamW at 2e-5, betas (0.9,0.99), weight decay 0.01 and gradient-norm cap one.
Use 100-update warm-up, cosine decay to 10%, EMA decay 0.99 and final EMA only.
Each arm has 32,000 first and 32,000 later sampled exposures. This is a
registered small-head intervention, not complete book pretraining. Reuse the
same Gaussian draw for the FM path and the candidate sampler. Checkpoint
recomputation draws no randomness. Independently verify initial tensors,
sampled-index digest and final CPU/CUDA random states between arms.

Before updating weights, compare the differentiable solver with the deployed
solver on the same actual training batch and noise. Checkpointed and ordinary
forward values must be bit-exact; CPU finite differences must verify an
actual network derivative. A frozen-decoder gradient test must reach flow
parameters without creating decoder gradients. Tests establish numerical
execution, not writing or language success.

Run one GPU0 worker under the existing mutex, with 48 GiB available host RAM
at entry, 24 GiB during execution, and at most 8 GiB allocated for the head.
Control fitting has a 1,800-second ceiling; candidate fitting has 3,600 seconds,
each including at most 600 seconds pressure waiting. Do not prolong a failed
attempt or change update count from observed quality. Save optimizer/RNG and
latest/previous checkpoints every 250 updates and retain failed attempts.
The full reader is not loaded for either fit or cached-state assessment.
Foreign GPU services remain untouched.

## Decision rules and the real language test

Assess final EMA from both arms on all 128 previously fixed positions with
the original seeds and targets: 32 first and 32 later training positions,
plus all 64 normal development first states. Sample each once with the
unchanged 32-step deployed solver, bypassing stopping only in this labelled
one-glyph diagnosis. Retain parent, original-flow and supplied-codec controls.
Independently recount every pixel score and inspect every original tile.
For the candidate, each training pool must have median foreground F1 at least
0.99 and at least 30/32 tiles at F1 at least 0.95. Report the matched control's
gates as well. If the candidate fails either gate, reject this recipe as a
sufficient physical repair and leave its whole-answer quality unmeasured.
Do not choose raw weights, another checkpoint, seed, coefficient or a subset.

If the candidate passes, run **both** final arms on all 216 unchanged complete
streams, including both input layouts, erasures and the original books.
Preserve the old parent and original-flow language controls. Use actual pixel
feedback, original per-case seeds, stopping and cap 128. Freeze whole-output
readings before reference scoring. The candidate pilot requires at least
10/32 proportional exact answers, 2/16 both-correct pairs and 16/32 readable
answers, retaining the union of eight exact proportional cases from the
parent/original flow and both original fixed exact cases. Also report and
compare every matched-control outcome; a candidate must not lose its matched
control's exact cases or reduce its complete exact/pair/readability counts.
At least one complete-answer or both-correct count must exceed that control
before attributing a language gain to this objective. Report stopping, every
book failure, all erasures and costs regardless of the pilot result.

A pilot pass requires untouched source-group confirmation and independent
review before any general-language claim. Broad book continuation, historical
forms and matched-quality efficiency remain full-goal requirements. On
failure, retain all controls and make a different evidence-based decision.
Do not repeat this fit or let more publication work substitute for the next
unresolved language experiment. The verified current manuscript remains the
negative-result baseline until a new completed result justifies revision.
