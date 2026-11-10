# Separate the first decision, coherent glyphs and autonomous feedback

The completed paired run reaches 16/32 clear complete answers on its reserved
questions, versus 11/32 for a fixed-body control. It fails both the complete
answer pilot and the training-pixel gate. The
[full result](glyph_tail_paired_answers_result.md) supplies the denominators,
raw outputs and limitations. Three different problems remain; a single claim
about diffusion, model size or source-token mismatch does not explain them.

**First decision.** Training first-glyph F1 is 0.893582, below later
reference-conditioned F1 of 0.974584. Generated feedback has not happened at
the first position. The previous schedule also averaged only 4.25 question
visits. The active repair freezes the learned body, caches its actual states,
and fits G/stopping in batches with equal first/later image-loss weights.
This directly tests exposure and loss allocation. It changes no runtime
representation and adds no character classifier. Success requires new-context
answers, not a low training loss.

**Coherent glyph selection.** An image model can minimize per-pixel loss while
producing a bitmap that is not any plausible character. For ordinary BCE,
with hidden state h and binary ink pixel X_p, the unrestricted marginal optimum
is

\[
 q_p^*(h)=P(X_p=1\mid h).
\]

For constant foreground/background weights w_1,w_0 it instead becomes

\[
 q_p^*(h)=\frac{w_1P(X_p=1\mid h)}
 {w_1P(X_p=1\mid h)+w_0P(X_p=0\mid h)}.
\]

Neither expression represents dependence between pixels. Thresholding uncertain
stroke marginals can combine incompatible forms. The implemented batchwise
foreground balancing has data-dependent normalizers, so it is not exactly
this constant-weight formula; the missing joint objective remains. Conversely,
unique physical training inputs and damaged output do not prove irreducible
multimodality: inadequate fitting is still a competing explanation.

The repository's earlier [joint exact raster experiment](joint_exact_raster_v55_result.md)
already used a normalized dependent pixel law and still failed its visible
language-generation requirements. The [released MIXAR baseline](mixar477_baseline_result.md)
and subsequent native instruction fit also failed useful Chinese answering.
The marginal argument describes the present glyph loss; it does not explain
every earlier model or establish that adding a joint pixel decoder is the
missing solution. A new intervention must use and test the actual language
states, rather than repeat those failed motor-only arguments.

[PIXAR (ACL 2024)](https://aclanthology.org/2024.findings-acl.874/)
documents noisy pixel-language generations under its likelihood objective
and uses adversarial training. This motivates checking a coherent-generation
objective, not declaring BCE mathematically incapable of learning language.

**Autonomous feedback.** Later reference-conditioned states see correct
previous images. At deployment the transition is instead

\[
 h_{t+1}=F(h_{\le t},E(\hat I_t)),\qquad
 \hat I_t\sim\pi_\theta(\cdot\mid h_t).
\]

The resulting state distribution depends on the writer. Improving a fit on
fixed reference-history states alone cannot establish stable rollout. The
earlier source-head feedback test diagnoses changed sequence advancement;
the paired learned model nevertheless answers some new questions correctly,
so the diagnosis is not an impossibility theorem.

The October 2026 revision of
[MIXAR](https://arxiv.org/html/2604.11575v2) adds free-running GRPO training.
Its 32x32 variant uses ground-truth tile/pixel rewards rather than a
discriminator; its appendix reports that the discriminator alternatives harm
that geometry. This is a specific follow-up candidate if the current
supervised repair leaves rollout failure. The paper's selected checkpoints,
large pretraining run and evaluation protocol are separate from this
repository's small fixed-trial evidence. Its results are not ours.

A bounded subsequent test could sample short actual-pixel rollouts, reward
whole-column correctness together with ink-sensitive fidelity to training
images, preserve a teacher-conditioned reconstruction anchor, and compare
against an unchanged parent on new contexts. Training-only pixel references
would not become a runtime Unicode or font path. Group rewards must vary,
blank images must not win by background area, stopping must be evaluated,
and every failed stream must remain visible. This test is **not implemented
or validated here**. A continuous conditional motor distribution is another
unimplemented option if factorized sampling itself proves inadequate.

The immediate sequence is deliberate: establish adequate first-decision
fitting, measure coherent whole forms, then train and test the actual feedback
distribution. No physics analogy substitutes for those measurements, and no
small adapted parameter count removes the inherited body or pretraining cost.
