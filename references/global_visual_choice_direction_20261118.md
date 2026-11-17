# Candidate: supervise the global choice of visible form before local drawing

The [occupancy comparison](occupancy_density_preflight_result_20261118.md)
is complete and fails. Its hierarchy reduces source-code events and improves
physical F1 over the flat law, but draws dense damaged forms with zero exact
masks. The flat law usually draws one or two pixels. Both use correct source
history and numerically verified probability replay. Their nonzero conditional
NLL differences do not establish correct global glyph selection.

The next candidate adds **training-only whole-image discrimination** to the
actual shared condition used by the physical writer. Keep the trained source
actor frozen and retain the occupancy architecture and its normalized image
law. This changes the learning signal on its global conditioning features;
it does not add a new output alphabet at inference. An ink-count constraint
could address the opposite density errors, but would not establish which form
the prompt requires. It remains an untested alternative, not the selected fit.

## Supervision and causal boundary

Build temporary equivalence classes from the exact binary masks in the existing
training-only prepared images. Identical physical masks share a label, including
the empty mask. Retain original Unicode aliases and source images without
relabeling them. No development output, reserved reference, font rendering,
teacher winner or rejected judge enters this class construction. It is source
image supervision, not a new source/image alias audit or semantic reward.
These are identities of visible forms, not meanings. A different font can
define another class, so this initial canonical-style test cannot establish
font invariance or historical semantic transfer.

Let h be the existing pre-target context, i the physical index within a known
span and L its length. Let s=C_theta(h,i,L) be the same contextual feature vector
added to every physical event by the image writer. A temporary nonlinear
classifier r_eta(c|s) predicts the complete target's physical-image class **before
any current-target mask enters the model**. Use a two-layer classifier on s;
physical position/context interactions need not be linearly separable.

For K training classes, a candidate objective is

\[
 \mathcal L=\mathbb E\left[
 -\log q_\theta(Y\mid h,i,L)/1024
 -\log r_\eta(c(Y)\mid C_\theta(h,i,L))/\log K
 \right].
\]

Freeze this coefficient and normalization before a prospective execution;
do not tune them on the completed support or development answers. Both terms
use complete authentic targets. The second term is an auxiliary classification
objective, so their sum is not described as the image law's exact likelihood.
The deployed image law q remains normalized and draws its own root, topology
and pixels. The helper classifier, class mapping and training targets must be
discarded from the deployment path. No classifier argmax may select a source
image, font or stored glyph. Training uses categorical image supervision;
it must not be described as wholly free of discrete labels.

For fixed S, expected classification cross-entropy decomposes as

\[
 \mathbb E[-\log r(C\mid S)]
 =H(C\mid S)+\mathbb E_S D_{KL}(p(C\mid S)\|r(C\mid S)).
\]

The intended finite-budget effect is to make the global condition discriminate
complete alternatives. Ordinary complete-image likelihood is already a proper
objective; local teacher forcing is not a probability error or illicit target
leak. The extra task changes gradient emphasis and inductive bias. It neither
creates knowledge absent from h nor guarantees that the physical decoder uses
the resulting features. High auxiliary accuracy with bad images is a failure.
Memorized training identities are not language generalization.

## Differences from completed interventions

The earlier [contrastive readout](contrastive_glyph_readout_protocol.md) adds
cosine-anchor classification to MSE/BCE for a separate readout G, with a frozen
primitive decoder and synthetic teacher-winner calibration states. It fails.
Here labels come directly from actual source masks, the global condition is
shared with a learned normalized physical writer, and no synthetic state,
teacher argmax or latent-anchor regression is added. The same fresh physical
initialization and real complete-source order are required.

The [canonical binding trial](canonical_glyph_binding_v44_result.md) changes a
small predictive residual toward fixed image-field differences and loses
conditional binding. Its shifted cosine scores do not qualify global choice.
The new candidate does not retrieve an evaluator candidate bank at inference
or optimize the failed raw normalized DCT target.

The [recognition-loss failure](autonomous_writing_loss_result.md) uses a reader
on generated/assisted images and can overlook visible duplicates. This proposed
helper sees only a **pre-generation condition**, receives exact training-image
labels and never scores generated images. It supplies neither a policy reward
nor a replacement judge. The completed predictive-KL test uses native vocabulary
probabilities; this candidate uses no native lexical head or teacher distribution.

Auxiliary hidden-feature supervision is established methodology; see
[Deeply-Supervised Nets](https://proceedings.mlr.press/v38/lee15a.html).
[REPA](https://arxiv.org/abs/2410.06940) supplies separate image-generation evidence
for representation supervision, using pretrained visual features. Neither paper
proves this image-identity objective, visual language understanding or consumer-GPU
foundation training. The proposal is an engineering hypothesis, not a new
learning theorem or physical law.

## Execution decision

First implement source-only class construction, prefix-independent helper
features, gradient reach to the actual shared condition, inference-without-helper
and exact reset checks. Use the completed 96-update tree run as the unchanged
control rather than rerunning it. Bind the same base initialization, three
whole records, source order, optimizer, sampling, all 552 physical positions,
helper initialization and resources before a real candidate preflight.

Require both complete source-form discrimination and actual writing improvement;
keep the original physical and full-language conditions. If discrimination
improves without usable images, reject this as a sufficient writing repair.
If neither improves, do not launch a larger fit or assume that a new feature
name solved conditional language. Preserve all outputs and earlier successes.
Implementation and numerical checks are the next work. Source preparation and
any real trial need their own prospective bindings; no neural experiment or
full fit is registered by this design note. Full independent language,
historical forms, stopping, untouched confirmation and matched-quality
efficiency remain unproved.
