# Recent alignment and latent reasoning: implications after V62

Primary-source follow-up reviewed on 5 November 2026. These works suggest
training mechanisms; none establishes an image-language singularity or a
single-4090 general language model. V62's rejected primary result is unchanged.

## Transferable findings and deployment boundaries

| Work and version | Finding | Limitation for ILM |
| --- | --- | --- |
| [PixelDense, 30 September](https://arxiv.org/html/2610.00483v1), with [released model card](https://huggingface.co/xsourse/PixelDense) | Separate semantic/geometric projections and a row-space orthogonality penalty improve natural-image generation over summed teacher losses. Four frozen teachers and the alignment projections are removed after training. The matched GenEval score rises from 0.7927 to 0.8093. | Its recipe uses two H200s, 10,000 updates and effective batch 256. The 1.1B released denoiser still requires a Qwen3-1.7B text encoder during inference. Its natural-scene depth/normal teachers are not writing-semantic teachers. This is not a complete image-input ILM. |
| [LARC, 30 October](https://arxiv.org/html/2610.42149v1), especially Sections 2.2 and Appendix A.1 | Continuous hidden-state steps supplement a supervised curriculum and information-gain RL. A frozen reference estimates shared-noise flow-loss improvement, with a diversity term to discourage redundant latent trajectories. | The inference trajectory alternates language spans and latent steps, retaining text and special tokens. Training uses eight A100s and approximately 300 plus 96 GPU-hours, excluding BAGEL pretraining. Its result does not demonstrate removal of language tokens or training on one consumer GPU. |
| [Visual Prompt Engineering, 14 October revision](https://arxiv.org/abs/2606.04457v2) | An internal semantic plan helps downstream image generation while preserving direct access to the original input. | The studied tasks are image generation/editing with multimodal backbones, not autonomous glyph-language learning. A semantic-only intermediate can lose details; retain the physical input path. |

## A narrow hypothesis for the next ILM experiment

The completed V62 mixture learns better bitmap density but no resolved
question-dependent answer gain. Additional local image modes are therefore
insufficient at that budget. The next hypothesis should supply an explicit
training signal for the meaning of the requested answer, alongside exact
writing structure, while keeping the deployed actor image-only.

One candidate uses **one offline semantic teacher and a target-image
structure loss**, rather than importing four natural-scene teachers. Let
`h0` be the student's causal state before its answer and `ht` its state before
answer block `t`. Cache a teacher target `Ts(X,Y)` for the full requested answer
on training examples only. Supervise a small projection of `h0` toward that
target. Separately supervise structure predicted from `ht` against actual
next-block pixels or their lossless field representation. The known answer
is a training target; neither the answer nor its teacher feature enters the
deployed forward pass.

An illustrative objective is

\[
L=L_{\rm image\ NLL}
 +\lambda_s d(P_s h_0,\operatorname{stopgrad}T_s(X,Y))
 +\lambda_v d(P_v h_t,T_v(Y_t))
 +\lambda_o\|\bar P_s\bar P_v^\top\|_F^2.
\]

This is a proposal, not a tested result or a claim of a new theorem.
The final term discourages overlap of the linear readout spaces. Even exact
row-space orthogonality does not guarantee independent gradients throughout
the nonlinear shared backbone or meaningful semantics. A teacher selected for
natural-image invariance may discard a decisive stroke or negation, so it
must first pass writing-specific calibration. No new weight, learning-rate
choice or quality gate is selected from V62's pending auxiliary transcripts.

Teacher preparation and student training would run sequentially on GPU 0.
Cached features can keep the teacher out of student-training memory and final
inference, with preparation cost and external pretraining reported separately.
The complete-line training cache already provides actual English and Chinese
images, but no new semantic-feature cache or training result exists yet.

Before production, specify the exact teacher, feature extraction, losses,
cached-target hashes, causal masking, matched control, memory preflight and
fixed held-out questions. Compare answer quality and normal-minus-blank gains,
retain independent offline deployment replay, and keep the final sealed split
unscored during development.

## Why self-rewarding is not the immediate shortcut

A reference can reward an internal step that increases its own likelihood
without improving the actual answer. V62 already demonstrates that the two
can diverge. An unqualified reference would risk reinforcing its own errors.
Rollout training remains a candidate after a trustworthy conditional reward is
established; the reference must stay frozen within the comparison and cannot
replace independent question controls. A long latent trace, lower pixel loss
or a clearer numeral is not evidence of correct reasoning.

## Current implementation choice

The repository's earlier V37–38 experiments already tested pooled semantic
targets and left a large gap between training and held-out answer planning.
Before adding another teacher representation, V63 tests a simpler signal:
the free-image generator's likelihood must assign two accepted answer images
to their own questions about the same paragraph. The paired contrast cancels
question-only and answer-only energy offsets. It is applied alongside positive
image likelihood, with a matched no-contrast arm. This is a training signal,
not a proof that the output will be meaningful.

The [new protocol](question_assignment_v63_protocol.md) retains actual free
generation, blank-question controls, shared-paragraph pairs and independent
offline deployment as required evidence. The real preflight passes and
production starts on GPU 0; no new language pass is established. Semantic
feature supervision above remains a candidate, rather than a claimed result
or an external-model dependency silently added to the actor.
