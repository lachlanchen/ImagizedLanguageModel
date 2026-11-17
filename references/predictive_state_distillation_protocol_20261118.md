# Test prediction transfer rather than hidden-state angle

The completed full-context model produces no readable complete normal answers.
Its no-update diagnostic does not establish opposing auxiliary updates. Saved
outputs remain visibly prompt-dependent, alongside recurrent damaged first
tiles. These findings select one new training hypothesis, not a working model:
replace the teacher-state cosine loss by the teacher's predictive KL divergence.
Keep the visual reader, complete contextual body, flow writer, codec, stopping,
feedback, other five losses and their weights unchanged.

## Mathematical change

For an authentic target-preceding student image state h, existing projection P,
saved causal teacher state u and frozen native teacher readout U, define

\[
 p_T=\operatorname{softmax}(Uu),\quad
 p_S=\operatorname{softmax}(UPh),\quad
 L_{\rm predictive}=D_{KL}(p_T\Vert p_S).
\]

Both distributions cover the complete original teacher vocabulary, including
terminal/control rows. No top-k renormalization or sampled denominator is used.
The temperature and replacement weight are both one. The original five other
terms retain their values and gradients; the original state cosine is replaced,
not added. P has the same parameters and initialization as the original control.
The source head receives no gradient. The student state is still computed from
actual source images with causal gold-prefix history during fitting.

Cosine can equal one for Ph=c u with any positive c while softmax predictions
change with c. More generally, near u the KL curvature in state coordinates is
U^T[diag(p_T)-p_T p_T^T]U. This weights predictive directions differently from a
state-angle loss. It does not prove that cosine caused the observed failure or
that predictive transfer will make the image writer readable. The writer does
not directly see U or an output identity; the physical rendering problem may
remain after better prediction transfer.

The probability objective follows established
[knowledge distillation](https://arxiv.org/abs/1503.02531). Work on
[hidden-state matching](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2fb462e23667ad5e6471a4e9af8e4774-Abstract-Conference.html)
also supports alternatives to state cosine, but does not validate this image
model, KL replacement, corpus or one-card quality. This is an engineering
comparison, not a new physical law or a novelty claim.

## Difference from completed failures

The [contrastive readout fit](contrastive_glyph_readout_protocol.md) trained a
continuous form regressor against one class-forced semantic anchor, adding
cross-entropy over 4,087 anchors to its MSE/cosine/pixel losses. It used synthetic
and genuine cached states, froze the contextual body and decoder, and failed
both physical routes. Here no anchor target, synthetic state or frozen decoder
regression term is added. The entire existing image-to-context graph receives
the *complete conditional teacher distribution* through its existing projection.

The completed [joint writer](joint_writer_physical_result.md),
[context-gradient](joint_context_physical_result.md) and
[ranking](global_glyph_ranking_decision.md) comparisons changed physical image
likelihood/gradient reach or blank preference on the older 7.55B actor. This
test changes neither image probability law nor whitespace objective. Earlier
[native visual-access supervision](native_visual_language_learning_direction.md)
already tried hard answer-label cross-entropy on frozen-source visual adapters
and failed natural QA. That failure remains relevant: predictive supervision is
not automatically a repair. The present comparison changes a soft state-transfer
objective on the complete 1.465B context/reader/writer graph, using the same
authentic teacher states as its unchanged cosine control. None of these
differences supplies evidence of success before execution.

## First-order implementation and CPU qualification

`ilm/visual_lm/predictive_state_distillation.py` streams the frozen CPU readout
in 2,048-row blocks. A first pass obtains both complete log normalizers; a second
computes KL and its exact-arithmetic student derivative

\[
 \partial L/\partial(Ph)= (p_S-p_T)^T U/B.
\]

FP32 arithmetic is explicit even inside BF16 model autocast. The saved autograd
tensor is only a state-sized derivative; no vocabulary-sized graph or readout
parameter is retained on GPU. It supports first-order differentiation only.
The numerical CPU tests compare loss and gradient with dense softmax across
non-divisible blocks, temperatures and a single-category edge case, check
record weighting, unchanged inputs, rejected invalid support and exclusion of
the training head from parameter groups/state_dict. These are implementation
tests, not language experiments.

## Fixed real-source update/reset preflight

Use the final learned-context actor at update 8,172, with its exact saved
training-only projection. Preserve the original actor and all old evidence.
Reuse the three complete, already prescribed *training* records: the 513-glyph
book, 846-glyph QA and 1,042-glyph longest-native QA. No record is cropped or
replaced, and no development/DRCD source enters fitting. The teacher's 7.5B
readout is dequantized from the existing authenticated shared archive on CPU;
the teacher body is not loaded and no new teacher inference is performed.

Run the original cosine control and KL candidate serially from identical saved
weights, using fresh ordinary FP32 AdamW moments for each arm. Six updates per
arm cycle through the three complete records twice, with fixed seed 202611181
plus update index, image/projection learning rate 2e-4, context rate 2e-5,
betas (.9,.999), epsilon 1e-8, decay .01 and global clip one. There is no schedule
or early selection in this finite execution. Both arms use the existing
host-moment AdamW implementation. This is additional adaptation of an inherited
failed actor, not a foundation pretraining experiment or replay of an old step.

Before updates, verify that the replacement shares precisely the original
projection state/parameter names and gives identical other five losses under
RNG replay. Retain finite complete-record losses, gradients, clipping and actual
updates. At initial and final state, measure predictive KL and conditional
writing on *every* next-span target position in all three records, including
terminal length decisions separately. Use correct source-image history and
supplied next-span lengths to isolate writing, the same eight Euler steps and
per-span seed 202611182 plus record/span indices. Save unmodified binary outputs.
This physical assay does not generate autonomous answers or prove semantics.

The candidate earns only preparation of a larger matched comparison if: all
six updates execute; its final KL is below initialization for every record; its
pooled conditional writing F1 exceeds both initialization and the cosine
control by at least .05; and its false-blank count does not exceed either.
All complete positions stay in the denominators. Lower KL alone fails this
support rule. This finite rule is not a language acceptance threshold and does
not alter any original complete-language gate. A pass still requires separately
registered complete-source training/evaluation. On failure, preserve outputs,
reject this six-update preflight as sufficient support and do not launch a full
fit, extend unchanged epochs or tune these thresholds. It does not disprove
all predictive distillation or authorize another unregistered intervention.

Reset every model/projection update afterward and verify original tensor hashes.
No learned successor checkpoint is retained or deployed by this preflight.
U and all lexical supervision remain training-only. Export remains the original
image model with no lexical input/output matrix, tokenizer, OCR, font renderer,
teacher service or catalogue selection. This transfers inherited text knowledge;
it is not language learned from images alone.

## Ownership and resources

Commit and push implementation, protocol, CPU result and hash-bound registration
before neural execution. Reuse the project GPU0 mutex and verify one live owner;
run at most one serial job. Entry: 48 GiB available host RAM, 20 GiB free GPU0,
48 GiB free disk, PSI some <=.5/full <=.1 and two CPU threads. Runtime reserves
24 GiB available host RAM, caps allocation at 22 GiB, monitored time at 1,800
seconds and pressure waiting at 300 seconds. An external 1,860-second deadline
with thirty-second termination grace bounds cleanup. Record source/readout
hashes, whole model counts and actual costs; preserve any failure without an
automatic retry. Protect foreign GPU1 services and stop this owner on completion.
No working ILM, image-only foundation learning or matched-quality saving is
claimed from registration, CPU checks, local loss or a reset preflight.
