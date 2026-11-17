# An algorithm for learning language through images: evidence and next decision

The project has not yet demonstrated a reliable image-input, image-output
language model. Joint supervised adaptation of the visual reader, contextual body
and conditional flow writer has now **failed its complete registered comparison**.
Both learned and frozen context give zero readable complete normal QA and zero
credited coherent complete long books. All 792 streams were reviewed and readings
frozen before reference scoring. The
[completed result](trainable_visual_context_result_20261117.md) supersedes this
assessment's earlier pending status.

The [failure decision](trainable_visual_context_failure_decision_20261117.md)
selected training-only gradient accounting before another objective or architecture
intervention. That [diagnostic is complete](visual_context_gradient_result_20261117.md)
and does not establish its opposing-update criterion. The subsequent
[saved-output analysis](visual_context_saved_dependence_result_20261117.md)
finds prompt-dependent variation alongside recurrent damaged first tiles.
The [completed predictive-KL update/reset preflight](predictive_state_distillation_result_20261118.md)
fails all support checks: pooled glyph F1 falls to .307754, below initialization
and the cosine control, and book KL increases. All twelve trial updates are
reset. This objective replacement does not qualify a working algorithm or a
full fit. The
[engineering goal](../docs/first-imagized-language-model-goal.md) retains the full
language, historical-form, independence and efficiency requirements. The model
and research discussion below describe the rejected recipe and conditional
decision rules; they are not evidence of achieved useful visual language.

Later [head calibration](nonlinear_feedback_alignment_result_20261118.md) and
[direct readout fitting](feedback_ridge_refit_result_20261118.md) lower native
coordinate error to .271039 on complete fitting support, while still failing
the unchanged .10 reader gate. This is not a new language score. The
[exact-query floor](frozen_query_identifiability_result_20261118.md) is .016775;
identical-query ambiguity does not by itself rule out that gate. The
[profiled feature-learning comparison](profiled_feedback_fit_result_20261118.md)
now completes twelve updates and reduces error to .228074131, still failing
all fixed reader-support conditions. Its
[actual image consequence](reader_writing_consequence_result_20261118.md)
does not improve pooled physical agreement on the three complete training
records (.353468785 control, .353352368 candidate). No autonomous language
score changes. The
[writer-layer conditioning comparison](writer_layer_conditioning_result_20261118.md)
now completes its real twelve-update-per-arm support test and also fails.
Actual pooled F1 is .339850462 for modulation and .342911305 for unchanged
conditioning, from the identical .348600259 start. The short QA improves, while
book and long QA deteriorate against control. All trial updates reset exactly;
all 2,208 emitted tiles are recounted and exported. No working algorithm or new
full fit is qualified. Another unchanged fit is not selected.

The subsequent [occupancy-density comparison](occupancy_density_preflight_result_20261118.md)
uses a lossless hierarchy and an equal-parameter flat control with the same
explicit empty/nonempty root. Eight CPU checks and all 68,948 saved-support
cell roundtrips pass. Both real heads complete 96 whole-record updates/reset.
Tree F1 reaches .455980 versus flat .004023, but both have zero exact masks and
ink at all four true spaces. Every generated image and physical probability
trace is recounted; no language is qualified. Opposite ink-count errors remain
despite correct probability replay. The next
[global visual-choice supervision candidate](global_visual_choice_direction_20261118.md)
is unimplemented. Its helper would supervise complete source-form identity
before local drawing and be discarded at inference. It requires separate
source/gradient/reset and actual-writing qualification, not a larger unchanged fit.

## The failure that must be solved

The [completed compact diagnostic](compact_visual_span_diagnosis_result.md)
found malformed first answer spans even with correct preceding images and
supplied next-span lengths. On its selected first spans, both models have zero
exact training and development glyphs. No generated answer has entered feedback
at that point. Thus accumulated feedback error cannot be the sole cause.
Direct codec reconstruction on the same selected targets is almost exact;
64 sampling steps do not repair conditional writing. Neither a better-looking
diagram nor more unchanged sampling establishes a remedy.

There are also failures after writing becomes substantially more readable.
The [completed white-target comparison](white_target_physical_result.md)
passes all seven readability groups but fails whitespace: 12/252 false-ink
positions and 54 false blanks, against limits of eleven and 37. Its autonomous
language was not measured. The last completed
[continuation language trial](visual_continuation_learning_result.md) gives
7/32 strict proportional answers, 0/16 both-correct question pairs and no
credited coherent complete long-book continuation. These are different actors
and tests; their scores must not be merged into a single progress curve.

The reader, conditional choice, physical writing, feedback and stopping must
work together. Existing evidence rules out treating any one of the following
as a sufficient solution: codec reconstruction, a wider flow head, a joint
pixel likelihood, more accurate marginal pixels, or a lower training loss.

## What published algorithms establish

Only primary papers are used here. Their reported results are external evidence,
not measurements of this implementation.

| Work | Relevant evidence | Limit for this project |
|---|---|---|
| [PIXAR, 2024](https://arxiv.org/html/2401.03321v2), sections 3 and appendix D | A decoder-only transformer generates rendered language directly as pixels. Adversarial refinement improves writing over its pixelwise likelihood baseline. | This establishes prior pixel-input/pixel-output generation, not a novel first ILM. Its reported pretraining uses sixteen V100s for about four days. The existing local adaptations already failed; repeating the name or recipe is not a new intervention. |
| [MIXAR v2, October 2026](https://arxiv.org/html/2604.11575v2), sections 3--4 and appendix K | Multilingual pixel generation uses larger patches for CJK and reinforcement learning on free-running outputs. | Chinese free-generation remains weak and declines slightly after its second stage. Its already-tested local release is unchanged. Neither its external training budget nor its OCR-based rewards qualify our rejected judges or establish one-card foundation training. |
| [UniCanvas, June 2026](https://arxiv.org/html/2606.04264v1), sections 3--4 and appendix 1.3 | Diffusion learns to put text into a writable image canvas, with explicit preparation for text rendering. | The implementation retains tokenized instruction conditioning and a large pretrained image-editing backbone, with substantial multi-GPU training. It is not evidence for an independent, image-only language learner trained from scratch on one 4090. The main text and appendix name different accelerator types; no reconciled compute estimate is inferred here. |
| [UniModel, November 2025](https://arxiv.org/html/2511.16917v1), sections 3--4 | Painted text and natural images share a visual generative formulation. | Its evidence is primarily qualitative; the paper explicitly retains distorted and misspelled text among its limitations. Its visual unification does not demonstrate reliable complete Chinese answers or the required resource budget. |
| [MEGABYTE, 2023](https://arxiv.org/abs/2305.07185), [BLT, 2024](https://arxiv.org/abs/2412.09871), and [Fast BLT, May 2026](https://arxiv.org/html/2605.08044v1) | Local encoders/decoders and a slower global context can reduce the cost of processing fine-grained sequences. Fast BLT also studies parallel drafts with autoregressive verification. | Bytes encode text, rather than observed glyph pixels. Applying the hierarchy to images is an engineering inference. These results neither certify our writer nor imply that removing Unicode removes language-learning cost. Fast BLT's large training corpus is explicitly unlike our local adaptation corpus. |

The useful common principle is to allocate substantial computation to language
context while handling physical detail locally. The research does **not**
establish that diffusion, adversarial training, reinforcement learning or a
particular hierarchy automatically learns semantics from glyphs.

## A complete visual causal model

Let X be the observed prompt image and Y(t) the next emitted image span. Define
a visual reader E, causal contextual model F and conditional writer W:

\[
 h_t=F_\theta(E_\phi(X),E_\phi(Y_1),\ldots,E_\phi(Y_{t-1})),
 \qquad Y_t\sim W_\psi(\cdot\mid h_t).
\]

At deployment E receives the **actual emitted pixels**, including errors.
The writer must not receive its current target as context. A sampled image
must not be silently replaced by a reference image, source embedding, font
render or nearest catalogue entry before feedback. A stopping decision is part
of the process, and reaching a cap is distinct from stopping naturally.

One explicit output law uses a local autoregressive writer. Losslessly partition
a 32-by-32 binary glyph into 128 physical 2-by-4 pieces b(t,j). Each piece has
256 possible eight-bit patterns. For T glyphs and a natural stop,

\[
 p(Y_{1:T},\mathrm{stop}\mid X)=
 \left[\prod_{t=1}^{T}(1-s(h_t))
       \prod_{j=1}^{128}q_\psi(b_{t,j}\mid h_t,b_{t,<j})\right]
 s(h_{T+1}).
\]

The patterns are physical pixel values, not character identities. Every binary
32-by-32 image is representable; no Unicode inventory or font lookup is needed
to unpack it. That representational support does not imply readable generation
or understanding of an unseen script. A normalized local density also does not
guarantee that the overall process stops almost surely. Nontermination and
cap-censoring must remain visible.

This factorization is standard probability, and a
[joint physical-pixel writer](../ilm/visual_lm/glyph_joint_density.py) already
exists and has failed qualification here. It is included to specify the target
problem, **not** to propose repeating that completed experiment. The completed
compact comparison retained its conditional flow writer. It
tested whether adapting the entire contextual body improves this integrated
visual process, without simultaneously changing the output law.

A flow field trained with squared error is not inherently restricted to a
single blurred answer: with adequate conditioning, capacity, optimization and
integration, it can represent a multimodal distribution. Conversely, exact
autoregressive likelihood does not guarantee useful greedy outputs. The earlier
failures support neither a blanket impossibility claim for flow nor an automatic
success claim for a joint density.

## The tested algorithm

The [implemented comparison](../docs/trainable-visual-context.md) performs the
following operations on complete real QA and book sequences:

1. Encode source glyph images with the verified fixed codec and a learned
   span reader. Predict its grouping and number of continuous context updates.
2. Process those updates causally through the contextual transformer. The
   treatment learns all contextual weights; the matched control freezes them
   while retaining gradients to the visual reader.
3. From each eligible prefix state, learn the next span's image distribution,
   length and terminal decision. Current/future target pixels never condition
   their own prefix state.
4. Use native feedback vectors and saved teacher states only as offline
   supervision. Keep the original six objectives and their weights fixed for
   this comparison. No rejected reward judge supplies learning signals.
5. Restore each prescribed deployment checkpoint and generate complete answers
   from prompt images, rereading every emitted span. Freeze literal readings,
   uncertainty and coherence assessments before new reference scoring.

The deployed model has 1,465,381,142 parameters, including its inherited
contextual body. It inherits text pretraining. The comparison is visual
adaptation of that knowledge, not image-only pretraining from scratch. The
single-GPU implementation also uses substantial host RAM for optimizer moments;
its resource account must include that memory and offline teacher preparation.
All 13,785 sequences and 8,172 updates per arm remain prescribed. A falling
flow loss or teacher-state loss is not an autonomous language score.

The earlier compact reader has a measured native-coordinate alignment
restriction. Allowing its contextual body to learn may permit another useful
representation despite that restriction; this is an inference, not a proof.
Teacher-state cosine alignment likewise does not imply correct next-image
probabilities. The completed complete-answer comparison rejects the resulting recipe.

## The decision after the comparison

The observed outcome is the fourth row: both arms fail conditional writing.
Reject the complete recipe and preserve the original results.

Use the final checkpoints and all fixed intermediate observations as registered;
do not select whichever checkpoint looks best after inspection. Retain all 792
prescribed streams and both separate cold replays. Do not fit on the untouched
confirmation pool or repair uncertain readings with answer knowledge.

| Observed final result | Consequence |
|---|---|
| Treatment passes the fixed complete-language conditions and control fails | Evidence for full contextual adaptation on this development task. Execute the separately bound untouched-source confirmation before a broader claim. Historical forms and matched-quality efficiency remain unproved. |
| Both arms pass | Report both successes and the actual paired differences. Success alone does not establish that contextual adaptation was necessary. |
| Treatment fails and control passes | Reject the claim that this contextual adaptation improves the tested task. Preserve the working control and investigate the measured deterioration before another fit. |
| Both arms fail conditional writing | Reject the complete recipe. Reuse the existing negative controls; a new intervention must isolate a specific remaining conditional-learning failure before any larger fit. |
| Writing becomes readable but answers or books fail | Treat language selection, continuation and stopping as unresolved. Glyph fidelity cannot replace the language conditions. |

These rules supplement rather than replace the original registration. They do
not authorize an extra diagnostic model run during the active serial comparison.
Any later intervention needs its own sources, unchanged control, training-only
preflight, fixed evaluation and resource ceiling.

## What would count as finding a working algorithm

The first useful result is complete, question-dependent Chinese answers and
coherent book continuation from actual image histories, with retained failures,
natural stops and untouched-source confirmation. The pilot floors remain
12/32 strict proportional answers, 4/16 both-correct pairs, 24/32 readable
answers and 4/12 coherent complete books, together with the original retention
and erasure requirements. Passing them is a bounded development success, not
the full project goal.

Replacing text by an injective image representation preserves the information
in that text; it changes the computational representation, not the amount of
semantic information that must be learned. The
[existing mathematical note](glyph_readout_and_clock_math.md) states this
distinction. No reviewed theorem proves frontier-level language ability,
thousandfold savings or automatic understanding of historical forms on one GPU.
An efficiency claim needs useful matched quality against the named local text
baseline and the complete cost account. Until those measurements exist, no
working algorithm has been demonstrated.
