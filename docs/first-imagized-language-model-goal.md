# Imagized Language Model: Engineering Goal

Updated: 2026-11-17. **Active; usable visual language remains unproved.**

## Objective and deployment boundary

Build a useful, independently running language model that reads visible writing,
understands the supplied prompt and generates readable answer images on one
RTX 4090. Its causal history must include its own emitted pixels, and it must
learn when to stop. Begin with modern-Chinese questions, passages and book
continuation; this development stage does not complete the full objective.

The full target includes simplified and traditional Chinese, historical forms,
handwriting and writing absent from Unicode. The eventual word-origin application
accepts Chinese or English questions and produces explanatory writing with
attested historical forms and provenance. Generalize the visual representation
to other writing systems, including cuneiform, after useful modern-Chinese
language is established. Raw pages, three-dimensional glyph streams, movies and
audio remain later extensions of this language objective.

Typed prompts may be rendered into images at the input boundary. Offline
teachers, OCR, text corpora and inherited foundation weights are allowed for
preparation and training; disclose their contribution and cost. Deployed answers
must come from the trained model: no live teacher, tokenizer, lexical output
head, answer lookup, OCR or font renderer in answer generation. Optional
transcription may read a finished image without changing its pixels. Transferring
inherited language capability and learning language from images alone are
separate claims.

## Current model decision: full-context recipe rejected

The [complete full-context comparison](../references/trainable_visual_context_result_20261117.md)
is terminal and fails. Both arms complete all 13,785 records and 8,172 updates;
all 792 prescribed streams, 180 native boards and 144 normal-book visible
prefixes have actual review receipts. All readings were frozen before reference
scoring. Final learned and frozen context each give **0/32 strict complete QA,
0/16 both-correct pairs, 0/32 readable complete QA and 0/12 credited coherent
complete long-book continuations**. Every prescribed snapshot has zero strict,
paired and readable normal QA. Do not choose an intermediate checkpoint.

Treatment's twelve normal long books stop naturally after 209–415 emitted glyph
positions; the control stops on two blank books and caps on ten. This establishes
a stopping difference without usable complete writing. Overall review retains
741 uncertain streams and 51 blank streams. Unreadability prevents credit and
assessment of latent semantic correctness; it is not evidence of a particular
wrong semantic answer. One model reviewer had prior development familiarity;
there was no independent human adjudication. Never repair frozen readings from
references, source weights, OCR, fonts or answer images.

Both original CPU fit audits and both separate strict cold replays pass. Each
cold replay checks the same one fixed final book input, outside the 792 language
streams, with exact pixels, stopping and feedback and no undeclared artifact,
font/tokenizer/corpus or network access. Neither execution nor natural stopping
qualifies language. Preserve the earlier failed admission and cold executions.
The original immutable controller status predates the separate completed review;
current authority belongs in the private handoff.

The whole deployed model has 1,465,381,142 parameters, including 1,409,410,048
inherited contextual parameters. Treatment fit takes 60,127.268 monitored seconds
with 11.647 GiB peak allocation; control takes 4,309.029 seconds with 7.200 GiB.
These fit scopes exclude inherited pretraining, earlier teacher preparation,
all generation and failed executions. Both arms fail useful quality; no
matched-quality efficiency or language-from-images-alone claim follows.

Use the [public hash-bound evidence](../references/evidence/trainable_visual_context_result_20261117.json)
and [failure decision](../references/trainable_visual_context_failure_decision_20261117.md).
Do not restart completed training, generation, audits, cold checks or visual
review; do not relax thresholds or repeat unchanged earlier reader, codec,
writer, native-history or sampling diagnoses. All old failures and successful
cases remain evidence. The preceding goal revision is preserved at Git commit
`6390e80e7ee3e113419279702c2f872448d8c88d`.

## Next work: diagnose the conditional training signal

Full contextual adaptation under the six equally weighted objectives was
insufficient. The next bounded investigation measures the conditional-flow
and auxiliary gradients on the exact final actors and authenticated complete
training records, with no optimizer updates or retained new model. Equal loss
weights do not imply equal gradients; this is a possible interference mechanism,
not an established cause. Negative cosine alone is not an AdamW update analysis.
Follow the failure decision's group accounting, complete hypothetical update
requirements, limits and actions for either result.

Before execution, separately register exact records, final actor and offline
projection identities, parameter groups, precision, source bindings and a finite
resource ceiling. Admit only one diagnostic owner and verify full-record support
without silently cropping after failure. This goal text authorizes preparation,
not an unregistered heavy run or another full fit. No development answers,
reserved DRCD records, rejected reward judge or policy fit enters this work.

The [diagnostic protocol](../references/visual_context_gradient_protocol_20261117.md)
and [56-source registration](../references/evidence/visual_context_gradient_registration_20261117.json)
are prepared. Four CPU checks establish copied AdamW arithmetic and unchanged
inputs, not language learning. The diagnostic retains the complete 513-, 846-
and 1,042-glyph training records, three RNG-matched gradient passes per record,
retained moments and the direct combined gradient. Commit and push its binding
before the sole 1,200-second monitored execution, with no optimizer updates.
Measure actual model/projection preservation and support for both final actors;
preserve any execution failure without an automatic retry.

If interference is supported under the relevant update direction, prepare one
prospective objective intervention with an unchanged control and training-only
update/reset preflight. Require actual complete conditional writing improvement;
gradients or reduced auxiliary loss alone cannot admit the next full fit. If the
evidence does not support interference, reject the explanation and choose the
next single change from the measured limitation. No working successor algorithm
has yet been qualified.

The earlier white-target comparison, both reward judges, isolated-cell teacher,
fixed encoder metric, label uniqueness and preprocessing audits remain terminal.
Do not recover old evaluators, tune rejected thresholds or reuse development
outputs as fitting data. Physical aliases and invisible source symbols remain
preserved; an exact pointwise Unicode condition's failure is not proof that
visual language is impossible. The reserved DRCD collection remains unrendered,
unscored and evaluation-only.

Every successor fit requires its own failure-specific, prospectively bound
training-only support/gradient/resource preflight on a fresh clone with all trial
updates reset. Bind complete sources, one intervention, unchanged control,
initialization, schedule, checkpoints, full evaluation, resource ceilings and
actions for either result. Direct supervised proposals need no reward judge;
policy proposals also need qualified complete-action rewards and exact emitted-
pixel probability replay. More unchanged epochs are not a selected intervention.

An ordered visual volume remains a representation hypothesis. Its completed
invertible CPU operator preserves order but does not establish compression,
faster language learning or coherent generation. Any future block must predict
future visible spans without receiving its own target block as context. Do not
replace language learning with supplied-answer reconstruction.

## Acceptance criteria

The unit of success is a **complete correct answer or coherent continuation**.
Copying, glyph fidelity, canonical bitmaps, loss, stopping, cold execution and
resource savings remain distinct claims.

1. Demonstrate complete modern-Chinese answers and coherent book continuation
   on held-out works and passage groups, using actual emitted pixels as history.
   Ask different questions about the same passage and retain question-, passage-
   and prefix-erased controls. Preserve raw outputs, natural stops, caps and
   uncertain readings; freeze review rules before generation and readings before
   new reference scoring. Canonical mismatch alone does not establish falsity.
2. Preserve the original complete-response pilot: at least 12/32 strict
   proportional answers, 4/16 both-correct pairs, 24/32 readable answers and 4/12
   coherent complete long books; all eight proportional and two fixed earlier
   successes; each QA-erasure arm at most two exact answers and a dependence gap
   of eight. Keep caps 128/640, emitted-image feedback and natural stopping.
   A physical failure leaves autonomous language unmeasured, not zero.
3. Confirm development improvements on untouched source groups. The reserved
   DRCD pool has 998 paired passages, 3,485 questions and 6,970 human answer spans.
   It remains evaluation-only, unrendered and unscored, excluded from fitting,
   reward/preference calibration and tuning. Its known-source screen does not
   prove semantic novelty or unknown inherited pretraining exposure. A separate
   committed confirmation execution is required after a qualified development
   comparison; this pool does not replace books or historical-form requirements.
4. Reproduce training and standalone inference within one 24-GiB GPU. Account for
   the entire model, inherited weights, preparation, teacher cost, wall time,
   peak VRAM and latency. Compare useful matched quality against a named local
   Qwen baseline before claiming efficiency or large-model parity. A prototype
   attached to a 7.522B inherited body is not a compact foundation.
5. Extend demonstrated language to held-out writing styles and attested
   historical or unencoded forms. Evaluate explanation, historical stage,
   glyph fidelity and provenance separately; label reconstructed forms.
   Demonstrate Chinese/English word-origin questions as part of the full goal.
6. Maintain Nature-style TeX/PDF and editable native PowerPoint figures from
   actual measurements, preserving original images and negative results. These
   artifacts support the evidence; publication does not qualify the model.

A development pilot alone does not complete this objective. Novelty, human-like
reading and broad language capability require their own evidence. The requested
1,000-fold reduction in compute and 1,000-fold performance improvement are
unproved stretch targets, not an assumption or a result. First establish useful
complete language and report matched-quality whole-model costs honestly.

## Completed constraints and data

The last [autonomous continuation result](../references/visual_continuation_learning_result.md)
is 7/32 strict proportional answers, 0/16 both-correct pairs, 14/32 readable
answers and no credited coherent complete long-book continuation. All 732 streams,
183 boards and 96 normal-book prefixes are reviewed. Both the original scorer and
independent four-QA-erasure audit reject the trial. Preserve all five checkpoints;
the better intermediate result does not replace its prescribed final result.

Reuse the completed [clean-history](../references/continuation_clean_history_result.md),
[trajectory](../references/visual_continuation_trajectory_result.md),
[joint-writer](../references/joint_writer_physical_result.md),
[context-gradient](../references/joint_context_physical_result.md),
[nonempty-ranking](../references/global_glyph_ranking_physical_result.md),
[whitespace](../references/ranked_writer_whitespace_result.md) and
[saved-score](../references/ranked_glyph_score_gap_result.md) findings. Books fail
with correct history, so a feedback-only explanation is insufficient. Earlier
native-reader, teacher-state, question-assignment, recognition-loss, copying,
writer, codec and released MIXAR controls are terminal; do not repeat unchanged
fits, evaluator recovery or diagnoses. Preserve contradictory readings and
original scores. The preceding full goal text is retained at Git revision
`2dda87d203ce8e137a0c5450fa9efd9c28c1f302`; historical directions are indexed in
[the decision archive](ilm-goal-decision-history.md).

Use the authenticated corpus and whole-work/passage-group splits. The original
13,785 training sequences contain 2,576,239 target glyph positions; the complete
matched continuation comparisons retain 13,766 eligible sequences. Exposures are
not unique facts or foundation-scale pretraining. Add data only with a declared
coverage or supervision reason, provenance and group splits. Preparation is not
student-language evidence.

The current 58/101/102-page reading/review/redline PDFs include the completed
white-target failure, later isolated-reader falsification, source/preprocessing
and label audits, rejected fixed encoder metric and six real full-context reset
updates. Native Figures 30--31 retain original output/reference pixels and
separate physical fidelity, reader qualification, update execution and language.
All 337 prior figure assets and 154 prior embedded image contents remain.
The [build evidence](../publication/ilm-nature/evidence/visual_context_evidence_article_build.json)
records all 261 pages in layout overview and 43 detailed pages. Preserve the
older 53/93/95-page baselines. These PDFs predate the now-completed full-context
failure. Add the final matched scores and original output images under a bounded
manuscript/figure revision plan; do not present their earlier pending statements
as current status. Local Nutstore delivery and cloud
propagation remain separate verification claims.

## Work-unit and resource rules

Read this goal and `artifacts/glyph_tail_current_handoff.json` at each model work
unit. Verify actual results and live process identity before acting. Record the
new evidence and next decision at the end. Keep private runtime details out of
commits and historical result files out of live scheduling authority.

Run at most one heavy project job under `artifacts/ilm_gpu0.lock`. Before a heavy
launch inspect host RAM, swap, pressure, owned processes, tmux and ports. The
existing full-reader entry requires 48 GiB available host RAM and 20 GiB free
GPU; retain 24 GiB host RAM, 48 GiB disk and at most 22 GiB allocated VRAM, or
stricter registered ceilings. Use two CPU threads. Protect foreign projects,
reuse shared weights and stop obsolete owned runtimes after evidence capture.
An observation timeout or serial phase transition does not authorize a duplicate
or restart.

Prioritize the failure-driven model comparison and complete-language evidence
over additional source preparation, copying repairs, covers or manuscript
polish. Commit and push related source/document changes while preserving ignored
private artifacts and unrelated work. Keep the session goal active until every
language, independence, resource and written-form requirement is proved.
