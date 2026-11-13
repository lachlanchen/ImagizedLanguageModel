# Imagized Language Model: Engineering Goal

Updated: 2026-11-14. Status: **active; usable visual language is unproved**.

## Objective

Build a useful, independently running language model that reads visible writing
and generates readable answer images on one RTX 4090. Begin with modern-Chinese
questions, passages and book continuation. Typed prompts may be rendered into
images at the input boundary. The model must interpret the prompt, write its
answer, reread its own emitted pixels and decide when to stop.

The full target includes simplified and traditional Chinese, historical forms,
handwriting and shapes absent from Unicode. The eventual word-origin application
accepts Chinese or English questions and produces explanatory writing with
attested historical forms. Establish modern-Chinese language performance before
expanding scripts, raw-page layouts, three-dimensional streams or audio.

Offline teachers, OCR, text corpora and inherited foundation weights are allowed
for preparation and training. Disclose their contribution and cost. Deployed
answers must come from the trained model: no live teacher, tokenizer, lexical
output head, answer lookup, OCR or font renderer in its answer-generation path.
Optional transcription may read the finished image without changing its pixels.
Transferring inherited language capability and learning language from images
alone are separate claims.

## Direction check for every work unit

The complete-sequence comparison is finished and fails its registered pilot.
All 732 image streams, 183 opaque boards and 96 visible book prefixes have been
reviewed; readings were frozen before new mapping/reference scoring. The final
model gives 7/32 strict proportional answers, 0/16 both-correct pairs, 14/32
readable answers and no credited coherent long-book continuation. Its 7.578B
whole deployed parameters do not establish the desired compact model.

- Use the [completed result](../references/visual_continuation_learning_result.md)
  and [failure decision](../references/visual_continuation_failure_decision.md)
  as the current evidence. Do not repeat the completed fit, generation, opaque
  review, optimizer audit or unchanged earlier diagnoses.
- The [joint-writer comparison](../references/joint_writer_physical_result.md)
  is complete: all 41,298 updates, independent audit, 124 conditional cases,
  33 opaque boards and 2,080 frozen candidate/control readings. The new head
  gives 970/1,040 original frozen credits versus the point control's 599/1,040,
  but development books 91/108 and training-fixed QA 90/101 fail the unchanged
  90% per-group prerequisite. Stop this actor before autonomous evaluation;
  its complete language is unmeasured. A later [pixel/blank-flag audit](../references/joint_writer_blank_flag_audit.md)
  finds17review inconsistencies, including4creditedemptyimages; original credits
  are retained with that limitation and are not corrected physical accuracy.
  Before claiming a new physical pass, independently verify that every credited
  output contains ink as well as satisfying the frozen visual review. Preserve
  contradictory readings and disclose them; an ink check cannot establish language.
  Preserve the fixed reader/context/
  feedback/stop path, all snapshots and original verdict. Its 2,642,368 new
  parameters do not establish a small whole model.
- The [probability diagnosis](../references/joint_writer_blank_density_result.md)
  is complete on all 124 cases and 12,597 positions. It finds 349 false blanks;
  only sixteen prefer their source image. Only one of fourteen selected
  development-book false blanks and no selected training-fixed-QA false blank
  do so, rejecting the registered search-only decision rule. No blank-prefix
  argmax inconsistency is observed. Four selected book source images are
  themselves empty; keep whitespace separate without revising the old gate.
- The next [controlled change](../references/joint_writer_context_learning_decision.md)
  trains continuation adapters and the initial query through complete-image
  likelihood, with the completed head-only trial as its unchanged control.
  The exact memory-bounded state-gradient implementation matches unsplit
  autograd. The [real-source GPU preflight](../references/joint_context_preflight_result.md)
  now passes all four cases at 17.795 GiB peak allocated VRAM, including both
  complete 513-glyph books and all 131 trainable gradient comparisons. All
  updates were reset. Preserve the original SIGTERM and its one recorded
  recovery; do not repeat this completed check. The
  [complete context-gradient comparison](../references/joint_context_learning_protocol.md)
  retains the full source exposure and every original physical/language rule.
  Its completed fit, audit and language result remain unproved.
  The [post-qualification evaluator](../references/joint_context_language_execution.md)
  is implemented with all 131 context/head tensors restored at every snapshot,
  the complete 708-stream plan and an independent check that credited candidate
  forms contain ink. Its CPU checks are implementation evidence; full-body
  generation and language remain unexecuted until qualification and a separately
  committed execution binding. The failed head-only actor remains unevaluated.
  Keep prompt memory, native body, image feedback and stop parameters frozen.
  Register sources, initialization, schedule, gradients, complete evaluation
  and resource limits before neural execution. Do not restart a completed
  recipe or launch the failed actor's blocked autonomous comparison.
- The [new saved-trajectory analysis](../references/visual_continuation_trajectory_result.md)
  is complete. Final normal long-book outputs have 98.12% of tiles above clean
  book ink's 99th percentile, with dense writing visible from the beginning.
  All 416 first-step contrasts change state; this does not establish language
  understanding. The [initial/final clean-history control](../references/continuation_clean_history_result.md)
  is also complete, with both long cache checks passing. Final proportional
  development QA reaches F1 .95 on 656/713 later supplied-history positions;
  sampled training books reach it on only about 9%. Do not repeat these controls.
  The selected replacement must address conditional writing without substituting
  it for complete autonomous language.
- Register one bounded change and an unchanged control before another fit.
  Preserve the full corpus, complete answers, paired questions, erasures, long
  books, earlier successes and whole-model resource accounting. More unchanged
  epochs, a selected intermediate checkpoint or looser thresholds are not the
  next experiment.
- Require every mathematical or architectural proposal to name a testable
  prediction, its control and an outcome that would reject it. Ordered visual
  blocks remain a candidate representation, with language benefit unmeasured.
- Update the manuscript and native figures from actual completed measurements.
  The active reading/review/redline PDFs now include the completed continuation
  and head-only failures, Figures 23--25 and the exact context-gradient methods.
  Their [45/80/82-page build and visual-review receipt](../publication/ilm-nature/evidence/complete_continuation_article_build.json)
  preserves the earlier evidence and distinguishes all-page layout overviews
  from detailed checks. Office vector export and versioned local delivery remain
  pending. Publication work must not substitute for resolving the measured model
  failure; the running context-gradient actor's language remains unmeasured.

Keep the goal active through these decisions. Completion requires the language,
independence, resource and written-form evidence specified below; neither a
development pilot nor a manuscript revision closes it.

## What counts as success

The unit of success is a **complete answer or coherent continuation**. Evaluate
whether the writing is readable, answers the actual question, uses the supplied
passage and ends naturally. Ask different questions about the same passage;
retain both successes and failures. Copying, codec fidelity, individual correct
glyphs, lower loss and a working inference executable are supporting diagnostics.

1. Demonstrate complete modern-Chinese answers and book continuation on held-out
   works and passage groups, using actual generated pixels as history. Include
   question-erased, passage-erased and prefix-erased controls. Preserve raw images,
   stops, caps and uncertain readings. Freeze scoring and output review rules
   before generation; inspect outputs before opening new reference mappings.
2. Confirm development improvements on untouched source groups. Keep the initial
   checkpoint and existing successful cases as controls. Report uncertainty,
   prior cohort familiarity and the absence of independent human adjudication
   where applicable. Canonical mismatch alone does not establish semantic error.
3. Reproduce training and standalone inference within one 24-GiB GPU. Report the
   complete model, trainable parameters, inherited weights, data preparation,
   teacher cost, wall time, peak VRAM and output latency. Establish useful matched
   quality against a named local Qwen baseline before claiming greater efficiency
   or large-model parity.
4. Extend demonstrated language to held-out writing styles and attested historical
   or unencoded forms. Check historical stage selection, factual explanation,
   shape fidelity and provenance separately. Label reconstructed forms.
5. Maintain the Nature-style TeX/PDF and editable PowerPoint figures from actual
   measurements. Show raw outputs and complete-language results across fixed
   checkpoints when those measurements exist. Unmeasured trials remain unmeasured.

A restricted development pilot is not completion of this objective. A small
trainable interface attached to a 7.522B inherited body is not a compact foundation
model. The desired lightweight model must account for its entire deployed body.

## Current evidence

The [complete-sequence continuation trial](../references/visual_continuation_learning_result.md)
covers 13,766 sources, 41,298 updates and 6,108,741 target glyph positions.
Proportional exact/readable/pair counts at the five fixed checkpoints are
4/5/0, 4/8/0, 6/13/0, 9/16/1 and 7/14/0, each over 32 questions and sixteen
pairs. Every normal book arm has zero credited coherent complete continuations.
Both the original scorer and the independent four-QA-erasure audit reject the
final pilot. The [public evidence](../references/evidence/visual_continuation_learning_20261113.json)
binds the full comparison. Review used one model reviewer with prior cohort
familiarity and no independent human adjudication.

The earlier [frozen balanced-writer comparison](../references/balanced_writer_matched_result.md)
is complete: all 216 streams, 54 opaque output boards, twelve book prefixes,
frozen readings and independent scores. All 216 initial reader states match the
parent bit-for-bit. The same current development cohort gives:

| Image-output actor, proportional prompts | Exact QA / 32 | Both-correct pairs / 16 | Readable QA / 32 |
| --- | ---: | ---: | ---: |
| Original parent | 6 | 0 | 11 |
| Direct-context flow writer | 7 | 1 | 14 |
| Balanced deterministic writer | 6 | 0 | 16 |

Balanced fixed-cell results are 2/32 exact, 1/16 pairs and 3/32 readable. No
complete book continuation is credited. It retains the parent's exact answers
but only six of eight union proportional successes, and fails its registered
pilot. All four QA-erasure arms score zero exact; one proportional
question-erased output is readable. These are reused development results,
reviewed by one model reviewer with prior cohort familiarity.

The [matched writing diagnosis](../references/balanced_writer_matched_diagnosis_result.md)
finds all 64 selected training tiles pixel-exact. Development first tiles are
0/32 exact with fixed cells and 4/32 with proportional prompts. These are bitmap
comparisons, not character or language accuracies. Exact training writing has
not solved complete-answer transfer.

Other completed controls constrain the next decision:

- Both [compact](../references/compact_visual_span_learning_result.md) and
  [wider-writer](../references/compact_visual_span_full_rank_result.md) fits have
  0/32 exact QA, 0/16 pairs and no credited complete book continuation at all
  five fixed checkpoints. Both cover 13,785 training sequences and 8,172 updates.
  Removing the measured writer rank restriction did not solve language.
- The [direct-context flow](../references/glyph_context_motor_result.md) fails its
  language pilot and loses an earlier exact answer. Its cold replay verifies
  execution independence; it does not qualify language quality.
- The [decoded-loss](../references/decoder_aware_motor_result.md) and
  [full-sampler endpoint](../references/rollout_endpoint_motor_result.md) controls
  fail their later-writing gates. Their complete-language trials were not run.
  Do not assign them zero language scores or extend their failed fits.
- The [native-reader diagnosis](../references/glyph_tail_native_reader_result.md)
  has completed all 448 helper calls and 192 bit-exact adapted image prefills.
  Adapted native proportional decoding gets 10/32 exact QA; source-adapter-zero
  gets 13/32. This exposes retained access and layout sensitivity without
  supplying independent ILM image answers. Its lexical decoder stays diagnostic.
- Coverage, first-writing, saved-state and pixel-feedback audits are complete.
  Reuse their authenticated artifacts. They do not identify a unique cause of
  failed language, and repeating them is not the next model experiment.

The [current publication revision](../references/manuscript_realized_writer_revision_plan.md)
is complete: 35/62/63-page reading/review/redline PDFs, verified native Figure 22
and a reproduced 160-page combined raster comparison. The local RealizedWriting
package and latest aliases are verified; cloud propagation is unverified.
No repeat publication build or completed-cohort generation is pending.
The correct-image-history study and completed continuation trial are newer than
these PDFs. Include both in the next bounded results revision; do not imply
they are already in the article. Failure-driven model work retains priority
over presentation-only revisions.

## Next model decision

The registered complete-sequence recipe is rejected. All five prescribed
snapshots, 660 core streams and 72 long-book streams are complete, with unchanged
scoring and the supplemental four-arm audit. Final fixed results are 2/32 exact,
4/32 readable and 1/16 both-correct pairs. Final proportional results are 7/32,
14/32 and 0/16. The three-quarter snapshot gives 9/32 exact but does not replace
the registered final model. Six of eight required proportional successes and
both fixed successes remain. Final erased QA exact counts are 0/1 for question/
passage erasure in each layout; the proportional normal-minus-erased gap is six.

All twelve final normal 128-cap books reach the cap. The final 640-cap candidate
stops on eight of twelve normal prefixes, while the matched balanced parent
stops on seven. Neither produces a readable coherent complete continuation.
Record-level EOS labels are not proof of a natural whole-document ending.

All 41,298 optimizer updates and 85 final Adam counters were audited. Cumulative
training child execution is 75,209.326 seconds at 18.568 GiB peak allocated VRAM.
Candidate evaluation costs 6,338.799 cumulative child seconds and parent long-book
evaluation 609.626 seconds, preserving failed attempts. Preparation, admission
queues, CPU/review work and inherited pretraining are separate. These values
show one-GPU execution, not matched-quality efficiency over an LLM. Those old
jobs remain terminal. The separate registered joint-writer pipeline is current
work; do not restart the completed continuation recipe or evaluator recovery.

The immediate model work follows the
[failure decision](../references/visual_continuation_failure_decision.md).
The joint-writer comparison from the final frozen reader is now also complete
and fails its physical prerequisite, despite improved conditional forms. Use
its frozen result and completed empty-image probability diagnosis for the
next controlled gradient-reach change; no complete-language result is available
for that actor. Keep
complete sources, unchanged controls and all full-language criteria. The
completed initial/final clean-history comparison
applies to this actual final reader; the older diagnosis remains a different
actor. Do not repeat either study or treat supplied-history reconstruction as
autonomous language. Keep the full objective active.

## Training and evaluation contract

### Ordered glyph-volume hypothesis

Treat visible writing as an ordered `[T,C,H,W]` stream. Investigate learned
compression into a shorter volume with explicit order and temporal detail;
unweighted ink accumulation aliases different orders. The
[volume analysis](../references/ordered_glyph_volume_research.md) implements an
invertible reference and checks all 13,766 eligible targets. Exact reconstruction
qualifies that operator only. Retaining all coefficients does not reduce their
information content, and fewer temporal positions do not establish fewer bits
or a model speedup.

The language hypothesis combines ordered visual compression with conditional
future-block prediction and a coherent image decoder. Evaluate predicted latent
states and actual emitted-pixel histories, not only clean autoencoder inputs.
Use completed earlier blocks for causal context; a block containing the target
must not enter its own prefix. Question/passage dependence, complete answers,
book continuation and natural stopping remain required.

This hypothesis is a candidate for a failure-specific comparison after the
current registered trial. Its CPU representation analysis does not alter that
trial's model, data, schedule or evaluation, and does not authorize another
concurrent GPU fit. Preserve the full language, independence, historical-form
and whole-model efficiency objective. Do not replace it with a recovery task.

### Existing data and prospective comparisons

Use the existing authenticated corpus first: 13,785 full training sequences and
2,576,239 target glyph positions, with whole-work and CMRC passage-group splits.
See [visual sequence training](visual-sequence-training.md). Training-only teacher
caches already exist. Add data when an observed coverage or supervision failure
justifies it, keeping source provenance and group splits. Inherited pretraining
exposure remains unknown.

At the core, learn the distribution of the next visible span and its end decision
from the prompt image and previous visible spans. During autonomous evaluation,
those preceding spans must be the model's actual emitted pixels. Span size,
continuous representation and sampling method are design choices to test; they
must not replace language evaluation. A discrete image grid alone does not make
language learning cheaper or establish human-like reading.

Before each fit, record the observed failure, one intervention, unchanged
control, complete-data exposure, fixed checkpoints, evaluation suite, resource
budget and action for either outcome. Finish that comparison before opening
another branch. Preserve the 216-stream current comparison and its original
scoring. For the next complete-response pilot, preregister thresholds before
training and retain the existing union successes: eight proportional and two
fixed exact QA cases. The last balanced pilot required at least 10/32 exact,
2/16 pairs and 16/32 readable proportional answers plus retention; it failed.
Do not relax that completed rule or substitute earlier different-cohort counts.
Set book-continuation criteria prospectively, and require untouched confirmation
before a usability or generalization claim.

Report copying, conditional writing, complete language, natural stopping,
standalone execution and efficiency separately. Never repair a transcription
from its reference, select favorable outputs or plot an unmeasured result as
zero. Freeze whole-response readings and expose uncertain output regions.

## Work-unit and resource rules

At each work-unit start, read this file and
`artifacts/glyph_tail_current_handoff.json`. Verify the latest completed result,
live process identity and the one unresolved measurement being addressed.
At the end, record the actual result, evidence paths and next decision. Keep
live scheduling details in the private handoff and historical instructions in
[the decision archive](ilm-goal-decision-history.md), not in this active brief.

Run at most one heavy project job, under `artifacts/ilm_gpu0.lock`. Recheck host
RAM, swap, pressure, project processes, tmux and ports before launching. The
existing full-reader admission requires 48 GiB available host RAM and 20 GiB
free device memory, with 24 GiB host RAM at runtime and a 22-GiB allocated-VRAM
ceiling. These host requirements are distinct from fitting one 24-GiB GPU.
Respect stricter registered limits and the shared-workstation policy. Preserve
other projects' services. Reuse shared weights and stop obsolete owned runtimes.
An observation timeout does not authorize a duplicate worker.

Prioritize the failure-driven model comparison and complete-language evaluation
over extra dataset preparation, copying repairs, covers and manuscript polish.
Update the paper when new interpretable model evidence exists. Commit and push
related repository changes; keep private runtime data, credentials and unrelated
work out of commits. Leave the session goal active until its required language,
independence, resource and written-form capabilities are demonstrated.
