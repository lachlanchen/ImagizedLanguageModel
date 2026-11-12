# Imagized Language Model: Engineering Goal

Updated: 2026-11-13. Status: **active; usable visual language is unproved**.

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

The immediate question is whether complete-sequence adaptation of all answer
layers improves autonomous, prompt-dependent visual language while preserving
the frozen prompt reader. Finish the registered comparison before changing that
question. The current implementation is an experiment toward the objective;
its 7.578B deployed parameters do not establish the desired compact model.

- Tie each work unit to training progress, complete-output evaluation, a
  failure-specific intervention or a verified wait. For a live, progressing
  worker within its resource limits, preserve the run and record a verified
  wait. Extra fits, repeated audits and publication-only changes do not advance
  the pending language measurement.
- Keep model, data, checkpoints and acceptance rules fixed through the current
  comparison. Evaluate all prescribed outputs and erasure controls. A completed
  fit, decreasing loss or attractive glyph samples cannot replace this decision.
- If the pilot passes, test untouched source groups before claiming usability.
  If it fails, identify the measured failure in reading, writing, question or
  passage dependence, continuation, stopping or generalization. Register one
  bounded change and an unchanged control; do not extend the same failed recipe
  or lower its thresholds. Several causes may remain unresolved.
- Require every mathematical or architectural proposal to name a testable
  prediction, its control and an outcome that would reject it. Select the next
  method from the evidence rather than novelty or an assumed physical analogy.

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

The [frozen balanced-writer comparison](../references/balanced_writer_matched_result.md)
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
The correct-image-history study below is newer than these PDFs. Include it in
the next bounded results revision; do not imply it is already in the article.
The ongoing model comparison takes priority over another presentation-only work unit.

## Next model decision

The [correct-image-history comparison](../references/reference_image_history_result.md)
is complete: all 64 traces, 1,490 target positions, sixteen reviewed boards and
8,561 independently checked displayed tiles. Every initial state and saved
pixel/stop replay is exact. Proportional later median F1 is 0.985185, with
475/713 positions at least 0.95; fixed is 0.573840 and 139/713. Both registered
physical gates fail. Original complete-language scores remain unchanged.

Correct history improves later writing, but initial and conditional errors
persist. A feedback-only explanation is insufficient. All four nearly exact
but nonidentical first images have readable original autonomous answers, so
mere state divergence does not establish a harmful pixel-feedback error.
Canonical mismatch and different answer lengths remain distinct from semantics.
Do not repeat the completed comparison or substitute it for autonomous language.

The earlier [feedback ceiling](../references/glyph_feedback_ceiling_result.md)
already shows that native-unit and first-character steps are not interchangeable,
even with exact feedback vectors. Preserve its different cohort and exclude its
diagnostic records from training. Read the completed mixture-density and
first-function-transfer results before proposing those routes again.

The [continuation-only implementation and real preflight](../references/visual_continuation_preflight_result.md)
now pass. All 36 answer layers receive real gradients through complete targets
of 1, 74, 184 and 513 glyphs, with 20,212,225 trainable parameters and 18.123 GiB
peak allocated VRAM. Prompt states and all prompt keys/values remain bit-exact
after updates; causal checks pass. Independent audit verifies four AdamW steps
for all 85 optimizer tensors and authentic complete targets. The source retains
its 7.522B inherited body and frozen image interfaces. Preflight is not language.

The [complete-sequence pipeline and fixed protocol](../references/visual_continuation_full_protocol.md)
are implemented. Source indexing has authenticated 13,766 eligible sequences:
9,004 complete QA records and 4,762 book records. Nineteen diagnostic records or
source-group exclusions are recorded. After the book-prefix split there are
2,036,247 target glyph positions per epoch, with complete targets up to 513
glyphs. The prescribed three epochs give 41,298 updates and 6,108,741 scheduled
target positions. These are data exposure counts, not measured language results.

Preparation and the committed registration are complete. All 13,766 prompts
are preserved; 45 natural-layout rejections use their complete original raster.
The first execution stopped on an event-logging field collision after one unsaved
update. Its initial checkpoint, failure and 139.558657-second child cost remain
preserved. The [registered recovery](../references/evidence/visual_continuation_full_logger_recovery_registration_20261112.json)
changes logging and registration identity handling, with unchanged model, pixels,
targets, schedule and quality criteria. Its fresh worker has started in
`artifacts/visual_continuation_full_logger_recovery_v2`; the earlier run must not
be overwritten or restarted. Read the [recovery record](../references/visual_continuation_logger_recovery.md).
The [independent launch audit](../references/evidence/visual_continuation_full_launch_20261112.json)
verifies exact fresh initial parameters, optimizer and RNG against the original
step-zero checkpoint. Restoring four and replaying five/six matches the actual
uninterrupted step-six state bit-for-bit, with all 85 Adam counters at six.
The first six scheduled updates cover 1,367 target positions; two discarded
replay updates cover another 437 positions. These checks establish execution
and recovery, with no complete-language score. The full fit has now finished.
The [completion audit](../references/visual_continuation_completion_result.md)
verifies all 41,298 scheduled updates, 6,108,741 target positions, five snapshot
hashes and all 85 final Adam counters. Evaluation failed before any output at
its host-RAM entry guard. The immediate work is resource-gated recovery of the
unchanged evaluators, followed by complete-output review and scoring.
Discard all four preflight updates. Each source gets one fixed and two natural
presentations. Save initial, quarter, half, three-quarter and final weights;
retain the verified real optimizer/RNG recovery evidence. Evaluate all
732 prescribed image streams, including the unchanged 216-input cohort and
matched long-book controls, before applying the frozen pilot criteria.
The final proportional pilot requires 12/32 exact answers, 4/16 both-correct
pairs, 24/32 readable answers, 4/12 coherent complete long-book continuations,
all eight union proportional and both fixed successes, and the registered
erasure checks. A pass must then survive untouched source-group confirmation.

After the original frozen-review scoring, run the
[independent protocol audit](../references/visual_continuation_protocol_audit.md).
The bound scorer's pilot expression checks proportional erasures only; the
registered wording covers all four QA-erasure arms. The supplemental report
enforces that literal rule while preserving original scores and thresholds.
Do not claim a pilot pass from the original summary alone. The live pipeline
and its model/data/schedule stay unchanged.

Do not substitute the old 3,176-question, sixteen-glyph-target subset, repeat
the completed preflight, extend an unchanged failed recipe, or turn a working
checkpoint into a language claim. Full training is complete; language generation,
review and evaluation remain incomplete. Keep the
one unresolved decision explicit: does adapting all answer layers over complete
visual sequences improve autonomous complete language while preserving source
prompt interpretation? The architecture, data and evaluation must answer that
question before another model branch or presentation-only revision.

### Completed fit and evaluator admission

The completed fit reports 18.567853 GiB peak allocated VRAM and 75,209.326 seconds
of cumulative training execution, including the preserved earlier failed child.
The final tensors and Adam moments are finite; all 85 trainable tensors changed.
These execution checks do not establish complete language or efficiency.

The first evaluator failed at the unchanged 48-GiB host-RAM entry requirement,
with zero output streams. Preserve that status, console and controller journal.
Its 6.520045-second child cost counts toward the candidate evaluation ceiling.
Use the [recovery launcher](../scripts/recover_visual_continuation_evaluation.py)
for a single serial attempt after admission. It adds 2 GiB of prelaunch headroom,
keeps the original evaluator's 48-GiB guard, preserves the GPU/pressure/mutex
checks and tightens each cumulative child budget with a cleanup margin. Record
resource-queue time separately; it is not model throughput. A failed new child
requires evidence preservation before another retry. No training, model, data,
generation cap, scoring threshold or historical-form objective changes.

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
