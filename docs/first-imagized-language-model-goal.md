# Imagized Language Model: Engineering Goal

Updated: 2026-11-12. Status: **active; usable visual language is unproved**.

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

## Next model decision

**Before another fit, distinguish conditional writing failure from failure after
pixel feedback.** Use the saved balanced, parent and flow trajectories and the
existing native-reader controls. A new measurement must change the training
decision; another first-tile audit or generic loss curve is insufficient.

The next measurement plan is a frozen-weight, correct-image-history comparison
on all 32 existing normal QA cases in both layouts. Reproduce the actual initial
states. At each reference position, measure the predicted image and stopping
probability while feeding the correct preceding glyph images through the same
image encoder and contextual cache. Compare with the saved autonomous histories;
retain unmatched lengths, alternate wording and all failures. Record terminal
positions explicitly. Reference history and supplied lengths make this a
**diagnostic, not autonomous language generation**.

Before execution, commit a bounded protocol, source/checkpoint/input hashes,
resource ceiling and interpretation rules. This plan is not a completed or
registered experiment. Any native-head readout is an external diagnostic and
must remain separate from the image actor. Reuse existing native outputs before
considering more helper calls; they are not a ground-truth oracle.

Choose the next intervention from the observed pattern:

- Failure already at the initial state or under correct image history directs
  attention to conditional transfer or reader/writer alignment. Training only
  on generated histories cannot by itself repair that demonstrated failure.
- Readable conditional writing under correct history followed by deterioration
  with emitted-pixel history motivates one controlled feedback-training change.
  It does not authorize adding a runtime teacher or replacing images with IDs.
- Premature or missing stops under correct history require a separately measured
  stopping change. A glyph cap or supplied target length is not learned stopping.
- Failures shared with the existing native control also require reader-access
  investigation. A writer-only change cannot be assumed to supply missing
  understanding. Ambiguous cases remain ambiguous until evidence resolves them.

Do not infer a unique cause from canonical pixel mismatch. Prefer one
mechanism-based change with a preserved control over adding model components,
more epochs, new modalities or unrelated physics terminology. The implementation
may be replaced when the evidence warrants it; preserve the rejected recipes.

## Training and evaluation contract

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
