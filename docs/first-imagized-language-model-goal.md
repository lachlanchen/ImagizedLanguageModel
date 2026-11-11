# Imagized Language Model: Engineering Goal

Updated: 2026-11-11

## Immediate direction

Build an independently running visual language model: visible questions and
passages in, readable and relevant answer images out, with its own emitted
pixels as continuation history. Demonstrate modern Chinese on one RTX 4090
before expanding scripts, modalities or size. Keep historical, handwritten
and unencoded forms in the full objective. Using pretrained language weights
is allowed and must be disclosed; transferring inherited language capability
and acquiring language from images alone are different claims.

**Current verdict: the usable-language goal is not achieved.** Both completed
compact fits received 0/32 complete exact QA answers, 0/16 both-correct question
pairs and no credited readable complete book continuation at every fixed
checkpoint. Each fit covers all 13,785 training sequences and 8,172 updates.
The wider writer removed the measured rank restriction but did not solve
writing. See the [V1 result](../references/compact_visual_span_learning_result.md)
and [controlled wider-writer result](../references/compact_visual_span_full_rank_result.md).

The wider review is complete: all 372 opaque pages were inspected before their
mapping and references were opened for scoring. Twenty-four pages were blank;
348 had uncertain whole-response transcriptions. This is one model review
with prior dataset familiarity and no independent human adjudication. The
trained cold child passes the access/pixel audit. Its wrapper ended before
saving verification, for an unknown reason; the original completed child
traces support CPU-only recovery without rerunning generation. Execution and
capacity checks do not establish usable language.

The [frozen-weight diagnostic](../references/compact_visual_span_diagnosis_result.md)
is now complete for both final checkpoints. Correct history and supplied
lengths still produce zero exact first-span glyphs among twenty training and
twenty-six development targets per model. Direct codec reconstruction exceeds
0.9998 foreground F1, while increasing sampling from eight to 64 steps does
not repair writing. Both encoders have 383-dimensional feedback affine families
for 2,048-dimensional native targets, imposing about 0.57 minimum relative
squared error on selected positions. This is a target-alignment restriction,
not a language-quality bound. A separate one-case precision probe identifies
BF16 sequence-shape sensitivity; the completed diagnostic uses matched cached
execution and retains the original consistency tolerance.

The [earlier actor's matched-cohort evaluation](../references/glyph_tail_compact_cohort_result.md)
is complete: **2/32 complete exact QA answers, 1/16 both-correct pairs and
3/32 readable normal QA outputs**. Neither compact final model retains those
exact answers. One readable output omits the reference's parenthetical alias;
it fails unchanged strict matching without establishing factual falsity.
All 120 outputs were reviewed opaquely and rescored independently from the
frozen readings. The 44 normal inputs contain 36 caps, including all twelve
book continuations. No readable coherent complete book output is established.
The two exact answers belong to one passage pair; this remains limited
development evidence. Both QA-erased conditions score zero. The pointwise
95% interval for either normal primary rate is 0–18.75%.

The fixed-book cold replay reproduces actual pixels, states, stopping and
feedback without observed forbidden input access. Inference takes 1,589.88
seconds for 120 cases at 17.786 GiB peak allocation, with zero new updates.
The actor retains 7.522B inherited visual/contextual parameters. No source
question, passage or training-PNG overlap is found in its recorded adaptation
data; inherited pretraining and all earlier ancestors remain uncertified.
Its earlier 16/32 clear-answer result uses a different cohort and metric.
Bodies, observers, adaptation histories and writers differ, so this is not a
causal model-size or matched-quality efficiency comparison.

The [full input-presentation comparison](../references/glyph_tail_input_presentation_result.md)
is complete: **6/32 complete exact QA answers and 11/32 readable outputs**,
versus 2/32 and 3/32 with fixed glyph cells. Both-correct pairs fall from
**1/16 to 0/16**, and the previously exact **169平方公里** answer becomes
unreadable. The registered retention rule fails. Keep both presentations;
do not adopt the proportional interface as a uniformly better adapter.
All 96 outputs, including 64 erased controls, were reviewed before scoring.
Both erased conditions have zero exact and zero readable answers. This is
one model review with prior cohort familiarity, without independent human
adjudication. Strict canonical mismatch does not by itself establish semantic
falsity; no favorable semantic score is substituted after generation.

Every full passage and question is preserved, with no answer-centered crop,
truncation or supplied output length. Median input height changes from 704
to 512 pixels. The frozen weights and actual-pixel output loop make zero
optimizer updates. Neural execution takes 1,179.29 seconds at 17.786 GiB
peak allocation. The fixed-question cold replay verifies signed pixels,
states, stopping and feedback in a separate traced invocation. Normal caps
fall from 24/32 to 13/32. The exact-answer increase is 12.5 percentage points
with a pointwise 95% passage-group bootstrap interval of 0–25 points; the
paired decrease is 6.25 points with an interval of minus 18.75 to zero.
This reused development cohort supplies no untouched confirmation. Font,
spacing and wrapping change together, so the cause is not isolated further.

**Next required work: register the helper-only native-reader diagnosis before
another fit.** The [bounded decision](../references/glyph_tail_native_reader_decision.md)
compares all 32 questions and 64 erased controls in both presentations, with
and without the learned tail adapters, plus the same complete text prompts:
384 image and 64 text calls (448 total). Adapted image prefills must reproduce
the actual glyph actor's first saved contextual states bit-for-bit. The source lexical
decoder and previous-symbol lookup are temporary diagnostic tools outside
the ILM. Their outputs and development answers must not become training data
or deployed answers. Register source identities, scoring and resource limits
before execution; this diagnostic is not yet implemented or run.

Useful native image answers with damaged glyph output direct the next bounded
intervention toward writing/feedback. Better unadapted than adapted native
answers identify an adaptation tradeoff; good text but poor image results
identify an input gap. If all routes fail, examine the source/task and
implementation before another fit. These are conditional decisions, not
causes already proved. Reject the unchanged compact joint recipe and a
sampler-only repair. Check the completed semantic glyph loop and frozen
diagnostics before repeating a training approach. A development improvement
needs untouched source groups before broader claims. Copying, finite losses,
rank checks and presentation quality do not replace language.

The Nature-style article now includes both failed fits and the completed
failure-localization diagnostic. Reading/review/redline PDFs have 27/47/47
pages, with three additional native PowerPoint figures containing seven
measured charts and original diagnostic glyphs. Both newly completed
matched-cohort and proportional-input controls remain outside that verified
revision. The [bounded manuscript plan](../references/manuscript_glyph_input_transfer_revision_plan.md)
records their measured gains, lost pair and retention failure for the next
TeX/PDF and editable-figure update. No new figure or paper result is claimed
from that plan alone. Model diagnosis and readable held-out answers take
priority over additional presentation work.

## Goal and acceptance criteria

Train and run a compact, standalone language model on one RTX 4090. Start with
Chinese. It reads visible writing and a question, understands their relation,
and generates a readable, complete answer as an image. The same model should
continue a passage from its visible prefix. Render typed questions at the
input boundary. Use the generated pixels as feedback for continuation, and
offer text transcription as an optional downstream operation.

Use substantial authentic books and complete question-answer sequences.
Offline teachers and existing pretrained weights can speed up training.
Report their contribution, along with corpus size, target coverage and
preparation cost. The deployed model generates from continuous visual states
with no tokenizer, OCR, teacher service, font renderer, lexical head or answer
lookup in the answer path.

1. Run held-out prompts from independent works and passage groups. Test
   readable complete answers, coherent continuation and learned stopping.
   Ask different questions about the same passage to measure whether the
   answer responds to the question. Score raw outputs against preregistered
   controls and retain failures alongside successes.
2. Establish transfer across written forms. Include held-out fonts and
   handwritten or historical examples. For a glyph-origin task, check that
   the answer identifies the historical stages and renders the attested forms,
   including shapes unavailable in Unicode. Record source evidence for each
   historical claim and label reconstructed forms accordingly.
3. Reproduce training and cold inference within the memory of one 4090.
   Report total and trainable parameters, peak VRAM, elapsed time and latency.
   Compare with a named local Qwen baseline on the same tasks. Establish
   comparable quality before claiming a compute advantage.
4. Maintain the Nature-style TeX and PDF with editable PowerPoint figures.
   Plot complete-answer and paired-question performance at fixed training
   checkpoints. Show the same raw output images across checkpoints. Record
   pixel fidelity and copying as diagnostics and disclose inherited weights
   and evaluation limits.

## Evidence required for the claim

The central hypothesis is that a continuous visual interface can support
language learning and generation. The decisive observations are correct,
question-dependent answers and coherent continuations on held-out source
groups, generated with the model's own emitted pixels as history. Copying,
codec reconstruction, a lower auxiliary loss, and an interface that executes
are supporting measurements. Each has a different interpretation.

An improvement on the current small registered cohort is a development
result. Confirm it on new, untouched groups before describing generalization
beyond that cohort. Track complete-answer accuracy, both-correct question
pairs, readability, natural stopping, and book coherence with their
uncertainty. Keep the initial checkpoint and erased-input controls in the
same comparison. A conventional text baseline also needs adequate task
quality: beating a weak baseline alone cannot establish a useful ILM.

The compact route adapts a pretrained language body through learned image
interfaces. It tests transfer of inherited language capability to visual
input, generation, and feedback. It does not establish acquisition of language
from images alone. Report this distinction in the model description and paper.
Efficiency and Qwen 8B parity remain hypotheses requiring matched-quality,
matched-hardware measurement, including data preparation and teacher cost.

## Current decision: separate visual reading from glyph generation

The baseline is the [compact visual span model](compact-visual-span-model.md).
It uses a frozen 1.409B-parameter contextual body and 12.87M trainable image
parameters, plus a training-only teacher projection. Its shared image encoder
reads observed and generated glyphs; the writer predicts a span and its
length, including an end decision. Generation rereads its actual output
pixels. The deployed answer path has no source vocabulary embedding matrix
or lexical output head.

The [verified visual corpus](visual-sequence-training.md) contains 13,785
training sequences: full public-domain passages and complete natural
question-answer sequences, with 2,576,239 target glyph positions. Whole works
and CMRC passage groups define the held-out partitions. Exact text, source
and pixel overlap have been checked; inherited pretraining exposure remains
unknown. The offline teacher schedule covers 2,380,620 causal context targets.
The first complete fit has 8,172 scheduled optimizer batches. Preserve all
registered targets and the source identities when resuming.

The completed serial GPU schedule was teacher extraction, the complete fit, and
[fixed-checkpoint evaluation](../references/compact_visual_span_evaluation_protocol.md).
Evaluate 32 questions from sixteen held-out contexts and twelve continuations
from two held-out books at initial, quarter, half, three-quarter and final
checkpoints. The erased-input controls at both endpoints, all raw streams and
the trained standalone replay are complete. The pipeline used one 4090 under
the shared project GPU mutex. The 372 outputs have been reviewed and scored.
Preserve these results and checkpoints as the unchanged control for the next
experiment. Actual progress and process ownership belong in the private
runtime handoff; scheduled work is not a completed result.

The earlier [natural first-image experiment](natural-first-writer.md) is a
secondary diagnostic. Preserve its completed cache shards and fixed protocol
for explicit resumption. Its completion is no longer a prerequisite for the
compact experiment. Changing this scheduling priority does not qualify or
reject the unfinished diagnostic.

The [pixel-alias audit](../references/visual_feedback_identifiability_result.md)
finds a minimum relative squared feedback error of 0.0007492351 imposed by
identical images with different training targets. It measures an auxiliary
objective, with the actual batch weights. It neither predicts language quality
nor establishes that the fitted encoder reaches that bound. Interpret it
alongside the completed fit and actual output review.

The [flow-head audit](../references/compact_visual_flow_rank_result.md) finds a
separate architectural restriction in V1: a 384-channel linear output predicts a
768-dimensional velocity. In the continuous model, 384 Gaussian-noise directions
remain uncontrolled. The registered wider comparison removes this tested
capacity restriction but still fails the complete-response evaluation. Preserve both models as
controls. The restriction is not a sufficient explanation for the failed
answers. The completed diagnostic also finds conditional-writing failure
before emitted feedback. The earlier actor retains two exact current answers
with fixed cells and six with proportional prompts, but loses its exact pair
under the latter presentation. Preserve both actual output sets. Diagnose
native reading, tail adaptation and glyph continuation before another fit;
neither representation is qualified as a generally useful language interface.

## Current evidence and unresolved tests

The [completed wider-writer comparison](../references/compact_visual_span_full_rank_result.md)
uses 48.55M trainable image parameters and an 8.39M training-only projection.
Its trained projection, centered output map and three tested local Jacobians
retain rank 768. The fit takes 1,945.56 seconds, including 150.14 seconds of
resource waiting, with 4.889 GiB peak allocated GPU memory. Its five-checkpoint
review and authenticated analysis are complete. Final normal generation has
13/44 stops, 31/44 budget-limited streams and 4/44 empty responses. All QA
controls also score zero. These measurements reject the rank repair as a
sufficient solution under this recipe. The zero sample differences and
degenerate bootstrap intervals do not prove zero population success.
See the [bound language evidence](../references/evidence/compact_visual_span_full_rank_learning_20261111.json).

The compact implementation passes CPU gradient and emitted-pixel feedback
checks. The first fresh CPU invocation of its untrained heads stopped without
writing. The teacher cache and the 8,172-update compact fit are complete. The
fit covers all 13,785 sequences and 2,380,620 state targets, with five retained
snapshots and 3.930 GiB peak allocated GPU memory. It took 1,759.83 seconds,
including 110.11 seconds waiting for host memory pressure. The five-checkpoint
image evaluation and trained standalone replay are complete. On the fixed
development cohort, every step has 0/32 complete exact QA answers, 0/16
both-correct question pairs and 0/12 credited readable complete book
continuations. One model reviewer inspected opaque output pages before opening
their mapping or references: 342 responses had uncertain whole transcriptions
and thirty were blank. Recognizable individual glyphs do not supply a complete
answer. There is no independent human adjudication. Final normal generation
hit the 128-glyph budget on 27/44 cases, compared with 0/44 initially.

The final checkpoint reproduced a 128-glyph book output, stopping status and
all emitted-pixel feedback hashes in a separate traced process. No forbidden
training/reference paths or IP connections were observed. This verifies the
standalone boundary for that invocation, without establishing language quality.
See the [language result](../references/compact_visual_span_learning_result.md),
[complete fit evidence](../references/evidence/compact_visual_span_fit_completed_20261111.json)
and [earlier independent optimizer check](../references/evidence/compact_visual_span_gpu_start_20261111.json).

The source-matched [Qwen3 1.7B text reference](../references/compact_visual_span_qwen_reference_protocol.md)
ran the same 44 normal prompts on CPU. Exact scoring gives 8/32 QA answers
and both answers in 2/16 pairs. Several other outputs are partial answers;
semantic adjudication is pending. Book coherence is unscored. This run supplies
an observable reference for this cohort; the matched-GPU efficiency comparison
and the larger Qwen baseline remain pending.

The [historical evaluation](../references/historical_visual_evaluation_protocol.md)
contains attested oracle, bronze and seal images. The fixed codec reproduces
20/27 single-tile forms exactly. Five tall bronze forms represented by four
tiles average 0.939 foreground F1. These are physical reconstruction results.
Historical reading, stage selection and factual etymological explanation
require trained inference and evaluation. General page segmentation, unseen
fonts and handwriting also remain unproved.

Earlier [contextual adaptation](../references/glyph_tail_paired_answers_result.md)
gives 16/32 clear complete answers versus 11/32 for its writer-only control.
The [balanced writer](../references/balanced_glyph_writer_result.md) subsequently
gives 18/32 versus its parent's 19/32 on a different cohort, despite better
physical qualification. Both retain a much larger pretrained body. The project
has no qualified usable ILM, demonstrated compute advantage, or evidence of
large-model parity.

## Rules that keep the work on track

At the start of each work unit, read this goal and the private runtime
handoff, verify the latest completed evidence, and name the next unresolved
measurement. At its end, record the actual result, retained artifacts, current
process ownership and the next decision. Replace stale run statuses. The
session goal remains active while any required capability is unproved.

Every next experiment needs a short decision record: observed failure, one
change, preserved control, fixed evaluation and success/failure rule, data and
checkpoint identities, GPU/time budget, and the action for either outcome.
Register it before execution. Finish scoring that comparison before choosing
another branch. If the result is negative, change the hypothesis from the
evidence; do not keep extending the same run without a new bounded reason.

- Before each experiment, write down the observed failure, one bounded change,
  the unchanged control, the primary language measurement and the resource
  ceiling. Freeze the prompts and quality criteria before generation. Finish
  and interpret that comparison before opening another model branch.
- Give the current failure-driven model comparison and held-out language
  evaluation priority over additional copying repairs, dataset preparation,
  cover images and manuscript polish. Reuse the substantial authenticated
  corpus and teacher cache unless a measured failure requires new data.
- Run one heavy project job at a time. Keep resumable shards and optimizer
  checkpoints, authenticate inputs on resume, and include interrupted work in
  cost reporting. Schedule secondary experiments explicitly.
- Diagnose failed answers with the actual input, output pixels, span boundaries,
  stopping and feedback trace. Distinguish visual access, glyph production,
  contextual language errors and accumulated generated-history drift. Choose
  the next bounded change from that evidence and compare it with the unchanged
  control. Preserve failures.
- The repaired writer still fails. Complete the selected native-reader
  comparison with the actual image prefills before another fit. Use its
  unchanged controls to separate visual access, tail-adaptation effects,
  conditional writing and generated-history drift. Helper-decoded or
  reference-supplied results are not autonomous ILM answers. Do not infer
  that a larger model, more epochs or unrelated physics will repair the
  failure without a testable mechanism.
- Keep inference independent. Offline teachers, OCR and text corpora may prepare
  or supervise training; they must not supply answers during deployed inference.
  Describe the inherited weights and training-only native-unit supervision.
- Maintain the Nature-style TeX/PDF and editable PowerPoint figures as an
  evidence record. Plot only measured complete-answer and paired-question scores
  at fixed checkpoints. Show the same unrepaired output images at each step.
  Report pixel metrics, language quality and resource costs separately.
- Accept a development improvement only from the frozen complete-response
  metrics, including readable writing, paired question dependence, coherent
  continuation and natural stopping. Confirm it on new untouched source groups
  before claiming broader generalization. Establish useful task quality before
  making matched-quality efficiency or large-model parity claims.
- Keep the goal active until the acceptance criteria are demonstrated. A queued
  job, finite loss, successful build, attractive figure, or newly proposed
  mechanism does not complete it. Historical and unencoded forms remain part
  of the full objective after the modern-Chinese language test.

## Capability scope after the first language result

The long-term target includes simplified and traditional Chinese, historical
and regional forms, handwriting and shapes absent from Unicode. Written forms
remain images throughout the deployed language path. A typed prompt is
rasterized at the boundary; optional OCR transcribes the finished answer
without altering its pixels. Arbitrary scans still require a verified
reading-order interface. The current tile-stream implementation has not
established that capability.

The first complete application is a visual word-origin book: an English or
Chinese question produces readable explanatory text and the relevant attested
forms on an answer-page image. Keep the proposed 200-question application
benchmark, with 100 Chinese and 100 English word-origin questions and balanced
rendered versus scanned prompts, for a later frozen evaluation. It remains
unprepared and unmeasured. Score factual correctness, readability, historical
form fidelity, provenance, unencoded-region preservation and layout. Distinguish
attested source evidence from model-synthesized forms. General language use
also requires fresh passages and instructions beyond etymology.

Modern Chinese complete-answer learning is the immediate step. Historical
transfer and the application benchmark follow demonstrated modern language
performance. Expand to multilingual books, raw-page reading and longer dialogue
from measured capability. Three-dimensional glyphs, character movies and sound
are later research extensions, each requiring its own data and evaluation.
The current two-dimensional experiment makes no claim about those modalities.

## Preserved decision history

The [decision archive](ilm-goal-decision-history.md) retains the earlier model
specifications, negative results, mathematical proposals and capability ledger.
Use the current criteria and priority above when scheduling new work. The
archive supplies evidence and context; it does not replace current run status.
