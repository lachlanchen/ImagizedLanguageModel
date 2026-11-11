# Imagized Language Model: Engineering Goal

Updated: 2026-11-11

## Immediate direction

The objective is an independently running visual language model: visible
questions and passages in, readable and relevant answer images out, with its
own generated pixels as continuation history. Begin with modern Chinese on one
4090; demonstrate that behavior before expanding scripts, modalities or size.
Using pretrained language weights is allowed and must be disclosed. Learning
a visual interface to inherited language knowledge and learning language from
images alone are separate research claims.

The first complete compact experiment has now failed its registered language
test. All five checkpoints received zero complete exact QA matches; none of
the 372 reviewed outputs established a readable complete response. The trained
cold replay passed. This establishes an executable, independent inference
path, while the usable-language objective remains unmet. See the
[complete result](../references/compact_visual_span_learning_result.md).

The next model experiment must test the identified writer restriction: a
384-channel linear velocity output cannot control all 768 codec dimensions.
Test a full-dimensional velocity path against the preserved compact baseline.
Keep the encoder, contextual foundation, codec, corpus, supervision, optimizer
settings and evaluation fixed initially. Register the implementation, seed,
exposure and resource budget before training. Improving the writer's rank is
the hypothesis to test; readable conditional writing is the required outcome.

The [registered comparison](../references/compact_visual_span_full_rank_protocol.md)
uses a 1,024-channel writer. Its initial projection, LayerNorm-centered map and
three tested local velocity Jacobians have rank 768. The trainer verifies the
unchanged encoder/length initialization and noise-stream preservation. These
capacity checks are complete; training and generated-language results for the
new writer remain unproved. Preserve the same full corpus and five-checkpoint
evaluation. Treat its final checkpoint as the primary comparison.

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

## Current decision: repair the measured writing failure

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
separate architectural restriction: a 384-channel linear output predicts a
768-dimensional velocity. In the continuous model, 384 Gaussian-noise directions
remain uncontrolled. The unchanged registered image evaluation now records a
writing failure. Test a full-dimensional velocity path before spending more on
the same narrow writer. Preserve the present model as a control. Removing this
restriction still requires actual image and language evaluation; the audit
does not establish the sole cause of failed answers.

## Current evidence and unresolved tests

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
- If a repaired writer still fails, localize the next failure using matched
  reference-image history versus generated-image history and visual-input
  controls. Measure visual access, conditional writing and feedback drift
  separately. A reference-supplied diagnostic is not autonomous inference.
  Do not infer that a larger model, more epochs or unrelated physics will
  repair the failure without a testable mechanism.
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
