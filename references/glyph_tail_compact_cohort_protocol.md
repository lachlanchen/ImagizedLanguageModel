# Earlier readable image actor on the compact evaluation cohort

Both compact models fail the fixed complete-answer evaluation. Their
frozen-weight diagnostic also finds poor conditional writing before generated
feedback. An earlier contextual-tail model produced 16/32 clear complete
answers on a different question cohort. This control measures that exact
retained model on the compact inputs before another fit. It does not attribute
any difference solely to parameter count, conditioning or an image loss.

## Preserved model and inputs

Use `artifacts/glyph_tail_adaptation_v2/fit_recovery_02/checkpoint_latest.pt`,
SHA-256 `5c434a9a9b0c6ddc0c3fe4005265e4f8ca5b7d0add9a74a80ed371806fa71085`,
with all 18,000 recorded updates and the original final-four-layer adapters.
Use `scripts/infer_glyph_tail_language.py:load_actor` and the unchanged
`SemanticGlyphLanguageActor`. The native vision/context foundation retains
7,522,463,984 inherited parameters, with the trained glyph bridge, contextual
adapters and source-derived fixed numeric task controls. No new training,
weights, tokenizer, renderer or runtime glyph bank is introduced.

Use all 120 original PNGs from
`artifacts/compact_visual_span_evaluation_v1/inputs/manifest.json`: 32 normal
QA questions, twelve normal book prefixes and the 76 erased-input controls.
Preserve all sixteen question pairs and both book groups. Generation receives
the original physical image. Keep the actor's existing host image preparation
and native bilinear observer; verify that the host step leaves the registered
binary input PNG pixels unchanged. Report the native observer geometry and its
pretrained image-processing cost. This differs from the compact tile encoder.

The source/model/input/code hashes are fixed in a CPU-only registration before
neural calls. Audit current QA IDs and whitespace-normalized original passage
groups against the recorded 3,176 admitted local training questions and the
complete 3,211-question source pool. Also check exact original training PNG
hashes. Do not change cases in response to overlap; retain and disclose it.
Exposure during inherited pretraining and all earlier book/copy ancestors is
not certified by this audit. Prior research/evaluation familiarity remains.

## Generation and recorded outputs

Use a 128-glyph cap, the same cap as the compact evaluation. The earlier glyph
writer is deterministic and uses no flow sampler or stochastic seed; record
that difference explicitly. The original learned stopping decision applies.
Never force the reference length or repair a generated glyph. Retain blank,
wrong, unfinished and capped responses in every denominator.

Load once and execute serially. Preserve the complete original signed glyph
result, its states, original feedback hashes and stopping. Convert signed
binary glyphs to unsigned ink losslessly and verify the round trip. The PNG
grid layout changes serialization only; it supplies no reference, font or OCR.
Verify each signed-image hash against the actor's actual feedback event and
bind the corresponding binary-ink hash and native embedding hash. Independently
check displayed PNGs against saved raw ink before review. Do not invent glyph
segmentation statistics for the native visual reader.

Generation must not open scoring references, training corpora or source
inventories. Registration contains input identities and audit metadata, not
answer text. Reuse the existing fixed controls and weights. A separate cold
replay uses the first fixed book case `book_01_normal`, selected before outputs,
with the same 128-glyph cap and source-dependent native observer. Trace access
and compare original signed pixels, states, feedback and stop/cap decisions.

## Language measurement and decision

Prepare all 120 review pages under opaque identifiers after generation.
Transcribe complete readable output even if nonsensical. Mark uncertain glyph
identities without repairing them toward a question or reference. Inspect the
visible book prefix for any claimed coherent continuation. Freeze all readings
before opening their case mapping and scoring references. Disclose model review,
prior familiarity and the absence of independent human adjudication.

Use the unchanged compact scorer: complete exact stopped QA answers out of
32, both distinct questions correct out of sixteen contexts, readable complete
responses, coherent complete book continuations out of twelve, and natural
stop/cap/empty counts. Compare normal versus question/passage-erased cases.
Use the same passage-group uncertainty analysis where applicable; only two
book works do not support a book confidence interval. Keep the earlier 16/32
clear-answer result on its different cohort separate from these exact scores.

A matched-cohort gain is a development control, not full language qualification.
It would support retaining readable capability while designing a smaller
interface. If the earlier model also fails this cohort, investigate task,
layout and answer-length transfer before treating its earlier restricted
success as a general solution. No outcome of this control selects a new
training recipe automatically. Fresh confirmation, useful language quality,
historical/unencoded forms and matched-quality resource measurement remain
requirements of the full objective.

## Runtime limits and preservation

Use only GPU0 and the existing exclusive project mutex. Reuse full-reader
admission: 48 GiB available host RAM, low actual memory stalls and 20 GiB free
GPU memory at entry; maintain 24 GiB host RAM and a 22 GiB allocated-VRAM cap.
The generation attempt has a 3,600-second deadline and at most 600 seconds
waiting on pressure. A body forward hook enforces the guard during long
answers. Keep another project's GPU1 service untouched.

Save each complete case atomically. A resume must match all original identities
and verify the original case files and feedback; it does not regenerate complete
cases. Preserve partial/interrupted attempts and include their costs in any
resource claim. Run no concurrent heavy project job. GPU0 and any owned GUI
must be released after execution. The model comparison is implemented in
`scripts/compare_glyph_tail_compact_cohort.py`; CPU preparation and generation
are separate phases. This protocol records no completed result.

## Prepared request

The [CPU registration](evidence/glyph_tail_compact_cohort_registration_20261111.json)
binds all 120 inputs and the retained final checkpoint. The
[overlap audit](evidence/glyph_tail_compact_cohort_overlap_20261111.json) finds
zero question-ID, normalized original-passage or original-training-PNG overlap
with the recorded local tail-training data; the complete 3,211-question source
pool also has zero current QA question/passage overlap. Inherited pretraining
and all earlier book/copy ancestors remain outside that certification.

A [CPU serialization check](evidence/glyph_tail_compact_cohort_serialization_20261111.json)
uses an actual retained 64-glyph capped output, verifies every original
emitted-image feedback hash, and round-trips signed pixels through binary ink
and the wrapped PNG grid exactly. It makes no new neural model call and does
not measure language. Model generation has not yet run for the new cohort.
