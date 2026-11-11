# Frozen full-prompt proportional-raster comparison

This registration implements the prospective
[input-presentation decision](glyph_tail_input_presentation_decision.md).
Use the unchanged earlier final checkpoint and all 32 current QA questions,
with question/passage-erased controls. The fixed-cell control remains the
complete matched-cohort run. This is a zero-update inference comparison.

CPU preparation authenticates each complete source prompt against the verified
visual corpus. The original newline becomes one ordinary space, with every
non-whitespace codepoint retained. The exact earlier `CompleteLineRenderer`
configuration and font/runtime provenance must match the training identity.
Measure each full prompt before rendering and set capacity from that extent.
No source-answer position, text or length determines a crop, capacity or input.

Pango color-isolated renderings identify passage, question and structural ink
at their original UTF-8 byte ranges. The three masks must be disjoint and
their union must equal every original foreground pixel. Controls erase the
corresponding original pixels, without rerendering the shortened prompt.
Normal and erased strips use the same original safe blank-column wrap ranges.
Every folded image must invert to its complete original strip. The existing
image-only CLI preparation must preserve every registered input pixel.
Reject preparation on any missing glyph, clipping, incomplete source,
noninvertible wrap, unsafe field mask or changed source identity.

Bind all 96 PNGs, renderer/mask audit, evaluation view, original control,
checkpoint and source hashes before generation. The evaluation view reuses
the byte-identical original references and unchanged scorer, with QA-only
input coverage. No book measurement is added. Runtime reads only registered
images, numeric task controls and frozen weights, and rereads its actual
emitted signed glyph pixels. Keep the deterministic writer, learned stopping
and 128-glyph cap. No tokenizer, font, source reference, answer bank or teacher
enters deployed generation. All ambiguous, capped and blank streams remain
in their original denominators.

One serial GPU 0 job uses the project mutex, 48-GiB host/20-GiB free-GPU
admission, 24-GiB runtime host reserve and 22-GiB peak allocated-VRAM cap.
The attempt has 3,600 wall seconds, including at most 600 pressure-wait
seconds. Guard every native-body forward. Preserve completed cases and
interrupted attempt costs, with source validation before resume. Do not touch
another project's GPU service.

Review all 96 raw outputs opaquely before their mapping and references.
Transcribe readable nonsense faithfully and mark unresolved glyph identities.
Disclose prior familiarity and model review without independent human
adjudication. Keep complete exact QA, both distinct questions correct,
whole-response readability, natural stopping and caps separate. Resample
whole paired passage groups for the same pointwise uncertainty analysis.
Compare all normal and erased cases and preserve the three old readable
outputs. A positive development decision requires increased exact QA,
paired success and readability, retention of all three readable cases and
no increase over 24/32 normal QA caps. Otherwise follow the helper-only
native-reader diagnostic decision before another fit. Any gain needs new
untouched source groups before a broader claim.

A separate traced replay uses `qa_01_a_normal`, fixed before outputs. Compare
original signed pixels, contextual states, stopping and actual feedback.
This execution check cannot qualify language. This protocol contains no
completed result.
