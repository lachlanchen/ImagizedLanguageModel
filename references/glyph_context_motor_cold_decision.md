# Fixed cold replay for the actual-context motor

The candidate's standalone inference entry point is
`scripts/infer_glyph_context_motor_image.py`. It takes one physical input
image, the authenticated parent/motor checkpoints, a fixed seed and an output
cap. It does not call the evaluation runner's training-data validation or load
a training manifest. It keeps the final EMA, frozen neural E/D and stopping,
one sampled motor trajectory per glyph and actual emitted-pixel feedback.
Code availability is not a passed runtime-independence audit.

Before inspecting any new candidate answer pixels, select the same first
fixed-cell normal book case used by the ancestor's cold control:
`fixed_cells__book_01_normal`. Use its original physical image, manifest seed
and128-glyph cap, with no supplied reference or target length. After completing
the registered generation stages, run that case in a new traced process using
the standalone entry point. Bind the source and both checkpoint hashes.

Compare every actual output pixel, contextual state, stop probability,
continuous feedback embedding hash, sampled noise and normalized motor feature
against the retained original case. Verify the displayed PNG and stopping
status. Inspect file and network access traces for forbidden training,
reference, vocabulary-weight, font, OCR, bank or teacher access. Source weights
and fixed numerical controls remain permitted and disclosed. An optional
downstream transcription must never alter output pixels.

Retain failed attempts and report setup/generation/pressure costs separately.
A successful replay is one execution and independence check; it cannot turn
an unreadable or incorrect answer into a language success. If the selected
case is capped or damaged, retain that exact failure. Do not choose another
case for a better demonstration. Keep one heavy project worker and the
existing full-reader resource ceiling; do not launch this replay while either
generation stage is live. Cold inference is not yet run or passed.
