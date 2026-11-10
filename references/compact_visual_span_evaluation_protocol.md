# Held-out evaluation of the compact visual span model

Freeze the input suite before the image heads train. Use sixteen groups from
the visual corpus's held-out CMRC contexts. Each group has two distinct
questions about identical visible evidence and disjoint canonical answers.
Select groups and questions by SHA-256 order. Require that neither answer
contains the other and that the question itself omits the answer. Restrict
canonical answers to the 128-glyph generation budget. These structural
criteria define a paired extractive reading test. Pretraining exposure of the
inherited Qwen weights is unknown.

Take six passages each from the completely held-out works 西遊記 and 欒城集.
End the visible prefix at punctuation, leaving 64–128 original glyphs and
preferring a 96-glyph suffix. Choose eligible passages by the same fixed
SHA-256 rule. Inspect the original output for a readable and coherent
continuation. Also record exact source continuation, since multiple plausible
continuations can follow a prefix.

Every normal question has two controls. In one, blank the question tiles.
In the other, blank the passage tiles. Keep the original number and order of
tile positions and the surrounding template. For books, blank the whole
prefix as a control. These conditions preserve positional cues, including
the length of the missing text. The suite has 44 normal prompts and 76
controls. Write references outside the inference input directory.

Run the fixed initial, quarter, half, three-quarter and final checkpoints on
the same 44 normal images. Also run all controls at the initial and final
checkpoints. Use the same seed, eight flow integration steps and 128-glyph
budget. Record actual optimizer steps and elapsed generation time. Preserve
every emitted tile, including empty or capped outputs, and the pixel hashes
recorded by the feedback loop. No reference-length input is used.

Review the actual generated pages under opaque identifiers before reference
scoring. Give the reviewer the visible book prefix for coherence judgments.
Score a complete QA answer when a clear transcription matches the canonical
answer after whitespace removal and the model stops naturally. Count pairs
with both complete answers correct and distinct image streams. Report
readability, stopping, exact continuation and coherent complete continuation.
Keep all 32 questions and all 12 book passages in the respective denominators.
Record the evaluator and uncertainty in the review receipt. Plot measured
normal-prompt counts across the five fixed checkpoints, with raw images from
the first registered QA pair and first book prefix at each step. Use the
same measurements in the TeX figure and editable PowerPoint figure.

The first standalone replay uses `book_01_normal` with the final trained
checkpoint in a fresh process. Compare tiles, feedback hashes and stop/cap
decisions against the suite output. Trace opened files and network calls to
check the model boundary. This cohort uses the corpus renderer and does not
test arbitrary scanned page segmentation, historical form recognition, or
an efficiency comparison with a named text model.

Generate the suite with `scripts/infer_compact_visual_span.py --manifest`.
Pass the completed generation directories to
`scripts/review_compact_visual_span.py prepare-review --runs`. Give the reviewer
the `pages` directory and `readings_template.json`. Record the reviewer or
transcription method in `evaluator`. Use empty text and mark uncertainty for
unreadable writing. For each normal book output, judge whether its visible
writing continues the prefix coherently, regardless of exact wording. Score
the saved readings with `scripts/review_compact_visual_span.py score`. Use
`scripts/verify_compact_visual_span_cold.py` for the separate trained replay.
