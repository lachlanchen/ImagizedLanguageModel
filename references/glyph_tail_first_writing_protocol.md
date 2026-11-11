# Saved first writing and supplied-image reconstruction

This CPU-only audit implements the
[bounded decision](glyph_tail_first_writing_decision.md) after the complete
native-reader comparison. It measures the original glyph actor without a new
autonomous generation or optimizer update. It does not substitute a first-image
benchmark for complete language answers.

## Data and compatibility before neural scoring

Keep all 192 saved QA streams: fixed cells and proportional input, 32 questions
each, normal and both erased conditions. All 64 normal cases are primary;
erased first images remain separately labeled controls. Authenticate the
original checkpoint, actual signed/binary/PNG streams, first states, stopping
and feedback. Each first state must match the completed adapted native
diagnosis. Keep original complete-answer scores unchanged.

Obtain the complete original target arrays from the authenticated visual corpus.
Use its Noto Sans CJK SC face, collection index 2, size 26, common baseline 26
and 32-square binary threshold. Before neural reconstruction or output-error
scoring, the same rasterizer must reproduce all 4,096 selected primitive
characters and every current target tile exactly. Verify font and source
identities, not just family names. Stop on a mismatch before interpreting
cross-image scores. Preparation performs no neural forward call.

Count both character membership and exact-bitmap membership in the original
4,087-image primitive set, for all 32 first positions and every full target
answer position. Report missing forms and complete-answer coverage without
removing any question. This inventory is evaluator-only and never installed
in the model. Membership is training coverage, not a hard output vocabulary or
a proof that a continuous decoder cannot render an unseen form.

## Frozen supplied-image control

Load only the current bridge on CPU. Supply each canonical target tile to its
frozen image encoder and decode the resulting normalized representation:

\[
 \widetilde I=\mathbf 1[\sigma(D(E(I_{\rm target})))\geq 1/2].
\]

Score all complete target positions in batches of 32, with FP32 CPU arithmetic
and two threads. Report the first 32 positions, all positions, and covered
versus missing primitive groups separately. This reconstruction supplies the
answer and measures physical support; it is not autonomous language and does
not use the native lexical decoder. No CPU recomputation replaces original
GPU output pixels. Do not add a vocabulary, teacher or font to the deployed
answer path.

For each original first image and supplied reconstruction, retain ink counts,
TP/FP/FN, pixel exactness, per-image foreground F1 and aggregate foreground F1.
Both blank arrays receive F1 one; a nonempty-versus-empty comparison receives
zero. An absent first emission is distinct from a blank image and receives no
first-image exactness or F1 credit. Keep initial stopping, whole-answer stops,
caps and emitted lengths. Canonical pixel disagreement does not prove an
incorrect first character or semantic answer; near-identical learned strokes
can retain the intended identity.

Report all normal cases and the descriptive groups defined by the previously
frozen native/glyph exact outcomes. Do not select only native successes,
change canonical matching or supply a post-hoc semantic metric. Show all
32 normal first-target/first-output comparisons at unchanged nearest-neighbour
scale, including failures. This follow-up and its displays are chosen after
the earlier outcomes, with prior output familiarity and no new blind review.

## Verification, budget and decision

Independently recompute every pixel counter from saved target, original and
reconstructed arrays; verify board/source hashes, all denominators and CUDA
remaining uninitialized. Bind the registration and actual checkpoint. The
preparation and reconstruction stages each have 600 wall seconds including
at most 120 pressure-wait seconds, two CPU threads, 4 GiB peak process RSS,
32 GiB available host RAM at entry and 24 GiB during execution. Pause for
memory-pressure averages above 0.5% some or 0.1% full. Hide CUDA and launch no
contextual body, external teacher or heavy GPU job. Preserve failed attempts.

Poor supplied-image reconstruction or missing training primitives motivates a
physical representation/coverage intervention. High reconstruction with early
conditional pixel deterioration directs attention before generated feedback;
it does not by itself identify the correct learning objective. Near-correct
first forms with poor complete streams require later writing, stopping and
feedback tests. Choose one bounded next intervention from these results and
the completed earlier failed recipes. None of these controls qualifies the
full independent visual-language goal.
