# Frozen correct-image-history comparison

## Question and fixed scope

The balanced writer reproduces all 64 sampled training tiles exactly but gives
only 6/32 exact proportional QA answers, zero both-correct proportional pairs
and no credited complete book continuation. Its initial states match the parent.
Determine whether later conditional writing also fails with correct preceding
images, rather than assuming generated-history drift is the only problem.

Use the frozen balanced checkpoint
`dcac4ee207614adca78a8b99cc4fe422b3197204d9028192bef30f111d61586d`.
Keep the reader, adapters, image encoder/decoder, writer and stop predictor fixed.
No fit, seed selection, sampler change, new corpus or autonomous generation.
Retain the existing parent/flow/balanced complete-response scores and raw outputs.

Use all 32 existing normal QA cases in both prompt layouts: 64 traces. Reuse the
authenticated 745 reference glyphs, giving 1,490 nonterminal predictions and 64
terminal boundaries. Supply only preceding reference images to the existing
pixel-feedback interface, with its original positions and contextual cache.
Predict one glyph and stop probability at each boundary, bypassing stopping.
The current glyph must not enter its own prediction. No glyph target exists at
the terminal boundary; never score its predicted image against an empty target.

## Identity and measurements

Commit implementation and this decision, then CPU-register source hashes,
checkpoint, all 64 prompts, reference raster archive, previous raw traces,
review scores and existing native evidence. Commit registration before neural
execution. Retain per-case states, predictions, feedback hashes, stop decisions,
geometry and resource reports. Reproduce every saved initial state bit-for-bit.
Re-evaluate saved autonomous states through the same frozen balanced bridge;
require exact emitted-pixel and stop-probability reproduction. First predictions
must match between the two histories whenever the saved actor emits a glyph.
These invariants qualify the instrument, not language quality.

Report per-position canonical foreground F1, exact bitmaps, 0.95 counts, first
versus later positions, per-case averages and the corresponding saved autonomous
positions where present. Missing autonomous positions remain missing. Display
all reference, correct-history and original autonomous streams without repair.
Record every premature stop and terminal stop at the original 0.5 threshold,
and initial/correct-history versus autonomous hidden-state similarity. Join the
existing full-response parent/flow/balanced scores and adapted native exact flags
by immutable case identity. Do not run the completed 448-call native study again.

Independently recount integer pixel counters and stop decisions from raw arrays,
verify all displayed pixels against those arrays, and inspect every board.
Keep reference text, history length and the supplied-history experiment label
visible. The comparison is on reused development cases, with prior reviewer
familiarity and no independent human adjudication.

## Interpretation fixed before execution

Canonical bitmap agreement measures reproduction of one supplied answer, not
semantic accuracy. Alternatives can have different glyphs, lengths and prefixes.
Position-aligned autonomous comparisons are descriptive, not semantic alignment.
A native exact flag is an existing diagnostic, not an oracle for every failure.
No conditional result changes the original complete-answer language scores.

For each layout, describe whether correct-history later glyphs reach median F1
0.99 and at least 90% of positions at F1 0.95; report the first position separately.
These physical thresholds do not qualify the model. If correct-history writing
still fails broadly, reject a feedback-only explanation of canonical writing
failure and investigate conditional transfer/reader-writer alignment before
another fit. If correct-history glyphs are readable but autonomous histories
deteriorate, examine the history dependence before a controlled feedback-training
change. Any such claim must survive inspection for valid alternate wording.
If correct-prefix stop false positives exceed 5% of nonterminal boundaries or
terminal recognition is below 80%, stopping needs an explicitly separated test.
Mixed or ambiguous evidence does not authorize a unique causal claim.

Select one bounded training intervention only after this comparison is finished.
Retain all eight union proportional exact answers and two fixed exact answers
in its prospective complete-response evaluation. Use untouched groups to confirm
any development improvement. Do not extend rejected writer fits.

## Resource ceiling

One GPU0 job under `artifacts/ilm_gpu0.lock`; no changes to GPU1 services. Reuse
the full-reader admission: 48 GiB available host RAM and 20 GiB free GPU memory
at entry, 24 GiB host reserve at runtime and 22 GiB allocated VRAM ceiling.
Allow 3,600 seconds total per execution attempt, including at most 600 seconds
pressure waiting. Preserve failed/interrupted attempts and resumable per-case
files. Verify owner PID/start/boot identity before resuming. No runtime is live
merely because a status file says so; never duplicate a live worker.

This diagnostic supplies reference images and length. It is neither independent
inference nor evidence that the ILM autonomously understands or generates the
reference answer. No learning, efficiency or usability claim follows from it.
