# Clean-history control for the completed full continuation model

The new saved-trajectory result finds dense first and later writing in the final
checkpoint. The previous correct-history study used an earlier actor. This
control tests the new checkpoint, without another fit or another autonomous
answer run. The comparison is the original initial snapshot against the final
41,298-update snapshot; neither snapshot is selected by this control.

Use all 32 normal development questions in both prompt layouts and all twelve
normal development book prefixes in the existing fixed layout. Recover their
complete target images from the authenticated development corpus, without
rendering answer text. Add twelve training QA and twelve training book records
in both layouts, selected by ascending SHA256 of source ID. Include the longest
complete training book by replacing the twelfth selected book if necessary.
There are 124 prompt/history jobs per checkpoint. Keep all target positions.

At boundary t, the input is the learned continuous query followed only by gold
images preceding t. Current and future target images must not enter their own
predictor. Measure the original balanced image loss, canonical tile F1 and exact
matches, output ink, premature stops and terminal decisions. Save all predicted
pixels and stop probabilities. Stopping is bypassed; these are assisted physical
predictions, never autonomous answers or new language scores. Preserve all prior
outputs, scores and uncertain readings.

First run the full longest training book in fixed layout at each checkpoint
through both parallel and incremental answer computation using identical gold
history. Keep the existing .002 maximum-absolute state tolerance and also report
every boundary's error and the fraction of changed thresholded pixels. If the
state tolerance fails, stop this study and resolve the discrepancy before any
new fit; do not silently enlarge the tolerance or interpret it as semantic loss.
This extends the earlier shortest-QA-only check to a long complete target and
the newly trained final checkpoint.

Report every job and aggregate training/development, QA/book and layout
separately. The observational diagnosis uses two fixed conditions: dense output
means more than 90% of predicted tiles exceed the source-kind clean 99th
percentile; accurate later writing means at least 90% of later tiles have
foreground F1 >= .95. Apply these descriptions to each group, without dropping
short targets. A clean-history pass does not establish autonomous language.

- If long parallel/incremental states disagree beyond .002, investigate the
  implementation or numerical distinction first.
- If final training predictions are already dense or inaccurate under correct
  history, reject a feedback-only repair. Investigate the conditional fitting
  objective, state-to-image representation or optimization before rollout work.
- If training succeeds but development fails under correct history, target
  conditional generalization and inherited language alignment; neither more
  unchanged epochs nor a copying repair is justified.
- If correct-history writing succeeds but the existing autonomous streams fail,
  generated-history stability or feedback alignment becomes the next mechanism
  to test, with an unchanged control and full-language acceptance rules.

Bind code, initial/final checkpoints, corpus, selected images, previous analysis
and original frozen language review before neural execution. Use physical GPU0
under the existing mutex, two CPU threads, 48 GiB available host RAM and 20 GiB
free VRAM at entry, 24 GiB host reserve, 22 GiB peak allocated VRAM, 1,800 seconds
total execution and at most 300 seconds of memory-pressure waiting. No model
updates, source output head, tokenizer, teacher, OCR or font lookup are permitted.
Save failed attempts; no automatic retry. The general ILM objective remains open.
