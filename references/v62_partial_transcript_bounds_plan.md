# Post-result sensitivity plan for incomplete V62 transcripts

5 November 2026. This plan is written after observing three anonymous invalid
reader responses, before associating their hashes with conditions or gold
answers and before calculating any actor transcript aggregate. Calibration
passed at 122/128 references with every blank/noise control empty. The original
diagnostic must finish unchanged and report its registered invalid-transcript
decision if any actor response is invalid. This is a post-hoc sensitivity
analysis, not a new primary benchmark or a replacement decision.

After the evaluator finishes and cleans up, reuse its first cached responses
only. Do not make model calls, retry invalid outputs, parse a truncated text
prefix, repair pixels, alter normalization or remove a case. Reconstruct the
same anonymous PNG hash from every saved primary prediction and check its
bytes against the existing anonymous image and recognition receipt. Keep the
same 256 IDs and all four original conditions.

For valid transcripts, compute the registered exact normalized text and
bounded character accuracy against all retained references. For each invalid
response, retain score interval [0,1] for each bounded metric. Empty actor
tensors have known empty text. Average interval endpoints over all 256 cases;
report valid/unknown counts per condition. CER is unbounded for unknown text
and receives no filled-in aggregate.

For paired differences, use per-case interval
`[lower_left - upper_right, upper_left - lower_right]`. Apply the registered
article-grouped bootstrap with 2,000 draws, seed 20266379, to both endpoint
vectors. Report a sensitivity envelope from the 2.5th percentile of lower
draws to the 97.5th percentile of upper draws. This covers the missing-score
assignments within that resampling scheme; it does not account for systematic
vision-reader errors, training-seed variation or reused-development selection.
Do not label the envelope an independent model qualification.

Write separate reports under `vision_transcript_diagnostic/partial_bounds/`
with protocol/source/data/primary/evaluator hashes. Leave the completed primary
and original diagnostic reports immutable. Any later paper paragraph must
identify the analysis as post-hoc, disclose invalid counts and calibration,
and preserve both failed decisions. No new training or final test is authorized
by this sensitivity plan alone.
