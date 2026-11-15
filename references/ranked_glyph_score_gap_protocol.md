# Source and emitted image scores at nonempty bitmap mismatches

The completed nonempty-ranking writer reduces blank outputs, but exact source
bitmaps change only from 4,603 to 4,609 of 12,597. Fixed-order examples include
recognizable simple marks that differ from their sources. The previous
whitespace diagnostic compares emitted and blank images; it does not report
whether the density prefers source or emitted images at nonempty mismatches.

This is a new descriptive calculation on the already completed diagnostic's
saved arrays, without a neural replay, optimizer step, new image or new review.
The cohort has already been examined. Binding this calculation before opening
its score differences does not make it untouched confirmation or independent
human adjudication. Preserve every old result, frozen reading and failed gate.

## Fixed calculation

Use all 124 cases and 12,597 supplied-history positions. Authenticate the
completed diagnostic's registration, report, probability vectors and original
source/output pixels. Retain all seven groups and every original case. Let
`S=log q(source|h)` and `E=log q(emitted|h)` be the saved complete-image log
probabilities at the same actual learned context. At positions where both
images contain ink and their binary bitmaps differ, count:

- source preference: `S > E + 0.001`;
- emitted-image preference: `E > S + 0.001`;
- numerical ties: neither strict inequality.

These three counts must partition every nonempty bitmap mismatch. Retain
exact bitmap matches, source-white positions, false blanks and false ink as
separate counts. Verify original totals: 12,597 positions, 252 white sources,
37 false blanks, 25 false-ink outputs and 4,609 exact bitmaps. Blank flags must
agree with the original binary arrays. Identical source/emitted images must
have log scores agreeing within 1e-5 nat. All vectors must be complete and
finite and all saved argmax disagreements zero. Any failure stops this analysis;
it does not authorize recomputing or repairing the old neural outputs.

For each of the three original book strata (development fixed, training fixed
and training proportional), require at least half of its nonempty mismatches,
rounded upward, to prefer the source by the fixed margin. A stratum with no
such mismatches does not supply evidence for this diagnostic rule. Report each
denominator and all seven groups; a pooled majority cannot replace the three
separate comparisons.

If all three pass, existing complete-path scores support a later test of where
greedy prefix decisions diverge: do not claim that complete-image source
ranking alone is the missing repair. If any fails, a search-only repair is not
supported by this rule for the observed nonempty mismatches; conditional
source-mode learning remains a candidate for a later controlled comparison.
Either outcome first waits for the running white-target trial's unchanged
qualification and language decisions. This calculation authorizes no new fit,
decoder, intermediate selection or autonomous generation.

## Interpretation and execution

Greedy physical-mask choices need not maximize complete-image probability.
Conversely, preferring a source over one saved image does not prove that any
practical search will find it, or that it beats every other image. The test
does not measure global optimality, prefix logits, glyph meaning, alternative
valid wording, complete answers or independent language learning. A bitmap
mismatch alone is not a semantic error. Results concern supplied source-image
history, not autonomous feedback.

Implement and CPU-test the count partition and decision, then commit the
implementation. Register exact new source hashes and existing input hashes
without opening probability arrays, and commit/push that registration before
the calculation. Use one CPU operation with two threads, at least 48 GiB
available host RAM, at most 2 GiB RSS and 180 seconds. Verify the original live
white-target worker identities and all 115 training-source hashes before and
after. Use the project CPU-analysis mutex; start no GPU job, heavy build or GUI.

Preserve failures and receipts. Publish the complete result and verify it with
an independent scalar count implementation over the same original arrays.
Retain source, output, probability and registration hashes. Local CPU fixtures
verify accounting only; neither fixtures nor this descriptive result establish
a useful language model or replace the current training experiment.
