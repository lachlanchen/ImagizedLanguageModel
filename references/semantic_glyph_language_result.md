# Partial image-language transfer through a complete-glyph loop

The continuous glyph loop produces some readable complete answers to unseen
Chinese question images, but remains unreliable. The inspected outputs
include `288条` and `1874年`, both new complete strings relative to the
508 supported training answers, and `2001年`. The last image is pixel-exact,
but that complete answer string occurs in training. Most other development
outputs become malformed repetitions and hit the 64-glyph cap. This is a
bounded improvement over the previous unusable image writer, not a working
general language model.

| Assessment | Cases | Exact canonical-reference images | Mean foreground F1 | Stops | Caps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed training controls | 8 | 8 | 1.000000 | 8 | 0 |
| Chinese development, normal | 16 | 1 | 0.354426 | 5 | 11 |
| Chinese development, question erased | 16 | 0 | 0.075368 | 1 | 15 |

The [complete raw outputs](evidence/semantic_glyph_language/raw_outputs/)
include every normal and erased case. All forty outputs were inspected on
ten source-order sheets before the current canonical-reference comparison.
These cases were viewed in earlier experiments and are not fully blinded or
sealed. There is no independent human semantic grading. The
[output-first notes](evidence/semantic_glyph_language/review_manual_review.json)
retain uncertainty about the damaged Latin forms and the channel-name
response rather than repairing strokes toward the references.

The [canonical scores](evidence/semantic_glyph_language/review_canonical_scores.json)
compare actual generated images with evaluator-only prepared glyph images.
The readable `288条` and `1874年` have foreground F1 0.987315 and 0.986612,
respectively, and are not pixel-exact. The channel-name response has F1
0.892499 and is not counted as a correct answer. An exact-bitmap-only
transcription records unmatched glyphs as unknown; it does not snap output
to a nearby font or replace generated pixels. All sixteen references have
prepared canonical coverage. Pixel exactness and this qualitative reading
are different measurements, not interchangeable accuracy estimates.

The runtime receives the original prompt PNG, preserves the native FP32
assistant state, predicts one complete 32×32 glyph and encodes the actual
emitted pixels before continuation. It uses no source vocabulary input/output
table, tokenizer, font renderer, character lookup or external teacher. Its
inherited 7.5B visual/contextual body remains present: this is neither a
small foundation trained from scratch nor a demonstrated efficiency advantage
over text decoding. The 47.128M bridge includes a CNN image encoder, neural
glyph decoder, continuous state readout and learned stopping.

Stage A trains glyph formation separately; its
[clear reconstruction result](semantic_glyph_bridge_result.md) does not alone
establish language. Stage B keeps that encoder/decoder and the large body
frozen, and fits 17,848,833 readout/stopping parameters. Source strings prepare
gold image sequences only. Four of the original 512 training questions are
unsupported by the initial anchors and are recorded as such; 508 questions
provide 3,855 glyph events and 508 terminal events. Development outputs and
hidden states are excluded from training.

Frozen native caching takes 731.67 seconds, peaks at 17.789 GiB and retains
all 4,363 causal states with hashes. On the first sequence, bulk processing
and one-glyph-at-a-time deployment differ by at most 2.29e-5. The current
target is never consumed before the state predicting it. Six thousand
small-readout updates take 42.47 seconds at 0.658 GiB peak. Autonomous
assessment takes 332.59 seconds including 10.01 seconds waiting for host
pressure, at 17.646 GiB peak. These timings exclude inherited pretraining
and independent review. The
[protocol](semantic_glyph_language_protocol.md),
[causal cache receipt](evidence/semantic_glyph_language/cache/report.json),
[fit report](evidence/semantic_glyph_language/fit/report.json) and
[assessment](evidence/semantic_glyph_language/assessment/report.json)
remain separate.

The CPU review verifies every PNG against its actual binary tensor, all
emitted-image hashes and every stop/cap decision. This verifies the tested
recording paths; independent re-encoding of the saved feedback embeddings
has not yet been completed. All raw tensors remain local and are identified
by hash. Public resource receipts omit controller command lines while
retaining the local originals' hashes and measured admission information.

The next frozen test changes passage years without further training. Its
two parents are explicitly selected from these observed successes; it is
an exploratory intervention, not a sealed benchmark. The
[prospective protocol](semantic_glyph_counterfactual_protocol.md) fixes
fourteen new normal date conditions and their controls before inference.
Tracking those changes would strengthen evidence for input-dependent
composition. It would not resolve the eleven capped normal development
streams, open-ended language, English or historical writing.
