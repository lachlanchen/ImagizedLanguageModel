# Reading a verse line and writing its companion

A model trained from scratch on one RTX 4090 D filled both missing cells in Li She's pictured line □□余也是何人. It supplied 不知 after reading 縱使雞鳴遇關吏. Exchanging the first line changed its prediction to 不是. This was the first exchange-sensitive exact repair among 6,362 confirmation examples. With an entirely blank second line it generated 不來不是是人人. A successful short repair and a full poetic response remain quite different achievements. [Input and output PNGs](../models/verse_canvas/README.md)

## Sources and controls

In the preceding book diagnostic the frozen visual reader saw two passages from the same volume with identical final two glyphs. Its joint likelihood preferred the original pairing with two supplied eight-glyph endings on 1,276 of 2,131 pairs (59.9%, volume-bootstrap interval 56.4–63.9%). It preferred both correct endings independently on 208. A repeated-glyph count reached 58.5% with half credit for ties. The image reader resolved 195 of those 347 ties correctly. The matched byte reader scored 60.3%. All 2,131 pairs tied after truncating both contexts to their common two-glyph suffix. The same passage and ending occur on each side of the joint comparison, cancelling additive single-passage and single-ending preferences. [Paired likelihoods](evidence/book_assignment_v1/results.json)

For writing, 57,607 Tang poem records yielded 202,647 fitting pairs of five or seven glyphs, from 3,100 author strings. Both sides of a held pair were excluded if either line occurred in the fitting set. The first experiment sampled 4,096 pairs from 106 held attributions. Each final masked writer had six or fewer remaining positions after its first commitment. The image model recovered 761/23,114 glyphs when it filled the most confident cell first and 744 when it filled the leftmost cell first. The difference was 0.07 percentage points (95% paired author-bootstrap interval −0.05 to 0.16). Seven complete two-cell gaps were restored by confidence-first writing and eight by left-first. A matched byte reader restored 15. [Initial comparison](evidence/verse_canvas_v1/results.json)

The Lean derivation shows that if action scores approximate the true mass of consistent completions within ε, committing the highest-scoring position and shape loses at most 2ε units of first-step mass. A concrete distribution with perfect scores gives 21/100 mass to its fully greedy line and 29/100 to the best line. Calibration at one step cannot establish exact recovery of a whole expression. [Checked derivation](../formal/span_list/VerseCommitment.lean)

## Visible-glyph supervision

The second experiment used the output classifier to identify visible glyphs during fitting. At inference the source features continue directly into the reader. The added training loss uses a canonical bitmap class for each visible cell. The confirmation cohort comprised all 6,362 previously reserved pairs from 156 other poet attributions. Each row below uses the same lines and gaps.

| Writer | Blank-line glyphs / 37,442 | Missing glyphs / 12,724 | Complete gaps / 6,362 |
|---|---:|---:|---:|
| Image | 1,281 | 456 | 20 |
| Image + identity | 1,392 | 502 | 17 |
| Bytes + identity | 1,483 | 592 | 30 |
| Aligned glyph counts | 1,587 | 526 | 24 |

The image writer's blank-line gain was 0.30 percentage points (0.17–0.42). Switching the pictured first line reduced its correct positions to 340. For 6,336 exchanges the donor shared the same poet attribution, and every donor had the same length. The original source contributed to 16 of 17 exact two-cell repairs. All four neural conditions and the aligned counts recovered zero blank reference lines. The identity-trained visual writer repeated at least one glyph in 3,464 of 6,362 blank outputs. Only 359 recorded lines repeated a glyph. The model inserted a pair already seen in fitting on 5,656 pages, versus 4,459 attested reference pairs. It repeated the same glyph in both gaps on 310 pages, versus 49 references. Raising the prior for familiar phrases would favour many current errors. [Phrase recount](evidence/verse_canvas_v1/phrases.json) [Predictions and intervals](evidence/verse_canvas_v1/confirmation.json)

With the source printed in Droid Sans Fallback, nearest-shape retrieval improved from 1,651 to 5,519 of 8,414 identities in learned feature space. Direct comparison of raw 16 × 16 ink densities identified 5,475. On the same unseen font both visual writers recovered 857 blank-line glyphs. The recognition gain transferred to some short restorations, with 350 correct gap glyphs for the identity writer versus 283 for the base. Optical similarity alone does not specify a fitting reply.

## Cost and reproducibility

Each writer contains 3,282,782 trainable parameters. The same 10,000 batches presented 960,000 pairs with 3,749,390 masked target cells. The original image and byte fits consumed 8.78 and 8.81 kJ of measured board energy. The supervised image and byte fits consumed 10.14 and 10.16 kJ and took 87.27 and 85.69 seconds. Peak allocated GPU memory for the supervised image fit was 343 MiB. Costs exclude host preparation. The measurements provide no evidence for an efficiency advantage at equal language performance. [Training receipts](evidence/verse_canvas_v1/ink-bound-fit.json)

The confirmation evaluator saved all predictions before its initial JSON summary failed on a NumPy integer. The recovery read those predictions, recomputed their counts and optical diagnostics, and verified the published examples via independent CPU PNG inference. The original confirmation inference energy and timing were lost. All four fitting receipts and the initial test inference receipts were retained. No optimizer was restarted.

[Training and inference](../models/verse_canvas/README.md) · [First protocol](verse_canvas_protocol.md) · [Confirmation protocol](verse_binding_protocol.md) · [Conditional masked generation](https://aclanthology.org/D19-1633/)
