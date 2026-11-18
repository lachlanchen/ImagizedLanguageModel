# Sentence context changes when the Chinese phrase writer stops

Two visual writers learned from 120,628 pairs of insertion sites in classical-to-modern Chinese. In each pair, the nearby writing and answer length matched, but the two passages called for different phrases. The model trained to match phrases to passages produced 2,529 exact stopped images across 24,048 sites. Its equally trained likelihood control produced 2,516 and the frozen parent produced 2,143. Whole-phrase counts from the enlarged corpus produced 3,554. The writer receives a pictured classical sentence, its insertion boundary and the preceding modern writing. [Figure 55](../publication/ilm-nature/figures/Figure_55_sentence_choice.png) · [All 48 contextual readings](evidence/sentence_choice_20261118/reading_gallery.pdf)

| Exact stopped phrases | Discovery | Replication | Holdout 1 | Holdout 2 | New chapters |
|---|---:|---:|---:|---:|---:|
| Sites | 5,376 | 5,413 | 5,987 | 6,435 | 837 |
| Frozen writer | 639 | 589 | 434 | 455 | 26 |
| Authentic likelihood | 739 | 685 | 523 | 531 | 38 |
| Paired assignment | 741 | 690 | 526 | 534 | 38 |
| Expanded phrase counts | 1,033 | 945 | 751 | 766 | 59 |

The new cohort contains all 84 eligible sentences from 75 chapters after 7,221 earlier training and evaluation chapters were excluded. All rendered successfully and 73 chapters contained insertions. The 95% chapter-bootstrap interval for the paired writer's exact-phrase gain on those chapters was 0.34–2.66 percentage points relative to the parent and −4.28 to −0.56 relative to counts. In every cohort the interval for the difference between the two new writers included zero. Neither writer generated any of the 7,504 test instances of phrases absent from the expanded training inventory. [New cohort and uncertainty](evidence/sentence_choice_20261118/reserved.json) · [Chapter exclusions](evidence/sentence_choice_20261118/reserved_roster.json)

## Drawing and ending

The paired writer correctly ended the phrase 一定 before the rest of the supplied translation began with 会. Its parent wrote 一定会. In another passage both generated 是 where 人之所以 required the insertion 所以. Both pictured answers reproduced on CPU from the public weights. Across all multi-glyph targets, extra training **reduced** complete successes from 138 to 60. The updated writer ended after a strictly shorter correct prefix 2,143 times, compared with 1,197 for its parent. [Original figure inputs](evidence/sentence_choice_20261118/figure.json)

| Active part of the paired residual | Exact of 24,048 | Exact multi-glyph of 13,041 | First glyph correct | Stopped |
|---|---:|---:|---:|---:|
| Frozen parent | 2,143 | 138 | 4,920 | 21,445 |
| Drawing only | 2,160 | 134 | 4,931 | 21,391 |
| Ending only | 2,529 | 67 | 4,920 | 24,024 |
| Drawing and ending | 2,529 | 60 | 4,931 | 24,021 |

Each control feeds its actual generated pixels back into the writer. Updating only the ending computation reproduced the aggregate gain in complete answers. The two sets of individually solved cases can differ even when their totals coincide. [Path-control protocol](sentence_choice_ending_control.md) · [All path-specific scores](evidence/sentence_choice_20261118/ending_control.json)

## Context and meaning

A study-involved model reader graded anonymously ordered, deduplicated alternatives from 48 distinct new chapters, with preceding and following modern text visible. Both new writers received 10 acceptable, 34 incompatible and four uncertain judgments. The frozen writer received eight acceptable, local counts 14 and recorded translations 47. Independent human review remains open. [Frozen grades](evidence/sentence_choice_20261118/reading_result.json)

On 797 pairs from different chapters with identical local visual features, the paired writer preferred the authentic assignment 453 times. Actual generation yielded 199 exact phrases with authentic sources and 193 with exchanged sources. Among the 679 pairs whose two first signatures differed, the authentic recorded first glyph outranked the rival at 688 of 1,358 sites. The model generated only 258 of these first glyphs correctly. At 1,093 of the 1,100 errors, it assigned greater probability to the emitted 48-pixel signature than to the recorded signature. Seven errors favored the recorded one. More accurate search alone cannot make the recorded signature the maximum at the other 1,093 sites. [First-glyph scoring method](sentence_choice_first_glyph_protocol.md) · [First-glyph scores](evidence/sentence_choice_20261118/first_glyph_probe.json)

## Training and use

The 240,000-sentence corpus yielded 1,234,724 eligible sites. Each residual used the same 120,628 training pairs in three passes and 6,720 pairs from separate training chapters for calibration. Both models learn from the probabilities of the complete image sequence and its ending. The paired arm also rewards the authentic assignment of the two phrases over their crossed assignment. For either arm, the 49,408 trainable parameters transform the source-boundary and pooled sentence states. Full-strength updates had the lowest authentic loss among the four prospective calibration choices. [Mathematical protocol](sentence_choice_protocol.md) · [Fit and costs](evidence/sentence_choice_20261118/fit.json)

Preparing the training pairs took 19.6 seconds. Preparing frozen trajectories, training and calibrating both models together took another 62.4 seconds with 1.89 GiB maximum allocated device memory on one RTX 4090 D. These figures exclude corpus acquisition, training the inherited weights, evaluation and external comparisons. Public CPU inference loads 2,100,046 neural parameters and compressed physical-frequency tables. Required files total 29.8 MB and the uncompressed count arrays occupy 76.5 MB. The requested efficiency advantage over a matched-quality general language model remains unmeasured. [Runtime accounting](evidence/sentence_choice_20261118/runtime_cost.json) · [CPU inference](../models/visual_edit_zh/README.md#conditioning-the-writer-on-the-source-sentence)

The training entry point is `scripts/train_sentence_choice.py` and the independent physical evaluation is `scripts/evaluate_sentence_choice.py`. The earlier paired corpus and frozen visual writer are prerequisites. The first-emission experiment will directly compare recorded glyph images against rival images the parent actually selected on training passages. Its ending rule will remain fixed so an improvement must enter through what is drawn. [Next training comparison](first_emission_learning_protocol.md)

The parent has now generated its real first image at all 241,256 training sites and 13,440 calibration sites. In training, 184,447 first signatures differ from the reference; 24 reference bitmaps cannot be reproduced exactly by the fixed renderer even when supplied with their measured signature. The next residual has yet to be trained. [Training preparation](evidence/sentence_choice_20261118/first_emission_preparation.json)
