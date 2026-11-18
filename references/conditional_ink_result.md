# What the phrase writer learns from the preceding strokes

A 50,072-parameter update to the released visual writer improved exact stopped insertions on four held-out cohorts. The largest relative gain was 405 to 434 on the third cohort. On the new cohort the writer scored 455 of 6,435, its parent scored 440 and whole-phrase counts scored 670. A model reader accepted 13 of 48 contextual insertions from either neural writer. The supplied classical sentence, insertion location and correct preceding modern writing define this task. [Original outputs and comparisons](../publication/ilm-nature/figures/Figure_54_conditional_ink.png) · [All 48 contextual cases as images](evidence/conditional_ink_20261118/reading_gallery.pdf)

| Recorded measure | Discovery | Replication | Previous holdout | New holdout |
|---|---:|---:|---:|---:|
| Insertion sites | 5,376 | 5,413 | 5,987 | 6,435 |
| Parent exact | 618 | 564 | 405 | 440 |
| Updated writer exact | 639 | 589 | 434 | 455 |
| Local whole-phrase counts exact | 868 | 808 | 651 | 670 |
| Updated writer exact, multi-glyph targets | 45/2,534 | 32/2,679 | 30/3,448 | 27/3,806 |
| Exact instances of unseen phrase types | 0/1,554 | 0/1,727 | 1/2,433 | 0/2,617 |
| Updated writer stopped | 4,983 | 4,917 | 5,174 | 5,679 |
| Exact among sites with conflicting phrases at identical local inputs | 83/395 | 54/433 | 37/534 | 40/599 |

All four chapter-bootstrap intervals for improvement over the parent lie above zero. On the new cohort the 95% interval was 0.05–0.43 percentage points. The writer trailed whole-phrase counts by 2.63–4.08 points. The expanded evaluation covers four sets of 1,024 source pairs; six or fewer pairs in each set could not be rendered. Of the newest cohort's 561 chapters, none contributed training material, and 520 supplied other sentences to older translation evaluations. The parent and the update share their original training corpus. [Full cohort results](evidence/conditional_ink_20261118/confirmation.json) · [New source roster](evidence/conditional_ink_20261118/confirmation_roster.json)

**A success with a specific limit.** Given the classical account of an eldest son anticipating his brother's pardon and the preceding modern rendering, the writer inserted 所以来 (“therefore came”) immediately before 告辞 (“to say farewell”). The parent wrote 所以这样. Across 64,384 training pairs 所以来 never appeared as a complete aligned insertion, but the sequence appeared in six modern training sentences, including once in a longer insertion. This result shows reuse of observed writing in a new insertion context. In the second figure example the model produced 一定会; the required insertion was 一定 and the next position already supplied 会. Both examples reproduce on CPU from the released weights. [Sequence exposure](evidence/conditional_ink_20261118/novel_sequence_exposure.json) · [Pixel counts](evidence/conditional_ink_20261118/physical_recount.json)

**Contextual acceptability.** A study-involved model reader evaluated anonymously ordered and deduplicated answers at 48 new sites across 48 chapters. Half the reference insertions were multi-glyph. The updated writer received 13 acceptable and 35 incompatible judgments, as did the parent. The phrase-count control received 20 acceptable, 24 incompatible and four uncertain. Recorded translations received 43 acceptable, one incompatible and four uncertain. Seven accepted responses from each neural writer differed from the reference. The small stratified sample used surrounding reference text. Independent human review remains open. [Frozen ratings](evidence/conditional_ink_20261118/reading_result.json)

**Which source sentence matters?** In 786 pairs from different chapters, both sites had identical nearby visual signatures and different target phrases. We exchanged the entire classical sentence and cursor while preserving each modern prefix. All local drawing and ending inputs were identical at corresponding reference positions. Summed probabilities favored the authentic assignment on 422 pairs (53.7%), the exchange on 362 and neither on two. The model regenerated 158 of 1,572 phrases exactly with the authentic source and 151 after exchange. It changed 648 outputs, including 270 first glyphs. Regrouping one discovery case into a different inference batch changed its output pixels, with the same exact-answer grade. The other 1,571 authentic outputs reproduced. The comparison uses the same batch layout for both assignments. It captures sensitivity to the broader source, including its length and register, and the difference in correctly chosen phrases remains small. [Protocol](conditional_ink_sentence_exchange.md) · [New-cohort exchange](evidence/conditional_ink_20261118/confirmation_exchange.json)

## Learning rule, inference and costs

Each glyph is drawn from twelve four-pixel measurements. The new component adds a learned correction to groups two through twelve. A 96-unit network reads the actual four neighboring 48-pixel signatures, one missing-neighbor flag per cell and a phrase-onset flag. For the current group, its eight coefficients mix eight learned 16-by-16 matrices. Each matrix is centered over its rows and columns. The previous four drawn bits select the row; the sixteen logits score the next four bits. Each beam branch uses its own physical history. This lets sentence context change the joint probabilities of stroke groups with 200,892 bytes of additional weights. The original renderer, whole-sentence reader, feedback, local count model and ending rule stay fixed. [Model](../ilm/visual_lm/conditional_ink.py) · [Training protocol](conditional_ink_protocol.md)

Preparation and fitting took 23.0 seconds including 18.7 seconds of fixed-logit extraction on one RTX 4090 D. The optimizer saw 558,955 training glyphs in each of three passes. The selected global multiplier was 1.0 on 68,483 calibration glyphs from separate training chapters. The parent had seen those chapters. Peak allocated GPU memory was 543.5 MiB. On the newest test cohort the mean reference-history loss for the eleven affected transitions fell from 8.088 to 7.875 nats per glyph. These times exclude creating the corpus and training the inherited writer. The evidence does not support general language competence or a thousandfold improvement over a language model at matched quality. [Fit](evidence/conditional_ink_20261118/fit.json) · [Released CPU model](../models/visual_edit_zh/README.md#condition-on-the-previous-strokes)

## Learning from longer sentence context

All four cohorts failed the registered language criteria. To test whether the sentence supplies meaning beyond nearby strokes, the next training corpus contains 240,000 distinct paired sentences (175,616 additional), 1,234,724 insertion sites and 350,406 distinct recorded phrases. Training and calibration pairs are matched on the four local physical signatures **and** insertion length while requiring distinct chapters and different answers. Preparation retained 120,628 training pairs, including 53,623 multi-glyph pairs, and 6,720 calibration pairs. All 1,715 chapters used by the four phrase tests are excluded. The corpus build verifies disjoint sites, distinct targets and exact matching at every retained pair. Prepared examples alone establish no additional model quality. [Preparation receipt](evidence/conditional_ink_20261118/sentence_choice_preparation.json) · [Matched learning comparison](sentence_choice_protocol.md)

## Reproducing the experiment

The public CPU inference example uses the released weights and accepts either pictured or typed input. To repeat the research fit, use a separate working directory with the preceding experiments’ prepared data, writer checkpoints and archived control streams. The scripts retain the first written results. Run these commands at the repository root of that copy. [CPU example](../models/visual_edit_zh/README.md#condition-on-the-previous-strokes)

```bash
export PYTHONPATH=.
python scripts/prepare_local_ink_holdout.py --conditional-ink
python scripts/train_conditional_ink.py
python scripts/evaluate_conditional_ink.py
python scripts/exchange_conditional_ink_sources.py
python scripts/read_conditional_ink_phrases.py
```

The final command prepares anonymous alternatives for reading. Record judgments before revealing the assignments, then pass the ratings file to `scripts/read_conditional_ink_phrases.py --score`. The reporting script recounts generated pixels and checks the two published examples on CPU. The gallery script draws the actual answer strips. The separate corpus preparation script constructs the source-matched pairs for the next experiment. [Reporting script](../scripts/report_conditional_ink.py) · [Gallery script](../scripts/render_conditional_ink_gallery.py) · [Pair preparation](../scripts/prepare_sentence_choice.py)
