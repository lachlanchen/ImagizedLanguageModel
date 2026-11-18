# Why the writer reproduces a passage instead of answering

Asked how many of five bronze seals remained after two were sent away, the new model wrote the entire pictured premise and stopped. Asked which script appeared in the right column of a printed handout, it again reproduced the entire passage. Both copies are pixel-exact. The expected answers were 三枚 and 小篆. The 628,104-parameter model can write legible Chinese in this setting. Choosing the content requested by the question remains the limiting step. [Figure 77](../publication/ilm-nature/figures/Figure_77_grounded_glyph_writer.png)

## Generation from an empty answer

| Training questions | Test questions | First glyph / 571 | First four / 571 | Complete and stopped / 571 | Requires new content ink / 300 |
|---|---|---:|---:|---:|---:|
| Correct | Correct | 213 | 144 | 32 | 0 |
| Correct | Rotated | 185 | 126 | 32 | 0 |
| Reassigned | Correct | 217 | 153 | 37 | 0 |
| Reassigned | Rotated | 215 | 152 | 36 | 0 |

All complete matches came from the 177 contiguous references. Both trained writers missed every rearranged reference (94) and every input-absent reference (300). A question-blind first-sentence rule still gave 66 complete matches. Testing the paired writer with the original question gained 42 correct four-glyph openings and lost 24 compared with rotation (net +3.2 percentage points, paired 95% bootstrap interval +0.4 to +6.0). Against the writer trained on reassigned questions it gained 34 and lost 43 (−1.6 points, interval −4.6 to +1.4). This misses the predefined opening and composition criteria. [Protocol](grounded_glyph_writer_protocol.md) · [Physical recount](evidence/grounded_glyph_writer_20261119/audit.json)

A strict comparison with the human reference localized the 539 errors. In 522 responses the writer chose a different glyph while both sequences still had content. Ten ended at a strict prefix of the reference and seven emitted the full reference before adding excess content. In 214 of the 300 input-absent cases the first disagreement occurred before the first reference bitmap absent from the input. Of the 144 correct four-glyph openings, 112 failed to complete the answer. The writer began by copying the first four passage glyphs in 247 cases and reproduced the entire passage in 33. Every output contained visible ink. Exact agreement can miss good paraphrases; this run has no independent human semantic grading. [Public probes and exact output images](evidence/grounded_glyph_writer_20261119/figure.json)

## Choosing between two supplied answers

A follow-up test paired length-matched examples within the original groups. Each pair used the same combined evidence strip with two genuine questions and two candidate answers. Each writer scored both complete candidates under each question, first with one passage order and then the reverse. Among 285 pairs, the matched writer preferred both corresponding answers under both orders for 25 pairs; the reassigned control did so for one. The matched writer gained 24 pairs and lost none (paired 95% interval +5.3 to +11.9 percentage points). Among 150 input-absent pairs the counts were 20 and one. This establishes a measurable effect of the original question pairing on the relative scores of given answers. [Follow-up design](grounded_answer_choice_protocol.md) · [Recorded probabilities and counts](evidence/grounded_glyph_writer_20261119/answer_choices.json)

The options also have a strong lexical shortcut. Choosing the answer with the highest Jaccard similarity between sets of question and answer bitmaps solved 231 of the same 285 pairs in both orders. Scoring the largest product of question-to-passage and passage-to-answer similarities across the two supplied passage regions solved 251. Both baselines give no credit for ties. Shared shapes alone distinguish many questions from their unrelated foils. The current writer still performs far below these simple comparisons. A more informative subsequent language test needs competing plausible answers in one passage and question changes that preserve much of the shared ink while reversing the requested relation. [Independent choice recount](evidence/grounded_glyph_writer_20261119/choice_recount.json)

## Reproduction and cost

One RTX 4090 D trained each writer from scratch on 191,007 retrieved pictured question–passage–answer triples and then on 2,565 existing training records. Each arm took 2,152 optimizer updates, about 40 seconds and 1.04 GiB peak PyTorch allocation. The writer can emit any of 7,715 training-answer bitmaps or copy an input shape. Targets and stops were reachable at all 14,860 scored positions under true prefix images. These measurements exclude data preparation and external baselines. The study measures no efficiency ratio at matched language quality. [Fit and four-condition evaluation](../scripts/train_grounded_glyph_writer.py)

With the local research weights and packed bitmap memory present, the following command runs the matched writer from a blank answer. Pass aligned image strips with the corresponding image flags to use an entirely pictured input. The command writes the raw bitmap stream, a PNG and a receipt with the actual length and stop decision. The corpus, checkpoints and raw human examples remain local under DuReader's research terms.

```sh
python -m scripts.read_grounded_glyph_answer \
  --passage-text '庫房原有五枚銅印，送走兩枚。' \
  --question-text '庫房現在有幾枚銅印？' --output /tmp/ilm-grounded-seals
```

The next prepared test uses two distinct questions originally written about the *same* passage in DRCD. Retain pairs only when both questions tie or favour the other answer by direct image-set overlap, for every recorded answer alternative. This leaves 7,108 training pairs on 3,905 passages and 1,042 development pairs on 521 passages. A physical recount checked all retained pairs and acceptable span variants using exact fractional similarities. This is an extractive reading cohort. Its preparation does not measure a new model or the ability to write absent content. [Preparation and audit](evidence/grounded_glyph_writer_20261119/prepared_followup.json)
