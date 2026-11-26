# A pictured dictionary reader

Given a pictured English or Chinese request and a structured page with two headwords, this reader writes the original English gloss and one attested Chinese glyph in the requested order. Each selected field contributes its original pixels. Both the question and the field labels enter a recurrent neural selector as images. The deployed model contains 94,866 parameters trained from scratch. An optional prompt renderer accepts keyboard input. [Example page](example/page.png) · [Original question](example/question.png) · [Emitted answer](example/matched.png)

```sh
python -m scripts.write_origin_values \
  --prompt 'For 遘, give the original meaning and then the seal form.' \
  --page models/origin_value_reader/example/page.png \
  --geometry models/origin_value_reader/example/geometry.json --output origin.png
cmp origin.png models/origin_value_reader/example/matched.png
```

Use `--question models/origin_value_reader/example/question.png` in place of `--prompt` for image input. The example gloss reads “Meaning to meet or to come across.” The last image is seal specimen S01257. The JSON supplies occupied cell counts, row extents and the value column. The selector operates on the pictured labels, and the answer copies the selected value cells. The model currently requires this layout, 32-pixel tiles and an English source gloss even when the user asks in Chinese.

Training used 5,112 requests across 1,202 families, four per entry. Held evaluation used 608 requests across 145 other families. Each page offers at least two distinct historical stages per entry. Source-target specimens were screened for identical raster content across training, development and held splits. The model learned from the last epoch of 12 complete passes. Both matched controls received the same data and initialization. [Data manifest](../../references/evidence/grounded_origin_pairs_v1.json) · [Training protocol](../../references/grounded_origin_selection_protocol.md) · [Weights and learning curves](matched/fit.json)

| Exact stopped image | Complete requests | With headword-image matching | Continuous control |
|---|---:|---:|---:|
| Original wording | 608 | 608 | 533 |
| Other headword on identical page, where both stages exist | 514 | 513 | 447 |
| Source values exchanged on the page | 608 | 608 | 533 |
| New phrasing on original pages | 608 | 359 | 147 |

On rephrasing, the match-aware reader earned 228/304 on English prompts and 131/304 on Chinese prompts. Both models replayed all 1,730 original and counterfactual replies on CPU. The additional reworded outputs were generated on CPU. The result demonstrates learned visual selection and ordering of supplied facts and images. Open-ended etymological explanation and unrestricted phrasing require further training and independent evaluation. [Raw original and counterfactual outputs](../../publication/ilm-nature/evidence/grounded_origin_answers.zip) · [Raw changed-wording outputs](../../publication/ilm-nature/evidence/grounded_origin_rephrased.zip) · [Independent recount](../../references/evidence/origin_value_reader_v1/score-audit.json)

To reproduce with the local Hanziyuan collection, run the following modules from the project root.

```sh
python -m scripts.prepare_grounded_origin_pairs
python -m scripts.train_origin_value_reader
python -m scripts.evaluate_origin_value_reader
python -m scripts.replay_origin_value_reader
python -m scripts.audit_origin_value_scores
```
