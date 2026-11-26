# Reading pictured spans for translation and historical reference

A change to the source reader raised the number of complete English–Chinese translations on four new books from 20 to 45 out of 192. Both arms used the same fitted visual writer. The candidate recovered source addresses from spans of pictured glyphs and ran the pretrained Marian encoder. The continuous reader used one estimated state per pictured cell. Replacing the smaller reader also changed the deployed parameter count. [Paper](../publication/ilm-nature/manuscript.pdf) · [Raw outputs](../publication/ilm-nature/evidence/pictured_path_answers.zip) · [Fixed comparison](pictured_path_translation_protocol.md)

| Whole translation, stopped and credited in both automatic readings | To English / 96 | To Chinese / 96 |
|---|---:|---:|
| Independent single-glyph writer | 0 | 5 |
| Continuous reader with the visual span writer | 14 | 6 |
| Image-derived source paths with the same writer | 31 | 14 |
| Source paths supplied from text with the same writer | 31 | 17 |
| Exchanged source image | 0 | 0 |
| Qwen3-8B Q4 given text | 84 | 83 |
| Aligned corpus reference | 58 | 65 |

The path reader gained 35 credits and lost ten relative to the continuous reader. The net 13.02 percentage-point difference gave a paired four-book bootstrap interval of 5.21–22.40 points. The automatic judge rated 123 of the 192 references correct and rejected every unrelated foil. The first newly credited Chinese case is a close judgment between similar answers. Independent bilingual human evaluations are outstanding. The complete archive includes 960 candidate and control output images. All reproduced on CPU. [Case votes and reference-informed scores](evidence/pictured_path_translation_v1/assessment.json) · [CPU replay](evidence/pictured_path_translation_v1/replay.json)

**Why correcting input recognition has limited returns.** For 143 requests the inferred path exactly matched the supplied source addresses. All 143 produced identical answer pixels, boundaries and stops. Among them, 103 were uncredited. Supplying the reference addresses for all requests gained four correct answers and lost one. A Lean proof bounds the failures that survive exact path repair, and the paired score audit binds this result to the physical outputs. [Proof](../formal/span_list/SourcePathRepair.lean) · [Recount](evidence/pictured_path_translation_v1/repair-audit.json)

**Compute and print.** The source-path writer uses 90.87M floating values into English and 89.46M into Chinese, plus indices and precomputed ink. The warm medians on one RTX 4090 D were 197.64 responses/s and 0.670 board J/request at batch 96. The continuous reader gave 167.19 and 0.756. A learned print normalizer restored all 6,596 source tiles of a changed face and all 192 answer images. Direct serif input reproduced none of the canonical answers exactly. No matched-quality advantage over the text baseline has been established. [Energy](evidence/pictured_path_translation_v1/gpu-cost.json) · [Typeface transfer](evidence/pictured_path_translation_v1/transfer.json)

**Running the translation.** From the project Python environment, the downloaded source-path weights and pinned Helsinki-NLP checkpoint revisions in the preceding span study support the following CPU example. It reads an original pictured sentence. Add `--normalize-print` for the evaluated alternate print face. [Pretrained checkpoint revisions and environment](optical_span_rollout_result.md)

```sh
python -m scripts.write_optical_span --reader path --to en --device cpu \
  --cells 18 --input publication/ilm-nature/figures/pictured_path/0020-source.png \
  --output answer.png
cmp answer.png publication/ilm-nature/figures/pictured_path/0020-path.png
```

**Independent historical reading.** A separate 94,866-parameter reader trained from scratch on pictured field selection bound the correct source gloss to an attested historical shape on 608/608 structured requests from held character families. Rephrasing the questions gave 359/608. It receives both source values on the page and learns which ones to write and in which order. This addresses open-set visual copying. Freely explaining an absent fact, learning translation through this same independent system and reading an unsegmented scan remain open. [Reader, runnable page and controls](../models/origin_value_reader/README.md)
