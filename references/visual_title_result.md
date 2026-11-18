# What the model learned from pictured Chinese article titles

The trained model wrote **战国无双3** from a pictured 417-cell paragraph and stopped after the fifth cell. Both the archived GPU output and a fresh CPU run with pictured inputs gave exactly the reference bitmap. On another article, the opening said that the page introduces transport in Lesotho. The model copied 这个页面介绍莱索托的交通. The recorded title was 莱索托交通. Both answers are readable; the extra words expose the difference between choosing a span and composing a title. [Figure 39](../publication/ilm-nature/figures/Figure_39_visual_titles.png)

| Target relation to full pictured prompt | Articles | Complete exact title | Contiguous control | Exchanged article |
| --- | ---: | ---: | ---: | ---: |
| One span | 724 | 629 | 630 | 0 |
| Distributed visible glyphs | 46 | 0 | 0 | 0 |
| At least one absent glyph | 64 | 0 | 0 | 0 |

The 223,620-parameter transducer trained on 2,362 pictured articles for 197.5 seconds on one RTX 4090 D. The complete fit and three-condition evaluation took 239.6 seconds, using 107.6 MiB peak allocated device memory. The intact model stopped naturally on 830/834 articles. A separate program checked all 2,502 saved image streams against the reference and verified that every nonempty output of the contiguous control occurs at a glyph boundary in its original source. Of 34 titles containing a shape never seen in any training input or title, 32 were reproduced exactly from visible evidence. [Registration](visual_title_transducer_protocol.md) · [Training and measures](evidence/visual_titles_20261118/result.json) · [Recount](evidence/visual_titles_20261118/recount.json)

An opening-span rule learned eight physical stopping delimiters using only training titles and got 580/834 development titles. The full model's advantage was 5.88 percentage points (95% interval by article 3.84 to 7.91). Of its 629 correct titles, 581 were exact article openings and 48 were elsewhere. These figures measure extraction from a corpus with stereotyped introductions and a single instruction. Other reasonable titles can differ from the recorded reference. [Opening rule](evidence/visual_titles_20261118/lead.json)

Only 490/13,419 training title positions required an absent image. During autonomous evaluation the model drew ten cells in total and drew none on the 64 missing-glyph articles. When supplied the actual preceding title glyphs, it selected drawing at 69/143 absent positions. Evaluating the ink head at every absent position gave 85 exact images: 37 left parentheses, 36 right parentheses and 12 spaces. All 44 absent Han positions missed. The feedback history and the conditional ink head are both implicated; the experiment does not identify a single sufficient repair. The registered composition criterion failed. [Conditional routing](evidence/visual_titles_20261118/routing.json) · [Exact physical identities](evidence/visual_titles_20261118/identity.json)

The uniform-superposition argument, the physical bit-averaging counterexample and the treatment of uncertain source alignments are derived in the accompanying mathematical note. The earlier independent writer scored zero readable answers; preserve its separate blind result. No matched-quality advantage over a general LLM has been measured. [Ordered ink and learning](ordered_ink_learning.md) · [Native free-writing evaluation](native_continuous_blind_result.md)

## Run the trained model

With the prepared titles under `data/visual_title_composition_v1/` and the final checkpoint under `artifacts/visual_title_transducer_v1/`, the CPU inference entry point accepts a typed article or an image of row-major 32-pixel cells. The following command uses the saved first pictured development article. It writes the complete binary stream and, for nonempty responses, a PNG.

```bash
PYTHONPATH=. python -m scripts.infer_visual_title \
  --image artifacts/visual_title_transducer_v1/pictured_first.png --cells 417 \
  --question-image artifacts/visual_title_transducer_v1/pictured_instruction.png \
  --weights artifacts/visual_title_transducer_v1/weights.pt \
  --output /tmp/ilm-first-article-title
```
