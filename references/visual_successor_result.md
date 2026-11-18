# What visual context tells a predictor about the next Chinese glyph

The final native image-language model ranked the **same candidate glyph first for both histories in 509 of 512 articles**. It chose the right glyph independently for both histories in only two articles. Each pair came from one held-out paragraph, shared its final visible glyph and offered two successors absent from either history. This identifies a concrete obstacle to writing a new word. Even with real candidate images and intact preceding text, the writer's strongest preference almost always survived a change of context. The result does not identify a unique cause of the independent writing failure. [Independent recount](evidence/visual_successor_20261118/audit.json)

A 121,312-parameter visual predictor fitted 1,047,284 next-glyph examples in 19.46 seconds on two CPU threads. It exceeded chance on paired assignment, but its differences from initialization and rotated supervision were uncertain. A model that counted the previous two bitmaps gave the strongest assignment score. It made 177 correct assignments, 47 incorrect assignments and tied on 288. [Visual measurements](evidence/visual_successor_20261118/visual.json)

| Predictor | Correct assignments, with half credit for ties | Both glyphs independently correct | Fit time on this test's training budget |
| --- | ---: | ---: | ---: |
| Native writer at initialization | 281/512 (54.9%) | 39/512 | — |
| Native writer after 13,785 updates | 280/512 (54.7%) | 2/512 | Different training corpus and hardware |
| Compact visual predictor at initialization | 273/512 (53.3%) | 6/512 | — |
| Compact visual predictor, rotated supervision | 260/512 (50.8%) | 65/512 | 20.52 s |
| Compact visual predictor, aligned supervision | 282/512 (55.1%) | 45/512 | 19.46 s |
| Character-indexed predictor, aligned supervision | 288/512 (56.2%) | 55/512 | 18.73 s |
| Smoothed physical trigram | 321/512 (62.7%) | 40/512 | Unmeasured |

The paired assignment asks which of two possible pairings has the greater total score. It can succeed when both histories individually prefer the same glyph. Reversing the first 31 glyphs of each history while keeping the final glyph lowered the compact visual predictor's assignment score by 8.2 percentage points (95% article bootstrap interval 2.3 to 13.9). Its advantage over rotated training was 4.3 points (−2.0 to 10.7), and its advantage over its own initialization was 1.8 points (−4.5 to 8.0). Neither neural visual model reached its prespecified learning criterion. The rotated control changed 1,033,560 training labels. [Comparison design](visual_successor_protocol.md)

## Cost of the input representation

The matched character-indexed model used 584,448 parameters. The two predictors consumed the same ordered passages and 1,086 optimizer updates. Visual minus character-indexed assignment accuracy was −1.2 points (95% interval −6.8 to 4.5). The visual fit used 4.82 times fewer parameters and took 3.9% longer in this single timing per condition. Source preparation, evaluation and process startup are outside the fit timers. Neither model generated free answers in this comparison. Time to a matched general-language quality level and a 1,000-fold speed advantage over Qwen remain unmeasured. [Character-indexed measurements](evidence/visual_successor_20261118/character.json) · [Cost comparison design](visual_successor_cost_protocol.md)

To reproduce the two experiments with the project's prepared source corpus and native checkpoints, run these commands from the repository root. They save weights, article identities, four candidate scores per article and measurements under `artifacts/visual_successor_v1` and `artifacts/visual_successor_cost_v1`. Input pages come from CMRC 2018. The successor test uses only the article text. [Source dataset](https://github.com/ymcui/cmrc2018)

```sh
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. python scripts/measure_visual_successors.py
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. python scripts/compare_visual_successor_cost.py
```
