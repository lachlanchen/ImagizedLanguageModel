# Mandarin word order after one epoch of glyph prediction

The 23.55-million-parameter image-language model preferred the grammatical sentence in 284 of 552 pairs (51.4%). The same reader at initialization preferred it in 299 pairs (54.2%). A model counting transitions between adjacent training bitmaps scored 57.2% with half credit for ties. The experiment found no reliable improvement in word-order preference over initialization.

| Optimizer updates | Correct preferences | Accuracy |
| ---: | ---: | ---: |
| 0 | 299/552 | 54.2% |
| 3,447 | 286/552 | 51.8% |
| 6,893 | 272/552 | 49.3% |
| 10,339 | 265/552 | 48.0% |
| 13,785 | 284/552 | 51.4% |

Each pair contained the same rendered images, shared its first cell and differed in order. All 23 eligible ZhoBLiMP paradigms contributed 24 pairs. Neither complete alternative occurred in the fitted training pixels. This screen permits overlap in words and phrases. The baseline counted 2,571,477 next-image transitions and 13,785 endings from the same training targets. It won 288 comparisons, lost 209 and tied 55. The neural model produced no ties at any checkpoint. [Benchmark and sample](visual_word_order_protocol.md)

The descriptive 95% interval from resampling whole paradigms was 43.8–59.4% for final accuracy, −19.6 to +14.7 percentage points for the change from initialization and −16.8 to +6.3 points for the difference from the bigram. Related templates limit inference to other constructions. The observed decline of 2.7 points does not establish that training harmed grammatical knowledge.

The scorer supplied both complete alternatives and used their conditional writing-code densities and learned termination decisions. It measured a reading preference. Each additional generated cell in the separate autonomous evaluation instead depends on the model's own previously drawn pixels. The two fits took 1,306.37 seconds combined on one RTX 4090 D. A matched-quality speed ratio requires a common achieved level of language performance. [Fit measurements](evidence/continuous_image_training_completed_20261118.json) · [Word-order measurements](evidence/visual_word_order_20261118.json)
