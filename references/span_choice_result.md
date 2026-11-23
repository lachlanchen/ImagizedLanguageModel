# Choosing answer images on Chinese reading questions

**Source correction, 24 November 2026.** Later record checks established that the official DRCD test split was already reserved. The affected traditional-script evaluations describe observed outputs on exposed material. [Exposure audit and replacement evaluation](drcd_source_exposure_correction.md)

On 384 questions from previously excluded paragraphs, the learned selector returned 285 pixel-exact answers. The frozen generic reader also returned 285. They differed on six scored outcomes, with three gains and three losses (two-sided exact paired test, p = 1). The likelihood-only specialist returned 178 and a selector trained with shuffled answer labels returned 225. On the established exchange task, the specialist still gave more complete replies.

| Test | Generic | Factual | Likelihood | Margin | Pooled |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh direct answers, 384 questions | 285 | 232 | 178 | 192 | 285 |
| Earlier direct answers, 256 questions | 187 | 147 | 119 | 121 | 181 |
| Constructed statements with historical name substitutions, 32 | 17 | 13 | 12 | 14 | 11 |
| Human factual answers, both generated orders, 48 | — | 34 | 28 | 30 | 33 |
| Complete passage exchanges, 128 | — | 29 | 79 | 70 | 63 |
| Complete reordered-clause tests, 48 | — | 13 | 37 | 36 | 34 |

An exchange passes only when every pictured question in both assignments leads to a correct, stopped answer in both requested orders. The pooled writer also passed both orders on 921/1,200 other DuIE passages. Three of four prespecified retention requirements failed. The reader remains optional. [Protocol](span_choice_protocol.md) · [Counts and hashes](evidence/span_choice_20261124/results.json)

The first recovered generic error illustrates what exact scoring measures. For **正训的意思是什么？**, the generic answer included quotation marks. The pooled answer **道地的** matched the reference. The first absent proposal illustrates a different boundary error. For **霓虹雀鲷属于什么科目？**, the returned **雀鲷科** gives the family, while the reference also names the class, order and suborder. Exact-match counts include such partially informative replies as errors. [Original image strips](evidence/span_choice_20261124/examples.json)

### Where selection fails

Among 99 incorrect direct replies, 15 had an exact strip available. The remaining 84 lacked it in every reader's top proposal. Keeping the best two, four or eight spans per reader increases maximum exact answers to 319, 337 and 346 of 384. At depth eight, 46 of the 84 missing answers become available. Choosing the correct member of that larger set requires separate training and a fresh test. The extra 46 are potential recoveries. [Prespecified diagnostic](span_choice_coverage_protocol.md) · [Per-question ranks](evidence/span_choice_20261124/coverage.json)

### Training and inference

Four frozen endpoint scorers read the same page and question. A 42,340-parameter neural selector pools probability on proposed strips with identical pixels. We trained on 7,000 broad human questions and 8,704 paired role questions, then selected an epoch on 1,408 other questions. The paired validation subjects were held out from selector fitting but belonged to the earlier specialist training pool. Every fresh question came from a distinct paragraph with a representable source answer within the fixed prefix. Reported exact accuracy therefore applies to this eligible subset. Processing the pictured training and validation cases took 81.0 seconds; fitting both selectors took 4.8 seconds on one RTX 4090 D. The contextual reader and image writer both inherit pretrained language weights, whose training cost is excluded. [Training receipt](evidence/span_choice_20261124/fit.json) · [Corpus hashes](evidence/span_choice_20261124/corpus.json) · [CMRC 2018](https://github.com/ymcui/cmrc2018) · [DRCD](https://github.com/DRCKnowledgeTeam/DRCD)

Lean 4.12 checks that a pooled reference with more than half the total mass must win. For any incorrect winner, at least half the mass lies outside the reference. An answer absent from the candidate set cannot be returned by selection. These proofs concern a decoder with given scores. Learning an open language or attaining a computational speedup are separate empirical claims. [Machine-checked proof](../formal/span_choice/SpanChoice.lean)

The repository includes the trained weights and supports CPU and GPU inference from pictured or typed prompts. For a source checkout with the previous local training artifacts, the data, training and evaluation entry points are `python -m scripts.prepare_span_choice`, `CUDA_VISIBLE_DEVICES=0 python -m scripts.train_span_choice`, and `CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_span_choice`. The evaluation keeps completed receipts immutable. [Run the selector](../models/span_choice/README.md) · [Editable figure](../publication/ilm-nature/figures/Span_choice.pptx)

