# Which Chinese answers require a new shape?

A reader can copy every stroke correctly and still answer a question with the wrong person or date. More extractive training helped the pictured passage reader select evidence. To test whether it can compose, we measured which answers a perfect selector could physically spell under three restrictions. The selector in this calculation knows the reference and may use any available shape repeatedly. The figures are upper bounds on exact reproduction of the selected human reference. [Passage reader results](traditional_passage_training_result.md) · [Writing corpus](composed_chinese_answers_preparation.md)

| Shapes available at inference | Reachable references out of 571 | Reachable among the 300 requiring an input-absent content glyph |
|---|---:|---:|
| Current question and passage | 271 | 0 |
| Current inputs and all 2,692 distinct training-answer bitmaps | 560 | 289 |
| Current inputs and all 3,601 distinct training bitmaps | 567 | 296 |

Let $V_i$ contain the distinct tiles in the two input images for case $i$, $T$ the set available from training, and $Y_i$ its reference tile sequence. A selector with unlimited reuse can form that sequence precisely when every tile of $Y_i$ belongs to $V_i \cup T$. The upper bound on the number of exact answers is $\sum_i \mathbf{1}[\mathrm{set}(Y_i)\subseteq V_i\cup T]$. Extra optimization cannot assign positive probability to an unreachable reference. Alternate wordings can still answer a question correctly. [Reproducible calculation](../scripts/measure_composed_answer_support.py) · [Counts by group](evidence/composed_answer_bitmap_support_20261119.json)

Across the 300 cases, 1,525 target positions require a bitmap absent from the inputs. Only 19 of those positions also lack that bitmap in training answers. Seven positions across four answers lack it anywhere in the complete retained training split. Most cases therefore test whether the writer selects a familiar shape in a new linguistic context. Four cases offer too little evidence to estimate performance on unfamiliar historical forms.

## A writer and a control

A direct extension of the reader would model the probability of each complete output tile. At each step a recurrent state would determine whether to stop, copy one pictured cell, or write a cell. For a nonterminal bitmap $b$, the conditional probability after deciding to continue can be written as

$$
p(b\mid h,x)=
\alpha(h)\sum_{i:\,x_i=b}p(i\mid h,x)
+(1-\alpha(h))p_{\mathrm{write}}(b\mid h).
$$

The copy term sums over all occurrences of identical ink. The writing term can first be tested by scoring actual training-answer bitmaps with a shared visual encoder. This finite emitter supplies a strong baseline for familiar-shape composition, with a known ceiling of 560. An open raster writer would have to generate the pixels of $b$ conditioned on language state; its result can be compared with the finite emitter. A normalized patch or bit factorization permits assigning probability to previously unseen images. Whether it produces the right strokes is an empirical question. Text pointer-generators provide the precedent for learning a copy/write mixture. PIXAR gives a precedent for autoregressive language generation in pixels and reports readability problems when training its pixel decoder. [See et al., 2017](https://aclanthology.org/P17-1099/) · [Tai et al., 2024](https://aclanthology.org/2024.findings-acl.874/)

For a squared pixel loss, the optimal prediction is the conditional mean of the candidate images. At a pixel shared by half the plausible answers, that mean is grey even if every training target uses black and white ink. This explains how a low reconstruction error can coexist with an illegible prediction. It does not identify the cause of each earlier failure. Evaluating the chosen bitmap and the completed, independently generated answer tests what a reader actually receives.

The next comparison should freeze the test questions, score free-running answers and learned stops, and record separately where a correct glyph first came from. Include a same-passage question exchange to test dependence on the request, and a blinded reading of answer relevance and factual support where multiple phrasings are reasonable. A subsequent rendering transfer test can vary the font or use an attested historical form. Results on the single canonical grid leave this capability open. Controlled experiments on pixel-text encoders also find that rendering size and layout affect generalization. [Yuan et al., 2026](https://aclanthology.org/2026.emnlp-main.2412/)
