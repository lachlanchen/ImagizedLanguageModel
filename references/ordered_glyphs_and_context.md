# What can a model learn from a stream of glyph images?

The Chinese writer fills a four-character gap between pictured passages. The input and emitted ink are observable, so we can ask two separate questions. Can a reader preserve distinctions that change meaning? Does its writer use those distinctions when it chooses a complete expression?

## Which physical encodings preserve order?

Consider 人咬狗 and 狗咬人. The same three glyphs describe opposite events. If the page representation sums three aligned ink masks, the two measurements are identical. Max pooling or a transparent black-ink overlay has the same symmetry. On an equally weighted pair, a decoder receiving only this measurement can distinguish the originals with at most 50% accuracy. Learning a strong prior can improve guesses on natural text, but the measurement has discarded the evidence for this distinction. Position-dependent channels, successive frames or ordinary row-major placement retain a physical signal for order. An invertible reshape from a stack to a grid retains every bit. Whether a trained reader actually uses this signal is an empirical question. [Shannon, *A Mathematical Theory of Communication*](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)

## How can we detect the effect of context?

For passage image \(X\) and answer image stream \(Y\), the ideal reduction in uncertainty is \(I(X;Y)=H(Y)-H(Y\mid X)\). We do not know the distribution of acceptable book continuations. A test with one recorded answer per passage can measure exact recovery and can penalize a reasonable alternative. A tractable test on this corpus pairs passages sharing the glyphs immediately beside the gap, with different recorded answers. Let \(s(x,a)\) be the learned log probability of the physical production path for answer \(a\), including stopping. Compare the two assignments using

$$
\Delta=s(x,a)+s(x',b)-s(x,b)-s(x',a).
$$

Any additive preference for answer \(a\) or \(b\) independent of the passage cancels. The earlier guided writer had positive margins in 1,956 of 3,056 held-out pairs. With each source used at most once, it had 356 positive margins in 548 pairs. The writer can use a feature beyond the two adjacent cells. Topic, spelling elsewhere in the passage or a repeated phrase could all produce that result. Identifying grammatical understanding would require controlled changes such as exchanging the subject and object while preserving the pictured glyph inventory. [Paired experiment and definition of the score](paired_physical_answer_protocol.md) · [Measured margins](evidence/expression_preference_20261118/paired_diagnosis.json)

## Why can better discrimination yield worse writing?

A loss comparing two answers can improve their ordering even when a third answer outranks both. In a matched training comparison, adding a preference loss increased pairs with both supplied answers correctly preferred from 336 to 585 of 3,126. Exact freely generated answers on the same 8,790 passages fell from 207 to 132. Even the subsequently released reference-trained writer used ten expressions for 54.6% of its answers. On the independent confirmation set, every emitted glyph was a training-known bitmap. The observed errors are principally in choosing the expression on this task. [Training and free-generation results](expression_preference_result.md) · [Output frequency distribution](evidence/expression_output_concentration_20261118.json)

Published work supplies two relevant precedents. PIXAR generated text in pixel space and examined noisy output. PIXEL-M4 included simplified Chinese and probed linguistic representations obtained from rendered text. Here the immediate experiment is to hold the renderer, optimizer, update count and evaluation fixed while varying the number of distinct training passages. Whole-answer recovery, rare-expression accuracy and changes in both answers of context-matched pairs can show whether greater variety buys language sensitivity at the same exposure budget. A claim about GPU savings requires equivalent task quality and accounting for all training stages. [PIXAR](https://aclanthology.org/2024.findings-acl.874/) · [PIXEL-M4](https://aclanthology.org/2025.emnlp-main.1504/) · [Project evaluation criteria](../docs/first-imagized-language-model-goal.md)
