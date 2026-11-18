# Answering two human-written passage questions from their images

A 101,144-parameter reader can extract and order two answers on unseen Chinese passages. Its performance depends separately on identifying the questions and finding their evidence. In one official development example it read questions about the length and operator of the Guangmao Railway. The output image reads “364.6公里；三茂铁路股份有限公司”. Reversing the questions exchanged the two answers. The released CPU command reproduces the original output pixels. [Checkpoint and example](../models/pictured_passage_zh/README.md)

## What changed in training

The passage reader and question planner use physical glyph images. The planner sees complete prompt images and chooses read, skip, punctuation-copy and read-and-stop actions. The first planner saw all 24,846 paired requests and 7,235 singles each epoch for six epochs. It recovered the full supervised plan for 6,077 of 6,086 internal paired requests. Fine-tuning the original passage reader for eight additional epochs assigned likelihood to all aligned occurrences of the correct answer. Mean training loss fell from 2.34 to 1.87, but fewer complete held-out answers were correct. [Training protocol](paired_passage_protocol.md) · [Fit](evidence/paired_passage_20261119/fit.json)

| Internal development, whole image and stop | Ordered requests, 6,086 | Both orders, 3,043 | Singles, 1,788 |
|---|---:|---:|---:|
| Original reader with supplied split | 440 | 220 | 434 |
| Original reader with learned split | 437 | 217 | 434 |
| Refined passage reader with learned split | 423 | 210 | 419 |
| Refined reader with question order erased | 212 | 0 | — |
| Varied prefixes, original reader | 438 | 218 | 434 |
| Same varied weights, constrained decoding | 438 | 218 | 434 |

The refined reader gained 107 ordered requests and lost 124 relative to the original supplied-split composer. A bootstrap over the 425 held-out passages gave a difference of −0.28 percentage points, with a 95% interval from −1.20 to +0.62. The first failure in corpus order substituted a description of an author for their name and educational history for the requested relationship. Only two internal targets occurred as a single contiguous span. [Original source and output images](evidence/paired_passage_20261119) · [Corpus](https://aclanthology.org/D19-1600/)

## Reading instruction words

The official development comparison has 836 distinct passages, 3,170 questions and 9,600 ordered requests. Twelve passages with 49 questions required glyphs absent from the source font. Every retained reference occurs in its passage. Each pair tolerates any combination of the official reference images, and exchanging the questions changes every accepted target. The following counts require the reader to stop and answer both orders correctly.

| Pictured instruction | Fixed-prefix training | Six-prefix training | Same six-prefix weights with constrained decoding |
|---|---:|---:|---:|
| 请依次回答这两个问题： | 518 | 516 | 518 |
| 依次作答： | 0 | 0 | 505 |
| 请按次序作答： | 0 | 461 | 517 |
| Answer in order: | 371 | 369 | 371 |

All columns use the same frozen passage reader on 4,800 pairs. The matched planners each received 1,506 optimizer updates on the same source questions and answers. The 461 gained pairs demonstrate transfer from varied instruction images to one held-out wording. With the shortest wording, the varied planner located both questions on most requests, but introduced extra passage reads. On 8,561 requests it read an extra region ahead of the two questions. A five-state maximum-score path permits a prefix, contiguous questions separated by copied punctuation, and an optional learned stop. Applying it to exactly the same logits recovered 505 pairs in this condition. It still failed on 4,295 pairs. Two ordered requests per official wrapper failed to stop. The official single-question score of the released model was 999/3,170. [Frozen transfer](paired_passage_external_protocol.md) · [Matched training](passage_instruction_protocol.md) · [Decoding comparison](passage_ordered_decoder_protocol.md)

We regraded the 83,140 saved official outputs for the two planner fits and the 41,570 official outputs from the constrained decoder. We independently restored the final checkpoint on CPU and compared the rendered PNG inputs and resulting pixels on the first successful exchanged pair and first failed request under the short instruction. The successful readings establish ordered source extraction. The dominant remaining error is selecting the correct evidence. Arbitrary answer writing and etymological explanation require a writer that can produce content beyond the visible spans. Single-GPU training and inference in this experiment provide no matched-quality compute comparison to a general language model. [Physical recount](evidence/paired_passage_20261119/ordered_audit.json) · [Measured figure](../publication/ilm-nature/figures/Figure_73_paired_passage.png)

## Mathematical context

Reversing two questions gives an intervention that preserves every input glyph and changes the desired output. Any deterministic reader invariant to that exchange can answer at most one order of a pair with disjoint target sets. The measured order-erased control answered zero pairs in both orders. Constrained decoding addresses a different ambiguity. Its inductive bias assigns zero support to re-entering the instruction prefix after reading has started. It still scores each admissible action with a learned function of local ink and relative position. Neither this constraint nor maximizing pixel likelihood supplies knowledge of which company operates a railway. That fact must be selected from the passage or learned from data.

Prior pixel-language models have measured semantic and syntactic information in rendered multilingual text. PIXAR evaluated autoregressive writing in pixel space. The experiment here tests a narrower causal question about order, training variation and the choice of evidence, with available source images and a partly structured decoder. [Kesen et al., multilingual pixel pretraining](https://aclanthology.org/2025.emnlp-main.1504/) · [PIXAR](https://aclanthology.org/2024.findings-acl.874/)
