# Reading two grammatical arguments from the same pictured sentence

In the held-out sentence about a church renovation, the reader returned 教堂 for the subject of 完成 and 主體大修工程 for its object. This is the fifth eligible test predicate and the first for which both complete images were correct. The matched reader recovered both arguments of 36/106 test predicates, compared with 1/106 after exchanging the role labels and 0/106 before adaptation. The matched advantage over exchange was 33.0 percentage points (95% sentence-bootstrap interval 24.3–41.9). It recovered both arguments for 11/36 predicates absent from the GSD training split. The grammatical questions used fixed wording and the natural treebank sentences retained canonical argument order. [Original images and controls](evidence/pictured_participant_reader_20261119/result.json) · [Figure 87](../publication/ilm-nature/figures/Figure_87_pictured_participant_reader.png)

| Complete response | Previous | Exchanged labels | Matched labels |
|---|---:|---:|---:|
| Both arguments on official GSD test (106 predicates) | 0 | 1 | 36 |
| New verbs in active and passive frames (64 questions) | 17 | 37 | 43 |
| Independently written new situations (64 questions) | 13 | 18 | 18 |
| All four answers correct in those situations (16 scenes) | 0 | 0 | 0 |
| Natural person questions (387) | 148 | 146 | 148 |
| Both natural questions correct (131 pairs) | 19 | 18 | 16 |
| Familiar reordered statements (404) | 344 | 344 | 344 |
| Descriptions with bronze names (32) | 30 | 30 | 30 |
| External Chinese XWinograd completion (504) | 28 | 28 | 28 |

On fresh wording the matched-minus-exchanged estimate was zero (95% interval −10.9 to +9.4 percentage points). It failed the registered transfer requirement. On the external Chinese XWinograd split all gold referents appeared in the pictured source. Seven items have identical option images. The visual reader received the sentence and a request to fill its blank without being given either option. This is stricter than the usual binary-choice evaluation. [External source and license](https://huggingface.co/datasets/Muennighoff/xwinograd) · [Archived outputs and counts](evidence/pictured_participant_reader_20261119/external_referents.json)

Each arm updated 103,426 parameters for 1,080 steps on one RTX 4090 D. The matched fit took 44.2 seconds and the exchanged control 57.2 seconds. This is incremental training on top of an existing image writer. The calibrated writer remains the reference for general use. The next experiment tests whether a continuous image input map can carry previously acquired contextual language knowledge into a reader with a source-image output. [Protocol](pictured_participant_protocol.md) · [Treebank and contributors](https://universaldependencies.org/treebanks/zh_gsd/index.html)
