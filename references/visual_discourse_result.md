# Predicting and writing language from optical cells

A causal optical writer trained from scratch improved prediction on unseen English and Chinese books and instruction responses. It often produced recognizable fragments and repeated them without answering the question. All four released demonstration images reproduce bit for bit from their actual PNG inputs on CPU.

| Held set | Pages | Initial bits per glyph | Trained | Trained with another prompt | Correct eight-glyph starts |
|---|---:|---:|---:|---:|---:|
| English fiction | 128 | 5.66 | 2.64 | 2.83 | 0 |
| Chinese fiction | 24 | 10.68 | 7.40 | 7.67 | 0 |
| English instructions | 128 | 5.49 | 2.16 | 2.78 | 10 |
| Chinese instructions | 128 | 10.22 | 5.62 | 6.47 | 3 |

The likelihood columns supply preceding target glyphs. The last column runs the writer on its own output. Several continuations of an instruction or book passage can be valid. Literal prefix matching alone gives an incomplete picture. On the first held English instruction the actual model repeats words and fails to give an energy-saving action. On the first Chinese instruction it echoes the subject of alliteration without giving five alliterating words.

| Held contrast pairs | Independently trained, both written | Paired training, both written | Byte input, both written | Paired training, both ranked |
|---|---:|---:|---:|---:|
| WinoGrande reserved families (1,015) | 60 | 62 | 84 | 134 |
| WinoGrande validation (237) | 15 | 12 | 16 | 27 |
| External Winograd (120) | 0 | 0 | 0 | 13 |

The paired optical writer produced the exact stopped answer on 18/240 individual external variants. Where the question requested the colour of a box containing a key, it reflected a red-to-blue change in Chinese. On 16 authored bilingual contrasts the paired writer included the desired term and avoided the opposing term on both variants only once. Those questions are investigator-authored diagnostics.

The two warm fits saw identical draws totalling 45,419,196 supervised cells each. Each continued arm saw 6,145,370 supervised cells and evaluated another 223,735 foil cells. Each representation had 8,000 warm updates followed by 2,000 fine-tuning steps per arm. For the deployed paired optical writer the two stages took 182.9 seconds and 25.92 kJ of device energy, with at most 0.70 GiB allocated on one RTX 4090 D. These numbers exclude the controls, data preparation, host and evaluation. Synthetic instruction data can share values and translated examples across languages. The assessment assumes an aligned preparation font.

The full contrast likelihood is a diagnostic. Two opposite signed log odds must reverse under a context change and overcome answer bias. The [Lean proof](../formal/span_list/TwoPageMargin.lean) establishes the exact integer-margin identity for this decision. Text quality and transfer require independent generation. [Schedule and raw scores](evidence/visual_discourse_v1) · [Model and real PNG examples](../models/visual_discourse) · [AI2 WinoGrande](https://winogrande.allenai.org/) · [GOLEM](https://aclanthology.org/2026.acl-short.39/)
