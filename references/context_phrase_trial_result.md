# What did the pictured phrase contrast teach the writer?

The 2.33-million-parameter writer learned to choose both assigned continuations for all 39 supervised pairs. On 15 new pairs, its only two-sided success distinguished belief from disbelief. On eighteen prespecified readings drawn from 192 new book passages, it produced zero faithful translations. A separate writer given positive rehearsal alone reached the same totals. Both parts of the registered improvement criterion failed. [Protocol](context_phrase_trial_protocol.md) · [Scores and complete emitted images](evidence/context_phrase_trial_20261119)

| Method | Training pairs, both choices / 39 | New pairs, both choices / 15 | Faithful book translations / 18 | Both independently written details / 15 | Edits on 192 passages |
|---|---:|---:|---:|---:|---:|
| Starting visual writer | 13 | 0 | 0 | 2 | 7,660 |
| Rehearsal | 39 | 1 | 0 | 2 | 9,076 |
| Rehearsal with crossed phrases | 39 | 1 | 0 | 3 | 9,160 |
| Source ink overlap | 32 | 1 | — | — | — |
| Phrase composer | — | — | 1 | 5 | 5,497 |
| Literal copy | — | — | 0 | 12 | 5,745 |
| Local Qwen3-8B reading text | — | — | 11 | 10 | 5,330 |
| Archive or authored reference | — | — | 15 | 12 | 0 |

Given 师从弟子学礼。, the crossed writer emitted 他们的老师跟弟子学礼。 The reader credited the teacher learning from a pupil. Given 父问子，子答之。, the same writer emitted 父亲说他的计问儿子，你回来答应。 The son's answer disappeared. Both examples came from the authored probes. The latter appears in the figure with its prespecified reversal. The credited teacher–pupil case was selected after scoring. [Original glyphs and editable figure](../publication/ilm-nature/figures/Context_phrases.pptx)

## Why the loss fell

Holding the pictured prefix fixed makes a simple test possible. A source-independent writer gives the same ordering of two phrases on both sides and cannot earn strict credit for the pair. Crossing the sources produced the correct sign on both sides of all 39 familiar pairs. Yet a count of candidate glyphs present in the source already solves 32. Original and crossed training also differ from evaluation in the wording of the supplied history. Training uses book-opening fragments such as “听见” and “后来”. Evaluation often asks about a particular role or event, for example “回答的是”. The results cannot isolate which change caused transfer to fail.

The test included 6 archive pairs and 9 authored pairs. Twelve pairs had both initial continuation glyphs absent from both source images. On the 30 one-sided choices, the starting, rehearsal and crossed writers scored 15, 16 and 15, respectively. Both trained writers ranked the same belief pair correctly in both directions. Each independently produced the exact supervised opening and phrase on only 5 of 78 familiar inputs. The autonomous writer receives no supplied prefix and must choose when to copy, retrieve, draw and stop. This limits what a correct conditional ranking can establish about generation.

A model reader assessed 46 distinct sources in method-masked order. This was a single judgment per distinct answer. The reference itself received 15/18 complete credits. Explicit comparisons and computed totals were graded separately from accurate translations of the given facts. The 192-glyph limit was reached before a learned stop 15 times for rehearsal and 16 for the crossed writer across 210 inputs. Neither produced a fully credited pair of complete translations. [Frozen grades](evidence/context_phrase_trial_20261119/reading/credits.tsv) · [Individual scores](evidence/context_phrase_trial_20261119/evaluation.json)

## Single-card resources and next test

Each continuation made 873 updates on a single RTX 4090 D, with one full replay of 55,872 original book pairs and 3,492 presentations of paired sources. Rehearsal took 39.8 seconds and the crossed fit took 34.2 seconds. These durations exclude inherited training, data preparation and text-teacher inference. They cannot support a training-efficiency claim against Qwen. [Checkpoints](../models/visual_edit_zh/context_phrases)

A higher priority than repeating these pairs is to train the **actual free writer on the same pictured instruction it will receive at deployment**. Pair a source with two different questions, and a question with two minimally different sources. In each case the model must emit a standalone answer image from its normal start state. Holding the question alone or source alone fixed makes shortcut learning measurable. Reserve unseen chapters, source constructions and question paraphrases. Measure whole-answer accuracy on these crossed inputs and independent book translation as separate outcomes. Require improvement against a writer with identical positive examples, source copying and a question-only predictor before scaling. MIXAR also studies the difference between supplied histories and self-generated histories. Its results on a different model and benchmark motivate testing the writer along its inference path. [MIXAR](https://arxiv.org/abs/2604.11575) · [PIXAR](https://aclanthology.org/2024.findings-acl.874/)
