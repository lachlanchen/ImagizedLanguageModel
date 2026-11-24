# Visual context recovers a previously missing sense of 之

A lexical filter had removed the bracketed grammatical person annotations from the writer's training data. All 211 retained occurrences of the glyph 之 were glossed *go*. We restored image supervision for direct objects tagged with first, second or third person and tested whether the revised writer could keep the motion and pronoun readings apart on a previously unseen printed context. [Data diagnosis](everyday_gloss_result.md) · [Registered comparison](pronoun_writer_protocol.md)

| Marked-location test | Examples | Parent | Aligned residual | Within-headword exchange |
|---|---:|---:|---:|---:|
| Canonical object pronouns in reserved works | 393 | 0 | 365 | 344 |
| Motion in reserved works | 8 | 8 | 4 | 2 |
| Both readings of 之 in reserved works | 8 pairs | 0 | 4 | 2 |
| Canonical object pronouns in reserved sections | 723 | 0 | 700 | 638 |
| Motion in reserved sections | 40 | 40 | 31 | 7 |
| Both readings of 之 in reserved sections | 40 pairs | 0 | 31 | 4 |

For the paired test the centre glyph is identical, the surrounding ink differs, and the answer has to switch from *it* to *go*. Each motion instance is matched with a different pronoun instance by a fixed content hash. The held-works motion set contains only eight examples. The parent retained more motion readings than either residual. Canonical person results were 987/1,011 for *it*, 71/87 for *me* and 7/18 for *you*. Much of the single-pronoun accuracy is available from the headword and its marginal target frequencies. [All scored replies, including failures](evidence/pronoun_writer_v1/held_replies.json)

The free-running writer first encodes a 17-cell printed window into its existing 160-dimensional state. A two-layer residual adds a context-dependent correction before the English speller. English target glyphs and stop events supervise the correction, with a quadratic penalty on state changes for retained lexical examples. The 15,410 training windows included 4,225 restored pronouns, 210 motion examples and 10,975 lexical examples. Twelve passes, including matched additional motion exposures, used 1,548 updates. The incremental fit took 9.70 seconds, 117.3 MiB peak allocated memory and 918 J on one RTX 4090 D, including encoding the residual training windows. The full model also required the earlier image writer and reader training. The original cuDNN pre-update attempt failed because the frozen GRU was in evaluation mode; the successful run used a training-mode GRU with no dropout and frozen weights. [Aligned fit](evidence/pronoun_writer_v1/aligned_fit.json) · [Control fit](evidence/pronoun_writer_v1/exchanged_fit.json)

With the broader pictured-question reader and its predicted source positions fixed, all-four classroom scores rose from 98 to 101 of 539 earlier passages. On the two new books, both glosses were correct at supplied positions for ten of 29 passages, up from five. The end-to-end score stayed at one. No question located both words across both languages on any of the 14 fronted-object excerpts. A new writer can repair a word only when the reader reaches it. [Complete paired and transfer scores](evidence/pronoun_writer_v1/result.json) · [Run on real input images](../models/pronoun_writer/README.md)

The standard object pronoun images *me*, *you* and *it* measure person. The source tags alone do not specify a human antecedent or distinguish singular from plural in every context. Lean verifies that one underspecified tag cannot determine two distinct reference translations without more evidence. The separate new-book test retained its original contextual reference lists. [Formal statement](../formal/span_list/UnspecifiedReferent.lean)

The derived classical Chinese windows and annotations are adapted from version 2.16 of the Classical Chinese Kyoto Universal Dependencies treebank, annotated and converted at Kyoto University's Institute for Research in Humanities. We selected fixed-width printed contexts and rendered canonical person glosses for this evaluation. The derived dataset is shared under CC BY-SA 4.0. [Source treebank](https://github.com/UniversalDependencies/UD_Classical_Chinese-Kyoto/tree/r2.16) · [Data licence](https://creativecommons.org/licenses/by-sa/4.0/)
