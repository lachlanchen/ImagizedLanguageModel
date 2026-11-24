# An everyday question can change the written answer, with limited transfer between books

On the same classical excerpt 還其珠, the complete visual reader drew *return* for the action and *pearl* for its object. On 買其櫝 it located the object, a box, and drew *capper*. All 116 replies to the independent classroom questions on the new books reproduced from the released CPU model with both inputs read from PNG files.

| Complete passages | Previous reader | Broader reader | Exchanged targets | Correct positions supplied |
|---|---:|---:|---:|---:|
| Four classroom questions, earlier five books | 0/539 | 98/539 | 0/539 | 363/539 |
| Four classroom questions, two new books | 0/29 | 1/29 | 0/29 | 5/29 |
| Eight original order-sensitive questions, earlier books | 230/539 | 227/539 | 0/539 | 363/539 |

The Chinese classroom pair improved from 4 to 240 complete earlier passages. The English pair dropped from 135 to 107. For four earlier conversational pairs, scores changed from 0, 33, 0 and 2 to 188, 66, 132 and 226. These results use free writing. Individually grammatical English words, a correct source character alone and spelling corrections after generation earned no extra credit.

The new books comprised 18 excerpts of the *Zhuangzi* and 11 of the *Han Feizi* after three overlap exclusions. Fourteen had a fronted object. Across all four classroom questions the new reader located both words on six of 29 passages, all with postposed objects. Narrow editorial gloss lists also matter. The model's *amass* for 聚 missed an accepted list including *gather* and *accumulate*. No readings were changed after the run.

With fixed writing weights the complete-score gain equals newly correct locations minus newly lost locations on passages where the writer can express both accepted answers. Lean checked this identity and its upper bound. The release audit recovered 98 repairs and zero losses on the earlier books, and one repair and zero losses on the new ones. This is an accounting result. Whether learning transfers requires the empirical test. [Lean proof](../formal/span_list/QuestionTransferCeiling.lean) · [Replay](evidence/everyday_gloss_v1/release.json)

Only 33 of 58 new target positions had any registered translation in the writer's original training records for the same printed glyph. All 28 correct supplied-position outputs occurred in that subset. Only six passages had training support for both target glosses. For the object pronoun 之 the writer had 211 occurrences, all glossed *go*. This identifies a missing supervision source as well as errors in finding the requested word. [Word-level coverage](evidence/everyday_gloss_v1/lexical_coverage.json)

Each matched fit used 19,067 clauses, twelve passes and 1,788 updates. The aligned fit took 26.1 seconds, 235 MiB peak PyTorch allocation and 2.63 kJ of device energy on one RTX 4090 D. These figures cover the cached focus fit. Source preparation, initial writer learning, cache construction and evaluation are additional costs. [Registered comparison and prompt sources](everyday_gloss_protocol.md) · [All scores](evidence/everyday_gloss_v1/result.json) · [Run the released model](../models/everyday_gloss/README.md)
