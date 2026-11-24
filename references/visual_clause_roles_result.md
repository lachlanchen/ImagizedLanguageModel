# Reading the same glyph in two grammatical roles

On 429 pairs from held classical Chinese material, the reader located the same glyph in both passages and drew both contextually registered English meanings on 110 pairs. The preceding reader completed 49. The 429 pairs reuse passages and represent 124 target bitmaps. The test includes only pairs with distinct registered glosses. For 言 the revised model wrote *speak* in 夫子言之 and *speech* in 汝安得爲此言. A differently worded action question on the first passage elicited *it*.

| Complete written outcomes | Prior reader | Distilled question reader | Glyph counts given the intended role |
|---|---:|---:|---:|
| Same glyph in different roles, 429 pairs | 49 | 110 | 38 |
| Same verb and object with opposite word orders, 18 pairs from four other books | 0 | 4 | 5 |
| Four classroom questions, 539 held passages | 101 | 253 | 179 |
| Four classroom questions, 29 earlier independent excerpts | 1 | 7 | 8 |

The joint reader trained on 27,776 clauses. Supplied the intended roles, uniform and order-balanced supervision each located both words in both passages on 8/18 opposite-order pairs. Exchanging endpoint supervision reached 9/18. The first three neural fits all misread an English classroom action question and received zero full bilingual credit. Adding 54 reviewed question pairs from a local teacher repaired that question. Both question classifiers recognized the same 9/12 independently reserved pairs of phrasings. None of the 429 meaning contrasts passed all twelve. The teacher runs only during data preparation.

The separate image replay checked 1,944 actual replies, including wrong locations and wrong glosses. This test measures short English answers in a fixed typeface. Long-form generation, manuscript handwriting and a compute comparison at equal language quality remain open.

[Protocol](visual_clause_roles_protocol.md) · [Question distillation](visual_question_distillation_protocol.md) · [Full scores](evidence/visual_clause_roles_v1/transfer_result.json) · [Image replay](evidence/visual_clause_roles_v1/replay.json) · [Model](../models/visual_clause_roles/README.md)
