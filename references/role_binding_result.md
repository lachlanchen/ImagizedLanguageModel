# Reading a pictured fact as a subject, relation and object

On 48 human-written Chinese questions the visual reader and its induced statement writer produced **33 complete correct sentences**, compared with five under disrupted supervision and three for the preceding evidence-conditioned writer. It returned the reference object on 43/48 as a short answer. The 48 were previously used to diagnose the old system. A second evaluation used 1,200 separate DuIE development passages with generated questions in wording withheld from the new fit. There the complete count was 841. Each returned subject and object came from physical pixels in the passage. [Complete component scores](evidence/role_binding_20261123/results.json)

| Source and question | Cases | Subject | Relation | Object | Complete statement | Disrupted complete |
|---|---:|---:|---:|---:|---:|---:|
| CMRC, original simplified questions | 29 | 24 | 25 | 24 | 17 | 2 |
| DRCD, original traditional questions | 19 | 18 | 17 | 19 | 16 | 3 |
| Independent programme notes and exchanged names | 32 | 30 | 24 | 22 | 16 | 2 |
| DuIE, separate sources with generated questions | 1,200 | 1,182 | 1,078 | 941 | 841 | 48 |

All counts in the complete column also matched the canonical target pixels exactly. Returning the correct object alone does not establish its role. In the first film programme the reader correctly credits 林遥 as director and 闻溪 as screenwriter. On the exchanged version it selects a stretch of prose as the director and extends the screenwriter's name by one character. Only two of the eight programmes pass both questions before and after the names are exchanged. With reference objects supplied, the reader gets 37/48 human statements and 24/32 programme statements. Both subject and relation errors remain measurable.

## What Lean establishes

A factual answer has three fields, each of which can be a sequence of bitmap glyphs. Given a fixed relation, the writing pattern consists of the observed material before, between and after the two participant spans. The constructive learner returns the first labelled pattern for that relation. Lean proves that this identifies its true pattern if examples of the relation exist and are consistent. Under these assumptions it suffices to observe the pattern once to express a new pair of names. Identifying the names and relation from real pages is a separate statistical problem.

Lean also proves that a decoder which collapses two inputs with different correct answers must fail on at least one, and that a correct representative generalizes across name exchanges if the prediction and target obey the same exchange law. The last assumption can fail in a trained network, as the film example shows. On every finite corpus the number of errors from the induced writer is bounded above by the sum of field errors under the same realization of the true and predicted fact. The proofs quantify over arbitrary glyph types and require no Unicode inventory. The build succeeded in Lean 4.12.0 without proof holes. [Checked source](../formal/role_binding/RoleBinding.lean) · [Compiler and axiom report](evidence/role_binding_20261123/lean.txt)

## Data and compute

The adaptation used 26,890 examples on 23,209 distinct pictured passages. These comprised 19,984 DuIE questions, 6,796 questions about deliberately exchanged participants, and 110 weakly labelled human training questions. Subject and full-passage overlap with the held-out sets was zero after script normalization. Only the subject, relation and object targets and the training-time phrase boundaries require text annotations. Inference accepts two image strips and outputs the selected and generated ink. [Frozen protocol](role_binding_protocol.md) · [Corpus counts and hashes](evidence/role_binding_20261123/corpus.json) · [Baidu DuIE release](https://github.com/PaddlePaddle/PaddleNLP/issues/447)

A 202,768-parameter trainable head uses contextual features from a frozen, previously trained CKIP image reader with 85,951,202 parameter elements. Computing the training features took 40.2 seconds on one RTX 4090 D. Six passes over those features took 40.1 seconds. The matched control used randomly permuted relation supervision and exchanged the subject and object endpoint targets on approximately half the records. Pretraining and visual calibration preceded this comparison. The induced writer covers twelve supervised relations and one alignment pattern per relation. Qwen3-8B parity, general discourse, and a thousand-fold saving at equal quality remain unproved. [Fit](evidence/role_binding_20261123/genuine_fit.json) · [Control](evidence/role_binding_20261123/shuffled_fit.json)

## Independent sentence generation

The separately fitted writer takes the predicted subject, relation and object as image strips. In response to a pictured order instruction it emits and feeds back each glyph until it stops. Its two displayed film credits read 林遥是《雨巷来信》的导演。 and 闻溪是《雨巷来信》的编剧。 Both are actual returns. Changing the order does not reformat a stored sentence after generation.

| Source | Both orders exact, predicted fields | Both orders exact, reference fields | Starting writer, predicted fields |
|---|---:|---:|---:|
| Human questions | 31/48 | 44/48 | 0/48 |
| Programme notes | 16/32 | 32/32 | 0/32 |
| Held-out DuIE passages | 810/1,200 | 1,124/1,200 | 0/1,200 |

Training used 8,000 records in both orders plus 4,096 short clauses from authentic DuIE training passages. In the first fit, some auxiliary reference cells were unreachable from either the finite writing bank or the pictured fields. Its mean negative log likelihood exceeded 10²⁶ and the weights were discarded from the comparison. The original receipt and checkpoint are preserved. The fresh fit filtered 63 ineligible candidates to obtain the same quota. It ran 1,676 updates and saw 756,642 target positions. Mean negative log likelihood fell from 0.718 to 0.562. The paired results above use free generation on held-out inputs. [Physical streams and outcomes](evidence/role_binding_20261123/writer_results.json) · [Accepted fit](evidence/role_binding_20261123/writer_fit.json) · [Rejected fit](evidence/role_binding_20261123/writer_unreachable_fit.json)

The natural-writing instruction gave both names in 34/48 human replies, but the names alone do not establish a sound sentence. Given a person, a school and the school relation, the writer supplied a birth year and doctorate absent from the three fields. A film response repeated a name until the output limit. Once the passage has been reduced to a triple, different biographies can give the writer identical inputs. Reliable free writing about further details requires retaining their evidence. [Original first-case readings](evidence/role_binding_20261123/natural_readings.json)

On 256 additional human questions sampled across topics, the specialized model returned 147 exact direct answers, compared with 187 for its frozen visual reader. The specialized pipeline should be evaluated within its stated domain. Replacing each programme participant by a pair of digitized inscription glyphs yielded 13/32 physically exact statements, zero of eight complete programmes and 14/32 outputs that commuted with the name exchange. These are experimental aliases, not historical spellings of the participants. [Independent transfer checks](evidence/role_binding_20261123/transfer.json) · [Original alias images and physical scores](evidence/role_binding_20261123/aliases/evidence.json)

The complete bundle and CLI reproduce three induced and two independently generated film outputs, pixel for pixel, in an isolated CPU process. [Model usage](../models/role_binding/README.md) · [Training commands](../models/role_binding/TRAINING.md) · [Hashes and CPU verification](evidence/role_binding_20261123/release.json)
