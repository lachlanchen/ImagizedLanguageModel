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
