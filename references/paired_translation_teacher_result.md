# Can a visual translator write the change it recognises?

The same 1.82-million-parameter writer that earned 6/96 on new books recognised the aligned sentence on 260/308 requests drawn from corpus-derived training pairs. When it had to compose its own answer, it fully preserved the source change in both directions for **3/77 families**. It did so for **46/51 constructed families**. A family contains two source variants translated into English and Chinese. These examples were in the broad writer's training set. [Paper](../publication/ilm-nature/manuscript.pdf) · [Original paired images](../publication/ilm-nature/evidence/paired_visual_answers.zip) · [Free-writing assessments](evidence/paired_translation_teacher_v1/free-meaning.json)

| Evaluation | Corpus-derived | Constructed |
|---|---:|---:|
| Audited families | 77 | 51 |
| Individual pictured requests | 308 | 204 |
| Aligned reference scores above the alternative | 260 | 204 |
| Both reference rankings correct in both languages | 38/77 | 51/51 |
| Free answer accepted for its own source, both orders | 125 | 200 |
| Own free answer alone accepted, both orders | 115 | 199 |
| Change preserved on both sources in one direction | 26/154 | 97/102 |
| Change preserved in both languages | 3/77 | 46/51 |
| Free answers pixel-identical to aligned reference | 21 | 103 |

The broad writer changed “Look at me, Mom.” to “Look at me, Harry.” when the Chinese source changed the addressee. It wrote “I did a funny dream.” for 我做了个好玩的梦 and “I made a nice dream.” for 我做了一个噩梦。 The second changes the sense of *nightmare*. The display uses the first successful and first unsuccessful Chinese-to-English contrast in fixed audit order among corpus-derived families whose two supplied translations passed the independent assessment in both languages. All 512 original answer images stopped. All 32 sampled isolated CPU outputs and stop decisions exactly reproduced the archive. An automated assessor credited paraphrases, and none of the 128 families reproduced all four target bitmaps exactly. [Original figure](../publication/ilm-nature/figures/Paired_visual_answers.png) · [Writer scores and replay](evidence/paired_translation_teacher_v1/writer.json)

## Corpus provenance

The earlier breadth report described all 147,456 variable training pairs as natural. The source arrays show that 1,032 were constructed role examples inserted into an earlier book-and-talk corpus. The other 146,424 came from OPUS (68,645), talks (61,416), the visual library (8,827) and a separate book set (7,536). The 24,576-pair repeated cohort contained 185 constructed rows. None of those 1,032 pairs appeared as an exact bilingual bitmap in the unmodified source arrays. Each pass separately included the previously reported 6,240 controlled pairs. Source tracing changes the interpretation of this contrast assay while preserving the published comparison of independent unseen-book answers. [Byte-level attribution and source hashes](evidence/paired_translation_teacher_v1/provenance.json)

## Qualifying the teaching signal

We mined 1,384 disjoint source edits from the broad training rows. The two native OPUS-MT directions preferred the aligned whole sentence by more than 0.10 nats per target unit on both rows for 890 families. Replaying all 64 sampled families with reversed batch order reproduced every threshold decision, with maximum score drift of 0.00000382 nats. Conditional scoring includes the end decision. This screen gives a relative preference even when both translations omit information. [Selection and original row indices](evidence/paired_translation_teacher_v1/selection.json) · [Directional checkpoints and scores](evidence/paired_translation_teacher_v1/teacher.json)

A separate quantized Qwen3-8B read the 128 reserved families one source at a time in both translation directions and candidate orders. Each candidate could pass or fail independently. All eight decisions agreed on only 69 families, below the required 112. Of the 69, 48 came from the 51 constructed families and 21 from the 77 corpus-derived families. Only 464 of the remaining 762 families had two corpus-derived rows. Verified pairs and weaker cases are included in the receipts. The unqualified bulk set supplied no student updates. A stopped worker left 964 completed qualification calls cached. Its 60 remaining calls completed under the same model, permutation and cache binding. The reported 12.49 seconds in the verification receipt measure the resumed process only. [All 1,024 decisions](evidence/paired_translation_teacher_v1/verification.json) · [Original protocol](paired_translation_teacher_protocol.md)

## What the scores can establish

Let the four stopped conditional scores be a,b,c,d in row-major order. The interaction (a − b) + (d − c) cancels additive offsets shared by a source row or target column. A single target bias can give both aligned answers a margin of at least m precisely when the interaction is at least 2m. Lean checks both implications for integer fixed-point scores. Positive interaction by itself does not ensure that the deployed writer ranks both answers correctly, or that either answer beats a third possible continuation. This distinction can be measured directly in the rank and free-writing columns above. [Lean proof](../formal/span_list/PairedTranslationChoice.lean)

Applied to the 154 directions of the corpus-derived audit at a margin of 0.10 nats per glyph, 90 already met both row margins. In 62 more, a different target bias for each family and direction could in principle meet both. Two had insufficient interaction for any such bias. The offset is a feasibility calculation that uses both labelled alternatives. It is unavailable when writing an open answer. At the first target glyph where the candidates differ, the writer preferred the aligned glyph on 234/308 requests. The highest scoring glyph was the aligned one on 151, the crossed one on 45, and a third choice on 112. In 83 of those 112 it already preferred the aligned glyph over the crossed one. Only 186 actual free answers reached the supplied common prefix. Every reached next action reproduced the archived greedy output. [Physical branch probabilities](evidence/paired_translation_teacher_v1/branches.json)

The autonomous writer exactly reproduced 30/308 complete phrase drafts in the corpus-derived set. Of the 125 credited free answers, 26 reproduced the draft and 99 changed its pixels. Editing made 14 answers newly exact and spoiled 18 exact drafts. The corresponding counts for the constructed set were 61 gains and 27 losses. Pixel changes can be punctuation or wording and do not by themselves establish a semantic improvement over the draft. [Exact draft comparison](evidence/paired_translation_teacher_v1/draft-copy.json)

The replacement teacher study starts from unmodified rows of the pinned OPUS-100 training split. Each proposed training contrast must pass its own semantic assessment in both directions and both candidate orders. A second model checks a separate reserve before admission to a matched student fit. [New corpus and qualification protocol](opus_paired_supervision_protocol.md)
