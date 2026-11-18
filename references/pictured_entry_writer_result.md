# Writing a lexical fact and an attested historical form

Given a picture asking for the original meaning and bronze form of 耳, a 28,032-parameter writer produced a complete sentence image containing *ear* and Hanziyuan B16241. A second pictured question on the same card elicited *ěr* and the Liushutong form L13682. On 142 cards from 284 unseen character families, 1,134 of 1,136 answers matched every reference pixel and ended with a learned stop. All eight answers were correct on 141 cards. The predeclared task criterion passed. [Runnable examples](../models/pictured_entry_writer/README.md) · [Figure 70](../publication/ilm-nature/figures/Figure_70_pictured_entries.png)

| Condition | Complete / 1,136 | Correct fact | Correct terminal form | All eight / 142 |
|---|---:|---:|---:|---:|
| Original wording | 1,134 | 1,134 | 1,136 | 141 |
| Alternate question frame | 1,132 | 1,135 | 1,134 | 140 |
| Evidence exchanged between entries | 1,134 | 1,136 | 1,134 | 141 |
| Synonyms replacing the field names | 225 | 228 | 1,129 | 0 |
| Original wording, questions erased | 0 | 0 | 0 | 0 |
| Original wording, card removed | 0 | 0 | 0 | 0 |
| Within-card derangement during training | 0 | 0 | 0 | 0 |
| Authored field parser, original wording | 1,136 | 1,136 | 1,136 | 142 |

For the two failures in the original wording, the pictured question asks about 裘 and the other entry is 文. The writer emitted the other headword and its gloss *culture* or pronunciation *wén*, while preserving the requested bronze ink. This is a complete-sentence error even though the historical image is correct. The evaluation counts each component separately. When only the question changed, the writer produced both correct complete images for 566 of 568 stage switches, 567 of 568 field switches and 566 of 568 headword switches. An independent recount checked complete results for all 15,904 evaluated streams. [Generation and component scores](evidence/pictured_entry_writer_20261119/evaluation.json) · [Recount](evidence/pictured_entry_writer_20261119/recount.json)

## What the model learns

The decoder starts without an answer prefix. Its learned actions choose visible headings, field names and values, emit one of 64 frequent training glyph images, or stop. It can therefore carry an unseen historical bitmap through the answer exactly. No example supplies a target action trace. The fitting procedure finds a shortest segmentation of the reference image stream and sums probability over actions with the same visible emission. Both training arms saw all 8,920 pictured requests eight times. The within-card control learned with the same images and a different pairing of questions and answers. It scored zero whole answers under the original wording and the alternate frame. The positive difference supports learning which visible material belongs in the answer. The authored parser reaches the ceiling on the regular card and question grammar. Copying from a source and generating an answer have precedents in pointer-generator architectures. [Protocol](pictured_entry_writer_protocol.md) · [See, Liu and Manning, 2017](https://aclanthology.org/P17-1099/)

Training the matched weights took 3.85 seconds on one RTX 4090 D, with 260.6 MiB peak PyTorch allocation. The deranged fit took 4.71 seconds. These times exclude rendering, corpus preparation, feature computation and alignment. The model uses four supplied delimiter images and exact visual matches of labels to parts of the question. Its output sentence frame is regular in the supervision. Historical glyph selection reproduces source evidence, and the meanings are those stated by Hanziyuan. These measurements do not establish open-domain language quality or a matched-compute advantage. [Corpus and source checks](pictured_entry_answers_preparation.md)

## Which question distinctions reach the writer?

In an additional diagnostic we replaced the field words with 原意 and 字音 in otherwise identical pictured questions. Across all 568 pairs, both sides produced bit-identical candidate features and the same candidate payloads. The two required answer images differed on every pair. From an identical initial state, greedy decoding takes the same action and receives the same feedback at every step. This bounds the best possible complete score of any deterministic writer using those features at 568/1,136 and its both-correct score at zero. Additional training on this representation cannot distinguish those inputs. The next comparison must carry the visual content of the complete question into the decision, teach lexical alternatives under matched exposure, and test further unseen formulations. Full passage translation retains its own evaluation. [Representation equality audit](evidence/pictured_entry_writer_20261119/equal_length_alias_audit.json)
