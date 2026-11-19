# Why a legible answer can still miss the question

The passage says that Qin Che lost his way and Zhuo Lin drew a map for him. Asked who drew the map, the causal writer answered “卓霖替他畫的。” Asked who needed directions, it reproduced the whole passage. The first answer identifies the person despite failing an exact-name score. The second makes the reader infer the answer all over again. The revised short-passage model returns the two names as images, 卓霖 and 秦澈. [Original images and chart data](evidence/pointing_causal_writer_20261119/figure.json) · [Runnable local model](../models/pointing_causal_writer/README.md)

## When the model can quote a fact

We presented two questions with distinct answers about the same passage. In the short-scene tests, we also exchanged the participants' names while keeping the questions fixed. The causal writer could generate a complete pictured caption interleaving modern Chinese and two ancient source shapes. It was weak at returning the requested name or fact from ordinary text. The frozen CKIP-based reader was much better at quoting that fact. A learned binary selection between these two actions preserved natural benchmark performance and historical writing, but initially misclassified every short role question. Each column below uses the same cases for all methods.

| Pictured reading or writing task | Writer alone | Span reader alone | Initial selection | Answer-sentence selection |
|---|---:|---:|---:|---:|
| Both facts on the same CMRC passage, out of 128 | 2 | 18 | 18 | 18 |
| Both facts on the same DRCD passage, out of 128 | 1 | 113 | 113 | 113 |
| Exact name in earlier short scenes, out of 64 | 0 | 46 | 0 | 46 |
| Exact name in 16 further scenes, out of 64 | 0 | 42 | 0 | 42 |
| Both historical orders, familiar request, out of 96 | 94 | 0 | 94 | 94 |
| Both historical orders, rephrased request, out of 96 | 1 | 0 | 1 | 1 |
| Exact open answer on original page, out of 571 | 11 | 1 | 8 | 2 |
| Exact answer requiring absent content ink, out of 300 | 0 | 0 | 0 | 0 |

Using 1,738 answer-bearing sentences from the original CMRC training passages gave the selector examples with short contexts and quoteable answers. Equal-length random excerpts from the same passages contained only 174 complete targets. This control recovered 11 names on the earlier 64 questions and four on the further 64. The answer-sentence selector recovered 15 of 32 question pairs and all four answers in six of the sixteen further scenes. It selected the span reader in every short scene. The fresh scenes were authored for this experiment. They support transfer across these examples, while prior benchmark exposure of the CKIP weights remains unaudited. More than one implementation can return longer, informative answers that the exact metric rejects. [Selection](evidence/pointing_causal_writer_20261119/selection.json) · [Short-context control](evidence/pointing_causal_writer_20261119/short_selection.json) · [Further scenes](evidence/pointing_causal_writer_20261119/transfer.json)

## When the model has to write

On 48 plates from held-out character families, each with two headwords and two orders, the single causal writer produced 188 of 192 complete stopped captions exactly. Its controls gave zero with uniform pointing and one with training questions exchanged within each plate. Rewording the requests on those same plates gave 21 correct individual captions. The separate 言 plate illustrates reversing oracle and bronze forms from a family represented in training. Historical forms are copied from the supplied source, while connecting words are emitted from a finite printed bank. The local interface supports sectioned strips and text rendered as strips. It does not segment a raw photographed page. [Returned answers and controls](evidence/pointing_causal_writer_20261119/scores.json) · [Both requests and the original source shapes](../publication/ilm-nature/figures/Figure_91_pointing_causal_writer.png)

Providing more documents did little for open writing. On 571 development questions the original page produced 11 matches to any available human reference. A learned evidence ranker gave eight, lexical ranking seven, and a reference-assisted selector fourteen. On 128 further questions requiring an alphanumeric character absent from the original page, these conditions all gave zero matches to any supplied reference. The candidate windows included the exact reference on eleven of these further questions. Thus even displaying the reference sometimes leaves the writer unable to deliver it. These tests measure exact references, and cannot count every possible valid paraphrase. [Evidence retrieval](evidence/pointing_causal_writer_20261119/evidence_ranking.json)

## Reproduction

The trainable pointer pools the probability of identical physical outputs across print generation and all matching source positions. It places the selected image back into the causal history. The fit uses the previously trained UER Chinese GPT-2 visual interface. The whole-span reader transfers a separate CKIP Chinese QA prior. Its 85.95 million parameters, the causal writer, the learned visual maps and output bank are all part of the combined deployment. The 148,033-parameter selector alone cannot read or write Chinese. The genuine writer fit updated 28.60 million parameters, including 246,914 new pointer parameters. It took 67.9 seconds and 3.23 GiB peak allocated memory on an RTX 4090 D. These costs exclude pretraining, existing visual interfaces, evaluation and the separate selection fits. No matched-quality compute efficiency gain has been measured. [Training receipt](evidence/pointing_causal_writer_20261119/pointing_fit.json) · [Complete exported model checksums](evidence/pointing_causal_writer_20261119/short_release.json)

The following repository commands use the prepared local corpora, the previous causal checkpoint and the fixed visual reader. Run on a fresh experiment directory or checkout, keeping the existing measured fits intact.

```sh
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
python scripts/train_pointing_causal_writer.py pointing uniform foil
python scripts/evaluate_pointing_causal_writer.py
python scripts/prepare_pictured_evidence_windows.py
python scripts/rank_pictured_evidence.py
python scripts/train_pictured_answer_selection.py
python scripts/train_short_passage_selection.py
python scripts/transfer_short_pictured_facts.py
python scripts/export_pointing_causal_writer.py --short-passages
```

The self-contained archive is `artifacts/pointing_causal_writer_v1/pointing-glyph-reader-writer.zip`. It includes the checked images, neural weights and inference code. The pure writer is also available by omitting the span-reader options. Training uses real modern Chinese reading questions and authentic historical image exemplars. [Human reading corpus and official DRCD split](https://github.com/DRCKnowledgeTeam/DRCD) · [Design and fixed cohorts](pointing_causal_writer_protocol.md) · [Short-context decision](short_passage_selection_protocol.md)

The remaining failure is supported composition in a new situation. A discriminating next comparison would vary which fact is true while holding the printed vocabulary constant, then ask for that fact in a grammatical sentence whose wording is absent from the passage. Score proposition, participant and order separately from pixel identity and verbatim match, with a held-out paraphrase audit. This would test precisely what selecting an existing reader or replaying a caption has left unresolved.
