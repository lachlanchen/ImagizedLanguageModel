# Which parts of a proposition survive image-to-image translation?

Six continued fits on one RTX 4090 D tested whether a 1,688,480-parameter visual writer could preserve the participants, polarity and quantity of an event across English and Chinese. The deployed writer receives 32 × 32 binary glyph cells, reads 16 × 16 local ink densities and emits a sequence of learned physical prototypes plus a stop decision. All reported translation outcomes use free generation. Selecting an aligned translation is a separate capability. [Runnable weights and PNGs](../models/translation_paraphrases/README.md)

## Changing a proposition

The first blind comparison reserved unordered actor or actor–object combinations from the structured training partition. Each contrast changes one fact. The 7,288 training pairs included 1,302 edits of human parallel sentences. After six passes and ordinary bilingual replay, complete exact answers rose from 5 to 143 of 144, with 71 of 72 pairs correct on both requests. Both ordinary likelihood and a first-divergence margin yielded this result. On 72 independently worded requests the two writers obtained zero exact answers and two versus three two-order meaning credits. With correct prior target glyphs supplied, the contrast writer preferred the right glyph over its rival on 64/72. It freely reached the common reference prefix on only 31/72. The 8B Q4 text model received 60/72 meaning credits in this panel. [Design and domain breakdown](evidence/proposition_translation_v1/design.json) · [Frozen votes and exact counts](evidence/proposition_translation_v1/result.json) · [Limits of contrastive evaluation](https://aclanthology.org/2021.blackboxnlp-1.5/)

## Recognition and generation

The next two fits used 240,000 unique eligible OPUS-100 training pairs plus matched replay. Each saw 2,208,000 directed sentences, containing 60,018,469 target positions. The aligned fit added a symmetric multi-positive batch retrieval loss to translation likelihood, weighted by 0.12. With the projection, the same encoder retrieved the correct Chinese image on 109/128 English inputs and the correct English image on 112/128 Chinese inputs. The matched likelihood encoder scored 8/128 and 9/128 with its raw states. The aligned encoder without the projection already scored 74 and 98. The retrieval head adds 26,080 parameters and is distributed alongside the writer. [Pinned corpus and filtering](evidence/bilingual_ink_v1/corpus.json) · [OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)

| Free generation, second comparison | Reworded /72 | Conversational /32 | Editorial /32 | OPUS /64 |
|---|---:|---:|---:|---:|
| Likelihood | 1 | 0 | 0 | 1 |
| Aligned, greedy | 1 | 1 | 0 | 2 |
| Aligned, forward beam | 4 | 3 | 0 | 2 |
| Aligned, reverse reranking | 5 | 3 | 0 | 4 |
| Qwen3-8B Q4, text input | 62 | 18 | 24 | 40 |

The beam and reverse scorer used identical stopped candidates. None of the 433 proposals for the 64 OPUS requests matched the corresponding reference, even after normalization. Some paraphrases still earned full meaning credit. Reranking added 2.97 s and 0.47 kJ over 360 requests. Across the 128 retrieval pairs the projection reproduced both directional top-one counts from original PNG grids on CPU. At a maximum ±0.01 change to every rounded cosine, the measured margins certify 107 English-to-Chinese and 111 Chinese-to-English choices. This is a statement about scores. Its application to damaged or unfamiliar writing would require a bound on the score change induced by those images. [Generation, beam coverage and retrieval](evidence/bilingual_ink_v1/result.json) · [CPU margins](evidence/bilingual_ink_v1/retrieval_margin.json) · [Lean 4 theorems](../formal/span_list/OpticalProposition.lean)

## Source paraphrases and natural language

Both final writers continued from the broader likelihood fit. Each received the same 333,448 directed presentations and 11,395,898 target positions. Sources in the intervention sometimes used a different wording for the same target. There were 5,986 structured event and quantity targets and 11,695 PAWS-X targets per pass, plus 24,000 ordinary parallel pairs. Only labeled-equivalent PAWS-X siblings supplied an alternative source. Chinese PAWS-X training uses machine translation. The official evaluation includes human translations. Sixteen previously unused non-equivalent sibling pairs supplied 64 further translation requests. [Training partitions](evidence/translation_paraphrases_v1/corpus.json) · [PAWS-X](https://aclanthology.org/D19-1382/)

| Free generation, final comparison | Canonical source | Varied source | Qwen Q4 | Reference | Foil rejected twice |
|---|---:|---:|---:|---:|---:|
| Previously examined rewordings, 72 | 1 | 1 | 58 | 72 | 72 |
| Original challenges, 16 | 0 | 0 | 14 | 16 | 16 |
| Conversation, 32 | 1 | 1 | 20 | 32 | 32 |
| Editorial, 32 | 0 | 0 | 21 | 30 | 32 |
| OPUS, 64 | 0 | 0 | 35 | 61 | 64 |
| New PAWS-X, 64 | 0 | 0 | 46 | 64 | 60 |

Both final writers exactly reproduced 144/144 controlled combinations. The varied writer produced **护士邀请了医生。** from “The doctor was invited by the nurse.” Swapping the participants elicited **医生邀请了护士。** The judge rejected the first twice and rejected the second in one order. The canonical-source writer produced **医生由了护士。** for the first. The published exact and semantic scores preserve the original judgments. On the first new English-to-Chinese PAWS-X case the source describes Wright moving from New York to Chapel Hill. The varied writer introduces the year 1000 and loses the destination. The errors show why an image of a well-formed sentence is insufficient evidence of understanding. [All aggregate outcomes, paired decisions and individual judge votes](evidence/translation_paraphrases_v1/result.json) · [Eight complete printed outputs and PNG inputs](../models/translation_paraphrases/examples/)

## A separate exact test of grammatical voice

After observing the grading discrepancy, we tested the frozen weights against one independently checkable target for every unexamined held participant pair and verb. Each family has two assignments of agent and patient, positive and negative polarity and active and passive voice. Both voices assert the same event and use the same canonical active Chinese target. Removing any family with an overlapping source or target across training or prior evaluations left 118 of 140 families, or 944 requests. With canonical versus varied-source training, exact stopped Chinese answers were **471 versus 471 of 472 active**, **0 versus 422 of 472 passive**, and **0 versus 76 of 118 complete eight-condition families**. This was a new, post hoc test of a construction used during varied-source training. The 944 requests share actors, verbs and families; they do not represent 944 independent natural-language encounters. [Exact results and weight hashes](evidence/translation_paraphrases_v1/voice.json) · [Reproducible sentence enumeration, overlap exclusions and scoring](../scripts/evaluate_voice_transfer.py)

## Resources and scope

| Continued fit | Wall seconds | Device kJ | Peak CUDA allocation, GiB |
|---|---:|---:|---:|
| Proposition likelihood / margin | 67.1 / 67.0 | 8.21 / 8.21 | 0.45 / 0.45 |
| Broad likelihood / alignment | 206.2 / 212.7 | 27.48 / 28.23 | 1.07 / 1.07 |
| Canonical / varied sources | 49.4 / 52.7 | 5.96 / 6.24 | 0.55 / 0.55 |

These are incremental fit costs. Earlier checkpoint training, data preparation, host energy and the larger evaluation models incur additional costs. Qwen3-VL-8B Q4 graded candidate texts in two blinded orders. Qwen3-8B Q4 generated a text comparison. They share a model family, and prior benchmark exposure is possible. Groups are small, related requests share families, and different panels produce different votes even for unchanged models. Only comparisons within the same panel use the same judging context. The main open problem is expressing all the facts of an unfamiliar sentence under free generation. Raw scan segmentation, independent historical explanations, wider font transfer and matched-quality efficiency remain open.
