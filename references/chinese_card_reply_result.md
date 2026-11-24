# Which fact did the guest ask for?

A single visual writer learned to read a human Chinese question and compose a Chinese answer using facts visible on a supplied venue card. The two questions below concern the same restaurant. The second asks where the restaurant is without using the printed field name 地址. These are original held questions and complete generated responses. The opening hours and address are copied from the evidence. The writer supplies the surrounding wording and stop. [Original page and answer bitmaps](../models/chinese_card_reply/examples/) · [Runnable checkpoint](../models/chinese_card_reply/README.md)

| Human question | Machine-written answer |
| --- | --- |
| 是吗，那这家店的营业时间你知道吗？ | 营业时间是周一至周日 11:00-21:30。 |
| 好的，就去它家了，你知道这个餐馆在什么地方吗？ | 地址是西四南大街60号(近缸瓦市)。 |

### Evidence from human dialogue

The official CrossWOZ training, validation and test splits yielded 5,097, 517 and 515 eligible turns. Each example needed one requested field, a resolved attraction, hotel or restaurant, a matching database value occurring verbatim in the human answer and a card with competing attributes. Only 515 of 4,238 held system turns passed those requirements. Entity resolution, formatting and selecting admissible records use annotations during preparation. The model receives the resolved card at inference. [Source database and human dialogues](https://github.com/thu-coai/CrossWOZ) · [Exact selection protocol](chinese_card_reply_protocol.md) · [Source hashes](evidence/chinese_reply_v1/corpus.json)

| Criterion | Pictured question | Question erased | Matched byte question |
| --- | ---: | ---: | ---: |
| Clean requested value, all 515 turns | 507 | 237 | 508 |
| Clean requested value, 59 new queries about new entities | 57 | 23 | 56 |
| Entire human wording, all 515 turns | 194 | 52 | 187 |
| Explicit relation with clean value | 476 | 237 | 463 |

A clean value requires the requested source span without another catalogue value or a number outside the source value and venue name. Among 114 pairs of natural questions asking for different facts on identical cards, 111 had two clean answers and none passed both after erasure. The model chose the correct field on 78/80 new questions omitting the literal field name. A character-trigram nearest-question control scored 71 and an available-field keyword heuristic with a domain prior scored 28. Whole questions and venue names can recur in the official split. The completely new question and new entity subgroup is small. [All scores](evidence/chinese_reply_v1/results.json) · [Lexical comparisons](evidence/chinese_reply_v1/lexical-baselines.json)

The human answers provided the word order and phrasing targets. The 515 generated answers had 136 distinct wordings with copied material masked. Twenty-nine replies used a masked wording absent from training. On 492 turns the writer emitted two or more Chinese characters outside the copied spans. Sixteen replies had none. On held turn 30 the hours were correct but the reply began “这是博以了，”. On turn 220 the model added “不是很合理的。” to a stated price. Both pass the value check. The first field error answered a hotel-type question with an address. These examples bound what the 507 clean values establish. No independent fluency rating was collected. [All original generations](evidence/chinese_reply_v1/outputs.json.gz)

All 515 held questions were rendered as PNGs and reopened through the released CPU inference path. All 515 resulting field choices and packed output images matched the batched GPU evaluation. Optional plain-text transcription matched 514 strings exactly. One catalogue tab and an ordinary space have the same blank raster. Five clean answers succeeded only with the optical input and six only with the byte input. [Physical replay](evidence/chinese_reply_v1/physical-release.json)

### Why order matters

On the identical printed record, exchanging 地址 and 营业时间 in “我想查地址，不是营业时间。” changes which answer would be true. The questions have the same multiset of glyphs. Any deterministic decision based only on that multiset must miss at least one of the two. The original writer often did. A separate 40,657-parameter network replaces the two detected field spans by identical visual markers, reads their surrounding ink in sequence, and scores the two positions. The resulting positional choice is bound to the original field and passed to the frozen answer writer. Each copy action retains the original source pixels. Learned positional choice, physical copying and sentence quality can be checked separately. [Lean proof of the paired decision](../formal/span_list/RequestedPosition.lean) · [Proof of source span transport and support](../formal/span_list/VariableInk.lean)

| Paired scope test | Original | With positional selector |
| --- | ---: | ---: |
| Single-field positive control | 18/18 | 18/18 |
| Three diagnosed two-field constructions | 9/54 | 54/54 |
| Twelve further two-field constructions | 58/216 | 180/216 |

The focus fit used 20 authored sentence frames with 24 prefix/suffix combinations, six field names and two reserved field names. Once the normalized physical query and anchor positions are identical across name swaps, paired routing success reduces mathematically to one positional decision. The 180 pairs span ten successful constructions. Two constructions gave zero of 18 each, including “麻烦报一下A，暂时别管B。” and “并非要打听A，要打听的是B。” No eligible natural question contained exactly two distinct literal available-field names, so the 515-turn score provides no natural-language validation for the added selector. The release exposes it as an option. [Scope protocol](chinese_focus_learning_protocol.md) · [Per-construction outcomes](evidence/chinese_reply_v1/focus-transfer.json)

### Training cost and scope

Reading uses 16 × 16 pooled ink from each 32 × 32 cell. A learned optical encoder predicts the requested relation. A causal writer conditions on the question, domain and chosen relation, chooses among 437 printable training bitmaps, two source-copy actions and stop, and can emit a sequence longer than the prompt. The formal results explain when a change of referent must propagate and why order information is necessary. They supply no guarantee that optimization learns unseen syntax or that pixel training saves energy.

The Chinese writer has 1,568,944 parameters. On a single RTX 4090 D, 8,000 updates consumed 109.83 seconds and 10.57 kJ of measured device energy, against 104.63 seconds and 9.91 kJ for its matched byte-input writer on the same 2,998,698 target positions. The focus fit took another 2.98 seconds and 0.27 kJ. These are device readings for fitting. A matched, general-purpose token LLM comparison and open-ended page writing remain outstanding. Earlier pointer-generator models learned a choice between generating and transferring a source element. This implementation binds exact ink spans and checks the outcome in the physical page interface. [Image fit](evidence/chinese_reply_v1/ink-fit.json) · [Byte fit](evidence/chinese_reply_v1/byte-fit.json) · [Focus fit](evidence/chinese_reply_v1/focus-fit.json) · [See et al., ACL 2017](https://aclanthology.org/P17-1099/)

The next informative test is naturally phrased Chinese that requires two facts in one answer, including exchanges of which one is requested first. Hold out conversations and report complete field coverage, order where specified, unsupported additions, and independently assessed phrase quality. A paired-field model must outperform both a single-field writer and independent concatenation of its two replies. Keeping an order-sensitive test alongside the human corpus makes a failed linguistic decision visible even when every printed glyph is clear.
