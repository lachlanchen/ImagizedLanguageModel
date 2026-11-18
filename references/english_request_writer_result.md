# An English question can select a Chinese fact and its historical form

With the requested dictionary record located in pixels, changing only the visual question encoder increased complete answers to the standard English frame from 249 to 1,080 out of 1,136. The writer recovered 48 of 96 independently worded English requests, compared with 19 before training and 21 with shuffled question assignments. The same output pathway still drew the lexical fact and the specific historical bitmap. [Figure 80](../publication/ilm-nature/figures/Figure_80_english_request_writer.png)

| Complete stopped responses | Parent | Shuffled | Matched |
|---|---:|---:|---:|
| Standard English, located / 1,136 | 249 | 310 | 1,080 |
| Standard English, full two-entry card / 1,136 | 74 | 80 | 273 |
| Crossed English roles, located / 1,136 | 230 | 270 | 267 |
| Independent English, located / 96 | 19 | 21 | 48 |
| Independent English, full card / 96 | 3 | 4 | 10 |
| Independent Chinese, located / 96 | 60 | 63 | 61 |
| DRCD pairs / 1,042 | 12 | 14 | 13 |
| DuReader / 571 | 15 | 14 | 14 |

Private local Qwen3-8B inference produced the test questions using the intended word, fact and stage. No answer, training example or candidate output was supplied. A review before fitting flagged 26 of the 96 English requests. Some asked whether or when a form existed, and one changed the headword. Among the remaining 70, the parent, shuffled and matched writers recovered 16, 20 and 32. The primary score includes all 96. The matched gain over shuffling was 28.13 percentage points (95% interval 15.63–40.63, resampling 24 cards). This is a test of synthetic independent wording. [Questions and recorded authorship](evidence/english_request_writer_20261119/authorship.json) · [Intent review](evidence/english_request_writer_20261119/intent_review.json)

The shared 628,104-parameter writer used 115,713 trainable question parameters. The writer and passage pathway stayed fixed. Matching versus shuffling compared identical imagery, targets, initialization, compute schedule and real-language rehearsal. Each arm trained on 107,040 English examples. The fits took 59.4 and 57.6 seconds on one RTX 4090 D and peaked below 0.45 GiB of PyTorch allocations. These figures exclude preparation, initialization and the independent test author. [Protocol](english_request_writer_protocol.md) · [Checkpoints, cost and metrics](evidence/english_request_writer_20261119/result.json)

In the crossed construction, four requests about a headword used identical visible glyphs in different orders. The matched writer completed none of the 284 four-question groups, and its single-answer score was three lower than the shuffled control. It also lost one exact natural DuReader answer relative to its parent and gave zero of 300 complete answers requiring new content. Both the broad English transfer and retention criteria failed. The narrower English gain does not establish flexible clause interpretation. [Independent archive recount](evidence/english_request_writer_20261119/audit.json) · [Global attention comparison](scoped_request_writer_result.md)
