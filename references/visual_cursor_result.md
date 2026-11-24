# Does following source ink improve visual question answering?

A writer trained on human conversation produced some complete answers from ordinary pictured passages. Its persistent copy mechanism did not improve the complete-answer count over independent source selection. Neither fitted writer answered both distinct questions on any of 192 passages.

| Model | CoQA English | CMRC simplified Chinese | DRCD traditional Chinese | Both questions correct |
|---|---:|---:|---:|---:|
| Frozen bilingual writer | 0/128 | 0/128 | 0/128 | 0/192 |
| Independent source positions | 5/128 | 3/128 | 4/128 | 0/192 |
| Persistent source position | 3/128 | 4/128 | 4/128 | 0/192 |
| Persistent, current question erased | 0/128 | 3/128 | 3/128 | 0/192 |

The persistent model answered the traditional Chinese question about a hollow rubber ball and a net with **網球**. When asked who contested the succession after Möngke Khan died, it wrote **1260年**. The stored reference names **阿里不哥**. The three correct English replies were all “no”. The 35 English items whose human reference was absent from the visible prompt gave zero complete answers with persistence and two without. Exact normalized scoring can miss appropriate paraphrases. The current model produced repetitive, incomplete responses to general English and Chinese instructions and failed both variants on every external Winograd pair. A book prefix even elicited “no”.

### Can the writer identify the question when the passage is identical?

We removed preceding English dialogue, gave each pair precisely the same passage prefix and crossed both official questions with both reference answers. On this diagnostic, scoring both correct answers above their competitors requires the model's preference to reverse when only the pictured question changes. The independent writer succeeded on 7/192 and the persistent writer on **0/192**. Even allowing a single joint assignment for both answers, the persistent model selected the correct assignment on 93/192 and the independent model on 103/192. Both possible joint assignments have identical total answer length. This makes an answer-length preference insufficient to identify the correct assignment. All 128 Chinese pairs had both reference strings in the common passage. The 144-cell tail supplied to the free branch made the original writer insensitive to the question on nearly all these long pages (189/192 ties within tolerance). More fluent decoding alone would leave the persistent model's measured question preference unresolved.

### Scope and reproduction

The encoder reads all visible pixels on an aligned page of up to 768 cells. Human answer text was rendered for training. No span offsets, answer choices or reference image were available at inference. Each of the two fits saw the same 2,400 batches and 1,120,936 target positions, with the visual backbone frozen. English answers use a single reference from the parquet release. Many English passages extend beyond the visible prefix. Shared earlier training collections have no universal contamination audit. The new networks each have 528,197 parameters in addition to the 2,117,313-parameter backbone. The persistent variant took 118.0 seconds and 12.66 kJ of additional GPU energy. Both ancestral stages plus this addition total about 301 seconds and 38.57 kJ, excluding evaluation, data preparation and host energy. Capability-matched savings remain unmeasured.

[Physical model and examples](../models/visual_cursor/README.md) · [Protocol](visual_cursor_protocol.md) · [Per-page decisions and energy](evidence/visual_cursor_v1) · [Earlier open-writing results](visual_discourse_result.md) · [CoQA](https://stanfordnlp.github.io/coqa/) · [CMRC2018](https://aclanthology.org/D19-1600/) · [DRCD](https://github.com/DRCKnowledgeTeam/DRCD)
