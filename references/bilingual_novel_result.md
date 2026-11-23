# Reading a Chinese novel passage in two question languages

A writer trained on matching English questions to the right fact in a Chinese novel gave 121 exact, stopped image replies on 748 English requests. Training with the same English questions exchanged between facts on the same page gave 54. Repeating the Chinese question gave seven. On Chinese requests, the aligned writer gave 232/748 exact replies. Each request and passage was supplied as a glyph image and the writer returned an image. The checkpoint met its fixed release criteria. [Run the writer](../models/bilingual_novel_writer/README.md) · [Preparation, training and selection](bilingual_novel_protocol.md) · [All scores](evidence/bilingual_novel_20261124/result.json)

The writer answered two questions in both Chinese and English on three of 374 held-out novel passages and one of 227 regular-print XQuAD passages. Both matched training controls scored zero on these complete-passage measures. The paired difference on BiPaR was 0.80 percentage points relative to the exchanged-English control, with a passage-bootstrap 95% interval of 0.00–1.87 and a two-sided sign-test probability of 0.25. The criterion was fixed before scoring and used a single training seed. A reproducible effect size across seeds or unseen novel titles remains unmeasured. [Paired comparisons and sample identities](evidence/bilingual_novel_20261124/result.json)

| Evaluation | Parent | Chinese repeat | English exchanged | English aligned |
|---|---:|---:|---:|---:|
| BiPaR Chinese questions | 93/748 | 213/748 | 228/748 | 232/748 |
| BiPaR English questions | 10/748 | 7/748 | 54/748 | 121/748 |
| BiPaR, all four replies | 0/374 | 0/374 | 0/374 | 3/374 |
| BiPaR, erased questions, both replies | 0/374 | 0/374 | 0/374 | 0/374 |
| XQuAD regular, Chinese questions | 58/454 | 71/454 | 60/454 | 67/454 |
| XQuAD regular, English questions | 6/454 | 5/454 | 15/454 | 27/454 |
| XQuAD regular, all four | 0/227 | 0/227 | 0/227 | 1/227 |
| XQuAD HanaMin, all four | 0/227 | 0/227 | 0/227 | 0/227 |
| CMRC regular, both replies | 14/197 | 18/197 | 17/197 | 16/197 |
| Familiar historical cards, both orders | 100/100 | 100/100 | 99/100 | 99/100 |
| Rephrased historical requests | 54/200 | 46/200 | 32/200 | 43/200 |
| English historical requests | 0/200 | 0/200 | 0/200 | 0/200 |
| Known 言 requests | 4/4 | 4/4 | 4/4 | 4/4 |
| Open answers needing absent content ink | 0/300 | 0/300 | 0/300 | 0/300 |

## What the images say

On a held-out passage from *The Great Gatsby*, the writer answered a question about where a person followed the narrator with 厨房 (kitchen). For a second question about who shared someone’s embarrassment, it wrote 黛西 (Daisy). It gave both answers correctly in both question languages. On an earlier passage, a question about monthly rent elicited 八十美元 (eighty dollars) in Chinese and 一片海景 (a sea view) in English. The second answer is legible and occurs in the passage, but does not answer the question. Figure 104 reproduces the original pixels. BiPaR question and reference excerpts were rendered from material by Yimin Jing, Deyi Xiong and Zhen Yan, licensed under CC BY-NC 4.0. [Figure 104](../publication/ilm-nature/figures/Figure_104_bilingual_novel.png) · [BiPaR](https://aclanthology.org/D19-1249/) · [License](https://multinlp.github.io/BiPaR/)

Of 627 incorrect English-request replies, 573 failed on their first glyph and 567 returned some contiguous part of the novel passage. The corresponding counts for Chinese were 428 and 447 of 516. Ordered-glyph F1 for English was 0.249 with aligned questions, 0.130 with exchanged English and 0.075 with repeated Chinese. All 23,619 non-whitespace English request glyph occurrences exactly matched the fixed optical inventory. On the 462 validation questions whose accepted answer strings had never been a supervised target in the book fit, Chinese and English questions yielded 102 and 40 exact answers respectively. The novel titles all occurred in training. Earlier text pretraining exposure is unknown. [Error locations and subgroup counts](evidence/bilingual_novel_20261124/error_audit.json) · [Input glyph coverage](evidence/bilingual_novel_20261124/question_inventory.json)

In a separate authored example the writer changed 三部词典 (three dictionaries) to 五部词典 (five dictionaries) when the Lin family’s count changed. Switching the Chinese question from the Lin to the Chen family left the reply unchanged on each page. Four of eight questions and edits passed. The initial Chinese example had been inspected before constructing the remaining seven. [All original bookshelf queries and replies](evidence/bilingual_novel_20261124/bookshelf/result.json)

## Single-GPU training and inference

Each of three fits consumed 49,283 examples, 538,340 target actions and 2,496 optimizer updates. The aligned fit took 175.5 seconds for optimization and checkpoint writing on an RTX 4090 D and peaked at 4.50 GiB of allocated GPU memory; the complete weights occupy 401,214,619 bytes. The model inherited a text-pretrained Chinese transformer. Its network has 87,087,170 parameters with 28,599,938 trained in this fit. Rasterization, data preparation and prior training are excluded from adaptation time. The visual interface uses a finite ink inventory, with copying from the supplied page for additional shapes. [Checkpoint and measured costs](evidence/bilingual_novel_20261124/bilingual_fit.json)

On the first 64 validation passages, warm batch-one image inference gave 42/128 exact Chinese-request replies in 5.97 seconds and 19/128 English-request replies in 3.63 seconds. Rerunning individually matched every archived image and stop flag. Local text Qwen3-8B Q8, with all 37 layers on the same GPU, gave 74/128 in 29.00 seconds and 63/128 in 29.88 seconds. The image writer completed zero four-reply passages on this subset, Qwen 18. This comparison measures different accuracies. General prose, Qwen-level comprehension and a thousand-fold matched-quality computational advantage remain unestablished. [Serving results](evidence/bilingual_novel_20261124/serving.json)

The Lean developments prove that incompatible answers require distinguishable queries and that exact greedy decoding requires every intended glyph and the terminal stop at its correct prefix. They also verify a sufficient score margin under bounded perturbations. These results explain what is necessary for the success criterion. The actual reading rates come from generating and scoring the held-out reply images. [Intent separation](../formal/span_list/BilingualReading.lean) · [Greedy decoding](../formal/span_list/GreedyReply.lean)

At the first glyph where the two annotated answers differed, the aligned writer ranked its intended glyph ahead of the other answer’s glyph in 603/748 Chinese and 494/748 English cases. The intended glyph won the unrestricted choice just 278 and 152 times. A third glyph won 440 and 536 times. Recomputed choices matched the archived autonomous action on all 741 Chinese and 742 English auditable prefixes. This points to the probability of the first answer glyph across the full output inventory as the next training target. [Measured action probabilities](evidence/bilingual_novel_20261124/conditional.json)

The self-contained research package includes the checkpoint, code, two household examples, and a 2,315-headword historical-image catalogue. An offline CPU run matched the saved English-request answer and both 言 reference reply images. A separate lookup for 言 placed the attested oracle and bronze bitmaps in the requested order, with both orders verified against the catalogue’s source tiles. The page presented to the model also contained a distractor headword. [Package and CPU checks](evidence/bilingual_novel_20261124/release.json) · [Lookup provenance and output positions](evidence/bilingual_novel_20261124/catalogue_serving.json)
