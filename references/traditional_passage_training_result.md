# What the pictured passage reader learned from traditional Chinese

A 101,144-parameter reader trained with additional human-authored traditional Chinese questions produced 1,270 complete accepted answer images on 3,505 held-out DRCD questions (36.2%). The preceding reader gave 500 (14.3%). Continuing to train on the old CMRC data for exactly the same number of optimizer steps gave 380 (10.8%). The mixed reader also improved on official CMRC singles, from 999 to 1,111 of 3,170. The checkpoint passed the predeclared retention criteria and is the default for pictured passage questions. [Run the model](../models/pictured_passage_zh/README.md) · [Protocol](traditional_passage_training_protocol.md)

| Held-out measure | Denominator | Preceding checkpoint | CMRC replay | CMRC + DRCD |
|---|---:|---:|---:|---:|
| DRCD single questions | 3,505 | 500 | 380 | 1,270 |
| DRCD pairs in both orders, 依次作答： | 4,638 | 104 | 58 | 613 |
| DRCD pairs in both orders, 請依次回答這兩個問題： | 4,638 | 14 | 4 | 77 |
| Official CMRC singles | 3,170 | 999 | 840 | 1,111 |
| Official CMRC pairs in both orders, 依次作答： | 4,800 | 505 | 353 | 601 |
| Internal CMRC singles | 1,788 | 434 | 381 | 461 |
| Internal CMRC ordered requests | 6,086 | 438 | 364 | 513 |

## Locating the gain

The first exact answer gained over replay is **1991年**. The question asks when nearly 50,000 people in India could speak Sanskrit. Replay supplied the correct year within a longer census report. Returning the year alone improved the strict score, although both outputs contained the requested fact. Of 947 newly exact answers over replay, 331 had previously included an accepted answer with extra ink. In 616 cases the previous output lacked a complete accepted answer or stop. This count establishes a change in the physical evidence selected. It cannot determine the acceptability of every alternative phrasing. The mixed reader also lost 57 former successes. The net difference was 25.39 percentage points with a 95% article-clustered bootstrap interval of 23.67 to 27.14. [Example answer](evidence/traditional_passage_20261119/gained_mixed.png) · [Recount](evidence/traditional_passage_20261119/containment.json)

The first remaining error asks which region began academic research on Sanskrit. The reference is **歐洲**. The mixed reader produced **德國學者陸特**, a German scholar. It found nearby relevant words and selected the wrong kind of entity. The study also exchanged distinct questions about the same evidence. Across 9,276 directed DRCD question pairings, the mixed reader's single-question output matched its own reference 3,398 times and the sibling's disjoint reference 75 times. Questions can recur across these pairings. The same passage facts elicited both correct orders on 613 pairs with the shorter instruction but only 77 with the traditional instruction. These rates limit the generality of the learned behavior. [First wrong output and reference](evidence/traditional_passage_20261119/audit.json)

## Data, training and computation

The added corpus contains 21,934 DRCD training questions on 6,579 passages. The 3,505 held-out questions occupy 994 passages from 379 articles. Exact passage images, subject titles and normalized text windows were screened for overlap with reserved releases. The preparation retains every aligned occurrence of an accepted source answer. The original and added training records share the same packed binary glyph representation. [Preparation and exclusions](traditional_passage_preparation.md) · [Delta Research Center corpus](https://github.com/DRCKnowledgeTeam/DRCD)

Both fits started from the same image-conditioned reader. At half of each epoch's positions, both used the same pictured CMRC records. The mixed arm replaced the other half with all DRCD questions once per epoch. The replay arm substituted balanced CMRC draws at similar passage lengths. The loss takes the negative logarithm of the total probability assigned to all accepted spans. Its implementation evaluates `-logsumexp(log p(start) + log p(end | start))` over the accepted intervals. At inference the planner chooses question regions and learns when to stop. The span decoder transfers the chosen original pixels into the output. [Training implementation](../scripts/compare_traditional_passage_training.py)

Each arm ran 5,504 AdamW updates on 175,472 question presentations on one RTX 4090 D. Mixed training used 81.80 million passage-cell exposures and 87.40 million padded positions; replay used 86.31 and 95.01 million. The fits took 43.0 and 45.1 seconds, respectively. The complete serial comparison including preparation and evaluation took 138.2 seconds. These times exclude the inherited model's training and original rendering. The mixed arm peaked at 277.3 MiB of PyTorch allocations. Parameter sharing allows a compact reader to reuse visual matching across scripts and sources. The experiment did not compare compute at matched general-language performance. [Run receipt](evidence/traditional_passage_20261119/results.json)

An independent scorer reread 214,503 archived output streams. The original checkpoint reproduced the preceding official CMRC archive digest. All three checkpoints replayed the illustrated cases on CPU, and the installed default reader reproduced the newly exact year through its PNG interface. The Nature-style article reports these findings in Figure 74. The model's emitter still requires visible source ink. It cannot supply an independently composed explanation that requires new glyphs absent from the pictured input. [Figure 74](../publication/ilm-nature/figures/Figure_74_traditional_passage.png) · [Editable figure](../publication/ilm-nature/figures/Traditional_passage.pptx)
