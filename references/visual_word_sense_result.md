# Writing the meaning of a character from its context

In 有是言也 the model drew **speech**. In 言及之而不言 it drew **speak** for the final 言. The source cells are Chinese images and the answer cells are English images. In both cases the queried glyph has exactly the same pixels. Its surroundings provide the evidence that can change the gloss. [First passage and emitted image](evidence/visual_word_senses_20261124/examples/KR1h0004_yan_1_answer.png) · [Second emitted image](evidence/visual_word_senses_20261124/examples/KR1h0004_yan_0_answer.png) · [Runnable model](../models/visual_word_senses/README.md)

## Learning and mathematical test

We used occurrence-level dictionary glosses supplied by the Kyoto Classical Chinese treebank, release 2.16. Its source texts include classical prose and poetry. We separated the eight works in its training file by source identity, reserved the available parts of the *Analects* and *Chu ci*, and removed cross-split or conflicting printed windows. The remaining six works provided 247,456 marked training occurrences with 2,193 distinct gloss strings. No evaluation passages or gloss annotations were needed to fit the visual features or the writer. The two held works have 6,987 and 7,173 eligible windows. Different translations can be valid for a short excerpt. [Source corpus](https://universaldependencies.org/treebanks/lzh_kyoto/index.html) · [Pinned edition](https://github.com/UniversalDependencies/UD_Classical_Chinese-Kyoto/tree/r2.16) · [Exact split](evidence/visual_word_senses_20261124/split.json)

For two passages with the same marked image and distinct reference answers, a deterministic headword-only reader cannot get both right. Lean also verifies that two independent draws from the same headword-only distribution have at most a quarter probability of jointly giving the two specified readings. The empirical control shuffled whole glosses *only between instances of the same central bitmap*. It preserved the frequencies of each headword and answer while severing their association with a particular context. Each fit used eight full passes. The scores come from the two fixed final checkpoints. [Lean proof](../formal/span_list/SameInkMeaning.lean) · [Protocol](visual_word_sense_protocol.md) · [All paired indices](evidence/visual_word_senses_20261124/pairs.json)

Each source cell is pooled from 32 × 32 to 16 × 16. Ordered convolutions encode the central cell, its local neighbourhood and its broader context. A recurrent decoder predicts a joint pattern at seven physical pixel locations, or a stop. A learned drawing network expands the chosen pattern into 1,024 pixel logits. Those seven pixels distinguish all 27 output glyph shapes used in training. A whole-gloss classification head assists training and is discarded for inference. The 1,180,977 release parameters include the optical input features, contextual encoder, recurrent writer and physical drawing network. [Training implementation](../scripts/train_visual_word_senses.py) · [Network and drawing head](../ilm/visual_lm/visual_word_sense.py)

## Which answers changed with the passage?

| Held passages | *Analects* | *Chu ci* |
|---|---:|---:|
| Exact emitted gloss and voluntary stop | 5,923/6,987 | 5,808/7,173 |
| Both readings correct in a same-character contrast | 213/566 | 82/403 |
| Both correct after within-headword shuffling | 4/566 | 5/403 |
| Both correct when the aligned writer saw only the central glyph | 0/566 | 0/403 |
| Training-count predictor using local ink, all passages | 6,306/6,987 | 6,528/7,173 |
| Training-count predictor using local ink, both readings | 211/566 | 104/403 |
| Both readings correct for 言 alone | 15/38 | 4/8 |

Headword-resampled 95% intervals for the gain in complete pairs over shuffled training were +31.6 to +42.4 and +12.6 to +25.3 percentage points. The simple count predictor was stronger on aggregate. It backed off from five printed cells to three, two and finally the head alone. The released writer reproduced none of the 69 test occurrences whose reference gloss string had never appeared in training. All drawn cells were legible under an exact physical match to the training alphabet. Four of the 14,160 replies reached the length limit without stopping. The frozen Chinese compositor, given the marked central location, answered 0/128 of the prespecified English-gloss requests. The identical-target replay on CPU reproduced 14,149/14,160 GPU streams and stops, including the failures. CPU strict counts were 5,921 and 5,807. [Full score table](evidence/visual_word_senses_20261124/result.json) · [Original images and first failed pairs](evidence/visual_word_senses_20261124/examples.json) · [CPU differences](evidence/visual_word_senses_20261124/cpu_replay.json)

We also held the immediate neighbours fixed. With three identical central cells, the visual writer solved 2/27 pairs in the *Analects* and 0/11 in *Chu ci*. With five fixed cells it solved 1/6 and 0/3. The single five-cell success is a warning about the reference labels. In *Analects* 16.2, 禮樂征伐 recurs in parallel clauses about who controls rites, music and punitive expeditions. The reference assigns 樂 the gloss *enjoy* in one clause and *music* in the other. Legge's independent translation carries the same sense across both. This pair receives its original numerical credit and supplies no positive evidence of correct semantic disambiguation at a distance. [Restricted pairs and counts](evidence/visual_word_senses_20261124/local_result.json) · [Legge's translation, chapter II](https://en.wikisource.org/wiki/The_Chinese_Classics/Volume_1/Confucian_Analects/XVI)

Qwen3-8B reading text selected 79/128 reference glosses when given two English choices. The visual writer selected 80/128 by assigning probabilities to the same two candidate images. They completed 20 and 18 of 64 distinct-headword pairs respectively. The result is compatible with a small difference in either direction on this narrow task. Candidate availability, typeface, corpus conventions and focus selection all limit the comparison. [Options and queries](evidence/visual_word_senses_20261124/comparison_roster.json) · [Comparison counts](evidence/visual_word_senses_20261124/comparison.json) · [Text responses](evidence/visual_word_senses_20261124/choices_text.json)

## How much does the typeface matter?

| Exact replies from the frozen writer | *Analects* | *Chu ci* |
|---|---:|---:|
| Unseen serif, direct input | 1,458/6,987 | 1,157/7,173 |
| Unseen serif, matched source ink | 5,429/6,987 | 5,410/7,173 |
| Printed UKai, direct input | 1/6,987 | 0/7,147 |
| Printed UKai, matched source ink | 1,209/6,987 | 1,091/7,147 |

A bank of 10,765 training bitmaps takes 1,378,048 bytes on disk. Comparing a new source cell to these shapes by stroke distance and pooled ink restored most of the original performance on serif input. Complete serif pairs rose from 26 to 205 and from 6 to 83. For UKai they rose from zero to 28 and 21. Twenty-six *Chu ci* windows contained a character unavailable in UKai and were excluded from both UKai conditions. Among distinct training-known shapes in the *Analects*, the matcher recovered the correct form for 999/1,130 serif shapes and 137/1,130 UKai shapes. Date, identity and spelling play no part in the matcher. Its repertoire is finite and an unfamiliar glyph can be replaced incorrectly. [Matcher and secondary protocol](visual_sense_ink_matching_protocol.md) · [Physical outcomes and font hashes](evidence/visual_word_senses_20261124/matching.json)

## Resource scope

The aligned eight-pass fit comprised 7,736 AdamW steps and 1,979,648 presentations on one RTX 4090 D. It took 38.75 s, 5.51 kJ and 325,737,472 bytes of peak PyTorch allocation. The shared output motor took another 0.41 s and 33 J. Both were trained from scratch. The training-only auxiliary head adds 353,073 parameters during fitting. The randomized control took 40.05 s and 5.56 kJ. Preprocessing, evaluation, the control and the text reference are outside the aligned fit's timing. A two-choice comparison on 128 prompts cannot establish parity with Qwen on general language, or a thousandfold compute reduction at matched quality. [Aligned fit](evidence/visual_word_senses_20261124/aligned_fit.json) · [Control fit](evidence/visual_word_senses_20261124/shuffled_fit.json) · [Physical motor](evidence/visual_word_senses_20261124/motor.json)
