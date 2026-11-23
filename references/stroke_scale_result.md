# Stroke variation and reading questions from images

The same image writer answered 43 of 394 Chinese questions in HanaMin after its glyph recognizer learned from changes in stroke width and proportions. Translation-only training gave 15. Six of 197 passages gained both correct answers and none lost both-correct status. In a second reserved family, FandolSong, two of 149 passages gained both answers and none lost them. Using the augmented reader everywhere cost one previously correct regular-print pair. Choosing a reader from the input pixels recovered that pair. On a separate translated benchmark the routed reader still completed zero two-question passages in either unfamiliar family. The previous released model remains the default. [Figure 103](../publication/ilm-nature/figures/Figure_103_stroke_scale.png) · [Reproduce the experiment](../models/print_routed_writer/README.md)

| CMRC or catalogue cohort | Questions | Shift only, exact (complete pairs) | Stroke and scale | Print routing |
|---|---:|---:|---:|---:|
| Trial regular | 394 | 97 (14) | 96 (13) | 97 (14) |
| Trial HanaMin | 394 | 15 (0) | 43 (6) | 43 (6) |
| Trial FandolSong | 298 | 18 (1) | 24 (3) | 24 (3) |
| Familiar historical requests | 200 | 200 (100) | 198 (99) | 198 (99) |

In the development sources, HanaMin rose from 13/196 to 35/196 answers. FandolSong fell from 20/158 to 17/158. The 95% paired bootstrap interval for the complete-passage gain on the HanaMin trial was 1.0–5.6 percentage points; FandolSong's interval included zero. Across just the 149 trial pages renderable in both families the gains were four and two complete passages. Erasing the question gave zero exact trial answers in both unfamiliar families. On the first scored miss the output was 4分钟内 (within four minutes), while the recorded answer was 4分钟 (four minutes). Exact-reference scoring rejected the extra glyph. [Original generated strips](evidence/stroke_scale_20261124/examples) · [All comparisons, including shared-source estimates](evidence/stroke_scale_20261124/result.json) · [Font coverage and omitted questions](evidence/stroke_scale_20261124/coverage.json)

## What the visual learner changed

The two models fitted 1,642,848 correspondence coefficients using the same four source fonts, 5,958 identities, 1,200 updates, 230,400 sampled identities, fonts and shifts. The recorded schedules have the same hash, and the repeated shift-only fit reproduced the earlier weight file byte for byte. Correct readings increased from 106,216 to 121,915 of 158,717 HanaMin trial occurrences and from 73,391 to 83,849 of 119,814 FandolSong occurrences. HanaMin wrong identifications fell from 6,619 to 3,847. These counts include repeated characters. The original and augmented fits took 27.3 and 8.3 seconds including table construction, respectively, with shared renderer caches warmed by the first fit. Each used under 0.33 GiB peak allocated GPU memory. The same pretrained language weights served both models. [Training runs and schedule](evidence/stroke_scale_20261124/presentations.json) · [Optical errors and punctuation](evidence/stroke_scale_20261124/optical.json)

An exact image-to-label function on an observed training collection exists if and only if identical images have the same target. Lean checks the equivalence for an arbitrary renderer and label set. The condition fails on 16 distinct augmented bitmaps spanning 39 presentations and three control bitmaps spanning 13 presentations. No claim about semantic generalization follows from the theorem. In a separate source-exchange probe, only two passages met the full equal-width, coverage and uniqueness criteria. Both models failed to answer all four questions on either exchange in any print. [Lean proof](../formal/span_list/VisualTransform.lean) · [Source-exchange results](evidence/stroke_scale_20261124/exchanged.json)

## Which question is being asked?

The router uses the shift-only correspondence if at least 80% of nonblank glyph occurrences on each strip are already in the fixed visual inventory. Otherwise it uses the augmented correspondence. Both paths share the same language model and retain the physical source pixels for copying.

| XQuAD question and print | Passages | Original exact | Routed exact | Original complete pairs | Routed complete pairs |
|---|---:|---:|---:|---:|---:|
| Chinese, regular | 227 | 58 | 58 | 7 | 7 |
| English, regular | 227 | 6 | 6 | 0 | 0 |
| Chinese, HanaMin | 227 | 3 | 22 | 0 | 0 |
| English, HanaMin | 227 | 1 | 2 | 0 | 0 |
| Chinese, FandolSong | 219 | 10 | 12 | 0 | 0 |
| English, FandolSong | 219 | 1 | 5 | 0 | 0 |

XQuAD pairs human translations with the original English questions from SQuAD development. All 472 initially selected identifiers occur in the locally stored SQuAD development corpus. Of the 454 questions covered in regular print, 23 identifiers had appeared in an earlier project's English fitting selection and eight in a previous assessment. Excluding their original paragraphs left 210 paired passages; routed Chinese HanaMin reading gave 18/420 answers and still zero complete passages. The upstream model's pretraining provenance is unknown. [XQuAD and original authors](https://github.com/google-deepmind/xquad) · [Previously seen English sources and separate recount](evidence/stroke_scale_20261124/earlier_english_use.json) · [Routed results](evidence/stroke_scale_20261124/routed.json)

On the first 64 HanaMin passages, serial inference on one RTX 4090 D gave 8/128 Chinese-question answers in 8.92 seconds and 2/128 English-question answers in 14.16 seconds. Qwen3-8B Q8 received the corresponding text and gave 85/128 and 74/128 in 28.03 and 27.47 seconds. It completed both questions on 29 and 26 passages. The image writer completed zero and its serial outputs matched the batched outputs on every scored case. Timings exclude model loading and image rendering. Different accuracy prevents a matched-quality efficiency claim. [Timing, placement and hashes](evidence/stroke_scale_20261124/serving.json)
