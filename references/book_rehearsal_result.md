# Learning a role reversal and reading it across typefaces

A pictured **Tom was helped by Mary.** produced the Chinese answer **玛丽帮助了汤姆。** Exchanging Mary and Tom in the passive source reversed their roles in the answer. The same writer failed to translate a description of a house and its author's reception from *Gulliver's Travels*. The plates contain the saved input and answer PNGs. All ten illustrated model replies were independently reproduced on CPU. [Original pixels](../publication/ilm-nature/figures/book_rehearsal/) · [Paper, pp. 50–52](../publication/ilm-nature/visual_language_paper.pdf#page=50) · [Run the weights](../models/book_breadth/)

## Holding the training budget fixed

Starting from the same 1,688,480-parameter model as the broad literary and talk writer, we replaced 3,219 training pairs with distinct, previously seen role constructions. The substitution included 1,711 active and 1,508 passive statements, affirmative and negative. It matched both source and target lengths in each slot. These pairs occupied 127,291 of 18,109,563 target positions per pass, including stops. Both writers received 11,148 updates and 108,657,378 target positions. The development criterion selected the sixth pass. The rehearsal fit took 234.3 seconds across the six passes and peaked at 1.21 GiB of allocated GPU memory. These figures exclude earlier training, data preparation and evaluation. The rendered literary translations may contain errors. [Protocol](book_rehearsal_protocol.md) · [Data substitutions](evidence/book_rehearsal_v1/design.json) · [Checkpoint receipt](../models/book_breadth/rehearsal/fit.json)

| Complete stopped Chinese image | Starting writer | Broad writer | Broad writer with rehearsal |
|---|---:|---:|---:|
| Active, out of 472 | 471 | 86 | 472 |
| Passive, out of 472 | 422 | 0 | 463 |
| All eight changes correct, out of 118 families | 76 | 0 | 110 |

Each family exchanges two agents and changes voice and polarity. Original control answer pixels and stops reproduced on every revisited request. All three exact natural answers survived rehearsal. The held combinations use a controlled writing construction. [Complete counts](evidence/book_rehearsal_v1/result.json)

## Can the model translate an unfamiliar passage?

Four further literary works and 24 further FLORES article URLs supplied 48 fresh passages. The preparation screened both language sides against the prior training sets and the current training images. Each passage was presented in both directions. The comparison measures freely written, stopped translations. A local Qwen3-VL evaluator received the readback of each output image and scored all candidates in two opposite masked orders. Qwen3-8B Q4 received the source as text. Exposure to these texts during external model pretraining is unknown. [Reservation and corpus digest](evidence/book_rehearsal_v1/fresh.json) · [Complete votes and controls](evidence/book_rehearsal_v1/result.json)

| Panel | Requests | Broad | Rehearsal | Qwen Q4 | Reference accepted twice |
|---|---:|---:|---:|---:|---:|
| Earlier literary, TED and FLORES | 144 | 5 | 4 | 79 | 140 |
| Four fresh novels and books | 48 | 0 | 0 | 22 | 47 |
| Fresh FLORES articles | 48 | 0 | 0 | 36 | 48 |

All 240 unrelated foils were rejected in both orders. Across the prior panel the grader awarded the rehearsal model three bitmap-exact translations and one alternative. The external comparator received 86/144 in the preceding assessment and 79/144 in this one. The model-based judge and its presentation of candidates limit precision. No independent human translation rating is available. With zero gains on fresh passages, rehearsal failed the specified joint transfer criterion.

The same failed 96 requests reveal a difference between conditioning and writing. With the reference target image and its stop supplied, the writer assigned the highest whole-stream likelihood to the true source among 24 same-domain, same-direction candidates on 90/96 requests. It preferred the original glyph order to the reversed source on 93/96. With the five distractors closest to the true source in input length, the true source won 93/96. The test can reward lexical matches, alignment and length cues. It supplies every preceding correct target glyph and earns no free-writing credits. [Scores by direction, including the complete ranks](evidence/book_rehearsal_v1/conditioning.json)

## The same sentence in other ink

Changing only the input to Noto Serif CJK SC Regular erased all 110 complete role families. The Hamming correction takes a nearby training bitmap only when twice the pixel distance is below that bitmap's separation from every rival. It recovered 833 of 834 distinct perturbed input symbols after four seeded flips per symbol, with no incorrect snap, and restored all 110 families. In serif it corrected 40 distinct symbols and left 793 incorrect ones unmatched. Lean checks the precise stability radius for a finite binary image and the upper bound on fixed-rubric gains when few complete replies change. [Lean source](../formal/span_list/InkSeparation.lean) · [Direct and Hamming results](evidence/book_rehearsal_v1/typography.json)

We then froze the language model and trained a convolutional reader with 1,423,653 additional parameters. It learned from 28,097 physical shape examples using canonical images and ten font faces, with seeded geometric and pixel augmentation. It received no sentences, translation targets or Noto Serif faces. The classifier corrects eligible unfamiliar images, with an exact-bitmap shortcut and a 0.5 confidence threshold. This adapter targets a known repertoire. Its combined deployed parameter count with the translator is 3,112,133. The 3,000 optical updates took 14.1 GPU seconds on the same RTX 4090 D. [Optical protocol](rehearsal_font_protocol.md) · [Fonts, sources and measured cost](../models/book_breadth/typeface/fit.json)

| Pictured source | Unaided complete families | With learned reader | Same natural outputs as canonical with reader |
|---|---:|---:|---:|
| Canonical | 110/118 | 110/118 | 240/240 |
| Four bit flips per symbol | 0/118 | 110/118 | 204/240 |
| Noto Serif CJK SC Regular | 0/118 | 110/118 | 239/240 |
| DejaVu Serif Condensed | 0/118 | 48/118 | 54/120 eligible |

The Noto Serif repair reproduces all four previously credited answers and all 96 failed fresh replies byte-for-byte, including stopping. DejaVu Serif Condensed covers the English-source subset. Coverage, correct role binding and correct natural writing each require a separate measure. Scanned pages, unconstrained drawing, arbitrary historical glyphs and matched-quality savings against an LLM remain open. [Full optical assessment](evidence/book_rehearsal_v1/typeface-reader.json)
