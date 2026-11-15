# Nonempty mismatches usually retain higher emitted-image probability

The [bound saved-score calculation](ranked_glyph_score_gap_protocol.md) is
complete on all 124 cases and 12,597 supplied-history positions of the completed
nonempty-ranking actor. It makes no neural call and changes no image, frozen
reading, gate or training run. An independent scalar implementation reproduces
the vector calculation from the original arrays.

Among 7,926 positions where both source and emitted images contain ink but
their bitmaps differ, the saved density prefers the source at **717**, the
emitted image at **7,206**, and neither by the fixed 0.001-nat margin at **3**.
Thus 90.92% of these canonical mismatches already favor the emitted alternative.
The original 4,609 exact bitmaps, 37 false blanks and 25 false-ink outputs remain
separate counts; the 252 genuinely white source positions are retained.

| Supplied-history group | Nonempty bitmap mismatches | Source preferred | Emitted preferred | Ties |
|---|---:|---:|---:|---:|
| Development book, fixed | 931 | 51 | 880 | 0 |
| Development QA, fixed | 302 | 44 | 258 | 0 |
| Development QA, proportional | 59 | 16 | 43 | 0 |
| Training book, fixed | 3,263 | 287 | 2,974 | 2 |
| Training book, proportional | 3,254 | 298 | 2,955 | 1 |
| Training QA, fixed | 94 | 15 | 79 | 0 |
| Training QA, proportional | 23 | 6 | 17 | 0 |
| Total | **7,926** | **717** | **7,206** | **3** |

All three book strata fail the fixed requirement that at least half of their
nonempty mismatches prefer the source: observed/required counts are 51/466,
287/1,632 and 298/1,627. A pooled count does not replace these strata. The
source-preference rule does not support a search-only repair of the observed
nonempty mismatches. This is separate from the completed blank-image comparison,
which also failed its own whitespace and added-erasure conditions.

## What the comparison establishes

At the same supplied-history state, let `S` be the complete log probability of
the source bitmap and `E` that of the existing emitted bitmap. If
`E > S + 0.001`, the source is below an already exhibited alternative in the
measured arithmetic. It therefore cannot be the highest-scoring complete image
in those scores. More exact maximization of that unchanged density cannot make
the source its maximum at such a position. This does not assert that the emitted
image itself is globally optimal or identify a unique training failure.

At the remaining 717 source-preferred mismatches, a higher-scoring source is
evidence compatible with a greedy prefix limitation. It does not prove that a
practical search will find the source, nor that the source outranks every other
image. Prefix logits and global optima are unmeasured. Numerical ties remain
uncertain rather than being credited to either side.

This calculation concerns canonical image prediction. Bitmap mismatch alone is
not semantic error: another glyph style or another complete continuation might
be valid. All contexts here receive preceding source images; autonomous
feedback, answer meaning, coherent books and natural stopping are unmeasured.
The cohort and some outputs were already examined. The prospective calculation
binding is not untouched confirmation or independent human adjudication.
Even a useful language model need not reproduce a reference book's exact next
glyph at every position. Forcing every reference bitmap to be the mode is not
a substitute for learning or evaluating complete language.

## Consequence and provenance

Conditional source-mode learning remains a candidate for later controlled
testing if the current white-target trial still fails. Whole-image ranking
against blank alone does not compare the source with other nonempty modes.
The running white-target objective adds a genuine-whitespace term; this result
does not establish whether its shared-parameter updates improve nonempty forms.
Complete its original fit, audit and qualification unchanged. Neither this
diagnosis nor a future readability pass substitutes for the complete-language
comparison. No new fit, decoder or autonomous generation is authorized here.

The implementation and calculation were committed before the score differences
were opened: implementation `c53fc216`, registration `f706ddd9`. The registration
binds four source files and 255 existing saved inputs. Thirteen CPU accounting
checks pass. The complete vector and independent scalar calculations retain
all cases, original bitmaps, blank flags and likelihood paths; no probability
array or frozen transcription was repaired. All 115 sources and original live
owners of the separate white-target fit remain unchanged.

The [complete counts](evidence/ranked_glyph_score_gap_result_20261116.json) and
[verification receipt](evidence/ranked_glyph_score_gap_verification_20261116.json)
bind the result, source and original input hashes and record CPU costs. The
[manuscript build receipt](../publication/ilm-nature/evidence/glyph_score_gap_article_build.json)
now records its inclusion in the 52/91/92-page reading, review and cumulative
redline PDFs, with native Figure 28 and its editable PowerPoint source. Results,
Methods, Discussion, summary and caption retain the distinction between these
canonical density diagnostics and useful complete language. All prior figures
and original failure images remain preserved; no new model output was generated.
