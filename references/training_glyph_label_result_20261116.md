# Source images are faithful; isolated Unicode recovery is not identifiable

The [registered full audit](evidence/training_glyph_label_registration_20261116.json)
is complete. All 13,785 training records, comprising 9,023 QA and 4,762 book
records, authenticate to their original source/group/text hashes and prompt
boundaries. Every one of 7,255,223 cells is byte-identical to the original font
rendering. There are 2,576,239 target positions. No development image or label,
teacher-unit array, model weight or API enters the label inventory.

The [result](evidence/training_glyph_label_result_20261116.json) nevertheless
rejects **exact context-free raw-character supervision**. The 9,599 source
codepoints produce 9,581 distinct masks. Eleven nonwhite masks have conflicting
raw labels, and the white mask includes 34 occurrences of U+200B ZERO WIDTH
SPACE, which is not removed by the original whitespace-only text normalizer.
Every collision and occurrence remains in the evidence; none is silently merged.

| Identical nonwhite source image | Raw source labels | Occurrences |
| --- | --- | --- |
| Ellipsis | U+2026 / U+22EF | 863 / 8 |
| Horizontal sound mark | U+30FC / U+3127 | 148 / 4 |
| Middle dot | U+00B7 / U+2022 / U+2027 / U+30FB | 7806 / 43 / 85 / 161 |
| Lowercase o | U+006F / U+043E | 16810 / 5 |
| Lowercase a | U+0061 / U+0430 | 20737 / 11 |
| Lowercase x | U+0078 / U+0445 | 1264 / 5 |
| Lowercase y | U+0079 / U+0443 | 3503 / 5 |
| Wave dash | U+301C / U+FF5E | 37 / 445 |
| Capital I | U+0049 / U+2160 | 3293 / 9 |
| Capital A | U+0041 / U+0410 | 5201 / 10 |
| Division slash | U+2215 / U+FF0F | 5 / 150 |

For an image-only pointwise decoder, let n(g,c) count character c at mask g.
At least sum over g of [sum over c n(g,c) minus max over c n(g,c)] raw labels
must be wrong: one fixed image cannot produce both conflicting answers. The
nonwhite conflicts alone force 388 errors on this empirical census. They are
rare and do **not** explain the much larger autonomous-language or isolated-
teacher failures. Context can help infer a script or meaning; pixels alone
cannot recover an invisible encoding choice.

The complete record labels and multi-label physical inventory are retained
privately with hashes, without correcting original images. The mask array is
byte-identical to the earlier 9,581-form selective-verifier bank. This audit adds
source-label accounting; it does not create a new corpus or repair the rejected
Hamming verifier. Runtime is 31.966 seconds at 0.208 GiB peak process RSS;
derived files total 10,536,820 bytes. Three CPU contract checks cover complete
extent, boundary mismatch, alias preservation and invisible nonspace labels.
No neural call or optimizer update occurs.

## Consequence for complete-answer learning

Separate optical-form identity from raw Unicode reconstruction. Actual image
forms provide an unambiguous training target even where several source encodings
share their pixels. Language context must still learn meaning, answer choice,
grammar and stopping. A form recognizer alone cannot establish those abilities.
The deployed actor continues to generate physical pixels, with no catalogue,
Unicode output head or font renderer.

A bounded candidate is to test the **existing frozen glyph-image encoder** as a
training-only visual metric, replacing the rejected verifier's bit-space
distance while retaining its complete-answer scoring and every original gate.
The original raw-character condition remains failed; this alternative makes no
claim to recover conflicting codepoints. Freeze its checkpoint, feature map,
metric and rejection rule before calibration. It must distinguish actual correct,
wrong and uncertain outputs on the full preserved cohort. No new fit is admitted
by this data result. The [saved preprocessing finding](isolated_reader_preprocessing_audit_20261116.md)
also remains a hypothesis, not an excuse to resume the rejected teacher.
