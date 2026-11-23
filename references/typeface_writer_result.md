# Reading Chinese across typefaces

The causal image writer answered 72 of 192 questions in UKai and returned both distinct answers on 13 of 96 pictured passages. Before adaptation it gave zero exact answers in that typeface. On the same questions printed in an entirely withheld font family, it gave 13 individual answers and passed zero pairs of 98. On a separate CMRC trial cohort in that family it gave 15 individual answers and passed zero pairs of 197. The checkpoint missed the complete-passage criterion for a general cross-typeface upgrade. It remains available for research and as an offline inscription-catalogue reader. [Figure 102](../publication/ilm-nature/figures/Figure_102_typeface_writer.png) · [Run the checkpoint](../models/typeface_writer/README.md)

Each table entry counts exact stopped individual answers, followed in parentheses by the number of passages with both answers correct. The parent is the preceding interval-distilled writer. “Map only” attaches a fitted visual correspondence to that frozen writer. Replay and mapped replay use identical additional question images and update the same language weights.

| Printed passages | Questions (pairs) | Parent | Map only | Typeface replay | Mapped replay |
|---|---:|---:|---:|---:|---:|
| Regular | 196 (98) | 72 (11) | 72 (11) | 71 (11) | 77 (16) |
| Noto Bold | 196 (98) | 39 (7) | 71 (10) | 50 (8) | 78 (17) |
| UKai | 192 (96) | 0 (0) | 65 (9) | 8 (0) | 72 (13) |
| HanaMin, excluded from adaptation | 196 (98) | 0 (0) | 15 (1) | 3 (0) | 13 (0) |
| Additional trial, regular | 394 (197) | 89 (13) | 89 (13) | 96 (16) | 97 (14) |
| Additional trial, HanaMin | 394 (197) | 0 (0) | 16 (0) | 3 (0) | 15 (0) |

The independent scorer compared the returned bytes with complete references and required the writer to stop. It accepted each cell in either the source or canonical typeface. On the additional trial's first gain over replay, the question asked for a fish's common name and the writer reproduced the recorded answer 俗名龙尖. On the first error it returned 算为泡泡堂 instead of the recorded 踢爆. The HanaMin development question-erasure control gave zero exact answers for the mapped writer. The 197 trial passages were screened against official training and development sources and the reserved DRCD material. Their historical presence in the parent transformer's pretraining is unknown. [Original pixels and four-arm comparisons](evidence/typeface_writer_20261124/result.json) · [Additional source selection](typeface_writer_additional_sources.md) · [Public CMRC data](https://github.com/ymcui/cmrc2018)

## What transfers between printed forms

The learned correspondence selects a regular-print embedding for recognized out-of-inventory images. A confidence threshold and top-two margin allow it to abstain. The visual pointer retains the unmodified input bitmap, including after a successful match. Two shapes can therefore share a reading representation and still produce different ink when copied. The Lean derivation also identifies when a change in prototype scores preserves both confidence and separation from competing readings. It assumes bounds on the score perturbations. [Checked derivation](../formal/span_list/TypefaceCopy.lean) · [Protocol and thresholds](typeface_writer_protocol.md)

On the UKai development inputs, 75,111 of 77,093 character occurrences followed the correct correspondence or already matched the canonical bitmap. On the additional HanaMin trial, 106,216 of 158,717 were correct, 45,882 led to abstention and 6,619 selected a different identity. Counts include repeated characters. In the original historical controls, 70 of 258 distinct inscription bitmaps already occurred in the fixed inventory. Of the other 188 the model abstained on 187 and mapped one to a printed prototype. It preserved all 192 familiar historical replies, 54/192 with rephrased instructions and the known 言 card in both orders. On earlier modern Chinese reading questions the mapped writer gave 146/359 exact answers. On DuReader it matched nine of 177 contiguous answers and none of the 94 distributed or 300 missing-content references. [Physical identification audit](evidence/typeface_writer_20261124/identification.json) · [Task scores](evidence/typeface_writer_20261124/result.json)

## An offline catalogue

The local index covers 2,315 headwords and 6,267 headword–stage attestations chosen from previously screened Hanziyuan plates. Given a headword and two stages, retrieval assembles the original glyphs on a page with an additional entry. A user can supply a typed request or their own question image. The same causal writer reads the resulting images and produces the answer strip. The archive records both catalogue labels and the original SVG digests. This retrieval task supplies evidence for the forms; a causal or chronological account of their development would require further source material.

For 中 the command retrieved bronze B00616 and oracle J00887 from entry E09560. With 名 on the same page as a distractor, the CPU writer returned the requested historical bitmaps in the correct order, including the complete modern Chinese labels. The extracted archive reproduced the local command's page, question, answer and provenance. It also reproduced both 言 answers and the HanaMin reading example without a teacher. [Actual response for 中](evidence/typeface_writer_20261124/zhong/answer.png) · [Catalogue receipt](evidence/typeface_writer_20261124/catalogue_release.json) · [Original catalogue](https://hanziyuan.net/#%E4%B8%AD)

## Training cost

The four-font mapping saw 23,812 printed views of 5,958 training identities over 230,400 sampled presentations. Fitting its convolutional metric and prototypes took 22.6 seconds including table preparation. Each subsequent image writer processed 32,457 examples in 1,638 updates and 400,724 answer positions on one RTX 4090 D. Replay took 117.4 seconds and mapped replay 114.4 seconds in the recorded training loop. Peak allocated memory was below 4 GiB for each writer. The deployed learned components comprise the earlier 86,588,258 parameters plus 1,642,848 fitted metric and prototype coefficients. The prototypes are serialized as frozen buffers. Adaptation uses a text-pretrained language backbone. Its upstream compute and the costs of acquiring the source corpus are outside these measurements. [Mapping fit](evidence/typeface_writer_20261124/correspondence.json) · [Replay](evidence/typeface_writer_20261124/replay_fit.json) · [Mapped replay](evidence/typeface_writer_20261124/mapped_fit.json)

## Isolating the remaining reading error

When supplied the correct regular-print embedding for each unambiguous representable input bitmap, the mapped writer answered 95/394 trial questions and passed 13/197 passage pairs. The learned map gave 15/394 and zero pairs. Out of 237,478 input occurrences across the two HanaMin evaluations, 5,543 had no unambiguous in-inventory correspondence. Regular print gave 97/394. Actual source images and the fixed physical output inventory remain distinct in the supplied-map test. [Intervention and remaining input conflicts](evidence/typeface_writer_20261124/supplied_identity.json)

In a warm batch-one trial run on one RTX 4090 D, the mapped writer returned 15/394 exact answers in 28.15 seconds. Assembling the pictured evaluation inputs and references took 1.86 seconds separately. The image results differed from batched output on 2 question(s). Qwen3-8B Q8, with all 37 layers on the same GPU, answered 142/394 from text in 104.20 seconds and passed 33/197 pairs. No matched-quality resource advantage was demonstrated. [Device placement, physical archives and costs](evidence/typeface_writer_20261124/serving.json)

## Questions about additional catalogue headwords

The same inventory supplied new cards for 100 development headwords outside the training headwords and earlier 48-card evaluation. Each question asked for two real forms in a particular order. Both requests for each headword shared the same page. The table gives results for the mapped writer; the archived comparison includes the frozen parent and replay controls.

| Request | Whole reply | Both forms in order | Both whole replies |
|---|---:|---:|---:|
| Familiar Chinese | 200/200 | 200/200 | 100/100 |
| Rephrased Chinese | 54/200 | 70/200 | 9/100 |
| Chinese in HanaMin | 25/200 | 46/200 | 0/100 |
| English, requesting Chinese labels | 0/200 | 44/200 | 0/100 |

Form-order scoring checks for the two requested physical bitmaps in the specified order and requires a stop. It can give credit when surrounding prose is wrong. The first mapped example passing both familiar requests, when available, is shown with its actual source and archive labels. [New catalogue scores](evidence/typeface_writer_20261124/catalogue_evaluation.json) · [Original card and responses for 盗](evidence/typeface_writer_20261124/catalogue_example)
