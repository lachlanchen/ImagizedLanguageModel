# How reliable is a local translation grade?

On 103 Chinese news sentences from 57 WMT 2020 documents, two quantized local models each chose the professionally accepted English translation in both candidate orders on **67/103** cases. Each correct translation had unanimous no-error judgments from at least three raters. Each rival had a major accuracy error marked by every rater. The professional translators saw the document and the local models saw its sentence. The cases were selected before model grading. [Human ratings and study design](https://aclanthology.org/2021.tacl-1.87/) · [Pinned original annotations](https://github.com/google/wmt-mqm-human-evaluation/tree/29acd6999aaea3586378c9dcb58d8cc4820cede3/newstest2020/zhen) · [Fixed panel](evidence/human_translation_calibration_v1/panel.json)

| Decisions on the same 103 cases | Qwen3-8B | Qwen3-VL-8B |
|---|---:|---:|
| Both candidate orders select only the error-free translation | 67 | 67 |
| The choice changes with presentation order | 26 | 20 |
| Judged separately, accept the error-free translation | 96 | 94 |
| Judged separately, reject the major error | 28 | 46 |
| Both separate judgments agree with the professionals | 24 | 40 |

The paired 95% document-bootstrap intervals for the change from comparative to separate grading were −52.1 to −31.9 and −35.7 to −16.8 percentage points. The models gave identical pairs of comparative ballots and missed the professional result on 14 cases. Agreement between two checkpoints can therefore retain a shared error. The single-candidate procedure frequently credits fluent sentences containing major errors. Preserve its individual scores as automated measurements. [All ballots and intervals](evidence/human_translation_calibration_v1/results.json)

## Independent reader and writer

The earlier lookup interface admitted 54/103 Chinese inputs. One sentence exceeded 96 cells. Among the other 48 exclusions, 45 contained a new physical bitmap and three contained an existing bitmap outside the Chinese source inventory. The installed face could draw every source character. The independent writer stopped on 52/54, and both model judges gave zero stopped answers a full-meaning credit. Only nine news sources and both of their complete reference alternatives fit the comparison. The independent writer preferred the human-approved wording on five, with the identical nine decisions after replacing each source and its phrase draft. Every displayed news answer is archived. All eight isolated replays matched. [Scores and input scope](evidence/human_translation_calibration_v1/writer.json) · [Source coverage](evidence/human_translation_calibration_v1/source-coverage.json) · [Original PNGs](../publication/ilm-nature/evidence/wmt_news_visual_answers.zip)

Passing new bitmap observations through the frozen reader extended its input coverage to 102/103. Existing output pixels and stops reproduced on all 54 previously supported cases. On the additional 48, hiding the physical unfamiliar ink while preserving its location changed 47 answers. Both conditions earned zero model-graded complete meanings. The physical condition stopped on 47 and the hidden condition on 48. Sixteen isolated replays reproduced the original results. [Route](open_news_ink_protocol.md) · [Every free answer](../publication/ilm-nature/evidence/open_news_ink.zip) · [Independent grades](evidence/human_translation_calibration_v1/open-meaning.json)

## Comparator reading the actual pictures

The local pretrained Qwen3-VL-8B comparator received all 103 printed sources, including the rejected sources, with the source text absent from the image-only prompt. The other checkpoint credited 97 pictured-source and 102 text-source responses. The paired image-minus-text interval was −9.76 to 0.00 percentage points across documents. On the 49 originally excluded sources the counts were 47 and 48. All 54 common input PNGs were byte-identical to those supplied to the independent writer. The comparator uses a large pretrained vision-language model and returns text. Its absolute scores depend on the permissive single-candidate grader. [Inputs, original outputs and intervals](evidence/human_translation_calibration_v1/image-teacher.json)
