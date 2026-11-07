# Full-page supervision improves copying but not natural QA

The completed curriculum adds 2,560 updates to the prior 4,096-update visual
adapters. Exact stopped requested-only copying improves from **27/32 to 29/32**
(English 16/16; Chinese 13/16). Natural Chinese QA remains **0/32**. The model
answers with `1998年` or `1990年` regardless of the actual requested fact.
The frozen auxiliary language head supplies these readouts; no trained
image-answer generator is demonstrated.

| Complete held-out assessment | Before curriculum | After curriculum |
| --- | ---: | ---: |
| Requested-only copies, exact stopped | 27/32 | 29/32 |
| Chinese natural QA, exact stopped | 0/32 | 0/32 |
| Chinese paired questions, both correct | 0/16 | 0/16 |
| Normal Chinese outputs unchanged by question erasure | 29/32 | 30/32 |
| English natural QA, exact stopped | 0/32 | 0/32 |
| English paired questions, both correct | 0/16 | 0/16 |

## Full transcripts and optimizer coverage

The specified curriculum uses the same 384 training images and excludes all
64 validation examples. There are 1,536 dense transcription updates (four
complete passes), 512 QA updates (two passes) and 512 requested-only copy
updates (four passes). Every optimizer record is present and finite; the
scorer verifies the prescribed order and every per-ID pass count. All original
image/cache/text bindings remain checked. There is no matched continuation
arm, so this observation does not isolate a benefit over ordinary continued
training.

On the separate eight-page full-transcription screen, the four short copying
pages improve from 0/4 to 4/4 exact complete transcripts. Their aggregate
Levenshtein character error falls from 0.603 to zero. The four full Chinese
QA-page transcripts remain 0/4 exact; error increases from 0.999 to 1.734.
Only two of the four terminate before the 1,024-decision cap. Unrelated,
repetitive paragraphs replace the actual source passages. Insertions can make
character error exceed one. This screen measures transcription of the entire
visible page, distinct from the 32 requested-only copying assessments.

The model trains the same 8,489,984 gated visual/fusion adapter parameters,
with the source language body/head and image writer frozen. The complete
adapted model has 2,839,267,040 parameters, plus the 126,892,531-parameter
Wan codec and inherited external pretraining. The new training invocation
takes **1,946.753 seconds** with **7.072 GiB peak allocated VRAM** on one
RTX 4090 D. These costs exclude source pretraining, acquisition and evaluation.
The final checkpoint SHA256 is
`016429ec5454cdd681693d881df4c6b7d52fb43dcae7b9e15857d2d9cd32b483`.

## English transfer and readable source pages

The prospective English cohort consists of sixteen SQuAD 1.1 articles with
two questions each. Its eighty calls per model comprise 32 normal pages,
32 passage-erased pages and sixteen shared question-erased pages. Selection,
complete passages, all annotated alternatives, source offsets, masks and seeds
are fixed before student calls. No English passage enters this optimizer.

Independent source-page OCR has **0.001522 character error** (22/32 full
pages exactly recognized); primary answer images are 31/32 OCR-exact. OCR is
offline calibration only. Both learned models nevertheless answer **0/32**
normal English questions correctly. Before/after question erasure changes
11/32 and 7/32 normal outputs, respectively; neither model has any both-correct
pair. Output changes alone do not establish understanding. Correct-answer
retention without the passage has a zero denominator in these conditions.

The official SQuAD 1.1 release is CC-BY-SA-4.0, Rajpurkar et al., EMNLP 2016.
This public development subset may occur in foundation pretraining. It is
neither a pristine foundation test nor an official benchmark replication.
CMRC passage images retain CC-BY-SA-4.0 attribution to Cui et al.,
EMNLP-IJCNLP 2019; copy strings are synthetic.

## Interpretation and reproducibility

Dense teacher-forced labels do not repair page-level selection at this budget
and interface. Short copying success, finite loss and low memory do not
qualify a language model. The complete registered observations must be kept
when evaluating an interface correction or another method.

The subsequent [native-resolution diagnostic](native_reader_geometry_result.md)
identifies a substantial source-interface effect. It does not retroactively
change these original-pixel results or establish that the learned adapters
work at the corrected geometry.

The [curriculum export](evidence/native_visual_alignment_result/result.json)
binds 13 reports/journals/readout files and reuses the preceding original input
rasters. The [English export](evidence/native_english_transfer_result/result.json)
binds 121 files, including all 112 input/control/reference PNGs and every model
response. Both scorers were executed on the complete cohorts.

[Curriculum protocol](native_visual_alignment_protocol.md),
[English protocol](native_english_transfer_protocol.md),
[curriculum commands](../docs/native-visual-alignment.md) and
[English commands](../docs/native-english-transfer.md) specify the runs.
