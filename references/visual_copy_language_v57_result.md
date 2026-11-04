# V57: real image-answer training, failed reading qualification

2026-11-05. Both final EMA arms and the frozen audit completed on physical
GPU 0. This is an independent image-input/image-output student. It uses no
tokenizer, Unicode IDs, OCR, global glyph inventory or external model call at
inference. The 19,007,523 parameters include the retained exact raster writer,
visual source attention, a copy gate and learned stopping.

Each arm trains 4,000 updates, batch four, on the same prepared CMRC questions
and natural Chinese continuation/motor exercises. The copy arm takes 278.572
seconds at 1.053 GiB peak allocated CUDA memory; no-copy takes 269.126 seconds
at 1.049 GiB. This excludes earlier V55/reader pretraining and data preparation.
Matched data/training RNG states are identical. Deterministic four-step
interrupted/resumed smoke model and EMA tensors match exactly.

| Primary 256-question result | Copy | Question pixels blanked | Matched no-copy |
| --- | ---: | ---: | ---: |
| Exact stopped answer | 10/256 (3.906%) | 1/256 (0.391%) | 0/256 |
| Multiset raster F1 | 15.718% | 5.553% | 6.582% |
| Mean emitted cells | 10.457 | 11.535 | 8.227 |
| Safety cap reached | 22.656% | 40.625% | 17.578% |

The paired full-minus-blank question F1 gain is 10.165 percentage points,
95% article bootstrap interval [7.043, 13.371]. Copy-minus-no-copy is 9.136
points [5.879, 12.483]. Exact-answer gains are 3.516 points [1.167, 6.061]
and 3.906 [1.613, 6.349], respectively. There are 198 primary articles;
2,000 grouped draws use seed 20265819. Blanking is an intervention, not an
alternative natural question drawn from the same distribution.

Only **1/393** shared-paragraph/disjoint-answer pairs has both answers exact.
All 2,677 primary copy-arm cells use observed-image copying; none uses the
retained free writer. These outputs therefore do not demonstrate novel free
answer synthesis. All 256 supplied V55 motor images remain exact.

Across all 1,321 development questions, mean event NLL is 1.329 copy versus
12.025 no-copy; sequence NLL is 8.377 versus 91.477. Reference images enter
teacher-forced likelihood only, never generation. Raster F1 is exact multiset
image overlap and does not enforce order; it is not official CMRC segmented
F1. Full audit cost is 1,117.083 seconds at 0.140 GiB peak allocated memory,
including the slow 2,106-image no-copy free-raster control.

Four of seven registered gates pass. Exact answer accuracy, absolute F1 and
both-correct paired questions fail. Decision: **`reject_reading_model`**.
The model shows limited question-dependent use of writing images, not a
usable general language model or Qwen-equivalent performance.

## What the saved errors reveal

A separately planned post-result diagnostic makes no model calls. It finds
43/256 correct first images, but only 10/43 become complete exact answers.
Fourteen outputs contain a complete reference prefix followed by extra images.
Only 9.766% have a retained reference length. Evidence selection and answer
boundaries both need improvement. This diagnostic changes no primary result,
decode rule or gate. It motivates the independently registered V58 image-block
comparison; it does not establish that comparison's outcome.

## Artifacts and reproduction

- Training/checkpoints: `artifacts/visual_copy_language_v57_20261105/{copy,no_copy}/`.
- Full audit/raw PNGs/tensors: `artifacts/visual_copy_language_v57_20261105/development/`.
- Error diagnostic: `artifacts/visual_copy_language_v57_answer_diagnostic/`.
- Dataset: `data/visual_qa/cmrc2018_short/` (CC-BY-SA-4.0).
- [Training and independent native-image inference](../docs/visual-reading-language.md).
- [Frozen protocol](visual_copy_language_v57_protocol.md), [event-law derivation](visual_copy_language_math.md).

The copy checkpoint SHA-256 is
`44ad7c18047a3d5f564b911321b1b225a6bd113e1330457ea53e823ea1c841f7`.
All other checkpoint and source hashes are recorded in the audit identity.
Training data and model weights remain local and ignored by Git. The paper
contains raw first audit outputs, including failures, with editable measured
charts. No image is selected for correctness or repaired by OCR.
