# English passage reading and erasure controls

This prospective diagnostic tests English question answering after the
[full-page visual-alignment experiment](native-visual-alignment.md). It is
evaluation-only. The completed training and its registered Chinese assessments
stay unchanged. No English passage or answer enters that optimizer.

The source is the official [SQuAD 1.1 release](https://rajpurkar.github.io/SQuAD-explorer/),
distributed under CC-BY-SA-4.0. The
[protocol](../references/native_english_transfer_protocol.md) pins the source
revision/blob, selection order, geometry, controls, seeds and scoring rules.
Sixteen distinct articles supply two questions each. All annotated alternatives
are preserved and every answer offset is checked against the original passage.
Source pretraining may already contain this public benchmark; this is not a
pristine foundation-model test or an official SQuAD evaluation.

## Prepare and calibrate on CPU

```bash
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. python scripts/prepare_native_english_transfer.py --output artifacts/native_english_transfer/prepared_v2
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. python scripts/score_native_english_transfer.py --stage calibrate --output artifacts/native_english_transfer/calibration_v1
```

The preparer verifies the pinned 4,854,279-byte source file and reuses it if
already present. Whole passages and questions are rendered at the existing
32-pixel font size, without answer-aware cropping. Overflow and missing glyphs
are recorded as preparation rejections. The output has 32 normal pages, sixteen
shared question-erased pages, 32 passage-erased pages and 32 separate reference
images. Input indexes contain no answer strings or annotated offsets.

Erasing the question preserves all context pixels. Erasing the passage leaves
the complete question at its original coordinates. All conditions of a passage
share the same latent and readout seeds. Correct answers retained without the
passage cannot establish that the model used that passage.

The executed CPU calibration recognizes 31/32 primary reference images exactly.
Full input-page OCR is exact on 22/32 pages, with aggregate character edit
error 0.001522. Raw punctuation/recognition errors remain in the report; no
case is replaced or repaired. This measures source readability, not the
student. OCR never enters model conditioning. All 112 images, selected cases
and input/reference indexes match the initial preparation exactly; v2 records
font coverage over the complete cohort.

## Run only after the active controller is terminal

Complete all 2,560 alignment updates and its registered transcription, full QA
and paired-question assessments before these additional calls. Keep one GPU
job at a time. Reuse the existing isolated runtime and model weights.

```bash
export ILM_NATIVE_PYTHON="$PWD/artifacts/external/native_unified_source/runtime/bin/python"
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_visual_alignment.py --stage evaluate --inputs artifacts/native_english_transfer/prepared_v2/inputs.json --checkpoint artifacts/native_visual_access/learning_v1/checkpoint.pt --output artifacts/native_english_transfer/before_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_visual_alignment.py --stage evaluate --inputs artifacts/native_english_transfer/prepared_v2/inputs.json --checkpoint artifacts/native_visual_alignment/learning_v1/checkpoint.pt --output artifacts/native_english_transfer/after_v1
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. python scripts/score_native_english_transfer.py --stage complete --output references/evidence/native_english_transfer_result
```

Each model makes 80 calls: 32 normal, sixteen question-erased and 32
passage-erased. The scorer preserves raw outputs, every source/reference/control
image, calibration records and reports. It reports strict stopped answers
against any annotated alternative, both-correct question pairs, erasure changes
and correct-answer retention without the passage. NFC and collapsed whitespace
are allowed; case, punctuation, quotes and word boundaries remain significant.

These remain auxiliary native text-head readouts. English transfer cannot
qualify Chinese understanding, direct image-answer generation, historical
writing or the complete ILM goal. Both completed adapter checkpoints answer 0/32 normal questions correctly
at the original 1024-square geometry. The first model emits `1999年` or
`1998年`; the dense curriculum emits `1990年` or `1998年`. No paired questions
are both correct. See the [complete result](../references/native_visual_alignment_result.md)
and the subsequent [native-resolution check](../references/native_reader_geometry_result.md).
