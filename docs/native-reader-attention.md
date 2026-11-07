# Learn question-dependent attention without replacing the glyph frontend

The [controlled canvas result](../references/native_canvas_control_result.md)
shows that white padding alone reduces the earlier dense reader's copying
from 29/32 to 7/32. Its original copying strips retain 13/16 Chinese copies,
while all natural QA remains wrong. The later low-view state-alignment
learner is not a qualified replacement.

This [prospective learning protocol](../references/native_reader_attention_protocol.md)
keeps the complete dense source and its calibrated original geometry. Its
learned visual frontend is frozen. Fresh rank-16 residuals train only the
native language body's q/k/v/o attention projections. Zero residuals preserve
the measured source before optimization. The test asks whether question
selection can improve without degrading the existing visual features.

The model remains an externally pretrained full source with a temporary
Unicode head. This is a reader prerequisite; it does not generate image
answers or establish a lightweight foundation model trained from scratch.
There is no completed learning-quality result yet.

## Execute the fixed experiment

Reuse the pinned source setup and completed dense checkpoint. The training
inventory is the existing 640 bilingual examples; copying uses the original
strips, with every source pixel verified against the square originals.
Offline caching stores only native image prefixes and their bindings.
Although the source training JSON contains unused transcript fields, no
transcript or text-state target enters neural conditioning. The cache receipt's
`original_transcripts_loaded=false` field is imprecise: the JSON parser does
read those unused fields. No separate text-prefix or text-state computation
is performed. The execution record retains this metadata limitation.

```bash
export ILM_NATIVE_PYTHON="$PWD/artifacts/external/native_unified_source/runtime/bin/python"
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_reader_attention.py --stage cache --output artifacts/native_reader_attention/cache_v1
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/run_native_reader_attention.py
```

The controller reuses completed code-bound stages and preserves incomplete
or live ones. It runs three actual preflight updates, 2,560 fresh main
updates, the complete 160-condition original-layout assessment and scoring.
All four passes are fixed in advance. The disabled/zero source replay,
FP32 causal check, every trainable gradient and all six existing quality
requirements must pass where applicable. Lower loss, finite gradients or
memory feasibility are not language results.

Inspect `artifacts/native_reader_attention/controller_status.json` and its
stage logs. Use one GPU job at a time. Cache/numeric checkpoints remain
local; the final export retains raw responses and a compact optimizer
journal with the full raw journal's SHA256.

## Resume after a verified interruption

After the prior controller and every child are terminal, use fresh explicit
output paths. The learner restores the actual optimizer/RNG state and copies
the old journal through the saved checkpoint update. An uncheckpointed tail
stays in the original directory.

```bash
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/run_native_reader_attention.py --resume artifacts/native_reader_attention/learning_v1/checkpoint.pt --learning artifacts/native_reader_attention/learning_resumed_v1 --after artifacts/native_reader_attention/after_resumed_v1 --score-output references/evidence/native_reader_attention_resumed_result
```

The resume path has not yet been exercised by a real interruption. Never
resume or restart a process merely because a tool observation expired.
