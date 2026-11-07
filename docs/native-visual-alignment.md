# Full-page visual alignment experiment

This experiment follows the [completed visual-access failure](../references/native_visual_access_result.md).
That model reads many short copying strings through its temporary text head,
but answers none of the 32 natural questions correctly. The new experiment
adds complete-page transcription supervision before returning to QA and
requested-only copying. It keeps the same images, source backbone, adapter
architecture and frozen language/image-generation components.

The [fixed protocol](../references/native_visual_alignment_protocol.md)
specifies the hypothesis, initializer, data, curriculum, optimizer and complete
assessments. This is auxiliary Unicode/token supervision. It does not provide
the required independent image-answer ILM, and transcription cannot substitute
for question understanding.

## Prepare existing data

First complete the existing [native source/cache setup](native-visual-access.md).
Reuse the isolated runtime and original 448-image cache; do not download or
duplicate weights. The completed predecessor adapter checkpoint is required.

```bash
export ILM_NATIVE_PYTHON="$PWD/artifacts/external/native_unified_source/runtime/bin/python"
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. python scripts/prepare_native_visual_alignment.py --output artifacts/native_visual_alignment/prepared_v1
```

Preparation reconstructs complete original source text and checks all 448
text SHA256 values against the original page geometry. It authenticates every
image/cache binding. Only the original 384 training IDs enter `training.json`.
The eight-page transcription screen has an input-only index and separate
references. Preparation makes no model calls or new images.

The QA training transcripts contain 2,387 distinct Han characters and 73,425
character occurrences, compared with 628/1,418 in the original short answer
labels. These record-level counts include paired passages and do not establish
the cause of the earlier failure.

## Execute sequentially

Output directories must be fresh. Each GPU stage holds the project GPU lock
and checks the shared-workstation memory limits. Preserve a live process until
it is terminal; do not restart because a tool observation times out.

```bash
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_visual_alignment.py --stage preflight --output artifacts/native_visual_alignment/preflight_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_visual_alignment.py --stage evaluate --mode transcribe --inputs artifacts/native_visual_alignment/prepared_v1/transcription_screen_inputs.json --checkpoint artifacts/native_visual_access/learning_v1/checkpoint.pt --output artifacts/native_visual_alignment/transcription_before_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_visual_alignment.py --stage train --output artifacts/native_visual_alignment/learning_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_visual_alignment.py --stage evaluate --mode transcribe --inputs artifacts/native_visual_alignment/prepared_v1/transcription_screen_inputs.json --checkpoint artifacts/native_visual_alignment/learning_v1/checkpoint.pt --output artifacts/native_visual_alignment/transcription_after_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_visual_alignment.py --stage evaluate --inputs artifacts/native_visual_access/full_validation_inputs.json --checkpoint artifacts/native_visual_alignment/learning_v1/checkpoint.pt --output artifacts/native_visual_alignment/full_validation_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_visual_alignment.py --stage evaluate --inputs artifacts/native_visual_access/question_controls_v1/inputs.json --checkpoint artifacts/native_visual_alignment/learning_v1/checkpoint.pt --output artifacts/native_visual_alignment/question_readout_v1
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. python scripts/score_native_visual_alignment.py --stage complete --output references/evidence/native_visual_alignment_result
```

The three-update preflight first reproduces all twelve predecessor readouts,
then optimizes the largest tokenized QA transcription, largest copying
transcription and largest QA answer. In the executed preflight, every adapter
receives finite gradients, all 316 adapter tensors change and the disabled
prefix remains exactly unchanged. The largest target has 404 decisions and
4,135 prefix positions. Peak allocated memory is 6.988 GiB. This establishes
execution only.

The complete run initializes again from the completed predecessor, discarding
the preflight state. Its 2,560-update schedule contains four full transcription
passes over 384 training examples, two QA passes over 256 examples and four
requested-only copy passes over 128 examples. The schedule is checked and saved
before updates. Checkpoints contain numeric adapters, optimizer/RNG state and
source/data/code/curriculum bindings. Resume unchanged code into a fresh path:

```bash
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_visual_alignment.py --stage train --resume artifacts/native_visual_alignment/learning_v1/checkpoint.pt --output artifacts/native_visual_alignment/learning_resumed
```

Evaluation loads fresh images and numeric model state, without training labels,
answer histories or a teacher service. The text head remains a diagnostic.
Report all raw responses, exact stopped copying, natural QA correctness,
both-correct question pairs and the erased-question controls. Even accurate
transcription does not establish image-answer generation. The completed [result](../references/native_visual_alignment_result.md) gives
29/32 exact requested-only copies, 0/32 Chinese QA and 0/32 English QA.
The four short full-page transcripts become exact, while the four complete
Chinese QA transcripts remain incorrect (aggregate character error 1.734).
The complete optimizer, all responses and failures remain in the evidence.

The completed before-transcription control gives 0/8 exact full transcripts.
Aggregate character edit error is 0.603 on the four copy pages and 0.999 on
the four QA pages. Copy-page transcription includes the complete visible
instruction and quoted content, so it differs from requested-only copying.
All four QA-page readouts collapse to `gōngfēi`. These immutable observations
measure the starting point, not a curriculum improvement. Before/after
transcription assessment uses order-sensitive Levenshtein error, with no case,
punctuation or quotation repair. Raw responses remain in the evidence.

The scorer authenticates all source evidence, the complete 2,560-update
journal and exact per-ID curriculum coverage. It keeps the full QA and
48-call question controls separate from transcription, reuses the existing
unmodified public input rasters and retains every new response/report. It
refuses incomplete optimizer schedules or assessments. To preserve only the
completed starting control before the learner finishes:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. python scripts/score_native_visual_alignment.py --stage before --output references/evidence/native_visual_alignment_before
```

A separate [prospective English transfer check](native-english-transfer.md)
completed after this learner and all registered assessments were terminal.
It adds question- and passage-erasure controls without changing training or
replacing any assessment. Both models fail all 32 English questions at the
original 1024-square geometry. A separate
[native-resolution diagnostic](../references/native_reader_geometry_result.md)
tests that source interface without changing these completed observations.
