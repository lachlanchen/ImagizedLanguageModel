# Image questions and generated glyph answers

The semantic glyph loop is a research model with limited demonstrated
Chinese reading and writing. It produces some correct short answers and
tracks eleven of fourteen changed-year conditions on two selected templates.
Most questions in the broader sixteen-case development cohort still fail.
See the [complete result](../references/semantic_glyph_language_result.md)
and [date interventions](../references/semantic_glyph_counterfactual_result.md)
for every output, uncertainty and measurement scope.

The neural model reads a prompt image, preserves its native contextual state,
generates a 32×32 glyph and encodes the actual emitted pixels before choosing
the next glyph or stopping. The runtime uses no character inventory, Unicode
decoder, tokenizer, OCR service, output font renderer or external teacher.
It retains an externally pretrained 7.5B visual/contextual body; the small
bridge does not replace that inherited knowledge or establish a small
foundation trained from scratch.

## Run an image prompt

The current verified bridge checkpoint is
`artifacts/semantic_glyph_language_v1/fit/checkpoint_latest.pt`, SHA-256
`3a81b4e6f4306b218d08e6988f84d688f4267a773356a8d4c9c470e496ae5141`.
Large learned weights remain local rather than in Git. The existing shared
Qwen3-VL archive and native GGUF loader are reused from the sibling LocalLLM
installation. Numeric task controls remain at
`artifacts/native_visual_transfer_v71_20261105/attempt01/structural_vectors.pt`.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. \
python scripts/infer_semantic_glyph_language.py \
  --prompt-image question.png --output artifacts/my_glyph_answer \
  --expected-checkpoint-sha256 3a81b4e6f4306b218d08e6988f84d688f4267a773356a8d4c9c470e496ae5141
```

`answer.png` contains the actual generated writing strip. `answer.pt` retains
physical glyph tensors, decision states, stop probabilities and feedback
hashes. `observed_prompt.png` records host preparation: large pages shrink
to a one-megapixel budget and white padding aligns dimensions to 32 pixels.
No text is inferred during that preparation. An empty stream has an empty
tensor and a white PNG placeholder. A cap is recorded as a failure, never as
normal termination. The default cap is 64 glyphs and can be set with
`--maximum-glyphs`; no desired answer length is supplied.

Inference does not require the training corpus, prepared glyph inventory or
font installation. It loads only the neural checkpoint, numeric controls,
local pretrained body and architecture code. The model checkpoint records a
completed fit, not general language qualification. New arbitrary questions
can produce malformed repetitions; inspect raw images rather than treating
the file's successful creation as a correct answer.

A [cold CLI qualification](../references/evidence/semantic_glyph_cli_cold/verification.json)
repeats the changed-2019 question image with bit-exact complete pixels,
stop probabilities and feedback hashes. Its system-call trace opens no
training-corpus, glyph-inventory or font files and makes no IP socket
connections. The raw trace stays local; its hash and source-bound receipt
are public. This qualifies the tested independent input path, not arbitrary
question accuracy.

## Train and inspect the complete experiment

CPU preparation uses fonts and source embeddings as offline supervision.
The first glyph stage selects training-only anchors, merges visually identical
forms, and learns a continuous encoder/decoder. The second stage caches
causal native states using gold previous glyph images, then fits only the
continuous readout and stopping. Every training and assessment stage runs
sequentially under the same project GPU lock.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/train_semantic_glyph_bridge.py \
  --stage prepare --output artifacts/my_glyph_primitive
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
python scripts/train_semantic_glyph_bridge.py \
  --stage train --output artifacts/my_glyph_primitive
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/train_semantic_glyph_language.py \
  --stage prepare --primitive artifacts/my_glyph_primitive --output artifacts/my_glyph_language
```

Then run `scripts/train_semantic_glyph_language.py` sequentially with
`--stage cache`, `--stage fit` and `--stage assessment`, using the deterministic
GPU environment from the inference command. Preserve failed stages and choose
a new output directory for a new experiment. Review all raw outputs with
`scripts/review_semantic_glyph_language.py --stage review`, record actual
output-first inspection in its `raw_review/manual_review.json`, and only then
run `--stage score`. The evaluator's canonical images never enter inference.

The frozen [primitive](../references/semantic_glyph_loop_protocol.md) and
[language](../references/semantic_glyph_language_protocol.md) protocols define
actual counts, reconstruction gates, losses, admission and stopping. They
distinguish clear familiar glyphs, complete training fit and unseen-answer
quality. The [independent feedback audit](../references/evidence/semantic_glyph_feedback_audit/report.json)
verifies all 2,781 emitted glyph embeddings across the completed seventy-two
assessment streams; it does not assign semantic grades.
