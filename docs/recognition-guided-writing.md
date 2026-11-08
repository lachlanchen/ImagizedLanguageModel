# Recognition supervision for an image writer

This experiment adds a frozen, training-only OCR loss to the existing paired
FLUX image generator. It tests exact copying before attempting semantic QA.
Neither a passing gradient check nor lower training loss qualifies a useful
ILM. The [prospective protocol](../references/recognition_guided_writing_protocol.md)
and [resize correction](../references/recognition_resize_correction.md) retain
the admission criteria and the first failed calibration. The
[completed 512-update comparison](../references/recognition_guided_writing_result.md)
still fails: 6/32 seen and 9/32 held-out OCR-exact copies, with visible content
errors. This recipe is not qualified for use or scale-up.

The student receives an input image, noisy answer image and the same generic
numeric condition for every example. The frozen reader scores the student's
decoded predicted-clean pixels against the training string and EOS. Gradients
pass through the reader and VAE to the student's adapters. Inference uses the
existing image-only runner: the reader, its tokenizer and target strings are
absent. This is externally pretrained model adaptation, not a new foundation
model trained from scratch or evidence of matched-quality efficiency.

Reuse the verified shared environment and source components. The commands
below name the completed paired-copy data layout, not a hidden answer lookup
used during inference. Source acquisition requires network access; admitted
model execution is offline and locks GPU0.

```bash
ILM_PYTHON=../LocalLLM/.local/image-generation/venv/bin/python
ILM_READER_CACHE=artifacts/external/writing_reader/hub
ILM_READER=$ILM_READER_CACHE/models--stepfun-ai--GOT-OCR-2.0-hf/snapshots/d3017ef2c2c1395888c8d635c5e0508bcb0ac78d
ILM_BASE=artifacts/external/image_edit_learning/hub/models--black-forest-labs--FLUX.2-klein-base-4B/snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d
ILM_CONDITION=artifacts/image_edit_baseline/condition/condition.pt
ILM_PAGES=artifacts/image_edit_learning/strip_pages/pairs.jsonl
ILM_PAIRS=artifacts/image_edit_learning/padding_trial_v1/cache/pairs
ILM_CONTROL=artifacts/image_edit_learning/padding_trial_v1/writing/updates.jsonl
ILM_RUN=artifacts/my_recognition_writer
mkdir -p "$ILM_RUN"

env PYTHONPATH=. "$ILM_PYTHON" scripts/calibrate_writing_reader.py --stage acquire --cache "$ILM_READER_CACHE"
```

Use the project's existing NVIDIA compatibility wrapper when the kernel and
user-space driver need it. Do not modify system drivers for this experiment.
Run these model stages sequentially; every stage requires a fresh output
directory. The trainer rejects an admission made with different code or pages.

```bash
ilm_gpu=("$HOME/.local/bin/with-current-nvidia" env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4 PYTHONPATH=.)

"${ilm_gpu[@]}" "$ILM_PYTHON" scripts/calibrate_writing_reader.py \
  --stage calibrate --snapshot "$ILM_READER" --manifest "$ILM_PAGES" \
  --output "$ILM_RUN/reader"

ilm_training=(--snapshot "$ILM_BASE" --condition "$ILM_CONDITION" \
  --teacher "$ILM_READER" --admission "$ILM_RUN/reader/report.json" \
  --manifest "$ILM_PAGES" --cache "$ILM_PAIRS" --control "$ILM_CONTROL")

"${ilm_gpu[@]}" "$ILM_PYTHON" scripts/train_recognition_guided_writer.py \
  "${ilm_training[@]}" --stage preflight --output "$ILM_RUN/preflight"
"${ilm_gpu[@]}" "$ILM_PYTHON" scripts/train_recognition_guided_writer.py \
  "${ilm_training[@]}" --stage train --output "$ILM_RUN/training"

"${ilm_gpu[@]}" "$ILM_PYTHON" scripts/infer_paired_image_cohort.py \
  --snapshot "$ILM_BASE" --condition "$ILM_CONDITION" \
  --checkpoint "$ILM_RUN/training/checkpoint.pt" \
  --inputs artifacts/image_edit_learning/full_strip_inputs.json \
  --steps 50 --guidance 1 --output "$ILM_RUN/inference"
```

The copying data preparation, numeric VAE cache and fixed-condition compiler
are documented in [paired image learning](paired-image-learning.md). A new
corpus needs explicit splits and separate evaluation; this controlled runner
intentionally fixes the existing first-32 cohort and 512-update budget.
To resume an interrupted training invocation, use `--checkpoint` with the
same implementation, input bindings and budget in a new output directory.
It restores optimizer and RNG state; it does not extend a completed run.

Evaluate every generated PNG separately using `scripts/score_paired_writing.py`
and its reference calibration for the same manifest. Keep the raw transcripts
and inspect the images. Use the entire cohort, not a few successful samples.
The training reader is not the independent CPU evaluator. Do not repair an
answer with OCR, a font renderer or the reference text before scoring it.
