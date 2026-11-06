# Learning from prompt and answer images

`scripts/train_paired_image_language.py` implements supervised image-to-image
learning with the undistilled FLUX.2-klein-base-4B generator. The model reads
question content through its native image branch and generates answer pixels.
Every example receives the same generic condition compiled offline. Inference
loads no tokenizer, text encoder, training corpus, answer cache or OCR module.
The complete pretrained generator and image codec remain necessary; locally
trainable adapters do not make their parameter counts disappear.

This implementation is a learning experiment. The earlier zero-shot baseline
failed all requested answers. No useful ILM, quality improvement or training
efficiency is established by adding a trainer or executing two updates. See
the [declared protocol](../references/paired_image_learning_protocol.md).

Use a Python environment with the installed compatible PyTorch/Diffusers
versions. Reuse verified source weights and the existing constant condition:

```bash
export ILM_IMAGE_PYTHON=../LocalLLM/.local/image-generation/venv/bin/python
export ILM_BASE_SNAPSHOT=artifacts/external/image_edit_learning/hub/models--black-forest-labs--FLUX.2-klein-base-4B/snapshots/a3b4f4849157f664bdbc776fd7453c2783562f4d
export ILM_FIXED_CONDITION=artifacts/image_edit_baseline/condition/condition.pt
```

The source snapshot must contain the pinned model index, transformer, VAE
and scheduler files; hashes are checked before model loading. If acquiring
it elsewhere, download only these components from the declared revision,
reuse a verified shared installation, and omit text-encoder/tokenizer weights.
Compile that constant once from the verified shared encoder if it is absent:

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. "$ILM_IMAGE_PYTHON" scripts/prepare_paired_image_condition.py --conditioner ../LocalLLM/.local/models/image-generation/z-image-turbo-f332072a --output artifacts/paired_condition
export ILM_FIXED_CONDITION=artifacts/paired_condition/condition.pt
```

This standalone compiler needs no earlier experimental checkpoint or training
run. It reuses the verified generic-condition implementation and encoder
weights, runs offline once, and exits before the generator starts. Its external
language pretraining is disclosed. Reuse an existing verified constant instead
of recompiling it for every question.

Preparation needs the existing `data/visual_qa/cmrc2018_short/train.jsonl`
and an installed Noto Sans CJK SC font. It renders complete passages without
an answer-aware crop. Train/validation QA articles are disjoint; the public
validation split is exploratory and held out from this optimizer.

```bash
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. "$ILM_IMAGE_PYTHON" scripts/train_paired_image_language.py --stage prepare --output artifacts/paired_pages
env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. "$ILM_IMAGE_PYTHON" scripts/train_paired_image_language.py --stage cache --snapshot "$ILM_BASE_SNAPSHOT" --condition "$ILM_FIXED_CONDITION" --manifest artifacts/paired_pages/pairs.jsonl --task copy --train-limit 32 --output artifacts/paired_cache
env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. "$ILM_IMAGE_PYTHON" scripts/train_paired_image_language.py --stage preflight --snapshot "$ILM_BASE_SNAPSHOT" --condition "$ILM_FIXED_CONDITION" --cache artifacts/paired_cache/pairs --task copy --train-limit 32 --output artifacts/paired_preflight
env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. "$ILM_IMAGE_PYTHON" scripts/train_paired_image_language.py --stage train --snapshot "$ILM_BASE_SNAPSHOT" --condition "$ILM_FIXED_CONDITION" --cache artifacts/paired_cache/pairs --task copy --train-limit 32 --updates 512 --output artifacts/paired_writing
```

Every invocation uses a fresh output directory and the project GPU lock.
Caching loads only the VAE, stores numeric image pairs, and retains every
target reconstruction and its pixel score at the declared answer geometry.
The preflight verifies that zero-initialized adapters preserve the actual
native prediction, then performs two real optimizer updates. It records
gradients, changed tensors, memory, time and the checkpoint. These are
execution observations, not language measurements.

Training uses uniform noise time, native velocity MSE, rank-16 adapters and
FP32 AdamW state with BF16 forward computation and gradient checkpointing.
The journal records each selected training ID and loss. Checkpoints include
optimizer and RNG state. Resume an interrupted run with `--checkpoint` and
the same arguments in a new output directory; completed budgets are rejected
instead of silently trained again. No earlier experimental checkpoint chain
is required.

Before scoring generated writing, calibrate the separate CPU evaluator on
every reference in the writing cohort:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. OMP_THREAD_LIMIT=1 "$ILM_IMAGE_PYTHON" scripts/score_paired_writing.py --stage calibrate --manifest artifacts/paired_pages/pairs.jsonl --output artifacts/paired_reference_ocr
```

For scoring, supply `--stage score --calibration .../report.json
--generations outputs.json` with a JSON list of `{id, path, sha256}` for the
actual output PNGs. Paths are relative to that index. The evaluator preserves
raw transcripts, expected strings and uncertain readings; it does not repair
the generated images. An eight-case display is explicitly distinguished from
the complete 64-case writing cohort. OCR scores are observations, not a
substitute for reviewing the actual pixels.

Deploy a resulting checkpoint with only the input image:

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. "$ILM_IMAGE_PYTHON" scripts/train_paired_image_language.py --stage infer --snapshot "$ILM_BASE_SNAPSHOT" --condition "$ILM_FIXED_CONDITION" --checkpoint artifacts/paired_writing/checkpoint.pt --input question.png --width 512 --height 256 --steps 50 --output artifacts/paired_answer
```

Omit `--checkpoint` to measure the same base generator before learning.
The input page must have dimensions divisible by 16 and at most 1,048,576
pixels. Declare the answer canvas independently of its expected content.
Guidance defaults to one, matching the conditional field used by this
training objective. The source base scheduler still performs 50 steps.
Both the actual postprocessed pixel tensor and an unrepaired answer PNG are
saved. Copying validation, natural QA, question-erasure controls and broader
language/length/style tests remain separate requirements. Use offline OCR
only after reference calibration and inspect the actual answer images.
