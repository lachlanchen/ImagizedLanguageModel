# Use a complete pixel-language checkpoint

`scripts/infer_complete_pixel_language.py` accepts typed prompts, a UTF-8 file,
an image page, or an image with a visible instruction. Typed input is rendered
outside the neural model. The actor receives physical pixels and generates
binary answer-image blocks with its own image history and learned stopping.
Its output is saved directly as tensors and PNGs, with no answer font renderer,
OCR service or reference lookup.

The model is a research checkpoint. At update 2,000,
[29 of 32 seen training answers are pixel-exact](complete_pixel_language_update2000.md);
unseen-prompt capability is not yet qualified.
Standalone GPU replay has not yet been qualified. Input preparation and
checkpoint-schema checks alone do not establish inference correctness; that
requires restoring a retained model and replaying its original PNG inputs and
complete emitted streams after GPU0 is free.

The [CPU interface receipt](evidence/complete_pixel_inference_cpu/receipt.json)
records image, typed and mixed-input preparation, exact preservation of the
owning image pixels, the actual checkpoint's full parameter schema and a
lossless page-formatting check on a previously generated answer. These checks
initialize no CUDA context and perform no new neural generation.

Prepare input without loading any model or using CUDA:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python \
  scripts/infer_complete_pixel_language.py \
  --text '资料：北京是中国的首都。问题：中国的首都是哪里？回答：' \
  --prepare-only --output artifacts/my_pixel_prompt
```

After the training process finishes, run the completed checkpoint on an image:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python \
  scripts/infer_complete_pixel_language.py \
  --training-run artifacts/complete_pixel_language_v1 \
  --prompt path/to/question.png --output artifacts/my_pixel_answer
```

Use `--text-file question.txt` for a typed prompt file. Add
`--instruction '请根据这页内容回答问题。'` to `--prompt page.png` for a visible
instruction above unchanged page pixels. Image page layout is preserved;
`--layout strip` explicitly folds
a binary horizontal strip at white columns. Missing font glyphs in typed input
raise an error; an image can carry forms outside the host font inventory.

The default checkpoint is the run's current completed checkpoint. Inspect an
unfinished research milestone only with an explicit override:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python \
  scripts/infer_complete_pixel_language.py \
  --training-run artifacts/complete_pixel_language_v1 \
  --checkpoint artifacts/complete_pixel_language_v1/milestone_models/actor_update02000.pt \
  --allow-incomplete --prompt path/to/question.png \
  --output artifacts/my_pixel_milestone_answer
```

Inspect a completed training milestone without loading another model:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python \
  scripts/inspect_complete_pixel_milestone.py \
  --run artifacts/complete_pixel_language_v1 \
  --stage train_update_02000 --output artifacts/my_milestone_review
```

The inspection requires the producer's complete 32-case report, validates raw
streams and stop decisions, independently scores full images and writes four
plates without resizing the original pixels. It does not recompute the neural
encoder. The terminal audit performs that separate check after model exit.

Generation obeys the single project GPU0 lock and workstation admission rules.
Do not start another model while training owns that lock. Input preparation is
available during training. The loader restores the actor's own state from the
authenticated frozen body, history codec and numeric controls; the earlier
native training checkpoint and cached first-block raster checkpoint are not
deployment dependencies. The required shared source weights remain substantial,
and this model is not a lightweight foundation trained from scratch.

`output/result.pt` retains the exact stream and stop trace. Nonempty answers
also have `output/answer.png`. Where wholly white columns permit safe wrapping,
`output/answer_page.png` provides a reversible page view; its source ranges are
recorded in `output/output_receipt.json`. No pixels are resized or repaired.
If a stream has no safe break, the original strip remains available. Empty
answers receive no fabricated placeholder image. A safety-cap event is recorded
as a failed termination, regardless of visual appearance.
