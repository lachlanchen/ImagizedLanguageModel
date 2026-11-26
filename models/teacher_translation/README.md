# Visual English–Chinese translation checkpoints

These 1,688,480-parameter writers read a grid of 32 × 32 glyph images, predict a sequence of bitmaps and choose when to stop. The corpus and distilled checkpoints were trained on identical pictured sources. The distilled checkpoint learned from screened complete translations by Qwen3-8B. Inference runs locally on CPU or GPU with the repository's shared optical inventory. The output repertoire is fixed.

Run the illustrated held Chinese sentence from the repository root with the project dependencies, including PyTorch and safetensors. Choose a fresh output directory.

~~~bash
python -m scripts.write_ordered_translation --model teacher-distilled \
  --image publication/ilm-nature/figures/teacher_translation/request-38-input.png \
  --cells 14 --to en --output /tmp/ilm-looking
~~~

The original question reads **她看了他一眼，却什么也没说。** The writer draws **“She looked at him and said nothing.”** in `answer.png`. The receipt reports 35 emitted cells and a stop. Pass `--model teacher-corpus` for the matched control, or replace the input image and count with `--text` followed by a quoted sentence to render a typed input. Both directions accept up to 96 occupied source cells in the canonical grid.

Across the complete 96-request panel, a masked source-only model reading credited three distilled answers, one corpus answer and 88 from the quantized text teacher. The registered reference-informed assessment credited zero, one and 58. These are experimental checkpoints for studying transfer. [Full result and limitations](../../references/complete_teacher_translation_result.md) · [Actual answer images](../../publication/ilm-nature/figures/teacher_translation)
