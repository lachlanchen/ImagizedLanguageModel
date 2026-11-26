# Wider visual translator

The checkpoint has 14,895,840 parameters. It reads aligned 32 × 32 bitmap glyph cells, attends to ordered windows of four source glyphs and autoregressively selects target bitmaps. Source images can contain unfamiliar glyphs. The output alphabet is the repository's fixed bilingual bitmap inventory. A separate learned stop action ends generation. At inference the checkpoint requires neither OCR nor a text-language-model service.

From the repository root, reproduce a held passive construction on CPU.

```bash
python -m scripts.write_ordered_translation --model wide \
  --image publication/ilm-nature/figures/prefix_recovery/case-145-input.png \
  --cells 23 --to zh --output /tmp/ilm-wide-passive
```

The expected output is **玛丽帮助了汤姆。** The output directory contains the generated PNG and stop receipt. For a different request, use `--text` instead of `--image` and `--cells`, or supply a grid saved by the same PNG writer. Inputs support 1–96 cells. A full book page needs separate segmentation.

On 64 new article requests the model earned **zero** whole-meaning credits, compared with **49** for quantized Qwen3-8B. It completed **100/118** controlled role families. [Measured results and actual outputs](../../references/visual_writer_width_result.md)

Training starts from the released optical rehearsal writer with function-preserving expansion from four to twelve attention heads. Zero-sum perturbations of copied columns allow head replicas to learn differently. Three passes on the same 231,337-pair corpus as the narrow control scored 49,519,224 target positions in each fit. These commands use the previously prepared book, talk and prefix-recovery data in the project workspace.

```bash
python -m scripts.reserve_wide_translation
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_wide_translation
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_wide_translation
```

[Predeclared comparison](../../references/visual_writer_width_protocol.md) · [Training receipt](fit.json) · [Complete language scores](../../references/evidence/wide_translation_v1/results.json)
