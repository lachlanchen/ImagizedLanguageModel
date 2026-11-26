# Writing after a mistaken glyph

These are two real 1,688,480-parameter bilingual translation checkpoints trained on one RTX 4090 D. Both accept a grid of printed 32 × 32 glyphs and draw the answer by choosing from the learned model's bitmap inventory. The reference fit observes aligned corpus prefixes. The recovery fit observes the starting writer's own mistaken prefixes during training.

From the repository root, translate this original image on CPU. Use a new output directory for each request.

```bash
python -m scripts.write_ordered_translation --model prefix-recovery \
  --image publication/ilm-nature/figures/prefix_recovery/case-145-input.png \
  --cells 23 --to zh --output /tmp/ilm-held-passive
```

The answer is in `answer.png`. The receipt contains the occupied glyph count, stop flag and packed pixel digest. On this held passive construction the model writes **玛丽帮助了汤姆。** The source is “Tom was helped by Mary.” Supply `--text` in place of the PNG and cell count to render a typed sentence through the same optical path. Select `prefix-reference` to run the matched control. Source images must follow the repository's glyph-grid layout and contain at most 96 occupied cells. Novel input can be presented optically. Output is limited to the current bitmap repertoire.

The recovery writer completed 111/118 controlled role families in the original font and with a trained reader on Noto Serif. It scored 0/144 on fresh literary, talk and encyclopaedic translation requests. Quantized text Qwen3-8B scored 96/144. No outside language model runs at inference. [Evaluation and training procedure](../../references/visual_prefix_recovery_result.md) · [Original input and output figures](../../publication/ilm-nature/figures/prefix_recovery)
