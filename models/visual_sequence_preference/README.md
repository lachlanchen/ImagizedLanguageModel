# Writing with whole-answer preferences

This English–Chinese writer accepts an image of 1–96 aligned printed glyphs and returns a generated PNG. It reads overlapping glyph windows in the source, constructs a phrase draft, then generates or copies one glyph at a time. The learned stop ends the answer. Text input uses the same printed input interface after rendering. The local Qwen models were used to select training comparisons and assess quality. The deployed model runs independently.

```sh
PYTHONPATH=. python -m scripts.write_visual_sequence_preference --text '对我来说也是如此。' --to en --output answer.png
PYTHONPATH=. python -m scripts.write_visual_sequence_preference --image source.png --cells 9 --to en --output answer.png
```

Supply the occupied cell count for a PNG. The reader accepts black ink on white in 32×32 cells, laid out left to right in rows of up to 32 columns. Each emitted glyph comes from the learned target print inventory. The source marker, palette, optical and phrase tables are shared with the parent draft editor. Use `--arm likelihood` for the compute-matched control. The numerical deployment payload is approximately 25.37 MB before interpreter, libraries, verification metadata and optional fonts.

To reproduce fitting and evaluation with the underlying local corpora and quantized grading models available:

```sh
PYTHONPATH=. scripts/with_ilm_gpu python -m scripts.prepare_visual_sequence_preference
PYTHONPATH=. scripts/with_ilm_gpu python -m scripts.confirm_visual_sequence_preferences
PYTHONPATH=. scripts/with_ilm_gpu python -m scripts.train_visual_sequence_preference
PYTHONPATH=. scripts/with_ilm_gpu python -m scripts.evaluate_visual_sequence_preference
PYTHONPATH=. python -m scripts.audit_visual_sequence_preference
```

The releases contain the final weights for both arms. Completed training fits are identified by their recorded bindings. [Measured capabilities, original images and limitations](../../references/visual_sequence_preference_result.md)
