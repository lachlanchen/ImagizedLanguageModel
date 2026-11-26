# Translating a pictured sentence into glyph pixels

The guided checkpoint writes five glyphs, 当然不是。, for the archived image of “Certainly not.” The following CPU command recreates the published output. Run it from the repository root with Python, PyTorch, NumPy, Pillow and safetensors installed.

```sh
python -m scripts.write_neural_glyphs \
  --target-training guided \
  --input publication/ilm-nature/figures/target_glyph_teaching/0049-input.png \
  --cells 14 --to zh --output answer.png
cmp answer.png publication/ilm-nature/figures/target_glyph_teaching/0049-guided.png
```

Inputs are aligned grids of 32 × 32 cells in reading order, with 1–96 occupied cells including spaces. Select the output language with `--to en` or `--to zh`. Use `--target-training likelihood` for the matched control. Both released checkpoints depend on the tracked prediction-field writer and its visual readers and motor. The loader verifies their hashes. The neural motor computes the output pixels from a selected visual code and feeds the result back. Its learned drawing vocabulary covers 3,717 shapes. Clean individual characters are common even when the sentence is wrong.

Both fits used 28,342 bilingual passage pairs in the same order for four passes. Only the guided writer used the offline text teacher probabilities. On 192 fresh requests it earned five whole-meaning credits. The matched writer earned four and a quantized text comparator earned 173. All guided images reproduced on CPU. Across all four conditions one control diverges between GPU and CPU, so the full replay command records the difference and exits nonzero. The archive retains the original graded image. [Training and uncertainty](../../references/target_glyph_teaching_result.md) · [All held images](../../publication/ilm-nature/evidence/target_glyph_answers.zip)

```sh
python -m scripts.replay_target_glyph_output
```

For the source collection, frozen parent weights, teacher revisions, split exclusions, fit order and assessment, see the [training protocol](../../references/target_glyph_teaching_protocol.md), [teacher cache receipts](../../references/evidence/target_glyph_teaching_v1/targets.json) and [training code](../../scripts/train_target_glyph_teaching.py).
