# Optical writers trained on paired translations

The two 1,688,480-parameter checkpoints read PNG glyph cells and write a sequence of physical glyph bitmaps. They differ in whether the training loss rewards assigning two complete translations to their sources. Both reuse the visual repertoire and geometry of the ordered translator. The experimental models scored 75/96 correct assignments and 0/192 freely written meanings on the held bilingual panel. [Matched results](../../references/paired_ink_result.md)

To reproduce the Chinese answer from the pictured English scientific sentence, run from the repository root with a new output directory.

```bash
python -m scripts.write_ordered_translation --model paired-assignment --to zh \
  --image publication/ilm-nature/figures/raw_ilm/natural-zh-input.png --cells 93 \
  --output /tmp/ilm-paired-zh
```

The resulting `answer.png` contains the original 24-cell reply. Its hash and stop status are recorded in `receipt.json`. The answer is legible and loses the source's scientific meaning. Use `--model paired-likelihood` for the matched control. The optional `--decode order` reproduces the source-order inference experiment. Both CPU and CUDA inference use the released weights locally. [Source and emitted PNGs](../../publication/ilm-nature/figures/raw_ilm/) · [Controlled role-reversal writer](../translation_paraphrases/)
