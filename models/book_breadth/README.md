# English–Chinese visual writers trained on book passages and talks

Both released checkpoints take a pictured sequence of up to 96 occupied 32 × 32 glyph cells. They emit a stopped sequence of physical bitmaps from a fixed training repertoire. The broad model saw 231,337 distinct translation pairs per pass, the narrow model 31,918. The corresponding held meaning scores were 5/144 and 0/144. The broader model also lost a previously learned passive construction. [Evaluation and limitations](../../references/book_breadth_result.md)

From the repository root, the following reproduces the *Jane Eyre* reporting clause from the original English input PNG. Use a new output directory.

```bash
python -m scripts.write_ordered_translation --model book-broad --to zh \
  --image publication/ilm-nature/figures/book_breadth/accepted-47-input.png \
  --cells 9 --output /tmp/ilm-book-reporting-clause
```

The output is **他问道。** The four-cell PNG and its stop receipt are in the output directory. The narrow checkpoint uses `--model book-narrow`. Use `--device cuda` for local CUDA or omit it for CPU. The paper preserves the whole original images, their line wrapping and their blank cells. [Recorded output](../../publication/ilm-nature/figures/book_breadth/accepted-47-broad.png) · [Full raw examples](../../publication/ilm-nature/figures/book_breadth/)

## Read an unfamiliar typeface

The rehearsal checkpoint reads **Tom was helped by Mary.** and emits **玛丽帮助了汤姆。** from the following source PNG. The optional optical reader enables this example in Noto Serif. It uses only observed pixels and stored weights during inference.

```bash
python -m scripts.write_ordered_translation --model book-rehearsal --to zh \
  --image publication/ilm-nature/figures/book_rehearsal/serif-input.png \
  --cells 23 --typeface-reader --output /tmp/ilm-serif-answer
```

The command writes the original grid to `answer.png` and its cell count, stop and packed-pixel SHA-256 to `receipt.json`. Omit the adapter for the original writer. The separate `--guard-ink` option corrects small bitmap perturbations using a certified separation rule. The language checkpoint completed 110/118 role families and scored 0/96 on fresh natural requests. [Pictured input](../../publication/ilm-nature/figures/book_rehearsal/serif-input.png) · [Expected output PNG](../../publication/ilm-nature/figures/book_rehearsal/serif_learned-output.png) · [Measured scope and training](../../references/book_rehearsal_result.md)
