# Writing with pictured source content

A frozen bilingual image writer reads a grid of 32 × 32 glyphs. A trainable head estimates which visual strips will occur in the translation. A predicted strip can receive a bonus on its first emission. A finite neural motor draws the answer. The reader, content predictor and writer run locally on CPU or GPU.

From the repository root, this command takes a PNG with 24 occupied cells in reading order and writes a Chinese answer. Include visible spaces in the count. The supported source length is 1–96 cells.

```sh
PYTHONPATH=. python -m scripts.write_source_supported \
  --input source.png --cells 24 --to zh --output answer.png
```

The two 1,557,508-parameter auxiliary heads were trained using exactly the same 268,342 pairs, true target sets and batches. The aligned head received its actual source sentence. The shuffled control received a different source from the same physical length bucket whenever the batch contained multiple examples. The parent is the released 3,196,224-parameter optical-strip writer. Its original weights and frozen glyph motor are shared. Use `--arm parent` or `--arm shuffled` for the controls.

With the verified visual-action corpora in place, reserve the fresh book panel, fit the matched heads and evaluate the outputs with these commands.

```sh
export PYTHONPATH=.
python -m scripts.reserve_source_supported_writing
python -m scripts.train_source_supported_writing
python -m scripts.evaluate_source_supported_writing generate
python -m scripts.evaluate_source_supported_writing replay
```

[Results and scope](../../references/source_supported_writing_result.md) · [Original book images](../../publication/ilm-nature/evidence/source_supported_writing.zip)

### Choose between available translation images

The single-source selector takes a pictured source and candidate translation PNGs. It scores each complete candidate twice, on the current source and against training-source images selected by length. It copies the chosen PNG and reports its zero-based index, conditional score, background score and difference. This mode requires supplied answers in the writer's print repertoire. It can also compare independently generated candidates.

```sh
PYTHONPATH=. python -m scripts.select_visual_translation \
  --source source.png --cells 24 --to zh \
  --candidate candidate-a.png 12 --candidate candidate-b.png 15 \
  --output chosen.png
```

The numeric visual-merge rules and background image addresses are included in the selection directory. Running the selector requires no external language model or training corpus.
