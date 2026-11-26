# Bilingual visual-strip restoration

The writer takes one to 96 ordered, aligned 32 × 32 glyph images and returns a PNG. It predicts the number of output strips directly from the source. Six passes fill a blank canvas, retaining more high-confidence strips at each pass. Each strip is a learned group of existing glyph bitmaps. The frozen motor emits ink from its finite repertoire. The free writer requires a target language and the occupied source cell count.

```sh
PYTHONPATH=. python -m scripts.write_strip_restoration \
  --input pictured-source.png --cells 18 --to zh --arm restore --output answer.png
```

Choose `mask` for the matched masking-only writer or `parent` for the earlier causal writer. CPU is the default. Use `--device cuda` for GPU inference. Source grids can wrap after 32 occupied cells. English whitespace occupies a cell. Input must use the grid geometry above.

Both bodies start from the released independent strip writer and fit the same 268,342 bilingual pairs for four additional epochs. Reproducing the training and evaluations requires the original local book and OPUS artifacts. The training set, ordered schedule, initial weights, programs and result weights are fingerprinted in each fit receipt. Both deployed visual writers use the shared strip table and verified neural glyph motor. The local language-model judge operates only during evaluation.

```sh
export PYTHONPATH=artifacts/library_teacher_v1/deps:.
python -m scripts.reserve_visual_strip_restoration
python -m scripts.train_visual_strip_restoration
python -m scripts.evaluate_strip_restoration generate
python -m scripts.evaluate_strip_restoration assess
python -m scripts.evaluate_strip_restoration replay
python -m scripts.measure_strip_assembly
python -m scripts.measure_visual_strip_gaps
python -m scripts.diagnose_strip_proposals
python -m scripts.time_strip_restoration
```

The evaluation archives include the original generated answer PNGs. The input exchange, supplied alternatives, supplied strip inventory and partly obscured target each test a different source of information. [Measured capabilities and limitations](../../references/visual_strip_restoration_result.md)

## Choose the order of a pictured answer

The supplied-choice reader accepts a source image and two or more pictured answers built from the same visual-strip inventory. One evaluation of the source and a blank target canvas provides position scores for every alternative. The command returns a byte-for-byte copy of the selected candidate. In this archived example the input says “Mary helps John.” The candidates have the same Chinese glyphs in different orders.

```sh
fig=publication/ilm-nature/figures/strip_restoration
PYTHONPATH=. python -m scripts.read_pictured_order \
  --source "$fig/Supplied_strip_restoration-source-roles-024-1.png" --cells 16 --to zh \
  --candidate "$fig/Supplied_strip_restoration-restore-roles-025-1-scrambled.png" 7 \
  --candidate "$fig/Supplied_strip_restoration-restore-roles-024-1-scrambled.png" 7 \
  --output chosen.png
```

The released reader chooses candidate two. Reversing the presentation order preserves the selected pixels. Free translation uses the separate writer command above.
