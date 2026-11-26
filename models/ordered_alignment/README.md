# Translate a pictured sentence with learned source order

The writer reads an aligned English or Chinese sentence image, follows learned source positions, and emits an answer image from a fixed print repertoire. It uses the local visual writer, bilingual glyph-strip probabilities and the direction-specific weights in this directory.

```sh
PYTHONPATH=. python -m scripts.write_ordered_alignment \
  --text 'It was the student whom David did find.' --to zh --output answer.png
```

For an image, replace the text argument with `--image source.png --cells N`. Use 1–96 occupied 32×32 printed cells in row-major order. Inference runs locally on CPU. The example produced 大卫找到了学生。 on the controlled test. On fresh book passages the model stopped on all 96 requests and received zero complete-meaning credits.

The training entry point is `scripts/with_ilm_gpu python -m scripts.fit_ordered_alignment` after `python -m scripts.prepare_ordered_alignment` has prepared the registered corpus. Run from the repository root with `PYTHONPATH=.`. Reproducing the fit requires the original book and corpus assets. The trained writer only needs its shipped model dependencies. [Protocol and results](../../references/ordered_alignment_result.md) · [Original images](../../publication/ilm-nature/evidence/ordered_alignment.zip)
