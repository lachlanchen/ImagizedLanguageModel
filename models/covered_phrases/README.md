# Translate pictured English or Chinese by source phrase

This local CPU writer reads a sentence image and writes an image by joining the translations of nonoverlapping source spans. The print face and 32×32 cell grid must match the training repertoire. Each translation uses the learned phrase tables, a smoothed target bigram model and the earlier optical inventory. No external model is called.

```sh
PYTHONPATH=. python -m scripts.write_covered_phrases \
  --text 'His Sundays were different.' --to zh --output sundays.png
```

Use `--image input.png --cells N` in place of `--text` for an image with 1–96 occupied cells. The example produces 他的周日被不同。 The passive marker makes the sentence ungrammatical. On 96 new book requests the writer reproduced 445 distinct reference strips and earned no whole-translation credits. It is useful for inspecting which visual source phrases contributed to a draft.

The [original image archive](../../publication/ilm-nature/evidence/covered_phrases.zip) includes `paths.json` with the half-open source span, emitted target strips and training observation count for every phrase. To reproduce training from the original corpora, run `scripts.prepare_covered_phrases`, `scripts.align_covered_phrases`, `scripts.fit_covered_phrases` and `scripts.write_covered_phrases --calibrate` as Python modules from the repository root. The alignment stage uses `scripts/with_ilm_gpu`. [Study, measurements and inference limitations](../../references/covered_phrases_result.md)
