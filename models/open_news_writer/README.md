# Open source-glyph news writers

The image interface accepts 1–96 aligned 32×32 source cells. Previously unseen cell bitmaps go directly into the existing visual encoder through their 16×16 pooled ink. Phrase hints come from contiguous runs of recognised cells. The learned English motor uses a finite printed repertoire. The two new checkpoints are trained for Chinese-to-English news translation. Their translator runs locally using the earlier phrase tables and optical model dependencies.

```sh
PYTHONPATH=. python -m scripts.write_open_source_cells --image source.png --cells 18 --to en --arm ink --output answer.png
PYTHONPATH=. python -m scripts.write_open_source_cells --text '委员们正在讨论这个问题。' --to en --arm hidden --output control.png
```

The ink and hidden arms each update 1,817,441 parameters. They start from the same broad visual writer and receive the same 1,368 professionally reviewed training sentences, schedule and earlier-corpus rehearsal. The hidden arm sees the mean source-repertoire observation at unfamiliar positions. The `parent` arm runs the original weights with the wider input interface. Each arm writes its actual selected glyphs and returns its learned stopping decision. Known glyphs include exact print shapes, so a new font or scan changes the input distribution.

Training and assessment use the original local ILM phrase tables and teacher installations. Restore the four professional rating files, prepare the first document split, reserve the external year, then run the matched fit and free-writing evaluation. The professional annotation preparation and external reservation retrieve missing pinned files. The calibration selection prepares the WMT 2020 exclusion file.

```sh
export PYTHONPATH=artifacts/library_teacher_v1/deps:.
python -m scripts.calibrate_translation_graders prepare
python -m scripts.prepare_open_news_training
python -m scripts.reserve_open_news_external
scripts/with_ilm_gpu python -m scripts.train_open_news_writer
scripts/with_ilm_gpu python -m scripts.evaluate_open_news_writer
```

[Measured transfer and original outputs](../../references/reviewed_news_writer_result.md) · [External evaluation protocol](../../references/open_news_external_training_protocol.md) · [Fit schedule and objectives](../../references/open_news_training_protocol.md)
