# Bilingual recurrent glyph writer

This model reads a pictured question and a short pictured passage. It can revisit the passage after a written phrase or emit one glyph at a time. The directory contains the dynamic and fixed optical heads and a byte-input experimental control. The optical command reuses the released shared book model. A real CPU-executed question on a museum page asks for two bronze counts; the writer returns 十七件 and misses the second count. [Original replies](examples/dynamic) · [Measured results](../../references/recurrent_evidence_result.md)

From the repository root, run the writer in the project's Python environment. Text inputs are rasterized before the model sees them.

```bash
python -m scripts.write_recurrent_evidence \
  --question '化石发现在哪里？' --passage '这块化石发现于尼日。' \
  --output /tmp/ilm-fossil-reply
```

For aligned PNGs use `--question-image`, `--passage-image` and their cell counts. One cell is a 32-by-32 monochrome image. Questions support 1–160 cells and evidence 1–600. The output directory is new for each invocation. See the separate English sentence writer for two-stage use. [Sentence writer](../visual_statements/README.md)

The fitting procedure expects the project's prepared paired corpus, Dolly split and authenticated DuReader archive at the locations in the preparation module. The three fits use one GPU serially. A new run directory keeps the published heads reproducible.

```bash
python -m scripts.prepare_recurrent_evidence
python -m scripts.train_recurrent_evidence --run-dir artifacts/recurrent-refit
```

The archived comparison can be evaluated with `python -m scripts.evaluate_recurrent_evidence`. [Shared corpus and model setup](../paired_evidence/README.md)
