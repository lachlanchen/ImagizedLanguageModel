# English–Chinese pictured draft editor

The editor reads 1–96 aligned binary glyphs from the trained print face. It builds a tentative translation from bilingual phrase tables, then writes the answer one physical glyph at a time until its learned stop. It requires no external language model for inference.

```sh
PYTHONPATH=. python -m scripts.write_glyph_draft_editor --text '对我来说也是如此。' --to en --output answer.png
PYTHONPATH=. python -m scripts.write_glyph_draft_editor --image question.png --cells 9 --to en --output answer.png
```

Supply the actual number of occupied 32×32 cells for a PNG. The input uses black ink on white in normal reading order and up to 32 columns. Use `--arm matched` for the separately fitted source-only comparison. The source marker, palette, learned visual strip and bilingual phrase tables reside in their original model directories. The study records a 25.37 MB combined numerical deployment payload for the full editor. The released writer supports the measured English and Chinese print repertoire. Its sole credited answer on the 96-request book test reproduces a correct phrase draft. [Training, dependencies, real outputs and limits](../../references/glyph_draft_editor_result.md)

From the prepared optical corpus and bilingual book editions:

```sh
PYTHONPATH=. python -m scripts.prepare_glyph_draft_editor
PYTHONPATH=. scripts/with_ilm_gpu python -m scripts.train_glyph_draft_editor
PYTHONPATH=. scripts/with_ilm_gpu python -m scripts.evaluate_glyph_draft_editor
PYTHONPATH=. python -m scripts.audit_glyph_draft_editor
```

Training reuses bound completed fits when available. Evaluation also requires the local quantized translation comparator and grader specified in the archived scores.
