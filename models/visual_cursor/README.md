# Visual source-position writer

This model reads a question and passage in one aligned PNG and writes the answer as a second PNG. It uses the released optical discourse writer together with a visual page encoder and a recurrent decoder. A posterior over page positions lets it reuse the next observed glyph, jump to another visible position, emit a training glyph or stop. It takes no candidate answers, tokenizer, text teacher or language-model server at inference. The text interface renders a prompt into the same 32-pixel cells.

From the repository root, reproduce the held traditional Chinese example. Its wrong output is preserved for comparison.

```bash
PYTHONPATH=. python -m scripts.write_visual_cursor \
  --image models/visual_cursor/examples/persistent/drcd/0/input.png \
  --cells 451 --output /tmp/ilm-reading
```

The stored question asks whom Kublai competed with. The writer replies **1260年**. Use `--text` for your own question and visible evidence. The training format is `问：… 原文：… 答：` in Chinese or `Q: … Passage: … A: ` in English. The display can recover known Unicode characters from the physical stream. Image input assumes a known cell geometry and row order.

Training draws from CoQA, CMRC2018 and DRCD and reuses the preceding optical book and instruction corpus. Place the source files at the pinned paths in `scripts/prepare_visual_cursor.py`. The discourse preparation and trained backbone must be available first. Existing authenticated terminal fits are reused.

```bash
PYTHONPATH=. python -m scripts.prepare_visual_cursor
PYTHONPATH=. python -m scripts.train_visual_cursor
PYTHONPATH=. python -m scripts.evaluate_visual_cursor
PYTHONPATH=. python -m scripts.diagnose_visual_cursor_questions
PYTHONPATH=. python -m scripts.replay_visual_cursor
```

The held results include the two fitted arms and the unchanged writer. Six archived PNGs, across three sources, exercise the physical inference entry point on CPU. [Results and controls](../../references/visual_cursor_result.md) · [Complete-question comparisons](../../references/evidence/visual_cursor_v1)
