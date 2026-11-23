# Pictured paragraph selection

This 61,925-parameter selector reads a question strip and ranks separately supplied paragraph strips. The existing Chinese glyph writer then reads the selected paragraph and the original question. Each strip contains a single horizontal line of 32-pixel square cells. Photographs and unsegmented pages need layout preparation. Typed input uses the host font to produce the same interface.

Extract the varied-instruction `page-evidence-writer.zip`, then extract `paragraph-selector.zip` into the same directory. Run from that directory with the writer's Python dependencies installed.

```bash
python -m scripts.respond_pointing_glyphs \
  --model writer --paragraph-ranker paragraph.safetensors \
  --passage examples/paragraph-left.png --passage examples/paragraph-right.png \
  --question examples/paragraph-question.png --output answer.png
```

Use repeated `--passage-text` and one `--question-text` for typed input. Single-paragraph calls bypass the selector. The selector can rank more than two supplied paragraphs, although the measured comparison used pairs. The writer uses a fixed output inventory plus bitmaps from the selected source. Its pretrained language backbone contributes to its performance and cost.

From the repository root, use `python -m scripts.train_paragraph_evidence --run-dir <new-directory>` followed by `python -m scripts.evaluate_paragraph_evidence --run-dir <same-directory>`. These research commands require the screened source splits and the varied writer recorded in the experiment protocol. The released archives support offline inference independently of those training artifacts.
