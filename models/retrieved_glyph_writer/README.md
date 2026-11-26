# Write with related bilingual passages

From the repository root, run the local CPU writer on a typed prompt rendered into its trained print face, or on the same 32×32-cell image grid used during evaluation.

```sh
PYTHONPATH=. python -m scripts.write_retrieved_glyphs --text 'It runs like this.' --to zh --output answer.png
PYTHONPATH=. python -m scripts.write_retrieved_glyphs --image question.png --cells 18 --to zh --output answer.png
```

The first command reproduced the four output cells **这样的。** in held case 0055. For PNG input, supply the actual occupied cell count in row-major order, up to 96 cells. Select English with `--to en`. The source must use the trained English–Chinese print repertoire. The writer makes its own next-glyph and stop decisions. The released book test yielded two automatically credited meanings in 96 requests. [All outputs and limitations](../../references/retrieved_glyph_writer_result.md)

The runtime loads the existing draft editor, its optical inventory and phrase tables, plus both `artifacts/ordered_alignment_v1/strips-0.npy` and `strips-1.npy`. They total 191.15 MB and match the two checksums under `strips` in the corpus receipt. A fresh checkout needs the original prepared corpus to reconstruct these arrays. The model builds one directional search index in memory and reads the stored development strength of 0.35. No external LLM is invoked for generation. [Parent weights and dependencies](../glyph_draft_editor/README.md) · [Corpus preparation](../../scripts/prepare_ordered_alignment.py) · [Corpus receipt](../../references/evidence/ordered_alignment_v1/corpus.json) · [Calibration](../../references/evidence/retrieved_glyph_writer_v1/calibration.json)
