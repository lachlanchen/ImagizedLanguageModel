# Visual word retrieval on a single GPU

The image writer reads a pictured passage and question, optionally with pictured history, and returns a PNG strip. It can copy word images, spell from observed glyph cells, or emit one of 2,048 physical words observed in fitting answers. This last route has a finite learned vocabulary. The checkpoint includes the reader, context model, prototype keys, glyph images, projection and writer. It runs independently of a teacher.

The released checkpoint handles aligned, space-separated English print in the training face. Inputs may contain up to 448 passage words, 32 question words and 48 history words. It predicts at most 16 answer words. The byte comparison uses ASCII fields. The input conventions and real example PNGs are documented with the four-pass writer. [Input format](../word_canvas/README.md) · [Accuracy and failure cases](../../references/word_memory_result.md)

Run from the repository root in its PyTorch environment. For typed inputs the command first renders the text into aligned glyph cells. For image inputs use the corresponding flags without the text suffix.

```bash
scripts/with_ilm_gpu python -m scripts.answer_dialogue_surface --device cuda \
  --model models/word_memory/ink \
  --passage-text "Lena placed the blue book on the table." \
  --question-text "Who placed the book on the table?" \
  --output artifacts/word-memory-answer.png
```

Fit a fresh copy using the original source preparation and released parent checkpoints. Each representation receives 102,400 training presentations. Evaluation writes into a fresh output folder.

```bash
python -m scripts.prepare_word_memory
scripts/with_ilm_gpu python -m scripts.train_word_memory --output artifacts/my-word-memory-models
scripts/with_ilm_gpu python -m scripts.evaluate_word_memory \
  --models artifacts/my-word-memory-models --output artifacts/my-word-memory-evaluation
```

The six evaluation arms retain each free answer on original and exchanged inputs, the strict physical decision at each concealed word and decoder certainty counts. The single-concealed-word test supplies the remaining answer words. Only the free-answer test measures autonomous writing. [Data setup](../compose_dialogue/README.md) · [Reservation and protocol](../../references/word_memory_protocol.md) · [Checkpoint lineage costs](../../references/evidence/word_memory_v1/cost.json)
