# Pictured passage writer

This 759,351-parameter model predicts an English final word or continues an English passage as a glyph image. The frozen intact and shuffled-training models, their fit receipts and the training-book frequency table are in this directory. Inference uses local CPU weights, PyTorch, safetensors, NumPy, Pillow, fontTools and the system Noto Sans CJK regular face. [Measured accuracy and original outputs](../../references/optical_passage_selector_result.md)

From the repository root, supply the text of a prefix or a row-major grid of aligned 32 × 32 cells in the training face.

```bash
python -m scripts.write_passage_word --text "When she opened the door, she saw " --output answer.png
python -m scripts.write_passage_word --image prefix.png --continue-words 3 --output continuation.png
```

Use `--cells N` for the occupied portion of a padded grid. Omitting `--continue-words` requests one sentence-final alphabetic run. Supplying it enables ordinary continuation with feedback until the learned stop or the word budget. The output grid is the actual concatenation of selected print cells. Image matching expects the canonical print face and alignment used by the model.

## Reproducing the comparison

Training requires CUDA-enabled PyTorch with fused AdamW. Data preparation also uses the repository's book-study dependencies, including PyArrow, requests and lxml. Restore the working dataset and verify the checked-in fits with the following commands. The restoration command uses the pinned sources and validates all five data hashes. The trainer verifies and reuses a completed arm.

```bash
python -m scripts.restore_passage_data
scripts/with_ilm_gpu python -m scripts.train_passage_selector
python -m scripts.score_passage_selector
python -m scripts.replay_passage_selector
```

To reproduce the actual fits and free generation, use a separate checkout, move both model-arm directories and the released `publication/ilm-nature/evidence/learned_passage_words.zip` aside in that checkout, then run the trainer, followed by `scripts/with_ilm_gpu python -m scripts.evaluate_passage_selector` and the scoring command. Keep the frozen frequency table and source manifest. Model construction uses no learned word-identity embedding. Training and inference both use the same spatial ink features and physical letter repertoire. [Training protocol](../../references/optical_passage_selector_protocol.md) · [Dataset receipt](../../references/evidence/passage_selector_v1/data.json)
