# Independent bilingual glyph writers

The two continued fits share a 10.48M-parameter image reader and motor. The preference checkpoint learned from reverse translation scores on its own drafts. The matched checkpoint received the same parallel sources and references. Both deploy locally with the repository's visual model weights and PyTorch, without a translation teacher at runtime. Their free literary translation quality remains low. The result below is a numbered heading credited by the automated reader. [Complete results and controls](../../references/sentence_reconstruction_result.md)

```sh
python -m scripts.write_sentence_reconstruction \
  publication/ilm-nature/figures/sentence_reconstruction/0161-source.png \
  answer.png --cells 11 --to zh
cmp answer.png publication/ilm-nature/figures/sentence_reconstruction/0161-preference.png
```

Inputs use reading-order cells of 32 × 32 pixels with one to 96 occupied cells including spaces. Use `--arm likelihood` for the matched control, `--arm parent` for the starting point, or `--compile-motor` to cache every output image and its visual feedback. The optional exact readback in the archive was applied after emission.

With the authenticated local library shards, original evaluation panels, palette and pinned OPUS-MT teacher snapshots installed, reproduce the fit and assessment from the repository root. The review uses the configured local Qwen and Qwen-VL evaluator. The replay operation uses CPU and the archived files.

```sh
python -m scripts.reserve_emitted_history
python -m scripts.collect_sentence_drafts
python -m scripts.train_sentence_reconstruction
python -m scripts.evaluate_sentence_reconstruction generate
python -m scripts.evaluate_sentence_reconstruction assess
python -m scripts.evaluate_sentence_reconstruction replay
python -m scripts.audit_sentence_reconstruction
python -m scripts.measure_sentence_reconstruction
```
