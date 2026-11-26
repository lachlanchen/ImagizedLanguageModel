# Independent bilingual glyph writers

These two 3,196,224-parameter language bodies read a grid of printed glyphs in reading order. The input image may contain one to 96 aligned 32 × 32 cells. The output is a PNG drawn by the frozen neural glyph motor. Each writing decision either stops or emits an ordered sequence from a finite repertoire. The single-glyph body emits one glyph at a time. The strip body can also emit a short sequence learned from adjacent training images. Both read pixels pooled from the same physical glyphs used to draw the answer.

To write Chinese from an English glyph grid on CPU, supply the occupied cell count, including intentional spaces.

```sh
PYTHONPATH=. python -m scripts.write_independent_optical_actions \
  --input source.png --cells 24 --to zh --arm strip --output answer.png
```

Use `--to en` for an English response. The command prints the number of returned cells and whether the model chose to stop. A grid may wrap across multiple rows. The 128-cell output cap applies to both writers. Proportional print and unsegmented page photographs need a separate reader. The public weight, repertoire, inventory and frozen motor files support inference independently from the training corpora. Decoded text is used only in evaluation.

Training uses both directions of 268,342 parallel pairs, ordered identically for both fits. Each fit has four complete passes. The data preparation verifies a lossless decomposition of every physical target. Book-level translation scores, exact input and output PNGs, alternate-source controls, CPU replay and measured training cost accompany the release. [Experiment](../../references/independent_optical_actions_result.md)

With the verified OPUS bitmap arrays and the parallel book library in place, the following commands rebuild the writing units, reserve the disjoint panel, fit both independent bodies on GPU 0 and generate the actual free translations.

```sh
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
python -m scripts.prepare_independent_optical_actions
python -m scripts.reserve_independent_optical_actions
python -m scripts.train_independent_optical_actions
python -m scripts.evaluate_independent_optical_actions generate
python -m scripts.evaluate_independent_optical_actions replay
```

The fit command verifies existing completed weights and resumes incomplete local fits from epoch checkpoints. Semantic assessment uses the project's local Qwen installations in a separate evaluation process.
