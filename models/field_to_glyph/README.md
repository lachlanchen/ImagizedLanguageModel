# Bilingual image-to-image writers

These checkpoints read a line or grid of 32 × 32 glyphs in reading order and write a PNG one cell at a time. Both use a neural stroke generator with 3,717 learned drawings. Their only inference inputs are the source image, the number of occupied cells (1–96, including spaces), and the target language. The writers need previously segmented input. Their ability to give a faithful literary translation is limited.

From the repository root with Python, PyTorch, NumPy, Pillow and safetensors installed, this example reads the pictured Chinese sentence 汤姆说。 and writes “said Tom.” on CPU.

```sh
python -m scripts.write_neural_glyphs \
  --source-field prediction \
  --input publication/ilm-nature/figures/field_transfer/0000-input.png \
  --cells 4 --to en --output answer.png

cmp answer.png publication/ilm-nature/figures/field_transfer/0000-prediction.png
```

The prediction and regression checkpoints restore their corresponding frozen visual readers from `models/readout_objective/{en,zh}` and the common motor from `models/neural_glyph_motor`. Their fit receipts bind all three parents by hash. The complete inference system has 10,481,570 parameters and 604,480 stored optical-code scalars. Each output cell is computed by a neural network and read back into the next step. The finite visual choice bank determines which shapes the model can write. The teacher, text codec and fonts are used outside deployed generation.

To reproduce all 768 archived answers, lengths and stops from their input PNGs, run the following on CPU.

```sh
python -m scripts.replay_field_writer_archive
```

In the primary two-order model assessment the prediction writer earned 4/192 whole-meaning credits, regression 5/192, the original motor 5/192 and Qwen3-8B Q4 with text 171/192. The experiment includes uncredited answers, wrong-source controls and both translation directions. [Results](../../references/field_transfer_learning_result.md) · [Raw images](../../publication/ilm-nature/evidence/field_writer_outputs.zip)
