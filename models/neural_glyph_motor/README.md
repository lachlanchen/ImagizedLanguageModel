# A compact translator with a learned pixel output

The model reads a pictured English or Chinese sentence and returns a binary PNG in the other language. It accepts one to 96 occupied cells of 32 × 32 pixels, in reading order. Include spaces in the cell count. The output network predicts each cell's 1,024 pixels and feeds their pooled ink back into the next decoding step. Its visual choices come from a finite learned repertoire.

From the repository root, this example runs entirely on CPU with PyTorch, NumPy, Pillow and safetensors.

```sh
python -m scripts.write_neural_glyphs \
  --input publication/ilm-nature/figures/writer_provenance/motor-0058-input.png \
  --cells 8 --to en --output answer.png
```

The pictured Chinese source means “Very true, he said.” On 192 held literary requests the motor reproduced every complete answer from its parent writer. The source-only automated assessment credited two translations. The model needs an already segmented grid, and its general translation quality remains poor. The shared motor learns from 3,717 known shapes. It has no font, OCR dependency or bitmap table at inference. The checkpoint contains the visual language writer, motor weights and optical choice vectors. [Input and answer images](../../publication/ilm-nature/evidence/neural_motor_answers.zip) · [Evaluation](../../references/evidence/neural_glyph_motor_v1/evaluation.json)
