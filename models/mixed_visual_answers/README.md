# Chinese image-question writer

The mixed writer reads a passage and a question, then writes a horizontal strip of glyph images. It can quote a source span, write a two-sentence factual answer, or return the historical shapes on a supplied catalogue card. Its 86.6-million-parameter network inherits a Chinese text-pretrained transformer. A fixed bitmap inventory and a pointer to the source provide its output shapes. The current model expects ordered 32-by-32 cells.

Extract `mixed-visual-writer.zip` into an empty directory. Install Python 3.10+, PyTorch, Transformers, safetensors, Pillow, NumPy and fontTools. Run from that directory.

```bash
python -m scripts.respond_pointing_glyphs --model writer \
  --passage examples/yan_oracle_page.png --question examples/yan_oracle_question.png \
  --output answer.png
```

The example asks for 言 in oracle-bone and then bronze script. Substitute `yan_bronze_question.png` to reverse the request. The catalogue plate is supplied with the question. For a natural reading question, use `human_page.png` and `human_question.png`. Add `--device cuda` for GPU inference. Typed requests are rendered with `--passage-text` or `--question-text` and a Noto CJK font. For two pictured paragraphs, repeat `--passage` and supply `--paragraph-ranker paragraph.safetensors`.

To reproduce the adaptation from the research checkout, run `CUDA_VISIBLE_DEVICES=0 python -m scripts.train_mixed_visual_answers`, then `CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_mixed_visual_answers`. Training uses the archived varied-instruction writer and screened CMRC, DRCD, DuIE and historical-form records. Evaluation stores raw pixel streams with offsets and stop flags. The standalone archive contains the complete inference weights, runtime and examples.
