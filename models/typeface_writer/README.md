# Reading Chinese questions across typefaces

The writer accepts two black-on-white glyph strips, a passage and a question. It returns an answer image and prints whether it stopped. Each input cell is 32 × 32 pixels. A learned mapping recognizes alternate forms of printed characters. A separate pointer can reproduce an unfamiliar source glyph from its original pixels.

Extract `typeface-writer.zip` and run the following in the extracted directory with PyTorch, Transformers, safetensors, NumPy, Pillow and fontTools installed.

```bash
python -m scripts.respond_pointing_glyphs --model writer \
  --passage examples/human_page.png --question examples/human_question.png \
  --output answer.png
```

The included reading question is printed in HanaMin, a family withheld during the adaptation. To try the inscription example, use `examples/yan_oracle_page.png` with either `examples/yan_oracle_question.png` or `examples/yan_bronze_question.png`. Add `--device cuda` for GPU inference. Typed passages and questions can be supplied with `--passage-text` and `--question-text` when a suitable local Chinese font is installed.

The standalone archive contains the entire writer and the visual correspondence weights. A pretrained Chinese transformer supplied part of its language knowledge. Historical forms in the example are on the supplied card. [Training comparison, actual replies and limitations](https://github.com/lachlanchen/ImagizedLanguageModel/blob/main/references/typeface_writer_result.md)
