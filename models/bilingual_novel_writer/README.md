# Ask a pictured Chinese passage a question

The bilingual research checkpoint accepts a Chinese passage and a Chinese or English question. It emits a horizontal image of its Chinese answer. Both inputs use 32 × 32-pixel cells. Typed inputs are rendered locally before inference. The model inherits a Chinese transformer and uses a learned writer with two ways to emit ink, a visual inventory and a pointer to a glyph on the input page.

The standalone noncommercial research release is `bilingual-novel-writer.zip`. Extract it and run the following command from the extracted directory with PyTorch, Transformers, NumPy, Pillow, safetensors and fontTools installed. The archive includes the weights, inference code, eight original example replies and the historical-form catalogue.

```bash
python -m scripts.respond_pointing_glyphs --model writer \
  --passage examples/human_page.png --question examples/human_question.png \
  --output answer.png
python -m scripts.ask_glyph_origin 言 --stages oracle bronze --output-dir reading
```

From the repository root, this command uses the fitted weights stored on the training workstation. CPU is the default. Add `--device cuda` to use a GPU.

```bash
python -m scripts.respond_pointing_glyphs \
  --model artifacts/bilingual_novel_v1/bilingual/page \
  --passage references/evidence/bilingual_novel_20261124/bookshelf/page_0.png \
  --question references/evidence/bilingual_novel_20261124/bookshelf/question_Lin_zh.png \
  --output answer.png
```

For a typed request use `--question-text '林家的书架上放着几部词典？'` in place of the question file. Use `page_1.png` to exchange the counts in the passage. The directory includes original answer PNGs from both pages for Chinese and English questions about each household. Each output has a stop flag and an exact-image score in the example receipt. [Example inputs and replies](https://github.com/lachlanchen/ImagizedLanguageModel/blob/main/references/evidence/bilingual_novel_20261124/bookshelf)

The authors of BiPaR distribute the training data under CC BY-NC 4.0. The fitted checkpoint is provided locally for noncommercial research. Raw data are in `data/raw/bipar/`, aligned book examples are in `artifacts/bilingual_novel_v1/{train,valid}.jsonl`, and the three locally fitted weight directories are `artifacts/bilingual_novel_v1/{bilingual,chinese_repeat,english_exchanged}/page/`. The training protocol lists the prior writer, commands and split details. [Training protocol](https://github.com/lachlanchen/ImagizedLanguageModel/blob/main/references/bilingual_novel_protocol.md) · [BiPaR, Jing, Xiong and Yan (2019)](https://aclanthology.org/D19-1249/) · [Dataset license](https://multinlp.github.io/BiPaR/)

On the first page, the pictured Chinese question about the Lin family produces 三部词典 (three dictionaries). Changing their count on the page changes that reply to 五部词典 (five dictionaries). Asking about the Chen family in Chinese produces the same answer as asking about the Lin family on both pages, despite different counts in the source. Four of the eight formulations receive an accepted answer. This small edit exposes a subject-selection error even when the glyphs are legible.
