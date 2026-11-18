# Classical Chinese visual editor

This research checkpoint accepts a pictured classical sentence and a pictured request to quote or translate. Each output cell is a 32 × 32 bitmap. Its 2,026,237 parameters include the source and question encoders, recurrent editor, glyph renderer and a predictor for the number of insertions allowed at each source position. The output generator runs independently on CPU and reads no corpus, tokenizer, glyph table or font.

From the repository root, with PyTorch, NumPy, Pillow and safetensors installed, reproduce one of the exact held-out edits using only a picture of the source and a typed instruction. Typed input uses Noto Sans CJK and the repository's rasterizer.

```sh
PYTHONPATH=. python scripts/infer_visual_edit_translation.py \
  --checkpoint models/visual_edit_zh/editor.safetensors \
  --source-image references/evidence/visual_edit_20261118/death_source.png \
  --question-text '请翻译成白话文' --output /tmp/ilm-classical-death
```

Open `answer.png` for the writing. The output directory also contains the packed bitmaps and the action and stop receipt. Supply a new output directory for each invocation. Use `--source-text` for typed passages and `--question-image` for a pictured instruction. Each input PNG must consist of ordered 32-pixel cells. Set `--source-cells` or `--question-cells` to exclude unused final padding. The model was trained on a single upright typeface and aligned cells. The supported tasks are quotation and classical-to-modern editing. Use `请原样抄录古文` for quotation. The optional `--joint-choice` setting evaluates the alternative action scorer.

On confirmation chapters the default decoder reproduced 2,045/2,045 renderable quotations and 30/2,048 reference translations. Three translations longer than eight glyphs combined visible source writing with a newly drawn Han glyph. The simpler local-edit control had lower overall edit distance. The checkpoint is available for reproducing these measurements. [Experiment, controls and training commands](../../references/visual_edit_translation_result.md)

SHA-256 of `editor.safetensors` is `705f020c14a1d2b81f55c4b4fc2650cff94e9710b5ded2ffd7251c618a0e025e`.

## Reproduce a measured change of wording

The optional likelihood-trained decoder contains the 242,901 updated parameters from the matched training experiment. With the pictured question to translate, it changed 何谓气？ into 什么是气？ on a previously unused chapter. It matched four complete references on 2,048 requests in that set. Add the following arguments to the invocation above, replacing its source image and output directory. [Original images](../../publication/ilm-nature/figures/Figure_49_edit_contrast.png) · [Results and controls](../../references/edit_contrast_result.md)

```sh
PYTHONPATH=. python scripts/infer_visual_edit_translation.py \
  --checkpoint models/visual_edit_zh/editor.safetensors \
  --decoder-update models/visual_edit_zh/likelihood_decoder.safetensors \
  --source-image references/evidence/edit_contrast_20261118/short_source.png \
  --question-text '请翻译成白话文' --output /tmp/ilm-classical-question
```

### Reproduce the sentence-diversity experiment

Figure 50 shows the larger-corpus editor on a successful calendrical expression and on a command whose meaning it changes. This research update contains the 242,901 decoder parameters trained on 192,000 distinct additional translation pairs. On two chapter-disjoint cohorts it gave 32 and 20 exact matches per 2,048 requests. Its aggregate glyph error was slightly higher than the repeated-passage control. [Experiment](../../references/edit_diversity_result.md)

```sh
PYTHONPATH=. python scripts/infer_visual_edit_translation.py \
  --checkpoint models/visual_edit_zh/editor.safetensors \
  --decoder-update models/visual_edit_zh/diverse_decoder.safetensors \
  --source-image references/evidence/edit_diversity_20261118/success_source.png \
  --question-text 请翻译成白话文 --output /tmp/ilm-solar-eclipse
```
