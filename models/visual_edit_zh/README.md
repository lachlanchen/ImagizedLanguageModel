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

### Generate from the visual span transducer

The research update `span_update.safetensors` holds the trainable parameters of a writer that sums over alternative alignments of pictured glyphs. On four constructed role-swap pairs the writer preferred the correct joint assignments. Its two independent responses never reproduced both recorded modern answers in a pair. The PNGs in Figure 51 show a one-glyph modernization and a failure that reached 192 output cells. [Study](../../references/visual_span_result.md)

```sh
PYTHONPATH=. python scripts/infer_visual_edit_translation.py \
  --checkpoint models/visual_edit_zh/editor.safetensors \
  --decoder-update models/visual_edit_zh/span_update.safetensors \
  --span-decoder action \
  --source-image references/evidence/visual_span_20261118/changed_source.png \
  --question-text 请翻译成白话文 --output /tmp/ilm-span-modernization
```

Use `--span-decoder prefix` and a fresh output directory to sum over compatible source positions as the image is written. The receipt gives the probability of the observed prefix (including termination if reached), its maximum probability-conservation residual and the number of decisions for which a discarded within-glyph prefix could still have beaten the selected output.

### Draw a missing modern phrase at a supplied boundary

The optional phrase writer reads a classical source, the number of source cells preceding the insertion, and any available modern prefix. It generates a new 32-pixel-high strip and a stop receipt. Use the published base editor with `phrase_update.safetensors` (SHA-256 `d02cd89140ece2a741841353c0aa3341e99117096695ba6ba69a6a84b24ee8eb`). The update includes the original pictured translation instruction. This specific task was trained with aligned, nonempty insertions and a correct preceding modern prefix.

From the repository root, the following CPU example returns the two pictured glyphs 没有 and stops. The source, prefix and answer can be inspected directly in the figure.

```bash
PYTHONPATH=. python scripts/infer_visual_phrase.py \
  --source-image references/evidence/visual_phrase_20261118/writer_source.png \
  --cursor 7 \
  --prefix-image references/evidence/visual_phrase_20261118/writer_prefix.png \
  --output /tmp/ilm-phrase-example
```

Use a new output directory for each run. `--source-text` and `--prefix-text` rasterize typed input in the training typeface. Omit the prefix when the insertion begins the modern sentence. On 10,789 sites the model produced 605 exact stopped phrases, including 20 multi-glyph insertions. A local training-count rule recovered 1,676. [Study and limits](../../references/visual_phrase_result.md) · [Original output images](../../publication/ilm-nature/figures/Figure_52_visual_phrase.png)
