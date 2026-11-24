# Optical Mandarin–English translation

The checkpoints read binary glyph images on a 32 × 32 cell grid and produce an answer PNG. Stride one gives the stronger original conversational translator. The specimen writer can carry an unfamiliar bitmap into its output. Ordinary bilingual translation still has a large gap to a local quantized 8B model. [Measured outcomes, learning curves and scope](../../references/ordered_visual_translation_result.md)

From the repository root, translate a pictured sentence that contains a historical specimen. The directory passed to `--output` must be new.

```bash
python -m scripts.write_ordered_translation \
  --image models/ordered_translation/examples/substitution-exact/source.png \
  --cells 28 --to zh --preserve-specimens --output /tmp/ilm-pictured-answer
```

The model emits the answer in `answer.png`. The receipt records the occupied cells, stop event and SHA-256 of their packed pixels. The original input asks about looking for a good job. The Chinese reference and emitted image preserve the specimen and express looking for work. Compare the first failure in `examples/substitution-error/` for an occupation the same writer loses.

For a typed sentence, the interface rasterizes the characters before inference.

```bash
python -m scripts.write_ordered_translation \
  --text 'The box contains fifteen coins.' --to zh --stride 1 \
  --output /tmp/ilm-coin-answer
```

The source can occupy 1–96 cells, including spaces. PNG input supplies its occupied length. The ordinary writer sees 16 × 16 local ink densities and emits physical prototypes drawn from training. The specimen mode keeps full-resolution unknown pixels separately. Use the ordinary appearance-sensitive `--stride specimen` for comparison. Requesting the meaning of an unknown symbol requires its shape to remain visible to the semantic reader. Full manuscripts with unknown layout require a segmentation stage outside this release.

To reproduce the matched conversational fits on GPU 0, run the following commands in order. Requirements include this repository's PyTorch environment, OpenCC, safetensors and its Noto CJK rendering face. Historical specimen training additionally reads the local catalogue SQLite database. The quantized reference assessment uses the project's separately installed local Qwen models. The released image writers run on CPU without those models.

```bash
python -m scripts.prepare_ordered_translation
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_ordered_translation
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_specimen_translation
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_ordered_translation
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_specimen_translation
python -m scripts.evaluate_specimen_plan
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_news_translation
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_news_translation
```

The human sentence pairs carry Tatoeba contributor attributions in the preparation records. Adapted examples retain credits beside the PNGs. Public checkpoints include their measured training receipts. [ManyThings bilingual archive](https://www.manythings.org/anki/) · [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/) · [News Commentary](https://data.statmt.org/news-commentary/) · [Historical specimens](https://hanziyuan.net/)
