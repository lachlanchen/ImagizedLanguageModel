# Visual page responder

Give the model a passage image and a question image. It uses a learned decision to read a span of the passage or compose a response with a recurrent glyph writer. The returned PNG contains the actual answer cells. Typed inputs are rendered into the training font before entering the model.

Use the extracted model archive with Python, PyTorch, Transformers 4.x, safetensors, NumPy, Pillow and fonttools. The regular typed-input renderer uses Noto Sans CJK SC. PNG input requires no fonts. From the extracted directory run

```sh
python respond.py --bundle . \
  --passage-text '秦澈迷了路；路口那張地圖是卓霖替他畫的。' \
  --question-text '畫地圖的是哪一位？' --output map-maker.png
```

For pictured inputs run

```sh
python respond.py --bundle . \
  --passage-image examples/fresh_sans_0_passage.png \
  --question-image examples/fresh_sans_0_question.png --output pictured-answer.png
```

For the attested bronze form of 言 on the supplied plate, run

```sh
python respond.py --bundle . --passage-image examples/yan_plate.png \
  --passage-cells 34 --question-image examples/yan_bronze_question.png \
  --output yan-bronze.png
```

The oracle and bronze questions each return the correct individual form when the 言 entry appears first. The combined chronological request returns a single glyph. The example files include both the returned and expected images.

Inputs use a row-major grid of 32 × 32 grayscale cells with dark ink on white. For a padded last line, supply the occupied lengths with `--passage-cells` and `--question-cells`. The model accepts already segmented streams. A page scan requires segmentation before inference. Typed text and PNGs can be mixed. Use `--device cuda:0` for GPU inference. The command verifies the packaged weights, saves a PNG, and prints which output action was used and whether generation stopped.

The reading branch inherits the contextual weights and extractive head of CKIP BERT Base Chinese QA. All runtime lexical embedding weights were removed. A convolutional glyph metric and dense regression carry the pixels into the contextual encoder. It scores contiguous source spans up to 64 cells. The composing branch uses image memory, source copying and learned stopping. Its strongest measured writing skill is reordering familiar forms of statements. Open DuReader answers remain at zero exact reference matches in 571 cases. [CKIP source and GPL-3.0 model license](https://huggingface.co/ckiplab/bert-base-chinese-qa)

The full archive is written to `artifacts/font_invariant_reader_v1/visual-page-responder.zip`. The repository commands are `PYTHONPATH=. python scripts/export_visual_page_responder.py` for packaging a fitted run and `PYTHONPATH=. python scripts/respond_visual_page.py` for inference with its extracted directory. The pinned contextual model, visual fitting protocols, controls and limits are in the study report. [Experimental results](../references/continuous_interface_result.md)
