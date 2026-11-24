# Writing two facts into a Chinese answer image

A guest asks “太好了。北京贵都大酒店的价格和地址分别是什么啊？” On the supplied hotel page, the released model writes **价格是1354元。地址是在北京西城区广安门内大街217号。** Both the question and the result are images. The 333,850 parameters run on CPU or CUDA without a language service. [Input PNG](cursor-examples/exchange-card.png) · [Answer PNG](cursor-examples/exchange-answer.png)

```bash
python -m scripts.write_chinese_two_fact \
  --input models/chinese_two_fact/cursor-examples/exchange-card.png \
  --output /tmp/ilm-hotel-answer
```

Run from the repository root with a new output directory. Add `--device cuda` for GPU inference. The directory contains the input, answer PNG, packed 32 × 32 output cells and an optional text transcription. Unrecognized output cells appear as U+FFFC in the transcript. Their pixels are preserved in the PNG and binary file.

For typed questions, replace `--input` with `--record card.json`. The renderer uses Noto Sans CJK and presents the same pixels to the reader as the PNG route.

```json
{"q":"这家酒店的价格和地址是什么？","domain":"酒店","name":"北京贵都大酒店","card":{"地址":"北京西城区广安门内大街217号","价格":"1354元","酒店类型":"高档型"}}
```

Each page has 80 columns with 32-pixel cells. The first three rows contain the question, domain and labelled venue name. The following rows contain one labelled property each. The program recognizes row prefixes and domain labels by their printed pixels. Questions and sentences use a fixed typeface. Prices, names and other values can contain unfamiliar marks. The reader chooses up to two of eight known properties of a single supplied venue. The writer draws from 446 ordinary glyphs and can copy the name or a selected source value. A changed typeface, a photograph of a page or a request for three properties exceeds this interface.

### Reproduce training and evaluation

Use the repository's Python environment with PyTorch, Pillow, NumPy, fontTools and safetensors. On Linux, use a CUDA-enabled PyTorch build for the fits. `ILM_TWO_FACT_RUN` selects a separate directory for the prepared records, checkpoints and scores. Raw CrossWOZ files are shared across runs. Preparation fetches missing files from the pinned revision and verifies their contents. The two modules train from scratch. The aligner command also fits the matched byte-input control.

```bash
export ILM_TWO_FACT_RUN="$PWD/artifacts/my-two-fact-run"
python -m scripts.prepare_chinese_two_fact
python -m scripts.train_field_alignment
python -m scripts.train_field_cursor
python -m scripts.evaluate_field_cursor
python -m scripts.extend_field_cursor
python -m scripts.write_chinese_two_fact \
  --input models/chinese_two_fact/cursor-examples/exchange-card.png \
  --output "$ILM_TWO_FACT_RUN/hotel-answer"
```

Completed fit receipts verify and reuse weights inside the selected run. Unset `ILM_TWO_FACT_RUN` to use the published checkpoints again. The optical path loads `alignment-ink.safetensors` and `cursor.safetensors`, together about 1.34 MB. It also uses the physical alphabet, checksums, printed label images, layout and optional transcription assets. The other checkpoints preserve earlier comparison models. [Training source](https://github.com/thu-coai/CrossWOZ/tree/df82c9fdff91b9b130f2d6b89110d3870ba6260e) · [Full results and failure cases](../../references/chinese_two_fact_result.md) · [Original trial receipts](../../references/evidence/chinese_two_fact_v1/)
