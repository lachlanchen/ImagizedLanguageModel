# Answer a Chinese question from a pictured paragraph

One real travel recommendation rates 全聚德烤鸭店(清华园店) at 4.7 and 四季民福烤鸭店（灯市口店） at 4.8. Give the program the same paragraph and the same pictured question, and change only the restaurant name in the third row. Its answer changes from **评分是4.7分。** to **评分是4.8分。** Both are generated PNGs. [First page](examples/restaurant-1-page.png) · [First answer](examples/restaurant-1-answer.png) · [Second page](examples/restaurant-2-page.png) · [Second answer](examples/restaurant-2-answer.png)

Run from the repository root. Use a new output directory for each answer.

```bash
python -m scripts.write_prose_fields \
  --input models/prose_fields/examples/restaurant-2-page.png \
  --output /tmp/ilm-restaurant
```

CPU is the default. Add `--device cuda` for GPU inference. The output contains the input page and a JSON record of selected properties and source intervals. When all requested values have been located, it also contains the answer PNG, packed bitmap strip and optional transcription. A missing interval produces a missing-property report. A substring selected with confidence may still be wrong.

For typed input, pass `--record record.json` in place of `--input`. The front end renders the record before model inference. This record reproduces the second restaurant example.

```json
{
  "q": "请问这家的评分是多少？",
  "domain": "餐馆",
  "name": "四季民福烤鸭店（灯市口店）",
  "text": "为您推荐：全聚德烤鸭店(清华园店)，评分4.7分；四季民福烤鸭店（灯市口店），评分4.8分，都是最有北京代表性的餐馆，您不妨试试。"
}
```

The question, domain and target name occupy the first three rows. The next one or two rows hold up to 160 cells of continuous prose with no per-property columns. Each row has 80 aligned 32 × 32 cells. The typeface is Noto Sans CJK. The model covers restaurants, hotels and attractions, with eight supported properties across the three domains. It uses an exact image occurrence of the supplied name when available. It can copy unfamiliar source cells but chooses its own ordinary wording from a fixed glyph bank. It often confuses prices belonging to different conditions and references such as 前者 and 后者. The two park examples preserve an actual failure. [Park input](examples/park-2-page.png) · [Returned answer](examples/park-2-answer.png)

The default reader was trained to distinguish named venues in the same text. The original single-venue reader remains available with `--single-venue`. It scored 184/184 on eligible natural single-venue replies. The name-conditioned reader scored 179/184 there and 472/539 on the four-answer, two-venue comparisons. The original reader passed zero of those comparisons. [Evaluation and costs](../../references/prose_fields_result.md)

## Fit and evaluate on one GPU

Use the repository's CUDA-enabled Python environment. The system uses PyTorch, NumPy, Pillow, fontTools, safetensors and the shared Noto Sans CJK face. Set a new run directory to keep your own arrays, weights and receipts. Preparation downloads missing official CrossWOZ files at the pinned revision and checks their hashes.

```bash
export ILM_PROSE_RUN="$PWD/artifacts/my-prose-run"
python -m scripts.prepare_prose_fields
python -m scripts.train_prose_fields
python -m scripts.train_prose_entity
python -m scripts.evaluate_prose_fields
python -m scripts.evaluate_prose_entity
python -m scripts.audit_prose_release
python -m scripts.audit_prose_names
python -m scripts.audit_prose_observability
python -m scripts.write_prose_fields \
  --input models/prose_fields/examples/restaurant-2-page.png \
  --output "$ILM_PROSE_RUN/restaurant-answer"
```

The two training commands fit the interval readers and matched controls. The question aligner and Chinese current-field writer are reused from the published two-fact model. Their original training instructions and receipts are available separately. Inference uses three learned networks totalling 486,442 parameters and no external language service. Unset `ILM_PROSE_RUN` to use the released weights. The optional external price evaluation reads the upstream DuReader 2.0 archive from `data/raw/dureader2/dureader_raw.zip`. [Question reader and writer](../chinese_two_fact/README.md) · [CrossWOZ source](https://github.com/thu-coai/CrossWOZ) · [DuReader](https://github.com/baidu/DuReader/tree/master/DuReader-2.0)
