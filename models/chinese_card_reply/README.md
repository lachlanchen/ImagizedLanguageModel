# Answering a pictured Chinese question

The released 1,568,944-parameter writer reads one pictured Chinese question and a venue record. It returns an answer PNG. The question, input evidence and emitted answer are available as ordinary images. The optional plain-text transcript decodes known output bitmaps. The model runs locally on CPU or CUDA.

```bash
python -m scripts.write_chinese_card_reply \
  --input models/chinese_card_reply/examples/question-b-card.png \
  --output /tmp/ilm-chinese-answer
```

Use a new output directory. The program saves the input, answer image, contiguous packed output cells and optional transcription. On the released card the question asks where the restaurant is. The answer reads **地址是西四南大街60号(近缸瓦市)。** Changing just the first row to the pictured question about business hours gives **营业时间是周一至周日 11:00-21:30。** A released failure asks for hotel type and produces an address. [Original cards and answer pixels](examples/)

For another question, provide `--record record.json` in place of `--input`. Supply the venue and its known fields. For example:

```json
{"q":"这家餐馆在什么地方？","domain":"餐馆","name":"砂锅居（西四店）","card":{"地址":"西四南大街60号(近缸瓦市)","营业时间":"周一至周日 11:00-21:30","评分":"4.6分"}}
```

The renderer uses Noto Sans CJK. Existing input PNGs use a fixed 80-column grid of 32-pixel cells. Each page contains the question, domain, venue name and labelled scalar rows. Supported domains are 景点, 酒店 and 餐馆. The same typeface and printed row prefixes are needed for exact field binding. The decoder writes Chinese from 437 learned printable choices and can insert the original name and value images. Unknown emitted marks appear as U+FFFC in the optional transcript. Inference uses neither a language teacher nor an external service.

For questions that explicitly mention two competing fields, `--focus` enables the separately learned positional selector. On twelve new controlled wordings it passed all swaps on ten. Both remaining wordings failed on every swap. None of the 515 held natural questions activated this option.

### Training and evaluation

```bash
python -m scripts.prepare_chinese_card_reply
scripts/with_ilm_gpu python -m scripts.train_chinese_card_reply
scripts/with_ilm_gpu python -m scripts.evaluate_chinese_card_reply
python -m scripts.audit_chinese_reply
scripts/with_ilm_gpu python -m scripts.probe_chinese_focus
scripts/with_ilm_gpu python -m scripts.learn_chinese_focus
scripts/with_ilm_gpu python -m scripts.evaluate_chinese_focus
python -m scripts.verify_chinese_reply_images
```

Preparation fetches the pinned CrossWOZ data, verifies Git objects and renders the official dialogue splits. Prepared training arrays live in `artifacts/chinese_reply_v1`. The original trainer verifies and reuses existing complete fit receipts. To fit from scratch, use a separate working copy without the generated fit receipts and weights for the desired arms. The optical and matched byte controls share the training schedule and physical writer. Full release inference requires only the model files, the corresponding hash receipts and code. [Measured results and limitations](../../references/chinese_card_reply_result.md) · [Official dataset](https://github.com/thu-coai/CrossWOZ)
