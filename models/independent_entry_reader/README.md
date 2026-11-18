# Read a pictured dictionary request

The shared-update reader writes a complete sentence image from a pictured question and a two-entry card. The included question about 明 elicits the gloss *bright* and a bronze inscription. Both were drawn from the visible evidence card. Run from the repository root with Python, PyTorch, Pillow, NumPy and safetensors installed.

```bash
python -m scripts.infer_independent_entry \
  --question models/independent_entry_reader/examples/gained_question.png \
  --card models/independent_entry_reader/examples/gained_card.png \
  --question-cells 24 --card-cells 74 --output /tmp/ilm-ming
```

The program saves `answer.png`, its packed 32-by-32 glyphs in `answer.bin`, and a stop and hash receipt. For the published failure case, use the two `missed_` inputs, 59 question cells and 63 card cells. The request is about 编 but the model writes the pronunciation and oracle form of 老. All question and card counts exclude the last row’s padding. Select `--arm candidate` to test the alternative learned update or `--arm replay` for the subsequent equal-compute control. The appearance file from `models/pictured_entry_writer` supplies the card delimiters and 64 pictured output literals. No teacher, OCR program or online service is needed for inference. [Exact standalone replays](../../references/evidence/independent_entry_20261119/standalone.json) · [Experiment and construction limits](../../references/independent_entry_result.md)
