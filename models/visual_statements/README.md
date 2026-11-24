# Writing a sentence from pictured evidence

The English sentence writer takes a question PNG and passage PNG and produces an answer PNG. Its two neural stages first read a short fact from the page, then express it using the question. Both operate on glyph images. The optional transcript is generated after the answer pixels. The inference path uses the shared trained book model and the released optical interval reader, and requires no external model service. [Full experiment](../../references/visual_statement_result.md) · [Original output pairs](examples/intervention)

From the repository root in the project Python environment (PyTorch, NumPy, Numba, SciPy, Pillow, PyArrow, fontTools and safetensors), run the exact held railway example on CPU.

```bash
p=models/visual_statements/examples/intervention/locomotive-original
python -m scripts.write_visual_statements \
  --question-image "$p/question.png" --question-cells 23 \
  --passage-image "$p/passage.png" --passage-cells 600 \
  --output /tmp/ilm-railway-answer
```

Open `answer.png` for the reply and `read_fact.png` for the intermediate. In this example the model writes “Matthew Murray designed Salamanca.” Both raw glyph streams are saved as packed bits and `result.json` includes the transcript and both stop flags. Use a new output directory each time. Text prompts can be supplied through `--question` and `--passage`; these are rendered into the same cell format as the PNGs. The model accepts up to 160 question cells and 600 passage cells. Input scans need to be aligned into 32-pixel cells first. The statement training is English.

## Reproduce the fit

Use the project's prepared recurrent corpus and SQuAD data. The preparation stage downloads the pinned QA2D split when absent and verifies both parquet files and reserved examples. The model starts from the published bilingual dynamic writing head. A separate fit writes its own weights and training receipt into the given directory.

```bash
python -m scripts.prepare_visual_statements
python -m scripts.train_visual_statements --run-dir artifacts/statement-refit
```

Use `--run-dir artifacts/statement-refit` with the writing command to load that fitted head. Run `python -m scripts.evaluate_visual_statements` to evaluate the released comparison on one GPU, and `python -m scripts.replay_visual_statements` to reproduce the four printed interventions on CPU. The saved data are under `references/evidence/visual_statements_v1`. [Upstream training](../recurrent_evidence/README.md) · [QA2D source](https://github.com/nli-for-qa/conversion)
