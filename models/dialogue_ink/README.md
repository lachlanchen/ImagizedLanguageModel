# Visual conversational writer

Give the writer a passage image, a question image and an optional preceding question and answer. It returns a PNG containing the answer. It can select words anywhere in its inputs, generate printed words one glyph at a time and decide when to stop. Both released checkpoints run on CPU. Inference uses the bundled visual weights and requires no tokenizer, OCR service or language-model server.

From the repository root, reproduce a held answer using the direct writer.

```bash
example=references/evidence/dialogue_ink_v1/examples
python -m scripts.answer_dialogue_ink \
  --passage "$example/energy-passage.png" --question "$example/energy-question.png" \
  --history "$example/energy-history.png" --output /tmp/energy-answer.png
cmp /tmp/energy-answer.png references/evidence/dialogue_ink_v1/energy-fraction.png
```

The pictured question asks how much energy incandescent bulbs convert to light. The writer returns **less than 5%**. On the next question about where the remaining energy goes, it produces an incorrect reply. Text can also be submitted with `--passage-text`, `--question-text` and `--history-text`. The interface renders typed input before inference. Use `--model models/dialogue_ink/word_code` to run the alternative word-image spelling model. The supported images use registered 32-pixel cells and visible spaces. Both models were trained on English printed in one face. The generator supports up to 16 words, with 24 glyphs per spelled word. The CLI reports the stop decision and writing actions alongside the PNG. [Experimental results](../../references/dialogue_ink_result.md)

## Reproduce training and evaluation

Use PyTorch with CUDA, safetensors, Pillow, NumPy and PyArrow, and the associated visual reader weights in this repository. Place the pinned CoQA training and validation parquet files in `data/raw/coqa/train.parquet` and `data/raw/coqa/validation.parquet`. The preparation script verifies their SHA-256 checksums. The released partition uses complete printable passages and reserves entire stories. The previous human exchange is supplied on each turn. Consult the original source-specific licensing terms for the underlying passages. [Dataset revision](https://huggingface.co/datasets/stanfordnlp/coqa/tree/0d9e9952f1ef6e5415492d3d84b5873259137e3c) · [CoQA and upstream sources](https://stanfordnlp.github.io/coqa/)

```bash
python -m scripts.prepare_dialogue_ink
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_dialogue_ink --output artifacts/refit-direct
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_dialogue_ink --word-code --output artifacts/refit-word-code
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_dialogue_ink --model artifacts/refit-direct --output artifacts/refit-eval
python -m scripts.audit_dialogue_ink artifacts/refit-eval artifacts/refit-audit
```

GPU energy measurement uses NVML. Set `ILM_NVML_RUNNER` to a compatible NVML environment wrapper when needed. The model JSON files record measured costs, sampling, checkpoint hashes and parameters. The word-code checkpoint also records its word-image pretraining.
