# Chinese page-conditioned image writer

Give the model a pictured Chinese passage and a pictured question. It generates a horizontal PNG, one 32 × 32 glyph at a time, until its own stop decision or the 48-glyph limit. Typed input is rendered before inference. The page writer runs without a second reader, a tokenizer or a remote service at inference. The underlying Chinese language weights were pretrained on text.

From the repository with the trained checkpoint present, run

```bash
python -m scripts.respond_pointing_glyphs \
  --model artifacts/page_writer_requests_v1/page \
  --passage-text '《江岸》由林青作词，1998年出版。' \
  --question-text '请介绍《江岸》的作词人，并指出同一段落写的是哪一年。' \
  --output answer.png
```

The standalone archive includes the weights, inference modules and measured input examples. Extract it and run the following command from the extracted directory with Python, PyTorch, Transformers, safetensors, NumPy and Pillow installed.

```bash
python -m scripts.respond_pointing_glyphs --model writer \
  --passage examples/passage.png --question examples/question.png --output answer.png
```

Use black ink on a white background with one horizontal row of 32-pixel square cells. Typed input also requires fontTools and the Noto CJK font specified by the CLI. The decoder accepts up to 1,024 positions including the page, two copies of the question, four separators and the output allowance. The writing repertoire combines a fixed bank of bitmaps with images on the source page.

### Reproducing the comparison

Training uses screened local DuIE passages, the existing factual writer initialization and the project’s source-split records. The raw-page writer and the factual-fields control receive the same questions, target glyphs and optimization schedule. From the prepared project run

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_page_evidence_writer --run-dir artifacts/new-page-writer
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_page_evidence_writer --run-dir artifacts/new-page-writer
CUDA_VISIBLE_DEVICES=0 python -m scripts.test_page_writer_transfer --run-dir artifacts/new-page-writer
```

To reproduce the question-wording comparison, run the familiar-instruction check before fitting the varied writer.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.probe_page_writer_wording --run-dir artifacts/new-page-writer
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_page_writer_requests --base-run artifacts/new-page-writer --run-dir artifacts/new-requests
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_page_writer_requests --base-run artifacts/new-page-writer --run-dir artifacts/new-requests
CUDA_VISIBLE_DEVICES=0 python -m scripts.test_page_writer_transfer --run-dir artifacts/new-requests --adapted-only
```

Use a new directory for each fit. Each run saves the exact corpus fingerprints, both checkpoints, physical output streams, stopping decisions and per-task scores. [Protocol](../../references/page_evidence_writer_protocol.md) · [Measured results](../../references/page_evidence_writer_result.md)

### Historical glyphs

The one-instruction checkpoint in `artifacts/page_evidence_writer_v1/page` returned both attested forms of 言 in oracle-then-bronze and bronze-then-oracle order on the known source card. The varied-instruction checkpoint is stronger on new date requests and failed both 言 prompts. The standalone packages contain the original demonstration page and both pictured requests. From either archive, run

```bash
python -m scripts.respond_pointing_glyphs --model writer \
  --passage examples/yan_oracle_page.png --question examples/yan_oracle_question.png --output yan.png
```

The oracle form is J04919 and the bronze form is B02976. [Original output from the one-instruction checkpoint](../../references/evidence/page_evidence_writer_20261124/yan/oracle_one.png) · [Measured answers in both orders](../../references/evidence/page_evidence_writer_20261124/inscription_cases.json)
