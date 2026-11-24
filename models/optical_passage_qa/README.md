# Reading English passages from glyph images

The model accepts an English page and a question as images. It marks the interval that answers the question and returns the original ink as a PNG. The released reader has 2,136,181 parameters, including the inherited optical encoder. Inference runs locally on CPU with PyTorch, NumPy, Pillow and safetensors. Typed input also requires fontTools and Noto Sans CJK Regular. [Measured performance](../../references/optical_passage_qa_result.md)

From the repository root, run the released model on two images.

```bash
python -m scripts.answer_optical_passage_qa \
  --passage references/evidence/optical_passage_qa_v1/examples/reserved-correct-page.png \
  --question references/evidence/optical_passage_qa_v1/examples/reserved-correct-1-question.png \
  --output answer.png
```

This produces **Lisbon**. The same passage with the second question yields **1822**. Either input can instead use `--passage-text` or `--question-text`. Typed input is rendered to pixels before inference. For image input, use a row-major grid of monochrome 32 × 32 cells, one cell per glyph, with blank cells between English words. The validated rendering uses the repository’s Noto Sans face. The limits are 192 passage words, 32 question words and a 16-word answer. The model always selects an interval. Unanswerable questions, natural camera images and arbitrary typefaces require separate validation.

### Reproduce the experiment

The data pipeline consumes the official SQuAD 1.1 training and development JSON files. Save the files at `data/raw/visual_semantic_teacher/squad_train.json` and `data/raw/squad_v1.1/dev-v1.1.json`. The script verifies both hashes. The regression check on MiniLM uses revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Its config, tokenizer and model.safetensors belong in `data/raw/visual_semantic_teacher/minilm/`. The released qualification result selects direct supervision. [SQuAD files and CC BY-SA 4.0 attribution](https://rajpurkar.github.io/SQuAD-explorer/) · [Pinned MiniLM release](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)

```bash
python -m scripts.prepare_optical_passage_qa
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_optical_passage_qa \
  --output artifacts/optical_passage_qa_v1/refit
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_optical_passage_qa
```

The evaluator defaults to the released weights. Select an independently fitted reader with `--model PATH` and put its evidence in a new folder with `--output PATH`. The human questions and answers reproduced in the evidence derive from SQuAD. Score both exact typography and normalized answers. The Lean file checks the question-dependence guarantee and the expressive limit of interval copying. [Proof](../../formal/span_list/OpticalPassageQA.lean)
