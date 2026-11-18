# Answer a pictured request about a Chinese character

The ordered reader takes two images, a question and an illustrated entry card. The sample card contains 耳, an English gloss, its pronunciation and two historical exemplars. Reading the question determines which fact and historical form appear together in the answer. The 51,776-parameter checkpoint runs on CPU. Its appearance assets come from the preceding pictured-entry writer in the same repository.

From the repository root, run

```bash
python -m scripts.infer_entry_question_reader \
  --question models/entry_question_reader/examples/meaning_question.png \
  --card models/entry_question_reader/examples/meaning_card.png \
  --card-cells 73 --output /tmp/ilm-ear-origin
```

Open `answer.png` in the output directory. This question gives the gloss *ear* and the bronze form B16241. Substitute `sound_question.png` and a fresh output directory to obtain *ěr* and L13682 from the same card. The question `cue_first_question.png` preserves the first request's meaning but reproduces the known placement failure. Inputs use the same 32 × 32 row-major glyph cells and four visible card delimiters as the [earlier writer](../pictured_entry_writer/README.md). Output contains the original packed glyph stream and a stop-and-hash receipt. Dependencies are PyTorch, NumPy, Pillow and safetensors. [Verified CPU examples](../../references/evidence/entry_question_reader_20261119/standalone.json)

The ordered checkpoint and the parameter-matched control are stored here. Training initializes both arms from the preceding visual entry writer and uses the prepared Hanziyuan images. The training and evaluation modules are `scripts.train_entry_question_reader` and `scripts.evaluate_entry_question_reader`. Their registered output locations preserve completed trials. [Data preparation and fixed evaluation](../../references/pictured_entry_request_order_protocol.md) · [Full experimental results](../../references/entry_question_reader_result.md)

To repeat the two fits and their complete evaluations with the prepared project data, use a new directory for the resulting weights and outputs.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_entry_question_study --output /tmp/ilm-order-study
```

The runner verifies the published source and corpus hashes and writes both checkpoints, learning curves, all six generated answer archives and `evidence/evaluation.json`. It expects the prepared `data/pictured_entry_answers_v1` and `data/pictured_entry_request_order_v1` directories on the training workstation. Only the pictured examples and the trained weights are needed for the standalone CPU command above.
