# Chinese causal glyph writer

This research checkpoint reads two horizontal strips of 32 × 32 black-on-white glyphs. The first gives the passage, the second the question. Each output step chooses a printed bitmap or ends the answer. The 7,990 proposals include training ink and distinct printed forms from the upstream Chinese model. The encoder also accepts other bitmaps. The trained output bank bounds which forms the writer can return. The ordered historical reader remains available for copying attested source images. [Measured results](../../references/causal_glyph_writer_result.md) · [Historical writing](../ordered_page_writer/README.md)

From the repository root, with the locally trained checkpoint available, run

```bash
PYTHONPATH=. python scripts/respond_causal_glyphs.py \
  artifacts/causal_glyph_writer_v1/role_0_passage.png \
  artifacts/causal_glyph_writer_v1/role_0_question.png answer.png
```

The self-contained local delivery includes the same checkpoint, inference code and four pictured examples. Extract `causal-glyph-writer.zip`, change into its directory and run

```bash
PYTHONPATH=. python scripts/respond_causal_glyphs.py \
  examples/role_0_passage.png examples/role_0_question.png answer.png --model matched
```

Python, PyTorch, Transformers, NumPy, Pillow and safetensors are required. The verified environment used Transformers 4.57.6 and PyTorch 2.11. CPU inference is available. Supply `--device cuda` for CUDA. The program writes a PNG and reports the number of glyphs and whether the model chose to stop.

The underlying language prior is the 12-layer UER Chinese GPT-2 trained on CLUECorpusSmall. Its causal blocks were transferred and its final four blocks adapted on DuReader. At inference a continuous map encodes the input and output bitmaps. The archive contains the weights needed for local inference. The training and development corpus remains local and is subject to the original DuReader terms. [Upstream model](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall) · [DuReader](https://aclanthology.org/W18-2605/)

To reproduce from the prepared pictured corpus and font metric, cache the upstream model at revision `c2c0249d8a2731f269414cc3b22dff021f8e07a3` and run the following commands in a fresh experiment directory tree from the repository root.

```bash
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
python scripts/train_causal_glyph_writer.py all
python scripts/evaluate_causal_glyph_writer.py
python scripts/diagnose_causal_glyph_choices.py
python scripts/audit_causal_answer_readings.py
python scripts/compare_causal_first_departure.py
```
