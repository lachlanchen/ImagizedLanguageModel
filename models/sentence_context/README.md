# Writing a pictured answer with a visible fact and page

Two independently loadable research writers receive ordered 32 × 32 glyph bitmaps for a question and source regions, select a visible fact, and emit an answer image. The drawing catalogue contains 6,346 bitmap entries including reserved cells. The writer can also copy a pictured input cell. Typed input uses the training renderer. Scanned text requires externally aligned glyph cells. Unicode transcription is optional after generation. Both writers scored below the earlier released system on complete replies to unseen questions. [Retained release](../fact_span_ranking/README.md) · [Full scores](../../references/visual_sentence_context_result.md)

From the repository root, use a fresh output directory for each run.

~~~bash
python -m scripts.write_sentence_context --arm surround \
  --input models/sentence_context/examples/statements/input.json \
  --output /tmp/ilm-statement-context
python -m scripts.write_sentence_context --arm direct \
  --input models/sentence_context/examples/exam/input.json \
  --output /tmp/ilm-chinese-direct
~~~

Outputs include `answer.png`, `fact.png` and a receipt recording stop, lengths, selected location and physical hashes. The two pictured cases and four requests about historical forms of 言 are provided with the release. The CPU is the default. Use `CUDA_VISIBLE_DEVICES=0` and `--device cuda` for GPU inference. Use `--question` and one or more `--passage` arguments to render text. The writer permits up to 1,024 decisions by default, including stop. Its checkpoints depend on the released fact reader, whole-span ranker and optical inventory. [Independent CPU replay](../../references/evidence/sentence_context_v1/cpu-release.json)

## Reproducing the comparison

Prepare the original fact-channel training split and the whole-span model according to their releases. The supplementary corpus preparation uses the pinned Dolly partition, DuReader, bilingual Alpaca and the training-derived expanded bitmap catalogue. Their local digests, admission counts and exclusion rules are in the references below. Run with the repository Python environment containing PyTorch, safetensors, NumPy, Pillow, RapidFuzz and the configured rendering fonts. The optional visual-model benchmarking and model-based review also need the locally configured Qwen weights. The inference commands above use only the ILM.

~~~bash
python -m scripts.prepare_sentence_context
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_sentence_context prepare
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_sentence_context direct surround
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_sentence_context
CUDA_VISIBLE_DEVICES=0 python -m scripts.probe_sentence_context
CUDA_VISIBLE_DEVICES=0 python -m scripts.probe_sentence_assignment
CUDA_VISIBLE_DEVICES=0 python -m scripts.probe_sentence_evidence
python -m scripts.verify_sentence_context_release
~~~

Training verifies existing released fits and resumes from private checkpoints. For a fresh timing run, use a separate checkout with a fresh private training work directory and empty direct/surround fit directories. Keep the released input models and data manifests. [Matched training protocol](../../references/visual_sentence_context_protocol.md) · [Optical corpus](../../references/evidence/sentence_context_v1/optical.json) · [Matched batches and training exposure](direct/fit.json)
