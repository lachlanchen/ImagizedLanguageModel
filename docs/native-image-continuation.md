# Native-context image-language training

The current full-corpus image writer is not a usable language model. It gives
2/128 exact Chinese development answers and 0/12 exact new copy answers.
Retaining full contextual history is the next tested design direction; it has
not yet established language capability. The [diagnostic results](../references/native_context_v80_result.md)
explain the distinction between source-reader content, glyph reconstruction and
actual autonomous image answers.

The new architecture accepts prompt pixels, retains the frozen native visual
and contextual body, and learns continuous image-history and writing layers.
Each emitted block is decoded to pixels and re-encoded before continuing.
This keeps the actual generated image in the feedback loop. It uses externally
pretrained weights and has 7.583B complete parameters, including 46.900M
trainable interface parameters. No tokenizer, OCR or teacher service is needed
by the deployed actor. These properties alone do not guarantee correct answers.

The [declared training protocol](../references/native_chinese_v80_protocol.md)
uses all 3,193 existing Chinese training records, batch four, 40,000 updates and
a bounded 32-prompt RAM pool. The original data, codec, numeric controls and
inherited weights are authenticated local prerequisites. The scripts refuse
missing or changed inputs rather than downloading substitute models or using
synthetic execution fixtures.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
  scripts/train_native_chinese_v80.py --run artifacts/native_chinese_v80_20261106
```

The same command resumes the latest checkpoint. Source and protocol changes
require a new run. The real-data preflight verifies prompt-memory identity,
finite gradients and exact optimization-state replay before resetting new
parameters for production. It is not a language evaluation.

Outputs stay under the selected run directory:

- `status.json`: current stage and update; `failure.json` records a failed run.
- `preflight/preflight.json`: actual-data execution and resume checks.
- `training/metrics.jsonl`: loss, optimizer and sample counts.
- `training/checkpoint_latest.pt` and `checkpoint_previous.pt`: new parameters,
  optimizer, RNG and replay cursors. Frozen source weights are not duplicated.
- `diagnostics/update_*/`: raw autonomous training-example images, not held-out
  evidence. Short image blocks form a complete generated writing strip.
- `development/`: all 384 final normal/question-erased image responses and scores.
- `verification.json`: independent physical-pixel scoring after the final audit.

After training completes, run `scripts/audit_native_chinese_v80.py --run ...`
with the same GPU environment, followed by the same command with `--verify`
and `CUDA_VISIBLE_DEVICES=''`. The fixed audit reports Chinese gates and English
as untrained transfer. The optional `scripts/continue_native_chinese_v80.py`
waits for the owned training job and performs those two phases sequentially.
Do not launch another training or generation process while this job owns GPU0.
