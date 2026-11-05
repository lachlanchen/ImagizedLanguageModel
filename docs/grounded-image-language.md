# Grounded continuous image-language actor

This candidate connects cached Qwen3-VL vision/contextual weights to a new
continuous image writer. The complete model passes all 19 real preflight
checks at **17.181 GiB peak allocated memory** on one RTX 4090 D, including
complete backward with allocated Adam moments. The
[result](../references/grounded_image_actor_preflight_result.md) records the
measurements and two preserved precision failures. Useful language remains
untrained and unqualified.

The model has **7.548B total parameters**: 7.522B inherited and frozen plus
25.58M trainable parameters. These include rank-eight contextual adapters,
two continuous queries, a 19.37M flow writer and a binary stop head. A small
trainable subset does not establish a small total model or comparative
efficiency.

## Image boundary and model

Input is a CPU FP32 grayscale tensor `[1,1,H,W]`, black `-1`, white `+1`, with
dimensions divisible by 32. Prepared writing strips use `fold_writing_prompt`
to wrap at white columns without losing pixels. Padded observed pages retain
their spatial layout. No character segmenter is required by the actor.

Image packing uses the archive's RGB mean/std of 0.5 and repeats still frames
along its two-frame kernel. Spatial coordinates and all three DeepStack
features are retained. Two learned vectors request the next 32-by-64 physical
block; they index no characters or words. Teacher forcing orders states as

```text
observed prompt, query pair 0, answer image 0, query pair 1, answer image 1, ...
```

Answer blocks are independently encoded before the bidirectional vision body.
The causal context prevents a query from seeing its own or future answer
images. Generation rereads completed output pixels through the same path.
The contextual vocabulary module is absent; no tokenizer, Unicode IDs, glyph
bank, OCR or teacher service enters inference.

The conditional flow learns

```text
Y_t = (1-t) U + t Y,       U ~ Normal(0,I)
L_flow = E ||v(Y_t,t,h) - (Y-U)||^2.
```

Explicit midpoint integration generates all pixels jointly, followed by binary
thresholding at zero. A class-balanced binary stop objective is separate from
white-space generation; its sigmoid score is not independently calibrated.
Sampling steps and the stop threshold must be fixed for a learning experiment.

## Precision and execution

Contextual parameters use BF16 storage with FP32 contractions/activations and
decoder-layer checkpointing. Layer-local weight casts avoid retaining a full
FP32 body. The vision encoder computes in FP32. Only writer contractions use
outer BF16 autocast; the entire contextual call disables it, including attention.

All 12,976 training rows are scanned to select the largest context and prompt
cases. Their complete backward, exact optimizer/RNG replay and future-pixel
isolation pass. Cached/full-prefix relative maximum error is 1.21e-5 against
the unchanged 0.02 limit. Six CPU tests check native normalization/patch order,
inverse spatial/time packing, owning storage and arbitrary unencoded patterns.

Use a new output directory to reproduce the complete check. The command
authenticates the existing shared archive, downloads no weights, calls no
model service and uses physical GPU 0 only. It requires an idle GPU 0, at least
24 GiB available host RAM and swap below 75%.

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python \
  scripts/check_grounded_image_actor.py --out artifacts/grounded_image_reproduction
```

The loader dequantizes retained parameters one at a time and omits vocabulary
matrices. The [archive guide](cached-visual-language.md) records the legacy
layout and M-RoPE provenance. Its original quantization remains attributed.

Core interfaces in `ilm/visual_lm/grounded_image_actor.py`:

- `conditions(prompt, completed_blocks)` returns causal query states.
- `training_loss(prompt, targets, generator=..., writer_blocks=16)` computes
  flow/stop losses. Complete target blocks exclude the cache's two white
  termination blocks.
- `generate(prompt, maximum_blocks=128, flow_steps=16, seed=0)` uses cached
  continuation and rereads generated physical blocks.
- `trainable_state()` / `load_trainable_state(...)` save and restore adapters,
  queries, writer and stop head, with the frozen body identified separately.

The preflight's two-update image is noise. Actual writing/instruction training,
independent quality tests, foundation-engine equivalence and final standalone
offline export remain outstanding. The general image-language goal stays open.

The subsequent [offline source check](../references/inherited_context_check_result.md)
matches all eight first-token choices and nine returned reference positions,
with 0.000264-nat mean coarsened distribution difference. Its tokenizer and
temporary source head exist only in that helper, never in the actor. This
small text-only sanity check supports the initialization; full equivalence
and image-language quality remain unproved.

The [writing-stage protocol](../references/grounded_writer_learning_protocol.md)
specifies 10,000 actual updates over Chinese/English image blocks before
instruction learning. `scripts/train_grounded_writer.py` includes checkpoint
resume verification, EMA, deterministic sampling and a fixed final component
evaluation. Reconstruction is a prerequisite output-path test, not a language
answer qualification.

Production is now running in the owned `ilm-grounded-writer` tmux session on
GPU 0. Artifacts and exact-resume evidence are under
`artifacts/grounded_writer_learning_v64_20261105/`; `status.json` records the
live PID/stage, and `report.json` will record completion and the fixed component
decision. Do not infer completion from a console line or loss decrease. Resume
an interrupted stage with the same command/environment as the protocol and
`scripts/train_grounded_writer.py --resume`. Preserve frozen sources and do
not start a second heavy job while the trainer is live.
