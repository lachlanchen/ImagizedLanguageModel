# Compact visual span language model

The model reads and writes 32 × 32 binary ink tiles in reading order. A frozen
7.42M-parameter convolutional codec maps each tile to a continuous vector. An
image encoder groups the visible tiles and learns one to three contextual
updates per span of up to sixteen glyphs. The 28-layer contextual body has
1.409B pretrained parameters from a local Qwen3 1.7B archive. The loader
materializes its contextual layers without vocabulary embeddings or a lexical
output head.

At each generation step, the model predicts the length of the next span. A
conditional rectified flow generates the span's continuous glyph vectors
together. The codec decodes those vectors into binary ink. The encoder reads
those emitted pixels and sends new continuous updates into the contextual
body. A predicted length of zero ends the sequence. Typed input can be
rendered outside the model. This implementation accepts a tile stream and
requires upstream reading-order segmentation for a raw page.

The 12.87M trainable image parameters learn span boundaries, update counts,
feedback vectors, output lengths and glyph flow on complete sequences. A
training-only projection can align image-conditioned contextual states to
offline teacher states. Native source alignment supervises boundaries and
feedback during training. Inference derives both from pixels. The compact
body and codec remain frozen during this fit, so training adapts inherited
weights to visual writing. See the [visual sequence corpus](visual-sequence-training.md)
and [teacher cache](visual-sequence-teacher-states.md).

## Verified interfaces

A CPU check used one complete book record with 50 glyphs and 46 spans. All
five image objectives were finite. Backpropagation through the checkpointed
contextual body produced finite, nonzero gradients in the encoder and writer.
The frozen contextual body and codec acquired no parameter gradients. In a
separate check with the reference length supplied, independently re-encoding
the emitted span reproduced the contextual input bit for bit. The CPU check
took 293 seconds and peaked at 4.71 GiB process RSS. It used untrained image
heads, zero optimizer updates and no cached teacher states.

The verified training schedule covers 13,785 complete sequences in 108 shards.
With a 1,024-position padded batch limit, one pass schedules 8,172 optimizer
batches and 2,380,620 supervised state positions. The fit requires the completed
offline teacher cache. It saves an initial snapshot and four equally spaced
checkpoints by shard exposure, recording each actual optimizer step. A rerun
with the same fit configuration resumes at the last completed shard. The
one-GPU fit and held-out evaluation are pending. See the
[CPU evidence](../references/evidence/compact_visual_span_interface_20261111.json).

## Run the fit

Once the teacher cache is complete and the project GPU is free, run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH=. python scripts/train_compact_visual_span.py
```

The run directory is `artifacts/compact_visual_span_fit_v1`. It contains the
resumable optimizer checkpoint and separate image-head snapshots. A standalone
model pairs the image heads with the verified contextual and codec weights.

The serial pipeline runs teacher extraction, fitting, and fixed-checkpoint
evaluation on GPU0. Pass the PID of the current project worker with
`--after-pid` to queue the pipeline behind it. The wait uses a Linux process
ID and kernel start time and allocates no model. Each GPU stage acquires
the existing project mutex. Teacher and fit attempts resume after a resource
deadline when the attempt saved a new shard or checkpoint.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  python scripts/run_compact_visual_span_pipeline.py --after-pid "$ACTIVE_ILM_PID"
```

The pipeline writes its status and logs under
`artifacts/compact_visual_span_pipeline_v1`. It generates all controls at
the two endpoints and the normal cohort at the intermediate checkpoints.
The resulting review directory contains the original answer PNGs under
opaque names and a readings template.

Evaluate complete answers with learned lengths, stopping and feedback
throughout generation. Preserve the original image stream and score
question dependence, legibility and correctness on held-out groups. See the
[engineering goal](first-imagized-language-model-goal.md).

## Standalone generation and held-out evaluation

Run inference with a gapless PNG grid of 32 × 32 tiles in row-major reading
order. If the last row has padding, give the actual tile count. The runtime
loads the contextual and codec weights plus the specified image heads. It
writes the original generated glyph stream, a display PNG and a record of
length, stopping and pixel feedback.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  python scripts/infer_compact_visual_span.py --image prompt.png --glyph-count 393 \
    --checkpoint path/to/image_heads_step_0000000.pt --output artifacts/compact_inference
```

The example checkpoint name illustrates the snapshot format. Select a trained
step for meaningful answers. An actual fresh CPU process loaded the initial
heads and read a 393-tile book prefix. The untrained length head stopped
immediately, emitting zero tiles. The system-call trace verifies the standalone
input and weight boundary for that invocation. It provides no evidence about
trained writing. See the [preparation and cold interface evidence](../references/evidence/compact_visual_span_evaluation_20261111.json).

The prepared evaluation suite uses 16 paired CMRC contexts, 12 book prefixes
and blanked controls. Its input manifest is in
`artifacts/compact_visual_span_evaluation_v1/inputs/manifest.json`. Use
`--manifest` in place of `--image` and `--glyph-count`. Use `--normal-only` for
intermediate checkpoints. The normal and control routes, blind image review,
scoring and trained cold replay are specified in the
[evaluation protocol](../references/compact_visual_span_evaluation_protocol.md).

The source-matched Qwen3 1.7B text reference has run on the same 44 normal
prompts. It matched eight of 32 QA answers and both questions in two of sixteen
pairs under exact scoring. Its CPU results provide the text baseline for
this cohort. The visual checkpoint scores and the same-GPU cost comparison
are pending. See the [text reference and scoring](../references/compact_visual_span_qwen_reference_protocol.md).
