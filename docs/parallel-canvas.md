# Whole-canvas image writing experiment

This implementation replaces the failed image-history continuation interface
with a decoder that predicts one complete 32-by-2048 answer image. It accepts
prompt-only visual memory, predicts 64 continuous tile coordinates jointly,
and decodes every tile through the fixed image codec. Generation has no text
vocabulary, reference image, source crop, teacher service or glyph lookup.

The current checkpoint fits seen answers but fails unseen requests. Its
[result](../references/parallel_canvas_result.md) reports 26/32 exact training
canvases, 0/12 new copies and 0/128 unseen Chinese answers. It is an experiment,
not a usable general language model. The full deployed runtime has 7.552B
externally pretrained parameters; the 16.102M trained decoder is only one part.

The [prospective protocol](../references/parallel_canvas_writer_protocol.md)
defines the model, training schedule, full cohorts and failed decision rules.
The [model implementation](../ilm/visual_lm/parallel_canvas.py) and
[training/evaluation CLI](../scripts/run_parallel_canvas.py) contain the actual
forward, backward, checkpoint, pixel inference and verification paths.

## Reproduce in the existing checkout

Use the shared source/codec installations and authenticated data caches already
present on this workstation. They are identified by hash in the protocol and
receipts. Do not duplicate model weights. Python is the existing 3.12 runtime
at `artifacts/external/mixar_env_py312/bin/python`; use it without resolving its
symlink. Commands below run from the repository root.

CPU preparation must use a new output directory:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python scripts/run_parallel_canvas.py \
  --stage prepare --run artifacts/parallel_canvas_reproduction
```

Run training and assessment sequentially on an otherwise free physical GPU0.
The installed `with-current-nvidia` wrapper selects userspace libraries matching
the current driver. A different machine should use its own correctly matched
driver installation.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. \
  /home/lachlan/.local/bin/with-current-nvidia \
  artifacts/external/mixar_env_py312/bin/python scripts/run_parallel_canvas.py \
  --stage train --run artifacts/parallel_canvas_reproduction

# After the training process exits, repeat the same environment with:
# --stage evaluate --run artifacts/parallel_canvas_reproduction
```

Training exercises real finite gradients and exact optimizer/sampler resume,
then resets the preflight before 2,000 production updates. Production resumes
from the current checkpoint. Any updates beyond the retained checkpoint are
archived and explicitly counted as replayed physical work. Earlier checkpoints
and failed outputs remain preserved. There is no development-selected model.

CPU verification reconstructs all 332 saved canvases from their predicted
coordinates and independently scores them against all accepted references:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python scripts/run_parallel_canvas.py \
  --stage verify --run artifacts/parallel_canvas_reproduction
```

## Infer from a physical PNG

The inference stage loads `model_final.pt`, the fixed source body, numeric
controls and codec. It does not load the prepared training records or caches.
Input image dimensions must be multiples of 32. The existing host adapters
can render ordinary text to a prompt image; no renderer runs inside the actor.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. \
  /home/lachlan/.local/bin/with-current-nvidia \
  artifacts/external/mixar_env_py312/bin/python scripts/run_parallel_canvas.py \
  --stage infer --run artifacts/parallel_canvas_v1 \
  --images references/evidence/parallel_canvas_result/inputs/training_00.png \
  --output artifacts/canvas_png_example
```

Use a fresh output directory. Each input produces an unchanged `answer.png`,
predicted `coordinates.pt` and header. The complete fixed canvas is retained,
including blank positions and erroneous marks. It has no learned stream-stop
event and does not promise arbitrary-length output. The example above is a
seen question, which must not be presented as new-question generalization.

The fresh-process audit replays all 32 training and twelve copy inputs with
byte-identical canvases and identical reader-memory hashes. A Python file-open
audit rejects the original training/data/cache paths. This is an execution
boundary check, not a complete operating-system sandbox or a language score.

## Evidence and costs

All [raw outputs and accepted reference images](../references/evidence/parallel_canvas_result/README.md)
remain available, with original bytes bound by the public receipt. Model
weights and coordinate tensors remain local under `artifacts/parallel_canvas_v1`.
The exporter is `scripts/export_parallel_canvas.py`; the separate predecessor
PNG comparison is `scripts/compare_canvas_pixels.py`.

Head training takes 72.025 seconds at 0.411 GiB peak allocated GPU memory,
including checkpoint I/O and excluding prior feature preparation and external
pretraining. Actual image inference retains the source body: the 44-call
fresh process peaks at 18.029 GiB allocated memory. These are different cost
boundaries, and no matched-quality efficiency advantage is established.
