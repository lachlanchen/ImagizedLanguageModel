# Reproduce the failed source-feature writing comparison

The [measured result](../references/source_canvas_result.md) is unqualified:
both arms fail every exact unseen QA and copying check. The commands below
reproduce a cached-feature diagnostic; they do not launch a usable chatbot.

Run from the repository root with the existing verified shared installations.
Use a fresh output path. Preparation authenticates all 3,257 admitted training
pairs and all 224 development inputs, then encodes complete reference images.
The previous eighteen exclusions remain in the receipt. No source weight
download or duplication is needed.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4 \
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
scripts/compare_source_canvas.py --stage prepare \
--run artifacts/source_canvas_reproduction
```

Check shared GPU and host memory before training. The runner requires physical
GPU0, deterministic operations and the project lock. It loads the 11 GiB frozen
feature cache into host memory, checks finite gradients, exact resume and the
largest input batch, then trains the two matched arms sequentially. Each arm
has 3,257 updates and four complete data passes. Final cached assessment writes
every full canvas and its predicted continuous coordinates without opening
reference images.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 \
MKL_NUM_THREADS=4 PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"$HOME/.local/bin/with-current-nvidia" \
artifacts/external/mixar_env_py312/bin/python scripts/compare_source_canvas.py \
--stage run --run artifacts/source_canvas_reproduction
```

After the model process exits, verify and score on CPU. Verification reconstructs
every generated PNG from its saved coordinates, checks all optimizer updates
and data draws, and independently checks pixel metrics. The original run had
no interrupted/replayed production updates. An interrupted run retains its
checkpoints and journals; the verifier rejects duplicate update histories
rather than silently claiming an uninterrupted run.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4 \
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
scripts/compare_source_canvas.py --stage verify \
--run artifacts/source_canvas_reproduction
```

The completed local run is `artifacts/source_canvas_v1`. Each arm contains
`checkpoint.pt`, `checkpoint_previous.pt`, `training.json`, `assessment.json`,
raw PNGs, predicted coordinates and update journals. `verification.json`
contains all scores. The public evidence folder retains all 512 unchanged raw
canvases, accepted reference images and fixed-order overview sheets. Model
weights and feature files stay local.

The exporter can create a separate, fresh evidence directory:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. \
artifacts/external/mixar_env_py312/bin/python scripts/export_source_canvas.py \
--run artifacts/source_canvas_reproduction \
--output artifacts/source_canvas_reproduction_public
```

Inference here consumes frozen per-ink-unit descriptors and a 52-state summary.
Those descriptors are nine-point pooled visual features, not full source grids
or native KV history. The source inputs and cached features belong to an
externally pretrained Qwen3-VL observer. A vocabulary-free writing head does
not remove that inherited pretraining. Actual PNG-input replay and a clean
deployed inference bundle are not qualified for this failed treatment.
