# Pinned MIXAR baseline: exploratory outcome

Date: 2026-11-05. External author checkpoint, not a locally trained ILM result.
The six prompts, decoding rule and maximum length were specified before the
run in [`mixar477_baseline_plan.md`](mixar477_baseline_plan.md).

The 477,145,984-parameter checkpoint executes image-in/image-out inference on
one RTX 4090 D using FP32 weights. Six prompts generate at most 11 two-patch
blocks before all streams meet the fixed blank-block stopping rule. Measured
generation takes 0.577 seconds and peaks at 1.798 GiB allocated CUDA memory.
The timing includes prompt prefill and generation, excluding weight loading,
rendering and external pretraining. The runtime receives image tensors and
spatial masks, with no tokenizer, OCR, output repair or external model call.

The outputs include readable English spans, incorrect factual continuations,
fragmented CJK and blank endings. The France-capital prompt does not yield a
clear correct answer. The water-freezing prompt continues with an unrelated
phrase about homes and businesses. The Chinese origin prompts do not give
usable explanations. This is a qualitative demonstration check, not an
accuracy benchmark; no statistical correctness claim follows from six prompts.

All prompts and unmodified generated strips are retained in
[`evidence/mixar477_baseline_20261105/`](evidence/mixar477_baseline_20261105/).
For example, the fixed Chinese-capital case is shown below; every case remains
available, including blank and incomplete outputs.

![Chinese-capital prompt](evidence/mixar477_baseline_20261105/case_03_prompt.png)

![Actual generated continuation](evidence/mixar477_baseline_20261105/case_03_generated.png)

The release revision is `c5d5bcfbad2f290296fd83810558f396c6a6333a`, with weight
SHA-256 `be5706028246c147998a141110fcc29c9ce2026623b31bd85d20ed0916659af8`.
The weights reside once in the shared Hugging Face cache. The author model
card specifies Apache-2.0. See the [author release](https://huggingface.co/TYTTYTTYT/llama-pixar-477m-p32-stage2)
and [versioned paper](https://arxiv.org/abs/2604.11575v2) for external provenance.
Its foundation training is not counted as local single-GPU training.

## Reproducible runtime

The working project environment reuses an existing Python 3.12.1 / Torch
2.3.0+cu121 installation through `venv --system-site-packages`. Only its local
packages add Transformers 4.57.3, pixar-render 0.5.1 and ManimPango 0.6.1.
Torch and CUDA installations are not duplicated. The renderer uses the pinned
release's bundled fonts and configuration.

System Cairo 1.18 and PyGObject 3.48.2 bindings provide the native renderer.
On this Ubuntu 24.04 host the missing `gi._gi_cairo` extension was extracted
from the matching `python3-gi-cairo=3.48.2-1` package into
`artifacts/external/native_bindings/`, without installing a system package.
Per-process system GLib preloads keep those bindings consistent with the
Python environment. The superseded Python 3.11 project environment was removed.
Shared Conda environment histories remain unchanged.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libglib-2.0.so.0:/usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0:/usr/lib/x86_64-linux-gnu/libgio-2.0.so.0 \
artifacts/external/mixar_env_py312/bin/python scripts/run_mixar_pixel_baseline.py \
  --snapshot /path/to/pinned/snapshot --device cuda:0
```

The runner validates the weight hash, explicitly imports native bindings and
requires Cairo's foreign-type bridge before importing the renderer. Its
automatic Conda-install fallback is never used by the completed run. All
framework versions, source hashes, rendering-config hash and actual tensors
are recorded under `artifacts/external/mixar477_baseline_20261105/`.

The checkpoint remains useful as an external comparison or adaptation source.
These outputs do not establish a high-quality language model or justify
relabeling external pretraining as the project's contribution. Subsequent
training must demonstrate coherent visible answers on held-out tasks.
