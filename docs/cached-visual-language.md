# Cached vision-language weights for an independent image actor

The subsequent [complete actor preflight](grounded-image-language.md) now
executes all retained vision/contextual layers and real backward at 17.181 GiB
peak allocated GPU memory, with an explicit precision repair. The CPU-only
inspection below remains a separate source-interface result.

The legacy combined Qwen3-VL 8B archive already available under `../LocalLLM`
can now supply its vision and contextual weights without exposing its language
vocabulary matrices. The new accessor reads the existing GGUF in place and
dequantizes one requested parameter at a time. External pretraining and the
archive's quantization remain part of its provenance.

This provides a verified weight interface and an executed image encoder. A
complete image-language actor still needs a contextual interface, image writer,
training and autonomous answer evaluation. No new actor is adopted by V63.

## What is verified

| Component | Completed evidence |
| --- | --- |
| Contextual body | All 397 retained parameter names/shapes match the installed native model on the meta device, with zero vocabulary rows. No full text-body forward executes. |
| Vision encoder | All 351 native parameters load from the archive and are finite. Its 27 layers encode two fixed 32×64 training-image crops, each producing two 4,096-dimensional vectors and three intermediate visual feature sets. |
| Legacy storage conversions | Separate Q/K/V projections are fused in the correct order. The flattened output/channel axis of the Conv3d kernel is restored. Independent real-weight contractions check the first/last visual projections and distinct channel/time inputs. |
| Repeatability | The second fixed crop reproduces every final/intermediate feature exactly on repeat CPU execution. This is local encoder repeatability, not standalone actor deployment replay. |
| Isolation | No CUDA context, shared teacher-service call, vocabulary lookup or language/image generation occurs. All frozen V63 source bytes remain unchanged. |

The original GGUF contains 459 vision tensors; fusing its split Q/K/V tensors
gives 351 native parameters without changing the scalar count. Together with
the contextual body, the retained source contains **7,522,463,984 parameters**.
Calculated BF16 storage is **14.012 GiB**. This excludes activations, gradients,
optimizer states and a writer; it is not a measured GPU-training fit.

The complete CPU inspection takes **28.9 seconds**, including the full model
blob hash, with **4,534.6 MiB peak process RSS**. Three real-archive tests pass
in the existing test environment. The full encoder executes in the unchanged
Python 3.12.1 / PyTorch 2.3.0 / Transformers 4.57.3 runtime; the separate tests
use the existing Python 3.10 / PyTorch 2.11.0 / Transformers 4.57.6 environment.

The [result note](../references/cached_visual_language_weight_result.md) and
[compact receipt](../references/evidence/cached_visual_language_weights.json)
record source hashes, exact inputs, numerical errors and scope. Full features
and the execution report are under `artifacts/cached_visual_language_20261105/`.

## Reproduce the CPU inspection

The pinned small upstream source documents the legacy layout and its omitted
M-RoPE sections. The script checks its hash; it does not execute that Go code.
No model download or training-environment installation is needed.

```bash
mkdir -p artifacts/vl_weight_reproduction/upstream
curl --fail --location --max-time 30 \
  https://raw.githubusercontent.com/ollama/ollama/a6d33251620eef7629c73b09e672f2df243eb363/compatmigrate/qwen3vl.go \
  -o artifacts/vl_weight_reproduction/upstream/qwen3vl.go

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=. \
artifacts/external/mixar_env_py312/bin/python \
  scripts/inspect_cached_visual_language.py \
  --compat-source artifacts/vl_weight_reproduction/upstream/qwen3vl.go \
  --out artifacts/vl_weight_reproduction
```

Use `--ollama-root`, `--gguf-source`, `--writing` and `--active-run` to point to
the existing model store, reader checkout, pinned complete-writing training
cache and V63 preflight respectively. A completed output report is immutable;
use a new output directory for another inspection. The command needs at least
24 GiB available system RAM and respects the shared workstation swap limit.

In the existing test environment:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=. \
python -m pytest -q tests/test_gguf_visual_language.py
```

`QwenVisualLanguageArchive` in `ilm/visual_lm/gguf_visual_language.py` exposes
`shapes()`, `read(name)` and `configurations(legacy_mrope_sections=...)`.
Native names start with `visual.` or `language_model.`. Vocabulary access and
unknown layouts fail explicitly. This legacy file omits M-RoPE sections, so
the caller must supply an attributed configuration; the checked inspection
uses `[24, 20, 20]` from the pinned upstream compatibility implementation.

The prepared interface reduces a future integration risk. It does not show
that the inherited contextual weights will interpret the intended visual
prompts, that a new writer will generate readable answers, or that joint
adaptation fits one 4090. Full-model execution and memory are now measured by
the linked subsequent preflight; readable writing and autonomous answer quality
still require actual learning and evaluation.

The subsequent [causal-prefix audit](causal-visual-prefix.md) verifies physical
input assembly and demonstrates why future answer images must be excluded
before this bidirectional encoder, even with a causal contextual decoder.
