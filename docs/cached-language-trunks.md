# Reusing cached transformer weights through a visual interface

An independent image-language actor can potentially reuse contextual weights
from a text-pretrained transformer while learning new image input/output
interfaces. External pretraining must remain attributed. Removing vocabulary
matrices does not establish visual understanding or preserve the original
language model's capabilities automatically.

The local weight-source implementation now reads the existing Qwen3 GGUF
archives without copying or downloading model weights. Its accessor exposes
only transformer parameters, excluding token embeddings and vocabulary output
matrices. It dequantizes one requested tensor at a time on CPU. No tokenizer,
model forward pass, text generation or shared inference service is used.

## Completed checks

| Cached archive | Retained transformer parameters | Calculated BF16 trunk storage | Excluded stored vocabulary matrices |
| --- | ---: | ---: | ---: |
| Qwen3 1.7B | 1,409,410,048 | 2.625 GiB | 2 |
| Qwen3 4B Q4_K_M | 3,633,511,936 | 6.768 GiB | 1 |

Every retained tensor name and shape matches the installed Qwen3 architecture
constructed on the meta device with zero vocabulary rows: 309 tensors for
the smaller trunk and 397 for the larger. The actual final norm, first query
projection and final feed-forward down projection are dequantized and checked
for finite values and exact expected shapes in each archive. Vocabulary access
is explicitly rejected. This is architectural compatibility and bounded real
weight verification; no full forward-equivalence or language test is claimed.
Two tests on the real cached archives independently verify every retained
parameter name against llama.cpp's official naming map; both pass.

All three inspected model blobs are authenticated against their local
manifests. The third, Qwen3-VL 8B, contains 576,388,336 stored vision-tensor
entries alongside its text trunk. That original inspection inventories it
only: the plain-Qwen3 accessor deliberately rejects that architecture.
A separate [legacy visual-language accessor](cached-visual-language.md) now
verifies the distinct vision/deep-stack/position mapping and executes the
real image encoder on CPU. It remains a preparation result, not a new actor.

Stored scalar totals can count duplicated tied matrices. In particular, the
1.7B archive contains separate stored embedding/output tensors; summing the
entire archive is not its original unique parameter count. Retained trunk
counts above exclude both.

The inspection takes 65.4 seconds on CPU, including complete model-file hash
checks, with 1,194 MiB peak process RSS. No GPU allocation, package installation,
training-environment modification or second model-generation job occurs.

## Use the weight source

Reuse the existing GGUF reader checkout, without installing into the active
training environment:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
artifacts/external/mixar_env_py312/bin/python \
  scripts/inspect_cached_language_trunks.py --out artifacts/trunk_inventory_reproduction
```

The default archive and reader locations are under `../LocalLLM`; use
`--ollama-root` and `--gguf-source` for other existing installations. The
completed report and complete tensor inventories are at
`artifacts/cached_language_trunks_20261105/`. A compact
[receipt](../references/evidence/cached_language_trunks.json) pins the blobs,
source code, measured inspection resources and exact checks.

`QwenTrunkArchive` in `ilm/visual_lm/gguf_language_trunk.py` provides
`configuration()`, `shapes()` and `read(parameter_name)`. `read` returns an
owning FP32 NumPy array for one retained transformer weight. The class is an
offline weight source, not a deployed image-language actor.

## What still has to work

A candidate actor would need a trained physical-image encoder, contextual
adaptation, a coherent image writer, stopping and independent image-only
deployment. Its image features must connect to useful contextual states;
that connection is not supplied by deleting the vocabulary. Source vocabulary
tables and an external language-model call must not enter the deployed actor.

The table is weight storage, not training VRAM. Reusing V63's FP32 parameter,
gradient, two Adam moments and EMA convention would require about **26.25 GiB**
for the smaller trunk alone and **67.68 GiB** for the larger, before activations
or image interfaces. That convention cannot fit either on a 24-GiB card.
Lower-precision/quantized optimizer states or parameter-efficient adaptation
are candidates requiring a real preflight; no fit or training-speed claim is
made here. The installed Transformers 4.57.3 GGUF tensor path dequantizes
weights, so the on-disk Q4 size must not be used as a dense-training estimate.

V63's matched image-language comparison remains the current GPU experiment.
Its sources and data are unchanged. Adoption of a larger cached trunk requires
a separately specified comparison, a full-model resource/causality/resume
preflight, autonomous image answers, question controls and a final evaluation.
The broader goal remains useful Chinese/English and historical writing, not
merely successful weight import.

Primary implementation references: [Qwen3-1.7B configuration](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json),
[Qwen3-VL-8B configuration](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json),
[installed-version Transformers GGUF importer](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/modeling_gguf_pytorch_utils.py)
and [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
