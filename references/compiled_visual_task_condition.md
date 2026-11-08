# Compiling the fixed visual task condition

The DeepGen candidate now has a CPU compiler for its constant task prompt.
The compiled condition accepts continuous image features and continuous query
vectors. Its runtime payload contains no token IDs, tokenizer, lexical table or
language output head. This is preparation for an image-input/image-output
baseline. It does not establish language learning or remove those components
from a complete generation model.

## What is compiled

The compiler uses the same generic instruction as the current image writer;
the changing question and passage remain in the input page. It evaluates the
released template, tokenizer, embedding rows and multimodal positions once for
a fixed image geometry. Both classifier-free guidance branches retain the
reference image, as in the released wrapper; their constant task text differs.

Let \(C\) contain the constant task vectors with zero image slots, \(M\) the
image-slot mask, \(V(x)\) the live visual features and \(Q\) the live continuous
queries. The runtime assembles

\[
H(x,Q)=\operatorname{concat}\bigl(C[M\leftarrow V(x)],Q\bigr).
\]

The attention mask and three-axis positions are also constant at the selected
geometry. There is no vocabulary decision in this assembly. Gradients must
remain attached to both \(V(x)\) and \(Q\), so the same component can be used
in later training. The runtime rejects a different feature count, image grid,
query shape or a KV cache. Separate compiled conditions are required for
different geometries.

The native wrapper resizes its reference to the requested output dimensions.
At 1024 × 1024, its semantic branch produces a 64 × 64 patch grid merged to
1024 visual features. This geometry choice does not validate resizing a full
page to a shorter answer image; that would need a separate experiment.

## Source and compatibility

The source is [DeepGen at revision
85c2ed257b9406d8531965c68e140dc7fdb81382](https://huggingface.co/deepgenteam/DeepGen-1.0-diffusers/tree/85c2ed257b9406d8531965c68e140dc7fdb81382).
Compilation checks the actual released files and complete weight-file hashes
before reading the small subset of embedding rows it needs. No base Qwen
checkpoint is substituted for the release's bundled VLM.

The checkpoint records Transformers 4.56.1. Installed Transformers 5.14.1 has
a different position-function signature and initializes masked padding
positions differently. The compiler executes the checksum-pinned
[4.56.1 position method](https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)
and the two exact released prompt-preparation methods. It does not change the
shared Python environment. This choice preserves the source version's
positions; numerical equivalence through a fully loaded generation model
still requires an actual replay.

## Running the compiler

First acquire the pinned source using `scripts/acquire_visual_semantic_source.py`.
Use the existing image-generation Python environment, with CUDA hidden:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. \
  ../LocalLLM/.local/image-generation/venv/bin/python \
  scripts/compile_visual_task_condition.py --stage layout \
  --output artifacts/visual_task_layout_1024

env CUDA_VISIBLE_DEVICES= PYTHONPATH=. \
  ../LocalLLM/.local/image-generation/venv/bin/python \
  scripts/compile_visual_task_condition.py --stage vectors \
  --layout artifacts/visual_task_layout_1024 \
  --output artifacts/visual_task_condition_1024
```

The layout folder is an offline compiler artifact and contains token IDs. It
must remain separate from the deployed `runtime/` folder, which contains only
`manifest.json` and `condition.safetensors`. Outputs use fresh directories.

The vector stage compares every assembled field against the exact native
methods using actual released task embeddings and continuous queries, with
two numerical image-feature probes. It also checks feature gradients, invalid
inputs and serialization. These probes validate assembly, not recognition,
reasoning or generated writing. No neural forward or GPU allocation occurs.

## Remaining model work

The full native generation path has not yet been compared with this compiled
path. Removing the complete model's lexical table and output head, loading a
self-contained visual/language body, measuring GPU memory and assessing
answer images remain separate steps. Retained language-model pretraining must
be counted and disclosed. The current registered QA experiment continues
independently; this compiler changes none of its weights, schedule or tests.

## Executed checks

The [compiler receipt](evidence/compiled_visual_task_condition.json) records
successful layout and vector stages at both geometries. All four returned
fields match the pinned native preparation methods exactly for both numerical
feature probes. The checks use 44 actual released embedding rows and the
released continuous queries. Gradients reach both live feature inputs;
invalid geometry, feature length and cache inputs are rejected, and the
serialized conditions reload without tensor changes. CUDA remains
uninitialized. The runtime payloads are 1,540,232 bytes at 512 × 256 and
8,926,864 bytes at 1024 × 1024. No generated answer, model-level vocabulary
removal or language-quality result is reported by these checks.
