# Cached language-backbone feasibility

5 November 2026. V63 production continues unchanged. This bounded CPU check
examines whether already cached weights can support a later independent
image-language actor without retaining a vocabulary input/output interface.
It does not start another training/inference job, change the current Python
environment, install packages, download/duplicate weights or call a shared
model service.

Reuse the existing llama.cpp GGUF reader source in LocalLLM. Authenticate the
actual local Ollama manifests and model blobs. Inventory the cached Qwen3
1.7B/4B and Qwen3-VL 8B tensors, separately counting transformer, vision and
vocabulary tensors. Stored scalar counts may include duplicated tied weights;
do not present their sum as the original model's unique parameter count.

For the two plain Qwen3 archives only, map every causal-transformer tensor to
the installed Qwen3Model names. Exclude token embeddings and output vocabulary
matrices. Construct the corresponding architecture on the meta device with
zero vocabulary rows, and verify every retained name and shape. No model
weights, attention computation or generation are executed in that model.
Dequantize only the final normalization, first-layer query matrix and final
feed-forward down matrix on CPU, one at a time, checking their actual shapes
and finite values. Record source hashes, resource use and complete inventory.

Record dense BF16 trunk storage as a parameter-count calculation, not measured
training VRAM. The installed GGUF import path may unpack quantized tensors;
the blob size is not the memory needed by a trainable dense model. Do not
infer full-model or adapter-training feasibility from header parsing alone.

Removing vocabulary tables does not turn a language model into a visual
language model. A future actor would still need a trained image-to-state
interface, a physical-image writer, stopping, causal rollout learning and
independent language evaluation. External text pretraining must be attributed;
no retained text-token interface or external model call may silently become
part of the deployed actor. Adoption requires a separate protocol and real
GPU preflight after the current matched experiment finishes.
