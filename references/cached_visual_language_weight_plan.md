# Cached visual-language weight compatibility

Prepared 5 November 2026, before execution. This is an independent CPU-only
preparation task while the frozen V63 aligned arm trains on GPU 0.

The existing Qwen3-VL 8B Ollama archive is a legacy combined text/vision GGUF.
Its vision projections are split into Q/K/V tensors and its Conv3d kernel has
a flattened output/channel axis. The plain Qwen3 accessor correctly rejects
this architecture. A separate accessor must reverse these storage operations
without exposing vocabulary matrices or silently treating the file as a
current llama.cpp vision projector.

Acceptance scope:

1. Authenticate the existing manifest/blob; reuse the installed GGUF reader.
   No weights are copied, downloaded or installed.
2. Map every retained text and vision tensor into the installed Transformers
   4.57.3 schema on the meta device. Reject unknown tensors and vocabulary
   access. Preserve Q/K/V ordering, Conv3d channel/time order and all three
   deep-stack mergers. Require an explicit source for missing M-RoPE sections.
3. Test the actual first/last vision QKV projections and flattened patch kernel
   against independent CPU contractions on distinct channel/time inputs.
4. Stream all real vision weights into one CPU encoder and run two fixed
   training-image crops, retaining all three intermediate visual features.
   This is an encoding check, not language/image generation or semantic proof.
5. Pin sources, input cache and outputs. Report CPU time/RSS. Verify that no
   CUDA context, teacher-service call, full text model execution or V63 source
   change occurs. Leave GPU 0's trainer and GPU 1's services alone.

The official Ollama migration implementation at commit
`a6d33251620eef7629c73b09e672f2df243eb363`,
[`compatmigrate/qwen3vl.go`](https://github.com/ollama/ollama/blob/a6d33251620eef7629c73b09e672f2df243eb363/compatmigrate/qwen3vl.go),
documents the legacy split projections, patch layout and missing M-RoPE
metadata. Installed Transformers model/configuration sources are the target
schema. The complete actor, image writer, training fit and useful independent
language remain outside this compatibility result. Architecture adoption must
follow the completed V63 comparison and a separate measured GPU preflight.
