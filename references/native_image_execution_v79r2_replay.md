# Cross-device replay of the native image execution check

The V79r2 GPU execution completed without optimization. Its first CPU verifier
stopped while loading an authenticated, locally produced checkpoint: PyTorch
2.3's restricted loader rejects the `TorchVersion` metadata class. This is a
verification failure, not a completed independent pass. Preserve that error and
all original producer sources and artifacts.

The supplemental replay authenticates every original artifact before loading it.
Only the locally generated `initial_adapters.pt` requires the trusted pickle
loader; all tensor-only files retain restricted loading. It repeats the original
numerical, causal, gradient, parameter-boundary and stop checks. The small fixed
codec re-encodes the original two emitted images on GPU and must reproduce the
original feedback hashes exactly. Those actual GPU vectors are retained. A CPU
codec replay then compares numerical vectors with maximum absolute tolerance
1e-4. CPU and GPU convolution byte identity is not required or claimed.

No inherited 8B model is loaded, no parameters are updated, and no output is
rescored as language. The original report remains unchanged. The supplemental
receipt distinguishes retained execution evidence from independently replayed
codec computation. Passing is permission to attempt training, not evidence of
language capability.
