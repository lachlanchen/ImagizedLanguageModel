# Full native image-language preflight

2026-11-05. All 477,145,984 model parameters train in the production route.
Four real updates with accumulation two complete on physical GPU 0 with
finite gradients in every parameter, including the native pixel projection,
first/last decoder layers and output projection. Peak allocated memory is
9.910 GiB. FP32 weights/optimizer/EMA use BF16 forward/backward and deterministic
eager math attention. No alternate GPU or external inference service is used.

An interrupted two-plus-two run matches the uninterrupted run exactly for
model weights, EMA, optimizer state, data RNG, Python RNG, NumPy RNG, CPU RNG
and CUDA RNG. The separate full-model causal test flips all answer/future
pixels: every earlier prediction remains bit-identical, maximum error zero.
The runtime reconstructs the actor from configuration and learned weights;
it does not load the original release weights for trained inference.

Torch 2.3 requires explicit device initialization before peak-memory reset;
the trainer sets the selected device before accounting. cuDNN reports rejected
candidate convolution plans and falls back to supported implementations;
deterministic algorithms stay enabled. The exact resumed result confirms no
relaxation was needed. These warnings do not establish language quality.

Receipts and reproducible smoke checkpoints are under
`artifacts/native_pixel_instruction_v61_preflight.json` and
`artifacts/native_pixel_instruction_v61_smoke_{full,resume}/`. Source hashes
match the frozen protocol and actual production trainer. The 1,000-update
run has started under `artifacts/native_pixel_instruction_v61_20261105/training/`.
No new development accuracy is inspected before its final EMA completes.
The Nature-style article remains at the completed V60 evidence until this
new free-writing audit finishes.
