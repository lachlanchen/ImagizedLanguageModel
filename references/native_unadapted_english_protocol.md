# Unadapted source check after failed English transfer

The completed predecessor and full-page curriculum both answer 0/32 normal
English questions correctly. This post-result diagnostic asks whether that
failure already occurs in the unadapted HQ source. It is not a new training
arm, a replacement for the completed observations or an official benchmark.

Use the already downloaded and authenticated Show-o2-1.5B-HQ checkpoint,
the same original source code, BF16 body, FP32 native VAE and cached incremental
readout. Add no adapters and run no optimizer. Preserve all eighty prospective
English input pages/conditions, full original geometry, per-passage seeds,
generic image instruction, top-one decoding and 96-decision cap. Read only the
input index during inference; keep annotated references outside the model.
Check source weights against the existing pinned hash. Do not acquire another
model or alter source code before this cheaper diagnostic is complete.

Retain every source response and report exact stopped alternatives, both-correct
pairs and both erasure controls using the existing English scorer functions.
The original text-supervised and dense-supervised models remain distinct
observations. A source success would identify adaptation-related regression;
a source failure would leave initialization, geometry and visual-language
alignment among the unresolved explanations. Neither observation alone proves
a unique cause or establishes an independent image-answer ILM.
