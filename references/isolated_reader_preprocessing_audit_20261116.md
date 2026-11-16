# Saved image processing explains the small visual budget, not the reading errors

The retained GPU0 teacher log records 402 image encodings, each with 81 visual
positions and 98 text positions. All 401 later requests explicitly reprocess
their prompt because the hybrid/recurrent cache cannot reuse its state. These
observations contradict a simple explanation that images never reached the
vision encoder or that the previous image was silently reused. They do not
verify every internal state or prove correct visual interpretation.

The 288×288 PNG, 16-pixel vision patches and spatial merge factor two give
$(288/(16\cdot2))^2=81$ merged positions. The logged dynamic pixel limits are
8,192–4,194,304; this input lies inside them and already aligns to 32 pixels.
The retained runner therefore agrees with the geometry in the pinned
[dynamic image preprocessing source](https://github.com/ggml-org/llama.cpp/blob/96278e39fc83e1d97c881e34bcec39ac7ea98820/tools/mtmd/mtmd-image.cpp#L932).
This verifies the expected size, not an independent replay of compiled C++.

The pinned Ollama release adds a 1,024-position minimum for four Qwen-VL
architectures but omits `qwen35` from that switch. The saved launch likewise has
no image-minimum override. The C++ warning concerns grounding accuracy, so it
must not be promoted into a demonstrated explanation for isolated-character
OCR failure. A larger visual budget is a concrete hypothesis, not an observed
repair. [Ollama's architecture switch](https://github.com/ollama/ollama/blob/c82ebbd5bfb9ec7d94d3894e9023db0fb224ff50/llm/llama_server.go#L999),
[Qwen projector defaults](https://github.com/ggml-org/llama.cpp/blob/96278e39fc83e1d97c881e34bcec39ac7ea98820/tools/mtmd/clip.cpp#L1491).

The installed runner reports revision `96278e39f`; upstream resolves that to
`96278e39fc83e1d97c881e34bcec39ac7ea98820`. The Ollama v0.32.6 tag resolves to
`c82ebbd5bfb9ec7d94d3894e9023db0fb224ff50`. Retained source files have fixed URLs,
SHA-256 digests and Git blob hashes. The cached C++ loader's Git blob matches
the pinned upstream directory entry. The stripped Go binary provides no VCS
revision; its exact correspondence to release source and a reproducible binary
build remain unverified. Existing nonconsecutive-position warnings are retained,
without assigning them a causal interpretation.

The [machine-readable audit](evidence/isolated_reader_preprocessing_audit_20261116.json)
binds the logs, binaries, source snapshots and geometry. It makes zero model
calls and changes no service, model, prompt, image, frozen reading or score.
The failed reader remains rejected. No third teacher run is launched. Proceed
with the separately registered training-only label audit; do not assume either
this preprocessing hypothesis or a small verifier already solves language.
