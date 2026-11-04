# Correlated native density: feasibility preflight

2026-11-05. V61 and its locked image audit have completed; no language pass is
claimed. Recent research is assessed in `physics_math_routes_20261105.md`.

Implement a small conditional mixture of complete Bernoulli image-block
distributions above the unchanged native pixel-language transformer. Each
expert predicts all 2,048 pixels from the current continuous state. Experts
have no stable character meaning, fixed glyph image, Unicode label or codebook.
The context remains a transformer, rather than a fixed-routing circuit.

Verify normalization by exhaustive small binary cases, single-expert equality
to the native Bernoulli law, whole-form correlations, finite gradients and
candidate scoring. Candidate generation enumerates expert modes and evaluates
their exact mixture probabilities; it is approximate MAP, not a global-mode
guarantee. These unit examples establish mathematical behavior, not language.

Run one real optimizer update on the first training image example with the
complete eight-expert actor, FP32 model/optimizer/EMA storage and BF16 forward,
from the pinned MIXAR release. Use physical GPU 0 only, eager math attention,
gradient checkpointing and the existing environment/weights. Measure actual
parameter count and allocated memory; verify finite nonzero gradients through
the image input, first/last contextual layers, output experts and gate. This
preflight is not a production training run or quality evaluation. No development
examples, OCR labels or source answer offsets enter it.

A later matched training protocol must fix font interface, data, optimizer,
seeds, decoding and held-out tests before any language claim. Do not amend
V61's result or adopt the eight-expert model based on this feasibility check.
