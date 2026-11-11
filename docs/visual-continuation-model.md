# Visual continuation with frozen prompt memory

This implementation preserves the imported source reader's prompt computation
and adapts how its contextual layers continue from glyph images. Its real
single-4090 preflight passes through a complete 513-glyph target at 18.123 GiB
peak allocated VRAM. Full language training and evaluation remain pending.
See the [measured result](../references/visual_continuation_preflight_result.md).

## Image boundary and state transition

The source visual reader receives physical prompt pixels and fixed numeric
structural controls. Its language embedding table and lexical output head are
absent. Typed questions can be rasterized before this boundary. Source weights
and offline supervision are inherited; this is adaptation rather than evidence
of language acquisition from images alone.

For each native layer, compute detached prompt memory
\(C_\ell(x)=(K^P_\ell,V^P_\ell)\) once. Start the answer with a trainable
continuous query \(q\). Subsequent answer inputs are the existing image
encoder's affinely denormalized features of preceding glyph images:
\[
 h^0_0=q,\qquad h^0_{t+1}=S(E(I_t)).
\]
The query initially copies the last numeric task-control vector; it contains
no case-specific answer, character identity or lookup table.

The implemented residual is **after** each native layer. With a causal answer
history and immutable prompt memory,
\[
 u^\ell_t=F_\ell(h^\ell_t;C_\ell(x),h^\ell_{<t}),\qquad
 h^{\ell+1}_t=u^\ell_t+g_t\frac{\alpha}{r}B_\ell A_\ell u^\ell_t.
\]
Here \(r=8\), \(\alpha=16\), and all 36 layers have trainable low-rank maps.
During prompt prefill the new residuals are bypassed, equivalent to \(g_t=0\).
For the answer query and continuation, \(g_t=1\). Thus the original prompt
memory stays unchanged while the answer transition can adapt to visual steps.
This retiming is learned; taking the first character of a native unit is not
used as a deployed policy. Its earlier failure remains documented in the
[feedback comparison](../references/glyph_feedback_ceiling_result.md).

The existing neural writer predicts a whole 32-square bitmap and the stop head
predicts an end event. At inference, logits are thresholded at probability 0.5,
and only the actual emitted binary image is re-encoded for the next step. Stop
is measured before emitting another glyph; an empty answer is possible.

## Training and cache correctness

A training sequence contains the query followed by the encoded complete
reference images. The first state predicts the first glyph; the final state
predicts stopping. The causal mask allows each answer query to attend to the
entire prompt and only its own and preceding input positions. The current
prediction's target image is not yet an input.

Only frozen prompt keys/values are cached. Every answer layer is recomputed
with its current residual parameters. Nonreentrant checkpointing discards
intermediate activations, including temporary FP32 contractions of frozen
BF16 source weights. The training cache's `update` operation is a pure
concatenation; recomputation cannot append to or lengthen prompt memory.
Incremental inference separately keeps the generated answer's keys/values.

The preflight uses the existing balanced physical image loss and balanced
stop/continue BCE. It trains the new query/residuals and readout/stop heads,
while source vision/context and the image encoder/decoder remain frozen.
The full fit must register its complete data exposure and optimization recipe
before execution. Reusing old lower-layer answer states would prevent those
layers' new dynamics from being trained and is not supported here.

## Implementation and reproduction

- [Visual continuation module](../ilm/visual_lm/visual_continuation.py) contains
  immutable prompt memory, pure training cache, residual dynamics and actual
  emitted-pixel inference.
- [Native numerical tests](../tests/test_visual_continuation.py) compare
  causality, gradients and cache behavior using real native decoder layers.
- [Full-GPU preflight](../scripts/preflight_visual_continuation.py) selects and
  binds complete training cases, performs four real updates and measures memory.
- [Independent audit](../scripts/audit_visual_continuation_preflight.py) checks
  the saved optimizer, authentic targets, prompt pixels and reported boundaries.

Use the existing shared source weights and Python environment. An existing
run is immutable; registration and its public file refuse overwrite. Reproduction
needs a new run/public registration identity fixed before execution. GPU0 uses
`artifacts/ilm_gpu0.lock`; the registered host/device/pressure limits remain.
The default completed run must not be restarted as pending work.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m pytest -q tests/test_visual_continuation.py
# Commands used for the now-complete, preserved registration and execution:
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/preflight_visual_continuation.py register
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. \
  python scripts/preflight_visual_continuation.py run
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/audit_visual_continuation_preflight.py
```

Preflight weights remain unqualified and must not initialize the full fit.
The existing corpus has substantial QA and book data; the previous short-answer
subset is insufficient exposure for the proposed complete-sequence experiment.
Complete readable answers, paired question dependence, coherent continuation,
natural stopping, retained controls and untouched confirmation decide success.
