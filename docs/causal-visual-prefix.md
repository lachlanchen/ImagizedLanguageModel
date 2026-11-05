# Causal inputs for a visual language reader

The image reader in an autoregressive actor must see only the visible prompt
and completed output images. A causal mask in the contextual model cannot
remove future answer information that a bidirectional image encoder has
already mixed into earlier features.

One valid construction is

\[
z_X=V(X),\qquad z_i=V(Y_i),\qquad
h_t=F_{\rm causal}(z_X,z_1,\ldots,z_{t-1}).
\]

The whole prompt X is observed. Each completed image block Y_i is encoded
independently, so future blocks cannot change its visual features. A model
could also recompute an encoder on each available prefix or use a genuinely
causal visual encoder; the implementation here prepares separate physical
inputs. The contextual body and learned image writer remain later work.

## Implemented physical interface

`prepare_teacher_forced_prefix` in
`ilm/visual_lm/causal_visual_prefix.py` takes a real training image, its known
prompt boundary and the number of completed answer blocks. It returns an
owning prompt page and owning completed blocks. It reads and validates only
the visible prefix. Changing future pixels to NaNs, or removing all future
pixels, leaves its outputs unchanged. Full target storage and future answer
length are not exposed through the returned object.

The companion `fold_writing_prompt` transforms the existing binary 32-pixel
writing strips into pages at most 1,024 pixels wide. It wraps only after a
wholly white column and pads each row with white pixels. `unfold()` recovers
every original pixel. A strip with no suitable blank column raises an error
instead of cutting ink. This transform uses pixel geometry, without source
strings, Unicode IDs, fonts, OCR or a language alphabet. Native scanned pages
require their own two-dimensional input layout.

All **12,976 prepared Chinese/English training prompts** pass the exact
round trip. None is rejected, and the largest page is 512×1,024 pixels.
Pixel preservation does not establish that the reader will use the new layout
correctly; task-specific training and complete-prompt evaluation remain needed.

## Actual encoder counterexample

The cached Qwen3-VL image encoder receives two 32×128 images with identical
completed left halves and different future right halves. Only the future
pixels change. The preceding two visual vectors change by a maximum absolute
coordinate difference of **0.90193** at the final output, with nonzero changes
at all three intermediate visual feature levels. Encoding the completed
32×64 block independently instead reproduces all four feature sets exactly.

This is one deterministic training-image counterexample to encoding complete
answer pages before causal decoding. The numbers are raw feature differences,
not language scores or a population estimate. V63 uses a different native
causal patch interface and remains unchanged.

The complete audit takes **47.6 seconds on CPU**, with **4,516.0 MiB peak process
RSS**, no CUDA context and no development/sealed examples. Three real-cache
interface tests pass. See the [result](../references/causal_visual_prefix_result.md)
and [receipt](../references/evidence/causal_visual_prefix.json).

## Reproduce

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=. \
artifacts/external/mixar_env_py312/bin/python \
  scripts/verify_causal_visual_prefix.py \
  --out artifacts/causal_prefix_reproduction

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=. \
python -m pytest -q tests/test_causal_visual_prefix.py
```

The audit reuses the existing model store and complete training cache. It checks
the shared-workstation memory limits and frozen V63 source hashes. Each output
directory retains its geometry journal, exact feature/input arrays and report;
use a new directory for another audit. The pure prefix adapter itself needs
neither model weights nor the encoder.
