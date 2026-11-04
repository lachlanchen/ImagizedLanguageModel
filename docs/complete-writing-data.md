# Complete Chinese and English writing images

The new host renderer preserves ordinary spaces, punctuation and full ink
bounds. It produces a binary 32-pixel-high strip with width padded to native
64-pixel image blocks. The font remains fixed at 24 pixels; no example is
cropped, shrunk or replaced by a missing-glyph box. Unsupported layout or
overflow raises an explicit error. The image actor can also receive a PNG
directly, without using a font or Unicode at inference.

This adapter addresses measured data problems. The supplied release renderer
clips ink and overflows on many complete training strings. The earlier frozen
cell helper deliberately rejects spaces, and therefore rejects every prepared
Dolly training prompt. The new implementation leaves V61/V62's source files,
image cache and registered comparison unchanged. A new font layout is not
evidence of better language understanding.

The candidate training cache is at
`data/general_instructions/complete_lines_train_20261105/`. It contains:

| Source | Complete accepted examples | Explicitly rejected |
| --- | ---: | ---: |
| CMRC Chinese reading | 3,193 | 18 |
| Dolly English instructions | 9,783 | 2,287 |

All eight Dolly categories remain represented. Rejections preserve IDs and
reasons: 2,192 sources exceed complete-image width, 95 exceed the vertical
ink area across both datasets, and 18 require a separate control/multiline
layout. No source text is truncated. Every accepted packed image has an exact
pack/unpack check and a file hash. Total packed files occupy about 341 MB.
Preparation takes 87.0 seconds on CPU, with no model call. Only original
training splits are used; development and sealed tests remain untouched.

`CompleteLineImageData` reads these files without Pango, source text or fonts:

```python
from ilm.visual_lm.complete_line_data import CompleteLineImageData

data = CompleteLineImageData("data/general_instructions/complete_lines_train_20261105")
pixels, prompt_blocks = data[0]
# pixels: [1, 1, 32, native_width], white +1 / ink -1
# prompt_blocks: spatial boundary for answer-only training loss
```

The dataset exposes physical pixels and a supervised-loss boundary. It
returns no character IDs, answer text, category or source-answer location.
This is a candidate cache for a later registered training experiment, not
data already adopted by V62 or a general-language benchmark result.

The renderer runs in the existing Python 3.12 environment with the verified
system Cairo/GI bindings. For example, from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libglib-2.0.so.0:/usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0:/usr/lib/x86_64-linux-gnu/libgio-2.0.so.0 \
artifacts/external/mixar_env_py312/bin/python scripts/render_complete_writing.py \
  --text 'Explain the origin of 言 in English and Chinese.' --out prompt.png
```

Preparing the candidate cache uses the same environment and
`scripts/prepare_complete_line_training_data.py`. The raw source partitions
retain their CC-BY-SA-4.0 CMRC and CC-BY-SA-3.0 Dolly attribution. The host
font/layout differs from both the old 26-pixel cell grid and released native
font configuration. Current actor weights have not qualified on this new
interface. Each future training comparison needs matching image preparation
and a genuinely unscored final evaluation.
