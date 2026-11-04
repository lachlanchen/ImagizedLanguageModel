# Training-only native rendering interface audit

2026-11-05, before new rendering measurements. The preceding goal turn made
progress: it implemented and preflighted V62, started its real single-head
training, and revised/synced the V61 manuscript with actual failed outputs.
V62 remains running; no duplicate GPU job or development inspection is allowed.

Audit the pinned released MIXAR font interface on CPU only. Reuse its bundled
fonts/configuration and the existing Cairo/GI/Pango bindings. Never invoke the
rendering library's automatic package/Conda installer. No model weights,
inference, teacher calls or GPU allocation enter this audit.

Measure every training prompt and first answer from the existing 3,211 CMRC
records, plus every flat training prompt and answer from the 12,070 prepared
Dolly records. Do not read the Dolly development or sealed-test splits.
Use full Pango layouts to count unknown glyphs, measure logical/ink extents,
identify vertical ink outside the 32-pixel canvas and horizontal overflow
beyond the released rendering capacity. Record the primary description and
registered bundled families. The borrowed Pango font-run accessor crashes in
this environment during preflight, so do not inspect it or claim that a
registered family is the family selected for a particular glyph. Unknown-glyph
counts still measure the actual complete layout. These are rendering
measurements, not language scores.

The native renderer's bounded separator calculation can conceal long-source
clipping. Determine required width from the uncropped layout before invoking
it, and reject overflow rather than silently truncating. Render only the
first four records of each source as PNG inspection evidence when safe. Keep
canonical Noto renders alongside them for comparison; different pixel images
are expected and do not imply that either font causes language failure.

Record source/config/package/font hashes, dataset hashes, software versions,
fixed sample IDs, exact issue counts and a conservative safe-layout fraction.
Preserve all source data and the V62 cache/optimizer. Any future native-font
training requires a separate registered data and evaluation protocol. Do not
adopt a renderer or claim better language based on this audit alone.
