# Complete shaped pages for preference-source preparation

The [completed cell inventory](chinese_preference_raster_result_20261116.md)
finds unsupported forms, four clipped isolated renders and formatting collapsed
to blank in 10,047 pairs. This preflight changes only the new source renderer:
shape complete text with native Pango, retain paragraphs and direction, measure
full ink and logical bounds, and draw an uncropped page. The existing font size
remains 26 pixels. No model, existing source string, old output or acceptance
rule changes. No preference training is admitted.

Use the existing Python3.12 rendering environment, system GLib/GObject/GIO
preloads and Pango1.52.1. Load the existing host binding helper by file, avoiding
the neural visual-package initializer. The primary font is Noto Sans CJK SC;
native contextual fallback is explicit. Record selected run-font descriptions,
matrices and checksums from version-bound serialized layout output and match
every checksum to actual mapped font-file bytes. Do not use the unsafe glyph-run
object traversal encountered during environment probing. Serialized layout is
debug evidence, not a permanent training format.
[Pango layout](https://docs.gtk.org/Pango/class.Layout.html),
[serialization contract](https://docs.gtk.org/Pango/method.Layout.serialize.html).

Keep width1024, padding8 and soft wrap at992 pixels. Preserve hard newlines,
tabs, combining clusters and bidirectional shaping without text normalization
or ellipsis. Set page height from complete logical/ink bounds, rounded up to32;
reject a page above16,777,216pixels or wider than the registered canvas. Reject
unknown font glyphs before emitting an image. Never substitute guessed forms or
truncate source text. Compare each support page with an expanded rendering:
require exact crop equality, zero outside ink and a protected white frame.
[Unknown-glyph check](https://docs.gtk.org/Pango/method.Layout.get_unknown_glyphs_count.html),
[ink bounds](https://docs.gtk.org/Pango/method.Layout.get_pixel_extents.html).

Preserve RGB source pixels. A separate binary derivative uses the existing
arithmetic-RGB-mean threshold128; its loss of color and legibility remain
unqualified. Do not describe binarization as lossless. For both arrays, save a
row-major32-by-32 tile stream and require exact reconstruction and PNG equality.
These tiles can cut across characters; they contain no lexical IDs. RGB source
support does not establish that the current binary writer can model RGB pages.

Select all registered real support cases from prospective training rows only,
before rendering. Include first occurrences of bar, three combining accents,
tab, carriage return, paragraph break, Arabic, Devanagari, emoji, joiner,
nonjoiner and replacement character. Add the longest prompt/chosen/rejected,
the row with the most distinct unsupported forms from the completed inventory,
and the row with the most source line feeds. Resolve ties by original row order;
deduplicate rows while retaining every selection role. Render all four fields
of every selected row, including empty fields. Keep source replacement characters
without guessing their intended content. This is a support cohort, not a
qualification of all10,735pairs or their machine-derived preference labels.

Fixed controls require newline/space layouts to differ, the composed/decomposed
acute fixture to render identically while retaining distinct source hashes,
the bar to fit its expanded reference, right-to-left shaping to appear, an
unassigned font glyph to be refused and an incomplete tile stream to be refused.
Save every outcome. If a source field has unknown glyphs or exceeds the bounds,
record its rejection and continue through the fixed cohort. Any source rejection
fails this preflight; do not omit the field or retune the cohort.

Admit at48GiB available host RAM; retain24GiB. Use one low-priority CPU process,
single numeric-library threads, no GPU/neural imports, at most120seconds and
1GiB peak RSS. Disable core dumping for this process only. Preserve environment
probe failures, partial journals and outputs; never restart because observation
expires. The original training owner and115bound sources remain unchanged.

If the source preflight passes, a later registered full-corpus preparation and
actual visual/semantic qualification are still required. Any failure needs a
failure-specific decision before another run. No outcome launches a student,
qualifies a reward, substitutes a smaller corpus or changes the active serial
white-target comparison.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libglib-2.0.so.0:/usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0:/usr/lib/x86_64-linux-gnu/libgio-2.0.so.0 \
  artifacts/external/mixar_env_py312/bin/python scripts/preflight_shaped_preference_pages.py \
  --output artifacts/shaped_preference_page_preflight_v1
```
