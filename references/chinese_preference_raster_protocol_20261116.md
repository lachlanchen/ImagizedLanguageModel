# Source-raster inventory for the prepared Chinese preference collection

The [new source preparation](image_preference_learning_direction_20261116.md)
preserves 10,735 complete preference pairs. It does not establish that their
symbols and formatting survive the target renderer. This CPU inventory inspects
that new source only. It neither repeats a completed actor diagnosis nor alters
the current white-target experiment.

Bind the exported rows, preparation report, source builder, audit implementation
and existing shared font before execution. Inspect all four writing fields:
system, prompt, chosen and rejected. Count every distinct code point and its
occurrences. Use the current corpus target geometry: Noto Sans CJK SC, face 2,
26 pixels, 32-by-32 cells, baseline y=26 and centered advance, with ink below 128.
No fallback font, resizing, source repair, label change or training is permitted.

For each visible, font-supported code point, draw both the original cell and
an expanded 128-by-128 reference at an integer-translated origin. Require the
reference crop to equal the original mask; count all ink outside the cell and
record frame contact separately. Fail the audit if the reference canvas itself
clips. Unsupported code points receive no invented replacement image. Empty
visible masks and identical raster groups are reported separately. Identical
forms are not automatically semantic errors.

The current cell policy maps whitespace and format controls to blank. Count
ordinary spaces separately from formatting collapsed to that same cell, and
flag combining marks for contextual layout review. Count complete source rows
affected by each issue, allowing overlaps. Report conservative inventory
candidates only when no field has one of those issues, no frame is touched and
the existing structural source check passes. Report proposed partition totals
and complete-answer lengths at 128, 513 and 640 code points. These are inventory
counts, not admitted training cases, full-image qualification or glyph counts.

Preserve the full source, all uncertain labels and every negative observation.
No sequence is truncated or rendered through an unreported fallback. Individual
cell checks cannot certify grapheme shaping, page layout, language correctness,
or the legibility of every complete answer. A deterministic source-only sample
plate may show the physical cells; label it explicitly as source-font evidence.

Run one low-priority CPU process, one thread per numeric library, no GPU or
neural imports. Admit at 48 GiB available host RAM; retain 24 GiB; stop above
120 seconds or 1 GiB peak RSS. Preserve partial files on failure. The original
model process and all 115 bound sources must remain unchanged.

If any unsupported, clipped or formatting-loss cases appear, preserve their
counts and require an explicit source-layout decision before any paired fit.
Do not repair this audit by changing font settings or discarding failed rows.
Even a clean inventory still requires complete-image and semantic-label
qualification, the actual-body likelihood/replay/resource preflight, and a
separately registered training decision after the current serial comparison.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/audit_chinese_preference_raster.py \
  --output artifacts/chinese_preference_raster_v1
```
