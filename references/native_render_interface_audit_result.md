# Native rendering audit: do not adopt the released font interface unchanged

The CPU-only audit measures all 3,211 CMRC training prompts/first answers and
all 12,070 prepared Dolly training prompts/answers. No development or sealed
test records, model calls or GPU allocations enter it. The pinned configuration
uses Pixeloid Sans at 35.556 physical pixels on a 32-pixel-high canvas, with
the release's bundled fallback fonts. All measurements and font/config/source
hashes are in `artifacts/native_render_interface_audit_20261105/report.json`.

| Complete training layout | CMRC | Dolly |
| --- | ---: | ---: |
| Records | 3,211 | 12,070 |
| Prompts with unknown glyphs | 520 | 1 |
| Answers with unknown glyphs | 23 | 2 |
| Prompts with ink extending outside the vertical canvas | 3,211 | 10,835 |
| Answers with ink extending outside the vertical canvas | 3,164 | 11,165 |
| Prompts exceeding complete-source horizontal capacity | 2,439 | 2,764 |
| Answers exceeding complete-source horizontal capacity | 0 | 1,923 |
| Both fields free of every measured issue | 0 | 109 |

These are conservative Pango ink/layout checks, not language scores. The first
fixed Chinese answer has an ink top of -3 pixels; its actual native PNG clips
upper strokes, while the canonical 26-pixel cell render preserves them. The
renderer's bounded separator calculation can hide long-source clipping, so
full uncropped layout width must be checked before rendering. Sample evidence
remains under the audit directory. Individual selected fallback font runs are
not inspected because the borrowed PyGObject font accessor crashes in this
environment; registered family names are not presented as selected fonts.

This does not explain V61's failure: V61 and V62 use the separate canonical
Noto interface. It rules out blindly adopting the supplied font configuration
as a data repair. Matching a released configuration is insufficient when it
clips or lacks source writing. A future renderer must preserve full geometry,
verify unknown glyphs and reject overflow explicitly.

A post-audit admission diagnostic also confirms a separate host-interface
limitation: the frozen V25 cell helper rejects ordinary spaces before drawing.
Its deliberate Chinese-stream policy is unsuitable for broad English
instructions, even where the font contains the letters. The source files
remain frozen, and `legacy_adapter_admission.json` records training-only
counts. A new adapter should preserve spaces and supported punctuation without
changing V62's prepared images or calling text models at inference.

The audit is reproducible in the existing Python 3.12 environment with
`CUDA_VISIBLE_DEVICES=''`, the documented system GLib/GObject/GIO preloads,
and `scripts/audit_native_render_interface.py`. Automatic dependency installers
are guarded out. This result changes the data-interface plan, not a completed
language qualification decision.
