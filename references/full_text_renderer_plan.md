# Complete Chinese/English prompt rendering

The training-only native-interface audit finds missing glyphs, canvas overflow
and clipped ink in the supplied release configuration. A separate post-audit
check finds that all 12,070 Dolly training prompts are rejected by the frozen
cell helper, principally because ordinary spaces are deliberately excluded.
The model's image-only boundary must accept complete English/Chinese writing
before that corpus can support broader language learning.

Implement a new host-only single-line renderer, leaving every frozen V61/V62
source and cache unchanged. Use existing system Noto CJK fonts, Pango/Cairo
shaping and the verified Python 3.12 bindings. Preserve ordinary spaces and
punctuation. Reject multiline/control input explicitly rather than silently
dropping it; callers may use the separately recorded flat dataset view.
Reject unknown glyphs, excessive height and full-source width overflow.
Center the complete measured ink bounds in a 32-pixel strip, preserve aspect
ratio/font size, and pad width to 64-pixel blocks. Never crop or shrink a
source to satisfy the capacity limit. Include a white frame to catch clipping.

Default to Noto Sans CJK JP at 24 physical pixels. A 26-pixel prototype rejects
ordinary mixed-script strings with descenders because their complete ink
height exceeds the protected 28-pixel area. Fix the default size before the
training-data audit rather than shrinking individual examples. Retain the
canonical family with proportional line layout and proper whitespace.
This is a new interface, not a claim of exact match to the release
or old cell rasters. Record font availability, primary file hash and render
configuration. Individual fallback selection is not inspected through the
unsafe borrowed font-run accessor. Return pixels and layout metadata, with
no model inference, OCR or text-model call.

Validate real bilingual strings, whitespace preservation, deterministic pixels,
white bounds and explicit rejection of unknown glyphs, controls and overflow.
Audit only the existing CMRC and Dolly training records; do not touch future
development/test questions or change the running matched density experiment.
Save first fixed prompt/answer examples without selection and record complete
source hashes. A later registered training protocol must evaluate any language
benefit; rendering completeness is not a learned-language result.

Cache complete accepted training images losslessly for a future experiment,
using at most 240 prompt blocks and 128 answer blocks, followed by two white
termination blocks. Keep every rejected ID/reason; never truncate a source or
modify the original partitions. The actor-facing cache contains physical bits
and the prompt's spatial boundary, without text or codepoint labels. This
candidate cache is not adopted by V62 and contains no development/test data.
