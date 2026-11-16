# Complete shaped pages pass the fixed source-support check

The new offline page renderer passes all 64 complete fields from the 16
registered training-source rows. All seven controls pass. Native shaping reports
zero unknown glyphs; every page equals its expanded reference crop with no
outside ink; and both RGB and binary tile streams reconstruct exactly. This
qualifies the bounded source-support implementation, not the complete corpus,
binary legibility, preference labels, a student model or language generation.
[Protocol](shaped_preference_page_protocol_20261116.md),
[registration](evidence/shaped_preference_page_registration_20261116.json),
[result and independent checks](evidence/shaped_preference_page_result_20261116.json).

The implementation is in
[`shaped_page_renderer.py`](../ilm/training_only/shaped_page_renderer.py), with
the executable check in
[`preflight_shaped_preference_pages.py`](../scripts/preflight_shaped_preference_pages.py).
It loads only the existing native host bindings, without importing the neural
visual-package initializer. Pango1.52.1 shapes complete text with its original
paragraphs, combining clusters, tabs and direction. Page height follows full
ink/logical bounds; the font remains26pixels. Unknown forms or bounds violations
are rejected before an image is admitted. Source strings are not normalized,
repaired, elided or truncated.

The cases were selected before rendering from prospective training rows only.
All18selection roles remain represented after deduplication: observed separator,
combining and format cases; Arabic, Devanagari and emoji; joiners and replacement
characters; longest prompt/chosen/rejected; the greatest variety of previously
unsupported forms; and the most line feeds. Every selected row supplies all four
fields, including empty system fields. This does not inspect or qualify every
row of the10,735-pair collection.

Eleven actual selected font files are authenticated against native serialized
font checksums, including the existing CJK, Devanagari, Tamil, Grantha, Sinhala,
Oriya, Takri, Gujarati, Latin/Arabic-capable and emoji fonts. The serialized
format is version-bound debugging evidence. It is not used as a permanent
training representation. No font files or shared libraries are installed,
copied or changed.

The physical codec splits an RGB or grayscale uint8 page into row-major32-by-32
patches and reconstructs the original bytes. A patch can span parts of several
characters; it is not a character ID. The maximum observed stream has8,128patches
for the most-line-feeds prompt. The longest chosen answer uses7,392patches.
These are larger than the current glyph-stream training examples. Neither
rendering nor lossless tiling proves that the current actor supports these
histories or that a consumer GPU can learn them efficiently. A later comparison
must bind geometry, actual-body support and cost; existing caps cannot silently
stand for the new pages.

An independent check reconstructs all64RGB and binary arrays by indexed tile
assignment, without using the renderer's inverse. It verifies all PNG, tile,
layout, input and actual-font hashes, and every original layout-text hash.
All64page layout overviews and six detailed source images were actually viewed.
This is labelled source-preparation review, not blind semantic adjudication or
a claim that every complete page has been read and judged legible.

## Binary conversion remains a failed full-source importer

The renderer retains full RGB pixels and separately exports the old arithmetic
RGB-mean threshold128. In source row139's chosen answer, the moon/star band at
coordinates `[9,48,71,78)` contains1,021nonwhite RGB pixels and zero black pixels
after thresholding. The source marks become white. This is a post-preflight
descriptive observation; no registered RGB gate changes.

An initial chromatic bounding rectangle also included surrounding antialiased
text, and its zero-black assertion failed. Preserve that failed check. The
subsequent record includes both contiguous nonwhite bands in the first100rows,
their exact coordinates and both image hashes. Selection followed visual
inspection; it is not a preregistered or blinded legibility score. The result
does not establish that every colored symbol disappears, and the binary
derivative is not promoted as faithful complete-source preparation.

The source replacement characters remain visible as replacement characters.
Faithful rendering cannot recover their intended missing content or validate
the machine-derived preference labels. RGB geometry, exact pixel transport,
binary legibility, semantic correctness and learned language remain separate.

## Execution scope and decision

The source preflight takes82.472seconds at0.609GiB peak process RSS. The separate
export verification takes1.753seconds at0.182GiB. The preflight timer begins
after module imports. Earlier environment probes include one missing-foreign-
Cairo error and two isolated native crashes; their elapsed time and peak RSS
were not captured and are excluded. The successful path uses native layout
serialization rather than unsafe glyph-run object traversal. All probe failures
remain recorded, and subsequent source execution disables its own core dumps.

There are zero model calls, optimizer updates, GPU allocations or neural imports.
The original training owner, all115bound sources, existing data, old outputs
and frozen readings remain unchanged. No source sample is an ILM generation.

Preserve the passed RGB support result and the negative binary observation.
Do not repeat the unchanged64-field preflight, repair its source or train its
unqualified derivative. Any full-corpus preparation needs a separate registered
decision, complete-image/label qualification and an actual-body resource check.
The current serial white-target trial and its unchanged final language decision
retain priority. No preference fit or usable-language claim follows here.
