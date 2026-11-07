# A visual writing state must preserve whitespace progress

Pre-execution specification, 8 November 2026. This is an input-state correction
for a future native-vision rereading loop, not a replacement language result.
The completed prefix probe and state-alignment learner remain unchanged.

A page containing only ink can lose the difference between a prefix and the
same prefix followed by an emitted white tile. For example, the fixed source
renderer may produce exactly the same complete padded pixels for `春` and
`春 `, while copying `春 风` requires different next emissions: a white space
and `风`. Verify this actual renderer case, including the declared observer
resize. Do not assume all sources or every original failed model have this
problem: the earlier image-history models retain sequence positions. The
issue applies to the proposed reconstruction of state from prefix ink alone.

If two equiprobable histories share the same observation but require distinct
next images A and B, any image-conditioned distribution q has
`(q(A|X)+q(B|X))/2 <= 1/2`. A deterministic next-image function also cannot be
correct on both. This elementary bound is about an aliased interface, not an
impossibility theorem for visual language learning. More training cannot
recover information deliberately removed from the observation.

Represent self-generated history as actual ordered 32-by-32 uint8 image tiles.
Each physical row has a 32-pixel writing band and a separate 32-pixel gutter.
Mark the **next** writable cell in that gutter, using a fixed black rectangle.
The marker encodes the already-emitted position, not a target length, language
symbol, reference or glyph identity. A white emission advances the marker.
No marker overwrites output pixels. Arbitrary written forms remain image
payloads; there is no font, tokenizer, OCR or character lookup in the runtime
state class. The image observer may consume the complete history page directly.

Implement immutable append, lossless page construction and inverse recovery
of the complete ordered payload and emitted length from the cursor pixels.
Test actual gray-valued image payloads at empty, single-tile, line-boundary and
longer histories through 4,096 tiles. Verify all source bytes and ordering,
including arbitrary all-white tiles. Retain actual pages and quantitative
hashes. Invalid/ambiguous gutter or nonwhite future cells must be rejected.

For 257 all-white histories of lengths 0..256, build a question/history page
and apply the existing native image-budget transform. Verify pairwise-distinct
observer inputs and exact inverse recovery before resize. This bounded check
does not prove that a learned encoder recognizes the marker, that all larger
resized histories remain distinct, or that any language can be generated.
The cursor semantics need a separately evaluated learned observer/writer.

Run on CPU only, four threads, after the current source/model jobs finish.
No optimizer, neural forward, source output projection or weight download.
Use the actual fixed renderer for the counterexample and supplied raster
payloads for the font-free runtime. Keep first-copy/source evidence separate
from generated ILM answers. The full goal remains active and unachieved.
