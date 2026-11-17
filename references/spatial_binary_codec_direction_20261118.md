# Test spatial structure at the unchanged 64-bit plan budget

The [equal-form comparison](balanced_codec_comparison_result_20261118.md)
improves training macro reconstruction but leaves validation F1 .923484
continuous and .820592 binary, with zero exact binary validation images. The
[saved-code analysis](evidence/binary_plan_information_20261118.json) already
rules out a collision floor for the original final codes on this finite support.
Equal exposure is not a sufficient repair. Do not extend either completed fit.

The next architectural hypothesis preserves spatial structure through the
bottleneck. Replace the global 2048-to-64 encoder projection and 64-to-2048
decoder expansion with shared pointwise maps, 128-to-4 and 4-to-128, on the
existing 4-by-4 field. The plan is four features at sixteen fixed sites: still
64 continuous values or 64 hard bits. Serialize sites in raster order with
four channels per site. The same bit decoder, XOR correction and normalized
joint image law remain applicable. No glyph identity, inventory, font, image
bank or larger plan is added.

This gives the drawing network a fixed spatial relationship between plan sites
and its convolutional field. It may make parts and strokes easier to share
across forms. It does not guarantee semantic geometry, exact translation
equivariance, faithful unseen glyphs or easier conditional language learning.
The number of projection parameters also changes, so compare this architecture
as a whole rather than attribute effects uniquely to locality. There are
1,767,365 codec parameters, versus 2,030,465 globally; all 1,766,209 shared
convolutional parameters retain the same original random initial bytes.
The common convolutions are untrained initialization, not inherited knowledge.

Four CPU checks qualify the complete site/bit ordering, shared initial tensors,
real continuous/binary gradient updates and exact resets, and compatibility with
target-free joint generation. This remains untrained code. Compare a new matched
spatial continuous/binary pair under the same equal-form source order, loss,
6000 updates, schedule, precision and resource ceilings. Reuse the completed
global equal-form controls without rerunning them. Keep every absolute fidelity,
macro exactness, empty-image and refinement-event condition, plus .02 gain over
the saved equal-form global binary control. The physical partition remains
reused development. Failure rejects this fixed architecture/recipe; success
only permits a separately registered conditional-image comparison. Full language,
stopping, historical forms, untouched confirmation and efficiency remain unproved.
