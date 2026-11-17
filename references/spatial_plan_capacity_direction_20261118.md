# Test sixteen features per spatial site with declared plan cost

The [spatial 64-bit comparison](spatial_codec_comparison_result_20261118.md)
improves held-form binary F1 to .847344, but remains below every absolute
fidelity condition, with zero exact validation images. Its continuous counterpart
also fails at .929688. Keep that architectural improvement and failure together.
No unchanged extension or conditional language fit follows.

Test one capacity change within the same 4-by-4 geometry: increase four features
per site to sixteen. This gives 256 continuous values or hard bits. All CNN
channels, shared convolutional initial values, equal-form source order, loss,
optimizer schedule, 6,000 updates per arm and quality/resource conditions remain.
Only the pointwise bottleneck capacity changes. There are 1,770,449 parameters,
3,084 more than the spatial-64 codec; all 1,766,209 common CNN values stay equal.
The 64-value special case exactly reproduces the preceding implementation.

More features can reduce the complexity of reconstructing local stroke detail.
That is a learnability hypothesis, not an information-theoretic claim that
the previous exact codes lost source identities. The previous unique-code result
does not guarantee a simple, stable, generalizing decoder. Neither code width
nor a better image motor creates semantic knowledge.

The cost is explicit: the binary prior now has thirty-two eight-bit events,
versus eight. The physical correction remains lossless and can represent every
binary image. Retain the same total plan-plus-correction mean ceiling, .60 of
97.30248883216336 source-tree events; do not report the old eight-event overhead.
These event counts are not entropy bits or end-to-end language efficiency.

Four CPU checks qualify exact 64-value compatibility, all 256 feature/bit
coordinates, shared initialization, actual updates and exact reset in both
modes, declared thirty-two-event cost, and target-free joint generation. The
candidate is untrained. Its comparison uses the saved spatial-64 pair as unchanged
controls, without rerunning them, and requires .02 binary validation-F1 gain
over .8473440743579628 in addition to every original absolute gate. The physical
partition is reused development. Failure rejects this fixed capacity recipe;
success only permits a separately registered conditional writer test. Full
language, untouched confirmation, stopping, historical forms and whole-model
cost remain unresolved.
