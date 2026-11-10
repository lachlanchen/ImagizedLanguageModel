# Spatial motor draws clean forms but fails uncertain conditions

The3,892,737-parameter spatial motor generates known clean-condition glyphs
with mean foreground F1 **0.998260**: all4,087 forms exceed0.95F1 and2,314
are pixel-exact. This is a working known-form image generator, trained on one
RTX4090D in **136.35s**, with **0.889GiB** peak allocation. The pretrained
language body is not loaded during this motor fit; its acquisition/pretraining
is excluded from these costs.

The same model still fails the combined prospective gate. Fixed55/45semantic
mixtures give mean selected F1 **0.810537**, only1.590%of forms exceed0.95F1,
and one is pixel-exact. Oracle-best candidate F1 is0.823132, versus0.998457
on clean conditions; references never select runtime output. Qualification
takes84.32s. Both fixed64-triplet sheets were actually viewed. Clean forms
are sharp; mixtures still yield damaged forms or another shape. Every candidate
and selected image is retained. No full language inference uses this motor,
and the original date-capable checkpoint is unchanged.

These tests establish a limited capability and a boundary: known confident
conditions can generate glyph images, but uncertain semantic vectors are not
a reliable route to the intended form. More rendering architectures alone do
not recover the lost identity. The next experiment should target G using real
image-conditioned states with broad glyph coverage, preserving the existing
language cohort. Synthetic class-forced hidden states did not transfer that
function. A copy curriculum can add genuine reader states without adding a
runtime class/answer bank; whether it transfers to QA remains to be measured.

[Prospective protocol](spatial_semantic_motor_protocol.md) ·
[All measured and sampled image evidence](evidence/spatial_semantic_motor_v1/)
