# Better random-noise rendering does not solve mixed glyphs

The decoder-only intervention fails its prospective primitive gate. Across all
4,087 forms, random-noise sigma0.6 mean F1 improves from **0.965706 to
0.992049**, with **99.657%** of forms reaching0.95 F1. Clean F1 remains
**0.999752**, compared with0.999940 before training, while exact glyphs fall
from4,048 to3,790. Hard unit-RMS55/45 mixtures improve from **0.761720 to
0.822751**, still far below the0.95 requirement, and no mixture is pixel-exact.
The full language assessment is therefore not run.

Training10,792,449decoder parameters takes26.60s and peaks at0.548GiB. The
reader, semantic readout, image encoder, affine buffers and stop predictor
remain bit-exact. Only canonical anchor/image pairs are used; no language
question or answer enters this fit. All three fixed64-triplet sheets were
actually viewed. Random-noise glyphs improve, but mixed-code outputs remain
composites with wrong or extra strokes. Every raw held-out glyph is retained.

A separate zero-update fixed-state diagnostic tests all forty saved initial
language states, without loading a body or generating full answers. Normal
Chinese first-image mean F1 rises from0.652553 to0.682402 with the new decoder;
exact first forms rise from1/16 to2/16, but the count withF1>=0.95 stays5/16.
Four anchored encoder/decoder passes give3/16exact with the original decoder,
but still5/16at0.95F1 and slightly lower meanF1. The new decoder plus four
passes also stays5/16at0.95F1. All160unmodified first-state outputs were
actually viewed on four sheets. This is no evidence of improved complete
language answers. The original date-capable checkpoint is unchanged.

Deterministic reconstruction of intermediate codes remains a plausible cause
of blended forms, alongside readout errors. These tests do not establish that
all glyph ambiguity is a simple two-anchor mixture. A conditional generative
motor can instead model a distribution over complete forms; its sampled image
must be checked physically, and any selection must use learned image features,
never a runtime character bank or reference answer. Such a model still needs
its own primitive and autonomous-language evidence.

[Prospective protocol](robust_glyph_decoder_protocol.md) ·
[All rendering evidence](evidence/robust_glyph_decoder_v1/) ·
[Fixed-state motor comparison](evidence/glyph_motor_states_v1/)
