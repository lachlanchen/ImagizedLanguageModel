# Separate semantic selection from glyph formation

The original complete-glyph checkpoint remains the only language checkpoint
with a verified restricted changed-date result. Both new broad readout
compilers are rejected before language inference. The contrastive variant
keeps the first compiler's architecture, initialization, states and6,000-update
schedule, adding a cosine-anchor discrimination loss. It gives mean pixel F1
**0.489500 on4,087 held-out synthetic states** and **0.795106 on774 genuine
states**, versus0.522572/0.829770 without that loss. Genuine pixel-exact glyphs
fall from180 to115. Its final training batch has78.125% nearest-anchor top-1,
which is neither held-out class accuracy nor successful physical rendering.
The run takes37.80s. All9,722 physical measurements cold-replay bit-exactly;
all held-out pixels and fixed64-triplet sheets are retained. Both sheets were
actually viewed and show damaged/composite glyphs. No neural language job uses
this rejected initializer.

The genuine histories are held out only from these compiler updates. The
original checkpoint previously trained on all508questions; this is retention,
not independently held-out language generalization.

A separate fixed-weight check uses each saved initial state from the original
forty language streams. Recomputing G and D reproduces every saved nonempty
first image exactly. For normal Chinese questions, evaluator-only cosine
ranking against the4,087 training anchors puts an allowed reference first
form at rank1 in **9/16** cases and within the first three in **11/16**. Only
**1/16** first images is pixel-exact. For example, the target first forms上,
亚,8and因rank first but have physical foreground F1 of roughly0.60–0.64.
These ranks do not repair output images or prove that full answers would be
correct. Seven cases still have the wrong highest-ranked reference form.

The same original D yields mean F1 **0.999941** under independent Gaussian
noise sigma0.03, **0.995495** at sigma0.3 and **0.962739** at sigma0.6 across
all4,087 anchors. A fixed30% second-anchor mixture gives **0.942542**; at45%
it falls to **0.758376**, with no exact glyph. These initial mixtures use raw
normalized-coordinate anchor magnitudes, which vary; they are a stress test,
not guaranteed dominant-class labels. The next prospective training/assessment
uses unit-RMS directions when defining a dominant mixture.

This localizes an actionable part of the failure: a roughly correct semantic
direction can still decode into damaged pixels. It does not isolate every
language failure or guarantee that decoder training will solve broad QA.
The next [decoder-only experiment](robust_glyph_decoder_protocol.md) freezes
G,E,stop and the native contextual body, and tests noise/mixture robustness
using only canonical anchor/image pairs. If its rendering gate passes,
autonomous language inference must establish whether better pixels also
improve subsequent actual-image feedback and complete answers.

[First compiler result](semantic_readout_distillation_result.md) ·
[Contrastive protocol](contrastive_glyph_readout_protocol.md) ·
[Contrastive evidence](evidence/contrastive_glyph_readout_v1/) ·
[Fixed-state geometry and decoder stress measurements](evidence/semantic_glyph_geometry/)
