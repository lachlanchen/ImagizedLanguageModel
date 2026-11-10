# Frozen-model date intervention

The initial glyph-language assessment contains readable complete `288条` and
`1874年`, with imperfect pixels, and a pixel-exact `2001年`. The latter full
string occurs in training. Most other development streams fail. This is
limited progress, not a useful general model. Test whether changing passage
facts changes the generated date without any further optimization.

Use the two previously observed year-question parents, Chinese development
indices 12 and 15. Their selection is explicitly based on current outputs;
this is not a sealed or randomly sampled generalization test. Fix all new
year values before inference: 1874, 1936, 1648, 1789, 2001, 2007, 2019 and
1993. For each parent, replace every occurrence of its answer year in the
passage, keeping the question unchanged. These altered passages are
counterfactual reading conditions, not claims about historical dates.

Reproduce each original normal PNG exactly with its authenticated existing
host renderer before preparing variants. Preserve original page wrapping,
verify unchanged strip width and safe blank-column boundaries, and verify
unchanged question ink. Erase the same original question-ink mask for each
control. Retain all 32 prepared images: sixteen normal and sixteen erased,
including the two original baselines. There are fourteen new normal date
interventions. Font and text are used only by CPU input preparation.

Run the unchanged completed glyph-language checkpoint, native visual body,
continuous image encoder/decoder/readout and learned stopping. No source
vocabulary projection, text decoding, font output, target length, answer lookup
or external teacher may enter inference. Each actual generated image is
re-encoded before continuation. Keep the existing uniform 64-glyph cap,
deterministic native computation and bounded exclusive GPU0 admission. Cap
the complete stage at 1,200 seconds with the same RAM/pressure monitoring.
Perform zero updates and retain every raw tensor and PNG, including failures.

Inspect all generated images in preparation order before comparing to the
requested year. Then report exact canonical-pixel matches separately from
visually legible dates, counts of stale original-year answers, wrong dates,
malformed images and caps. Compare each intervention's erased control.
Do not repair missing strokes, suppress a failed variant or extend training
after seeing this cohort. Retain source/output hashes and state the two-parent
selection limit. Successful tracking would support bounded input-dependent
date composition; it cannot establish open-ended language, English, historical
glyphs, a small foundation trained from scratch or an LLM efficiency advantage.

Preparation uses the existing Python3.12 native Pango renderer environment
with the documented GLib/GObject/GIO preload. Frozen-model inference uses the
same deterministic native runtime as the initial glyph-language assessment.
